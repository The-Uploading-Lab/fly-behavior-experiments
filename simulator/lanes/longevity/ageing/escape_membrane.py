"""Read giant-fibre voltage in the full fly and test its adaptation hypothesis.

The isolated escape circuit has no subthreshold adaptation, while the
walking theta applies adapt_a=2 to every neuron, including both GFs.
An optional GF-only override tests that difference without changing the
rest of the network. It is a HYPOTHESIS, not a measured giant-fibre value.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lanes.longevity.ageing import integrated_probe as IP
from lanes.longevity.ageing import integrated_puff as PUFF
from lanes.longevity.ageing.fixed_inputs import FixedSensoryInputs, parameter_fingerprint


def run(candidate, *, protocol="puff", gf_adapt_a=None, duration_ms=800.):
    candidate = dict(candidate)
    if gf_adapt_a is not None:
        if gf_adapt_a not in (0., 1., 2.):
            raise ValueError("Use the declared GF adaptation values")
        candidate.update(id=candidate["id"]+f"-gf-adapt-a{gf_adapt_a:g}", gf_adapt_a=gf_adapt_a)
    original_run = IP.JA.cns.CNS.run
    diagnostic = {}

    def neural_run(net, *args, **kwargs):
        gf = np.flatnonzero(net.meta["cell_type"].astype(str).eq("DNp01"))
        if len(gf) != 2 or net.meta.iloc[gf]["banc_888_id"].nunique() != 2:
            raise ValueError("Expected the two canonical giant fibres")
        if kwargs.get("record_voltage") is not None:
            raise ValueError("Existing voltage recording must be preserved")
        before = net.p.adapt_a
        previous_regularity = net.p.spike_reg_per_neuron
        values = np.broadcast_to(np.asarray(before, dtype=float), (net.N,)).copy()
        if gf_adapt_a is not None:
            values[gf] = gf_adapt_a
            net.p.adapt_a = values
        kwargs["record_voltage"] = gf
        fixed_inputs = None
        if candidate.get("fixed_sensory_layout", False):
            if candidate["fixed_sensory_layout"] is not True:
                raise ValueError("fixed_sensory_layout must be a boolean")
            fixed_inputs = FixedSensoryInputs(net.meta)
            net.p.spike_reg_per_neuron = fixed_inputs.regularity(previous_regularity, net.p.spike_reg)
            previous_drive = kwargs["drive_fn"]

            def fixed_drive(t_ms, feedback):
                indices, rates = previous_drive(t_ms, feedback)
                return fixed_inputs.apply(indices, rates)

            kwargs["drive_fn"] = fixed_drive
        try:
            diagnostic.update(banc_888_ids=net.meta.iloc[gf]["banc_888_id"].astype(str).tolist(),
                              adapt_a=values[gf].tolist(),
                              adapt_b=np.broadcast_to(np.asarray(net.p.adapt_b), (net.N,))[gf].tolist(),
                              threshold_mv=np.broadcast_to(np.asarray(net.p.v_threshold), (net.N,))[gf].tolist(),
                              scope="GF-only adaptation hypothesis; raw native-tick voltage reader",
                              runtime_parameters=parameter_fingerprint(net.p))
            result = original_run(net, *args, **kwargs)
        finally:
            net.p.adapt_a = before
            net.p.spike_reg_per_neuron = previous_regularity
        diagnostic["fixed_input_layout"] = None if fixed_inputs is None else fixed_inputs.describe()
        voltage = np.asarray(result["voltage"])
        times = np.arange(len(voltage))*float(net.p.dt)
        for label, a, b in (("pre", 300., 550.), ("response", 600., 700.)):
            window = voltage[(times >= a) & (times < b)]
            diagnostic[label] = {"mean_mv": window.mean(axis=0).tolist(),
                                 "min_mv": window.min(axis=0).tolist(),
                                 "max_mv": window.max(axis=0).tolist()}
        diagnostic["voltage_trace_1ms"] = np.column_stack((times, voltage))[::round(1./net.p.dt)].tolist()
        return result

    with IP.patched(IP.JA.cns.CNS, "run", neural_run):
        result = (IP.run(candidate, "walk" if protocol == "walk" else "escape", duration_ms=duration_ms)
                  if protocol in ("visual", "walk") else
                  PUFF.run(candidate, sham=protocol == "sham", rate_hz=450., duration_ms=duration_ms))
    result["gf_membrane"] = diagnostic
    result["runtime_theta_sha256"] = IP.identity({"base_theta_sha256": result["theta_sha256"],
        "gf_adapt_a_by_id": dict(zip(diagnostic["banc_888_ids"], diagnostic["adapt_a"]))})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--protocol", choices=("puff", "sham", "visual", "walk"), required=True)
    parser.add_argument("--gf-adapt-a", type=float, choices=(0., 1., 2.))
    parser.add_argument("--duration-ms", type=float, default=800.)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = run(json.loads(args.candidate.read_text()), protocol=args.protocol,
                 gf_adapt_a=args.gf_adapt_a, duration_ms=args.duration_ms)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"out": str(args.out), "result": result["result"],
                      "gf_membrane": {k:v for k,v in result["gf_membrane"].items()
                                      if k not in ("voltage_trace_1ms", "runtime_parameters", "fixed_input_layout")},
                      "runtime_parameters_sha256": result["gf_membrane"]["runtime_parameters"]["sha256"]},
                     allow_nan=False), flush=True)
    if result["review_required"]:
        raise SystemExit("Runtime or authority advanced: completed result saved for review")
