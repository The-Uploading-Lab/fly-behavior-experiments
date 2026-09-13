"""Apply a paired antennal pulse/sham to the same full-CNS candidate.

The first lane's visual stimulus is disabled. Actual JO-A/JO-B spikes enter
the graph; the giant fibre is not externally driven. Both declared rates
are uncalibrated to physical tap strength and do not model an LC4 treatment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lanes.longevity.ageing import integrated_probe as IP
from lanes.longevity.ageing.threat_input import AntennalPulse


def run(candidate, *, sham=False, duration_ms=3000.0, seed=0, rate_hz=60.0):
    original_run = IP.JA.cns.CNS.run
    original_evaluate = IP.W.evaluate
    state = {"pulse": None, "pre_spikes": 0, "during_spikes": 0, "post_spikes": 0}

    def evaluate(cache, meta, edges, theta, **kwargs):
        state["pulse"] = AntennalPulse(meta, sham=sham, rate_hz=rate_hz)
        if theta.get("force_spikes_ms"):
            raise RuntimeError("Antennal protocol refuses externally forced spikes")
        return original_evaluate(cache, meta, edges, theta, **kwargs)

    def neural_run(net, *args, **kwargs):
        pulse = state["pulse"]
        original_feedback = np.asarray(kwargs["feedback_rows"], dtype=np.int64)
        extra = pulse.rows[~np.isin(pulse.rows, original_feedback)]
        all_feedback = np.concatenate((original_feedback, extra))
        positions = np.array([int(np.flatnonzero(all_feedback == row)[0]) for row in pulse.rows])
        drive = kwargs["drive_fn"]
        regularity = net.p.spike_reg_per_neuron
        reg = (np.full(net.N, float(net.p.spike_reg)) if regularity is None
               else np.asarray(regularity, dtype=float).copy())
        reg[pulse.rows] = 1.0
        net.p.spike_reg_per_neuron = reg

        def drive_puff(t_ms, feedback):
            if feedback is not None:
                key = ("pre_spikes" if t_ms < pulse.onset_ms else
                       "during_spikes" if t_ms < pulse.onset_ms + pulse.duration_ms else "post_spikes")
                state[key] += int(np.asarray(feedback)[positions].sum())
            idx, rates = drive(t_ms, None if feedback is None else feedback[:len(original_feedback)])
            return pulse.apply(t_ms, idx, rates)

        kwargs["feedback_rows"], kwargs["drive_fn"] = all_feedback, drive_puff
        try:
            return original_run(net, *args, **kwargs)
        finally:
            net.p.spike_reg_per_neuron = regularity

    with IP.patched(IP.JA, "LIGHTOFF_HZ", 0.0), \
            IP.patched(IP.W, "evaluate", evaluate), \
            IP.patched(IP.JA.cns.CNS, "run", neural_run):
        result = IP.run(candidate, "escape", seed=seed, duration_ms=duration_ms)
    result["protocol"] = "antennal-sham" if sham else "antennal-puff"
    result["sensory_input"] = state["pulse"].describe()
    result["sensory_input"]["regularity_shape"] = 1.0
    result["sensory_spikes"] = {k: v for k, v in state.items() if k != "pulse"}
    result["result"]["disabled_visual_stimulus"] = result["result"]["stim"]
    result["result"]["stim"] = {k: v for k, v in result["sensory_input"].items() if k != "banc_888_ids"}
    result["limitations"].append("antennal rate is an assumed input, not calibrated to the filmed tap")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--sham", action="store_true")
    parser.add_argument("--rate-hz", type=float, choices=(60., 450.), default=60.)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = run(json.loads(args.candidate.read_text()), sham=args.sham, rate_hz=args.rate_hz)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": str(args.out), "result": result["result"],
                      "sensory_spikes": result["sensory_spikes"],
                      "flight_min_upright": result["flight_min_upright"]}, allow_nan=False), flush=True)
    if result["review_required"]:
        raise SystemExit("Runtime or authority advanced: completed result saved for review")
