"""Global synaptic-efficacy ageing hypothesis, default age zero is a no-op.

Every represented neuron's chemical and electrical output is assigned the
same efficacy q(day)=2**(-day/45). The 45-day half-life is ASSUMED, not fitted
or measured. This is a falsifiable global-weakening scenario, not an accepted
biological clock. Neuron identities, synapse counts, signs, intrinsic cell
parameters, imposed inputs, muscles and programmed flight support are fixed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from lanes.longevity.ageing import integrated_probe as IP
from lanes.longevity.ageing import escape_membrane as EM
from lanes.longevity.production import capture as CP

MODEL = "global-synaptic-efficacy-half45-v1"
DAYS = tuple(range(0, 91, 10))


def efficacy(day):
    if not np.isfinite(day) or not 0 <= day <= 90:
        raise ValueError("Declared age range is 0 through 90 model days")
    return float(2. ** (-float(day) / 45.))


def array_hash(x):
    a = np.ascontiguousarray(x)
    return hashlib.sha256(a.tobytes()).hexdigest()


def apply_to_network(net, day):
    """Apply after balancing so normalization cannot cancel the perturbation.

    W is post x pre, Wg contains graded-release columns, and G is the
    electrical adjacency. G and its degree must both change so a constant
    voltage field still has zero subthreshold gap current. W.data is also
    the fast scatter engine's _data view; mutate in place to preserve it.
    """
    q = efficacy(day)
    if net.N != len(net.ids) or len(set(map(str,net.ids))) != net.N:
        raise ValueError("Aging requires every canonical, unique neuron ID")
    assigned = np.full(net.N, q, dtype=np.float32)
    audit = {"model": MODEL, "age_days": float(day), "efficacy": q,
             "half_life_days": 45., "half_life_status": "ASSUMED; no day calibration",
             "assigned_neurons": int(net.N), "neuron_ids_sha256": IP.identity(list(map(str,net.ids))),
             "per_neuron_efficacy_sha256": array_hash(assigned),
             "neurons_with_chemical_outputs": int(np.count_nonzero(np.diff(net.W.indptr))),
             "chemical_stored_edges": int(net.W.nnz),
             "chemical_positive_weights": int(np.count_nonzero(net.W.data > 0)),
             "chemical_negative_weights": int(np.count_nonzero(net.W.data < 0)),
             "matrices": {}, "age_zero_noop": day == 0}
    for name in ("W", "Wg", "G"):
        matrix = getattr(net,name,None)
        if matrix is None:
            audit["matrices"][name] = None
            continue
        before = array_hash(matrix.data)
        before_sum = float(np.abs(matrix.data).sum(dtype=np.float64))
        if day != 0:
            matrix.data *= np.float32(q)
        audit["matrices"][name] = {"before_sha256": before, "after_sha256": array_hash(matrix.data),
            "stored_weights": int(matrix.nnz), "absolute_weight_sum_before": before_sum,
            "absolute_weight_sum_after": float(np.abs(matrix.data).sum(dtype=np.float64))}
    if getattr(net,"G",None) is not None and day != 0:
        net.gap_deg *= np.float32(q)
        net.gap_pair_weights = net.gap_pair_weights * np.float32(q)
    if not np.shares_memory(net.W.data,net._data):
        raise ValueError("Aged matrix disconnected from the fast scatter weights")
    if day == 0 and any(v and v["before_sha256"] != v["after_sha256"] for v in audit["matrices"].values()):
        raise AssertionError("Age zero changed the model")
    audit["interpretation"] = "Global reduction of represented synaptic efficacy. Does not delete neurons or model mortality, compensation, molecular aging or type-specific vulnerability."
    audit["sha256"] = IP.identity(audit)
    return audit


def run(candidate, day, protocol, duration_ms):
    original_init = IP.JA.cns.CNS.__init__
    records = []

    def aged_init(net,*args,**kwargs):
        original_init(net,*args,**kwargs)
        if net.N != 188508:
            raise ValueError(f"Expected full 188508-neuron model, got {net.N}")
        records.append(apply_to_network(net,day))

    with IP.patched(IP.JA.cns.CNS,"__init__",aged_init):
        result = EM.run(candidate,protocol=protocol,duration_ms=duration_ms)
    if len(records) != 1:
        raise RuntimeError(f"Expected one full network, got {len(records)}")
    result["global_ageing"] = records[0]
    result["ageing_runtime_identity"] = IP.identity({"baseline_runtime_theta":result["runtime_theta_sha256"],
        "aging":records[0]["sha256"]})
    return result


def capture(candidate_path,day,protocol,duration_ms,out):
    candidate = json.loads(Path(candidate_path).read_text())
    original_run = IP.JA.run
    completed = []

    def captured_run(*args,**kwargs):
        with IP.patched(IP.JA,"run",original_run):
            result = run(candidate,day,protocol,duration_ms)
        completed.append(result)
        return result["result"]

    # Use CP's escape entry point for all protocols; EM selects quiet walking
    # internally. This avoids the historical treatment walk/rest schedules.
    args=SimpleNamespace(out=Path(out),behavior="escape",arm="control",seed=0,
                         duration_ms=duration_ms,onset_ms=600.,land_s=2.5)
    with IP.patched(IP.JA,"run",captured_run):
        CP.run(args)
    if len(completed)!=1:
        raise RuntimeError("Expected one captured integrated run")
    result=completed[0]
    receipt=args.out/"integrated-receipt.json"
    receipt.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    manifest_path=args.out/"manifest.json"
    manifest=json.loads(manifest_path.read_text())
    manifest.update(behavior="walking" if protocol=="walk" else "escape",age_days=day,
        candidate=result["candidate"],candidate_sha256=result["candidate_sha256"],
        runtime_theta_sha256=result["runtime_theta_sha256"],global_ageing=result["global_ageing"],
        ageing_runtime_identity=result["ageing_runtime_identity"],integrated_protocol=result["protocol"],
        stimulus=result.get("sensory_input",result["result"]["stim"]),
        integrated_receipt_sha256=IP.sha(receipt),review_required=result["review_required"],
        limits=result["limitations"]+[result["global_ageing"]["interpretation"]])
    manifest_path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    if result["review_required"]:
        raise RuntimeError("Complete result saved, but source/authority review required")
    print(json.dumps({"completed_utc":datetime.now(timezone.utc).isoformat(),"age_days":day,
        "protocol":protocol,"out":str(out),"efficacy":efficacy(day),
        "aging_neurons":result["global_ageing"]["assigned_neurons"],
        "native_gf_ms":result["result"]["gf_spikes_ms"],
        "takeoff_ms":result["result"]["t_off_ms"],"timing_s":result["timing_s"]}),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate",type=Path,required=True)
    ap.add_argument("--day",type=int,choices=DAYS,required=True)
    ap.add_argument("--protocol",choices=("walk","puff","sham","visual"),required=True)
    ap.add_argument("--duration-ms",type=float,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    capture(a.candidate,a.day,a.protocol,a.duration_ms,a.out)
