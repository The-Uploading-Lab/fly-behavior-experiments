"""Record numerical states for the fixed, paired development body comparison."""
from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from lanes.longevity.ageing import escape_membrane as EM
from lanes.longevity.ageing import integrated_probe as IP
from capture_delivered import extra_parity, walking_receipt
from lanes.longevity.ageing.capture_candidate import parity
from read_body import METRICS, escape_readout, walking_readout, validate_state

MODELS = {"original": "candidate-steady.json", "gain105": "candidate-antennal-gain105.json"}
REFERENCES = {
    ("original", "puff"): "integrated-2026-09-13-0002/steady-puff-450.json",
    ("original", "walk"): "integrated-2026-09-13-0020/baseline-walk.json",
    ("gain105", "puff"): "integrated-2026-09-13-0036/antennal-gain105-puff.json",
    ("gain105", "walk"): "integrated-2026-09-13-0036/antennal-gain105-walk.json",
}


def verify_delivery():
    path = HERE / "delivery-antennal-gain105.json"
    delivery = json.loads(path.read_text())
    for entry in delivery["files"]:
        if IP.sha(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError(f"delivered source changed: {entry['path']}")
    return IP.sha(path)


def readout(integrated, state, protocol):
    state = validate_state(state)
    result = integrated["result"]
    row = (walking_readout(walking_receipt(integrated), state) if protocol == "walk"
           else escape_readout(result, state, 2.5))
    row.update(seed=integrated["seed"], model_id=integrated["candidate"]["id"],
               candidate_sha256=integrated["candidate_sha256"],
               runtime_theta_sha256=integrated["runtime_theta_sha256"],
               protocol=protocol, recorded_gf_active_bins_ms=result["gf_spikes_ms"],
               gf_event_list_capped_at=10, ttm_spikes_by_side_ms=integrated["ttm_spikes_by_side_ms"],
               flight_clear_fraction=integrated["flight_clear_fraction"],
               flight_minimum_upright=integrated["flight_min_upright"],
               touchdown_ms=integrated["touchdown_ms"],
               internal_body=result["walking_body"],
               internal_walking_regression_raw=integrated.get("walking_regression"),
               body_metric_scope="movement describes displayed winged body; internal body metrics separate")
    if protocol == "walk":
        row["schedule_is_imposed"] = False
        regression = integrated["walking_regression"]
        row["internal_standing_pass"] = bool(regression["standing"] or regression["standing_v32"])
        row["cohort_retention_pass"] = row["retention_pass"] and row["internal_standing_pass"]
        row["internal_gait_interpretable"] = row["cohort_retention_pass"]
        if not row["cohort_retention_pass"]:
            row["censor_reason"] = "failed_standing_in_at_least_one_body"
            for key in METRICS:
                row[key] = None
            row["windows"], row["bouts"], row["sensitivity"] = [], [], []
    else:
        flight_ok = not row["body_takeoff"] or (
            integrated["flight_clear_fraction"] is not None and integrated["flight_clear_fraction"] >= .95)
        row["cohort_retention_pass"] = bool(row["eligible"] and not row["inversion_after_input"]
                                             and row["upright_supported_finish"] and flight_ok)
    return row


def seeded_call(original_run, chosen_seed):
    if chosen_seed not in (0, 1, 2, 3):
        raise ValueError("only existing demo seeds 0–3 are declared")

    def call(candidate, protocol, seed=0, duration_ms=3000.):
        if seed != 0:
            raise ValueError("unexpected seed already supplied by the delivered wrapper")
        return original_run(candidate, protocol, seed=chosen_seed, duration_ms=duration_ms)
    return call


def parameter_fingerprint(params):
    def describe(value):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError("object-valued neural parameter")
            return {"dtype": value.dtype.str, "shape": list(value.shape),
                    "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (list, tuple)):
            return [describe(v) for v in value]
        if isinstance(value, dict):
            return {str(k): describe(v) for k, v in value.items()}
        return value
    described = {field.name: describe(getattr(params, field.name)) for field in fields(params)}
    return {"sha256": IP.identity(described), "fields": described,
            "scope": "all live SimParams fields at native CNS.run entry; array values hashed by dtype, shape and bytes"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--protocol", choices=("walk", "puff"), required=True)
    ap.add_argument("--seed", type=int, choices=(0, 1, 2, 3), required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if Path.cwd().resolve() != ROOT or args.out.exists():
        raise ValueError("run from the checkout with a new output directory")
    delivery_sha = verify_delivery()
    candidate_path = ROOT / "lanes/longevity/ageing" / MODELS[args.model]
    candidate = json.loads(candidate_path.read_text())
    before = IP.source_hashes()
    bodies = []
    original_body = IP.JA.ColourBody
    original_neural_run = IP.JA.cns.CNS.run
    runtime_parameters = []

    def observed_neural_run(net, *a, **kw):
        runtime_parameters.append(parameter_fingerprint(net.p))
        return original_neural_run(net, *a, **kw)

    class ObservedBody(original_body):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            bodies.append(self)

    duration = 6000. if args.protocol == "walk" else 3000.
    with IP.patched(IP.JA, "ColourBody", ObservedBody), \
            IP.patched(IP, "run", seeded_call(IP.run, args.seed)), \
            IP.patched(IP.JA.cns.CNS, "run", observed_neural_run):
        integrated = EM.run(candidate, protocol=args.protocol, duration_ms=duration)
    if len(bodies) != 1 or len(runtime_parameters) != 1 or integrated["seed"] != args.seed:
        raise RuntimeError("body/neural run count or actual simulation seed differs")
    state = validate_state(np.asarray(bodies[0].rows))
    row = readout(integrated, state, args.protocol)
    seed0_parity = None
    if args.seed == 0:
        reference = ROOT / "lanes/longevity/ageing" / REFERENCES[args.model, args.protocol]
        seed0_parity = extra_parity(integrated, json.loads(reference.read_text()), parity)
        seed0_parity.update(reference=str(reference), reference_sha256=IP.sha(reference))
    changed = [path for path, h in before.items() if IP.sha(ROOT / path) != h]
    drift = bool(changed or integrated["review_required"] or verify_delivery() != delivery_sha)
    args.out.mkdir(parents=True)
    (args.out / "integrated-receipt.json").write_text(json.dumps(integrated, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.out / "body-state.npz", body_state=state)
    (args.out / "readout.json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
    manifest = {
        "schema": "longevity-matched-body-numerical-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "protocol": args.protocol, "seed": args.seed,
        "seed_scope": "reused original demo seed; no independent validation or animal replication",
        "candidate": candidate, "candidate_file_sha256": IP.sha(candidate_path),
        "delivery_manifest_sha256": delivery_sha, "declaration_sha256": IP.sha(HERE / "MATCHED-BODY-COHORT.md"),
        "reader_correction_sha256": IP.sha(HERE / "MATCHED-READER-CORRECTION.md"),
        "source_sha256": IP.source_hashes(), "changed_sources": changed,
        "full_runtime_parameters": runtime_parameters[0],
        "files": {p.name: IP.sha(p) for p in sorted(args.out.iterdir())},
        "seed0_parity": seed0_parity, "runtime_review_required": drift,
        "cohort_retention_pass": row["cohort_retention_pass"],
        "scope": "new numerical development replay; not a reviewed candidate film",
        "command": [sys.executable, *sys.argv],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    if drift or (seed0_parity is not None and not seed0_parity["exact"]):
        raise RuntimeError("numerical result saved but requires runtime/parity review before continuing")
    print(json.dumps({"out": str(args.out), "retention_pass": row["cohort_retention_pass"],
                      "seed0_parity": seed0_parity, "timing_s": integrated["timing_s"]}), flush=True)


if __name__ == "__main__":
    main()
