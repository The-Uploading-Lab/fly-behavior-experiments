"""Record a reviewed candidate through the first lane's existing capture path.

No renderer or second simulation is introduced. The production capture
instruments the same integrated run, then checks it against its selection
receipt before it can be used as matching footage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from lanes.longevity.ageing import escape_membrane as EM
from lanes.longevity.ageing import integrated_probe as IP
from lanes.longevity.production import capture as CP


def parity(actual, expected):
    differences = []
    for field in ("candidate_sha256", "runtime_theta_sha256", "seed", "protocol", "duration_ms"):
        if actual.get(field) != expected.get(field):
            differences.append(field)
    for field in ("gf_spikes_ms", "ttm_spike_ms", "t_off_ms", "z_final", "upright_final",
                  "winged_body_disp_mm", "walking_body"):
        if actual["result"].get(field) != expected["result"].get(field):
            differences.append("result."+field)
    for field in ("flight_min_upright", "flight_clear_fraction", "touchdown_ms", "walking_regression"):
        if actual.get(field) != expected.get(field):
            differences.append(field)
    return {"exact": not differences, "different_fields": differences,
            "scope": "same model and seed; scientific outputs compared, wall time excluded"}


def run(candidate_path, protocol, expected_path, out):
    candidate = json.loads(Path(candidate_path).read_text())
    expected = json.loads(Path(expected_path).read_text())
    if expected["review_required"] or expected["seed"] != 0:
        raise ValueError("Capture requires a reviewed development-seed-zero receipt")
    if CP.JA is not IP.JA:
        raise RuntimeError("Production and integrated runners imported different models")
    original_run = IP.JA.run
    completed = []

    def candidate_run(*args, **kwargs):
        # Production's caller instruments Body and evaluate. Restore only
        # this entry point while the integrated runner invokes it, avoiding
        # recursion and preserving production's pose and neural capture.
        with IP.patched(IP.JA, "run", original_run):
            result = EM.run(candidate, protocol=protocol,
                            duration_ms=expected["duration_ms"])
        completed.append(result)
        return result["result"]

    args = SimpleNamespace(out=Path(out), behavior="walking" if protocol == "walk" else "escape",
                           arm="control", seed=0, duration_ms=expected["duration_ms"],
                           onset_ms=600., land_s=2.5)
    with IP.patched(IP.JA, "run", candidate_run):
        CP.run(args)
    if len(completed) != 1:
        raise RuntimeError("Expected exactly one integrated simulation")
    actual = completed[0]
    comparison = parity(actual, expected)
    comparison.update(expected_receipt=str(expected_path), expected_sha256=IP.sha(expected_path))
    (args.out/"integrated-receipt.json").write_text(json.dumps(actual, indent=2, allow_nan=False)+"\n")
    (args.out/"parity.json").write_text(json.dumps(comparison, indent=2)+"\n")
    manifest_path = args.out/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(candidate=actual["candidate"], candidate_sha256=actual["candidate_sha256"],
                    runtime_theta_sha256=actual["runtime_theta_sha256"],
                    stimulus=actual.get("sensory_input", actual["result"]["stim"]),
                    integrated_protocol=actual["protocol"],
                    scientific_parity=comparison,
                    integrated_receipt_sha256=IP.sha(args.out/"integrated-receipt.json"),
                    review_required=actual["review_required"])
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    if actual["review_required"] or not comparison["exact"]:
        raise RuntimeError("Capture saved but does not pass candidate parity; do not present it as matching")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--protocol", choices=("puff", "visual", "sham", "walk"), required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.candidate, args.protocol, args.expected, args.out)
    print(json.dumps({"out": str(args.out), "scientific_parity": result["scientific_parity"]}))
