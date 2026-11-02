"""Verify a completed control experiment and report paired descriptive changes."""
from pathlib import Path
import argparse
import hashlib
import json
import statistics

import numpy as np


def analyse(path):
    summary = json.loads((path / "summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError(f"Experiment is {summary['status']}; preserve partial rows without treating it as complete")
    rows = summary["rows"]
    seeds, null_seeds = summary["noise_seeds"], summary["null_seeds"]
    fixed = ["base", "stiffness-replay", "stiffness-mean", "activation", "delay-uniform", "inhibition-unscaled", "motor-zero"]
    expected = {(m, s, None) for m in fixed for s in seeds}
    expected |= {(m, s, ns) for m in ("stiffness-shift", "delay-shuffle") for s in seeds for ns in null_seeds}
    actual = {(r["mode"], r["seed"], r["null_seed"]) for r in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError("Missing or duplicated planned trial")
    for r in rows:
        case = path / r["case"]
        if json.loads((case / "result.json").read_text()) != r:
            raise ValueError(f"Changed result record: {case}")
        for name, expected_hash in r["files"].items():
            if hashlib.sha256((case / name).read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"Changed output: {case / name}")
    bases = {r["seed"]: r for r in rows if r["mode"] == "base"}
    comparisons = []
    for r in rows:
        b = bases[r["seed"]]
        with np.load(path / r["case"] / "mechanics.npz", allow_pickle=False) as arrays:
            delays = arrays["delay_ms"]
            # Frozen September SimParams.dt is 0.1 ms. The engine uses this
            # exact rounding rule; the delay-vector mean is pre-quantization.
            slots_ms = np.maximum(1, np.rint(delays / .1)) * .1
            nominal_mean = float(delays.mean())
            delivered_mean = float(slots_ms.mean())
        comparisons.append({"case": r["case"], "mode": r["mode"], "noise_seed": r["seed"], "null_seed": r["null_seed"],
                            "standing": r["historical_metrics"]["standing_v32"],
                            "net_displacement_mm": r["mechanical_metrics"]["net_displacement_mm"],
                            "arm_minus_base_displacement_mm": r["mechanical_metrics"]["net_displacement_mm"] - b["mechanical_metrics"]["net_displacement_mm"],
                            "historical_speed_if_standing_mm_s": r["historical_metrics"].get("speed"),
                            "sustained_posture_loss_s": r["mechanical_metrics"]["sustained_posture_loss_s"],
                            "nominal_delay_mean_ms": nominal_mean,
                            "rounded_delivery_slot_mean_ms": delivered_mean})
    return {"kind": "paired conditional simulation sensitivity; not animal replication",
            "noise_blocks": len(seeds), "assignment_seeds": len(null_seeds), "trials": len(rows),
            "median_trial_wall_seconds": statistics.median(r["wall_seconds"] for r in rows),
            "inference": "Descriptive only; shared reference trials are not independent assignment replicates",
            "comparisons": comparisons}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    args = ap.parse_args()
    result = analyse(args.directory.resolve())
    print(json.dumps(result, indent=2, allow_nan=False))
