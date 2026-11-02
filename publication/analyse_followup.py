"""Analyse the fixed six-block follow-up, preserving crossed null assignments."""
from pathlib import Path
import argparse
import json
import statistics

import numpy as np

from analyse_controls import analyse

SEEDS = list(range(2026110301, 2026110307))
NULL_SEEDS = list(range(2026110311, 2026110315))
RANDOM_MODES = {"stiffness-shift", "delay-shuffle", "stiffness-shift-after-startup", "delay-narrow-clock-mean"}


def bootstrap_mean(differences, rng, count=20000):
    """Rows are noise blocks; optional columns are crossed assignments."""
    a = np.asarray(differences, dtype=float)
    if a.ndim not in (1, 2) or not np.isfinite(a).all():
        raise ValueError("Finite one- or two-dimensional paired differences required")
    draws = []
    for _ in range(count):
        rows = rng.integers(0, a.shape[0], a.shape[0])
        if a.ndim == 2:
            cols = rng.integers(0, a.shape[1], a.shape[1])
            draws.append(float(a[np.ix_(rows, cols)].mean()))
        else:
            draws.append(float(a[rows].mean()))
    return np.quantile(draws, [.025, .975]).tolist()


def followup(paths):
    cases, summaries, comparisons = {}, [], []
    for path in paths:
        report = analyse(path)
        summary = json.loads((path / "summary.json").read_text())
        if summary.get("protocol_version") != 2 or summary["null_seeds"] != NULL_SEEDS:
            raise ValueError("Not the fixed follow-up protocol")
        summaries.append(summary)
        for row, comp in zip(summary["rows"], report["comparisons"], strict=True):
            key = (row["mode"], row["seed"], row["null_seed"])
            if key in cases:
                raise ValueError("Duplicate case across shards")
            cases[key] = (path / row["case"], row)
            comparisons.append(comp)
    fixed = {"base", "stiffness-replay", "stiffness-mean", "activation", "delay-uniform", "inhibition-unscaled", "motor-zero"}
    expected = {(m, s, None) for m in fixed for s in SEEDS}
    expected |= {(m, s, a) for m in RANDOM_MODES for s in SEEDS for a in NULL_SEEDS}
    if set(cases) != expected:
        raise ValueError("Incomplete fixed 138-trial grid")
    for field in ("release_commit", "source_hashes", "packages", "python", "machine", "numerical_environment"):
        if any(s[field] != summaries[0][field] for s in summaries):
            raise ValueError(f"Shards differ in {field}")
    trials = []
    for key, (path, row) in cases.items():
        mode, seed, assignment = key
        base_path, base_row = cases[("base", seed, None)]
        with np.load(path / "mechanics.npz", allow_pickle=False) as a, np.load(base_path / "mechanics.npz", allow_pickle=False) as b:
            q = a["quat"][-400:]
            tilt = np.degrees(np.arccos(np.clip(1 - 2 * (q[:, 1]**2 + q[:, 2]**2) / np.sum(q*q, axis=1), -1, 1)))
            height = float(a["body"][-400:, 2].mean())
            if mode == "stiffness-replay":
                for name in b.files:
                    np.testing.assert_array_equal(a[name], b[name])
                if row["neural_counts_sha256"] != base_row["neural_counts_sha256"]:
                    raise ValueError("Identity replay changed neural activity")
            if mode == "stiffness-shift-after-startup":
                np.testing.assert_array_equal(a["stiffness"][:200], b["stiffness"][:200])
                np.testing.assert_array_equal(np.sort(a["stiffness"][200:], axis=0), np.sort(b["stiffness"][200:], axis=0))
            if mode == "delay-narrow-clock-mean":
                slots = np.maximum(1, np.rint(a["delay_ms"] / .1)).astype(np.int64)
                original = np.maximum(1, np.rint(b["delay_ms"] / .1)).astype(np.int64)
                if int(slots.sum()) != int(original.sum()) or int(np.ptp(slots)) > 1:
                    raise ValueError("Clock mean control failed")
        trials.append({"mode": mode, "noise_seed": seed, "assignment_seed": assignment,
                       "net_displacement_mm": row["mechanical_metrics"]["net_displacement_mm"],
                       "difference_from_base_mm": row["mechanical_metrics"]["net_displacement_mm"] - base_row["mechanical_metrics"]["net_displacement_mm"],
                       "late_posture_success": bool(tilt.mean() < 51.68 and height > 1.),
                       "late_tilt_deg": float(tilt.mean()), "late_height_mm": height,
                       "historical_posture_and_motion_gate": row["historical_metrics"]["standing_v32"]})
    rng = np.random.default_rng(2026110321)
    arms = []
    for mode in sorted(fixed | RANDOM_MODES):
        rows = [r for r in trials if r["mode"] == mode]
        rows.sort(key=lambda r: (r["noise_seed"], r["assignment_seed"] or -1))
        diff = np.array([r["difference_from_base_mm"] for r in rows])
        if mode in RANDOM_MODES:
            diff = diff.reshape(len(SEEDS), len(NULL_SEEDS))
        blocks = diff.mean(axis=1) if diff.ndim == 2 else diff
        arms.append({"mode": mode, "trials": len(rows), "noise_blocks": len(SEEDS),
                     "assignment_count": len(NULL_SEEDS) if mode in RANDOM_MODES else None,
                     "posture_success_count": sum(r["late_posture_success"] for r in rows),
                     "historical_gate_success_count": sum(r["historical_posture_and_motion_gate"] for r in rows),
                     "mean_displacement_mm": statistics.mean(r["net_displacement_mm"] for r in rows),
                     "mean_paired_difference_mm": float(blocks.mean()),
                     "median_block_difference_mm": float(np.median(blocks)),
                     "block_differences_mm": blocks.tolist(), "all_paired_differences_mm": diff.tolist(),
                     "paired_difference_range_mm": [float(diff.min()), float(diff.max())],
                     "exploratory_bootstrap_95_interval_mm": bootstrap_mean(diff, rng)})
    return {"protocol": "FOLLOWUP_PROTOCOL.md", "trials": len(trials), "noise_seeds": SEEDS,
            "assignment_seeds": NULL_SEEDS, "bootstrap_resamples": 20000, "bootstrap_seed": 2026110321,
            "interpretation": "Exploratory conditional simulation sensitivity; small-sample bootstrap, not animal uncertainty or an equivalence test",
            "source_hashes": summaries[0]["source_hashes"], "hosts": [s["execution_host"] for s in summaries],
            "arms": arms, "trial_rows": trials, "delay_and_historical_readouts": comparisons}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directories", nargs="+", type=Path)
    args = ap.parse_args()
    print(json.dumps(followup([p.resolve() for p in args.directories]), indent=2, allow_nan=False))
