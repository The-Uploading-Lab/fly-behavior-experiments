"""Recalculate the manuscript's descriptive comparisons from archived rows.

Uses only Python's standard library. Run from any working directory.
"""
from pathlib import Path
import hashlib
import json
import math
import statistics as stats

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / "data" / name).read_text())


def verify():
    count = 0
    for base, manifest in [(ROOT, ROOT / "manifest.json"),
                           (ROOT.parent, ROOT.parent / "research-history/manifest.json")]:
        for item in json.loads(manifest.read_text())["files"]:
            path = base / item["path"]
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != item["sha256"]:
                raise ValueError(f"Source hash mismatch: {path}")
            count += 1
    return count


def analyse():
    verified = verify()
    muscle = read("muscle-configuration-rows.json")
    by = {(r["who"], r["seed"]): r for r in muscle["results"]}
    seeds = list(range(46, 76))
    if len(by) != 60 or len(muscle["results"]) != 60:
        raise ValueError("Expected 30 distinct paired seeds")
    stands = lambda r: r["upright"] > .90 and r["z_end"] > 1.0
    a = [stands(by["champ", s]) for s in seeds]
    b = [stands(by["cand5", s]) for s in seeds]
    counts = {"both": sum(x and y for x, y in zip(a, b)),
              "candidate_only": sum(y and not x for x, y in zip(a, b)),
              "predecessor_only": sum(x and not y for x, y in zip(a, b)),
              "neither": sum(not x and not y for x, y in zip(a, b))}
    n = counts["candidate_only"] + counts["predecessor_only"]
    k = min(counts["candidate_only"], counts["predecessor_only"])
    p = min(1., 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    muscle_result = {"n_pairs": 30, "predecessor_standing": sum(a),
                     "candidate_standing": sum(b), "paired_counts": counts,
                     "standing_difference": (sum(b) - sum(a)) / 30,
                     "exact_discordance_p_two_sided": p,
                     "median_gated_displacement_mm": {
                         w: stats.median(by[w, s]["raw"] if stands(by[w, s]) else 0
                                         for s in seeds) for w in ["champ", "cand5"]},
                     "changed_parameters": {
                         k: [muscle["champ"].get(k), muscle["cand5"].get(k)]
                         for k in sorted(set(muscle["champ"]) | set(muscle["cand5"]))
                         if muscle["champ"].get(k) != muscle["cand5"].get(k)}}
    rows = read("delay-replay-rows.json")["rows"]
    by = {(r["arm"], r["seed"]): r["record"] for r in rows}
    seeds = list(range(201303, 201309))
    arms = ["base", "flat6", "flat15", "noshort", "nolong", "perm"]
    if set(by) != {(a, s) for a in arms for s in seeds} or len(rows) != 36:
        raise ValueError("Expected six arms on six paired seeds")
    delay = {a: {"n": 6, "standing": sum(by[a, s]["standing_v32"] for s in seeds),
                 "median_speed_mm_s": stats.median(by[a, s]["speed"] for s in seeds)}
             for a in arms}
    paired = {}
    for a in arms[1:]:
        diffs = [by["base", s]["speed"] - by[a, s]["speed"] for s in seeds]
        paired[a] = {"median_base_minus_arm_mm_s": stats.median(diffs),
                     "positive_pairs": sum(x > 0 for x in diffs), "differences": diffs}
    calcium = read("calcium-and-walking-summaries.json")["calcium"]
    cal = {}
    for cell in ["L1", "L2"]:
        retention = [r["simulated"] - r["observed"]
                     for r in calcium["retention"] if r["cell"] == cell]
        peaks = [r["simulated_s"] - r["observed_s"]
                 for r in calcium["peaks"] if r["cell"] == cell]
        cal[cell] = {"retention_n_conditions": len(retention),
                     "retention_mae": stats.mean(map(abs, retention)),
                     "retention_rmse": math.sqrt(stats.mean(x*x for x in retention)),
                     "peak_n_conditions": len(peaks),
                     "peak_time_mae_s": stats.mean(map(abs, peaks))}
    obs = read("walking-observer-results.json")["results"]
    walking = {"trace_phase_combinations": len(obs),
               "combinations_with_half_second_bin_support": sum(
                   any(r["has_half_second_support"] for r in x["rows"]) for x in obs)}
    return {"kind": "recalculation of stored results; no new simulation or animal experiment",
            "verified_input_files": verified, "muscle": muscle_result,
            "delay_arms": delay, "delay_paired": paired, "calcium": cal,
            "walking_support": walking}


if __name__ == "__main__":
    output = ROOT.parent / "outputs/publication"
    output.mkdir(parents=True, exist_ok=True)
    result = analyse()
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
