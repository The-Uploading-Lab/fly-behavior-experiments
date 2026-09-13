"""Reanalyse saved demo receipts without executing a neural or body model.

Statistics describe simulation trials. They do not estimate biological
treatment effects. No frames or trajectory windows count as extra replicates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ESC = ROOT / "lanes/longevity/escape"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(successes: int, n: int) -> list[float] | None:
    if n == 0:
        return None
    if not 0 <= successes <= n:
        raise ValueError("successes outside denominator")
    z = 1.959963984540054
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, center - half), min(1.0, center + half)]


def distribution(values) -> dict:
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    if not len(x):
        return {"n": 0, "mean": None, "median": None, "sd": None, "q25": None,
                "q75": None, "min": None, "max": None}
    return {"n": len(x), "mean": float(x.mean()), "median": float(np.median(x)),
            "sd": float(x.std(ddof=1)) if len(x) > 1 else None,
            "q25": float(np.quantile(x, .25)), "q75": float(np.quantile(x, .75)),
            "min": float(x.min()), "max": float(x.max())}


def movement_windows(track, width_s=.25, threshold_mm_s=.5) -> dict:
    t = np.asarray(track, dtype=float)
    if t.ndim != 2 or t.shape[1] != 3 or len(t) < 3:
        raise ValueError("track must contain at least three (time, x, y) rows")
    if not np.isfinite(t).all() or not np.all(np.diff(t[:, 0]) > 0):
        raise ValueError("track must be finite with strictly increasing times")
    if width_s <= 0 or threshold_mm_s < 0:
        raise ValueError("invalid movement window or threshold")
    cuts = np.arange(t[0, 0], t[-1, 0], width_s)
    cuts = np.append(cuts, t[-1, 0])
    xy = np.column_stack([np.interp(cuts, t[:, 0], t[:, k]) for k in (1, 2)])
    dt = np.diff(cuts)
    delta = np.diff(xy, axis=0)
    dist = np.linalg.norm(delta, axis=1)
    speed = dist / dt
    moving = speed > threshold_mm_s
    path = float(dist.sum())
    net = float(np.linalg.norm(xy[-1] - xy[0]))
    state_changes = np.r_[0, np.flatnonzero(np.diff(moving.astype(int))) + 1, len(moving)]
    bouts = [{"moving": bool(moving[a]), "start_s": float(cuts[a]),
              "end_s": float(cuts[b]), "duration_s": float(cuts[b] - cuts[a]),
              "boundary_censored": bool(a == 0 or b == len(moving))}
             for a, b in zip(state_changes[:-1], state_changes[1:])]
    heading = np.arctan2(delta[:, 1], delta[:, 0])
    turn = np.angle(np.exp(1j * np.diff(heading)))
    valid_turn = moving[:-1] & moving[1:]
    turn_dt = .5 * (dt[:-1] + dt[1:])
    turn_rate = (float(np.degrees(np.abs(turn[valid_turn])).sum() / turn_dt[valid_turn].sum())
                 if valid_turn.any() else None)
    return {"duration_s": float(dt.sum()), "moving_fraction": float(dt[moving].sum() / dt.sum()),
            "moving_speed_mm_s": float(dist[moving].sum() / dt[moving].sum()) if moving.any() else None,
            "window_path_mm": path, "net_displacement_mm": net,
            "straightness": net / path if path > 0 else None,
            "moving_abs_turn_rate_deg_s": turn_rate,
            "window_s": width_s, "moving_threshold_mm_s": threshold_mm_s,
            "windows": [{"start_s": float(a), "end_s": float(b), "speed_mm_s": float(v),
                         "moving": bool(m)} for a, b, v, m in zip(cuts[:-1], cuts[1:], speed, moving)],
            "bouts": bouts}


def analyze_walk(record: dict) -> dict:
    result = movement_windows(record["track"])
    upright = float(record["upright_final"])
    duration = result["duration_s"]
    start = record["track"][0][0]
    end = record["track"][-1][0]
    scheduled_s = sum(max(0.0, min(end, b) - max(start, a)) for a, b in record["schedule_s"])
    result.update(arm=record["arm"], seed=record["seed"], upright_final=upright,
                  final_upright=upright >= .5, posture_coverage="final_endpoint_only",
                  target_walk_fraction_input=record["walk_fraction_input"],
                  realized_walk_fraction_input=scheduled_s / duration,
                  inference="conditional_on_imposed_leg_schedule; retrospective_demo_replay")
    result["sensitivity"] = [
        {k: v for k, v in movement_windows(record["track"], width, threshold).items()
         if k not in ("windows", "bouts")}
        for width in (.125, .25, .5) for threshold in (.25, .5, 1.0)]
    if not result["final_upright"]:
        result["censor_reason"] = "failed_final_upright_screen"
        for key in ("moving_fraction", "moving_speed_mm_s", "window_path_mm", "net_displacement_mm",
                    "straightness", "moving_abs_turn_rate_deg_s"):
            result[key] = None
        result["sensitivity"] = []
        result["bouts"] = []
    return result


def analyze_escape(record: dict, window_ms=100.0) -> dict:
    onset = float(record["flash_ms"])
    off = record["t_off_ms"]
    predeparture = off is not None and off < onset
    takeoff_latency = off - onset if off is not None and off >= onset else None
    takeoff = takeoff_latency is not None and takeoff_latency <= window_ms
    gf = record.get("gf_latency_ms")
    triggered = gf is not None and 0 <= gf <= window_ms
    raster = np.asarray(record["raster"], dtype=float)
    prespikes = raster[raster[:, 0] < onset, 3].sum() if len(raster) else None
    return {"arm": record["arm"], "seed": record["seed"], "p_lc4": record["p_lc4"],
            "stimulus": record["stim"], "onset_ms": onset, "window_ms": window_ms,
            "eligibility": "predeparture_excluded" if predeparture else "historical_initial_posture_not_recorded",
            "eligible": not predeparture, "neural_trigger": triggered, "body_takeoff": takeoff,
            "gf_latency_ms": gf, "takeoff_latency_ms": takeoff_latency if takeoff else None,
            "takeoff_censored_ms": takeoff_latency if takeoff else window_ms,
            "takeoff_observed": takeoff,
            "pretrigger_count_in_recorded_10ms": int(prespikes) if prespikes is not None else None,
            "upright_final": record["upright_final"], "final_upright": record["upright_final"] >= .5,
            "peak_height_mm": record["z_peak"], "airborne_s": record["airborne_s"],
            "flight_scope": "programmed_power_attitude_and_landing",
            "inference": "retrospective_body_demo_replay; not_circuit_only_probability"}


def group_summaries(rows: list[dict], behavior: str) -> list[dict]:
    result = []
    for arm in dict.fromkeys(r["arm"] for r in rows):
        group = [r for r in rows if r["arm"] == arm]
        s = {"behavior": behavior, "arm": arm, "n_trials": len(group),
             "seeds": [r["seed"] for r in group], "final_upright_n": sum(r["final_upright"] for r in group)}
        if behavior == "escape":
            eligible = [r for r in group if r["eligible"]]
            n = len(eligible)
            for key in ("body_takeoff", "neural_trigger"):
                k = sum(r[key] for r in eligible)
                s[key] = {"n": n, "successes": k, "fraction": k/n if n else None,
                          "wilson95": wilson(k, n), "scope": "fixed_model_demo_trials"}
            s["takeoff_latency_ms_responders"] = distribution(r["takeoff_latency_ms"] for r in eligible)
            s["gf_latency_ms_responders"] = distribution(r["gf_latency_ms"] for r in eligible)
        else:
            s["posture_coverage"] = "final_endpoint_only"
            for key in ("moving_fraction", "moving_speed_mm_s", "window_path_mm", "net_displacement_mm",
                        "straightness", "moving_abs_turn_rate_deg_s", "realized_walk_fraction_input"):
                s[key] = distribution(r[key] for r in group)
        result.append(s)
    return result


def write_csv(path: Path, rows: list[dict]):
    keys = list(dict.fromkeys(k for r in rows for k, v in r.items() if not isinstance(v, (dict, list))))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in keys})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite prior analysis: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest, walks, escapes = [], [], []
    specs = {
        "walking": [(a, s) for a in ("control", "starved_10h", "starved_20h") for s in range(4)]
                   + [("ibuprofen", s) for s in range(4, 8)],
        "escape": [(a, s) for a in ("control", "starved", "half_starved") for s in range(4)]
                  + [("ibuprofen", s) for s in range(4, 8)]}
    for behavior, trials in specs.items():
        for arm, seed in trials:
            name = (f"2026-09-13-walk-arms-{arm}-s{seed}.json" if behavior == "walking" else
                    f"2026-09-12-jumpfly-arms-{arm}-s{seed}.json")
            path = ESC / name
            raw = path.read_bytes()
            d = json.loads(raw)
            if d["arm"] != arm or d["seed"] != seed:
                raise ValueError(f"identity mismatch: {path}")
            row = analyze_walk(d) if behavior == "walking" else analyze_escape(d)
            row["source"] = str(path.relative_to(ROOT))
            row["source_sha256"] = hashlib.sha256(raw).hexdigest()
            (walks if behavior == "walking" else escapes).append(row)
            manifest.append({"path": row["source"], "sha256": row["source_sha256"], "bytes": len(raw),
                             "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()})
    summary = group_summaries(walks, "walking") + group_summaries(escapes, "escape")
    out = {"schema": "longevity-production-analysis-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
           "reader_sha256": digest(Path(__file__)), "protocol_sha256": digest(Path(__file__).with_name("PROTOCOL.md")),
           "exposure": "retrospective_saved_demo_trials", "biological_n": 0,
           "source_model_attribution": "historical_inputs; original_receipts_lack_complete_engine_hashes",
           "summary": summary, "walking": walks, "escape": escapes, "sources": manifest}
    (args.out / "analysis.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    write_csv(args.out / "walking-trials.csv", walks)
    write_csv(args.out / "escape-trials.csv", escapes)
    (args.out / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "walking_trials": len(walks), "escape_trials": len(escapes),
                      "escape": [{"arm": s["arm"], "body_takeoff": s["body_takeoff"]}
                                 for s in summary if s["behavior"] == "escape"]}, indent=2))


if __name__ == "__main__":
    main()
