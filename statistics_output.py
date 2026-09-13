"""Export comparable per-record statistics, preserving missing data and failures."""
import csv
import json
from pathlib import Path
import numpy as np
from portable_integrity import ROOT, sha

ARM = {"control": "control", "ibuprofen": "ibuprofen", "calorie_restriction": "deprivation_b_half"}


def read(path):
    return json.loads(Path(path).read_text())


def write(out, name, rows):
    path = out / f"{name}.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_movement(wet, model):
    speed = wet["projected_silhouette_lengths_per_analysis_s"]
    digital = model.get("speed_silh_s")
    ratio = speed / digital if digital else None
    low = wet["speed_window_sensitivity_min_BL_s_240"]
    high = wet["speed_window_sensitivity_max_BL_s_240"]
    model_low, model_high = model.get("sensitivity_min_silh_s"), model.get("sensitivity_max_silh_s")
    return {
        "clip_id": wet["clip_id"], "measurement_id": wet["bout_id"], "condition": wet["condition"],
        "time_since_first_recording_s": wet["capture_elapsed_s"], "recorded_at_cest": wet["recorded_at_cest"],
        "behavior": wet["behavior"], "walking_summary_eligible": wet["walking_summary_eligible"],
        "comparison_scope": "walking window" if wet["walking_summary_eligible"] else "motion reference only; model behavior not matched",
        "duration_s": wet["analysis_duration_s"], "wet_speed_silh_s": speed, "digital_speed_silh_s": digital,
        "wet_over_digital_speed": ratio, "digital_percent_slower": 100 * (1 - digital / speed) if digital is not None and speed else None,
        "speed_difference_digital_minus_wet_silh_s": digital - speed if digital is not None else None,
        "wet_speed_sensitivity_min_silh_s": low, "wet_speed_sensitivity_max_silh_s": high,
        "digital_speed_sensitivity_min_silh_s": model_low, "digital_speed_sensitivity_max_silh_s": model_high,
        "wet_over_digital_sensitivity_min": low / model_high if model_high else None,
        "wet_over_digital_sensitivity_max": high / model_low if model_low else None,
        "wet_path_silh": wet["projected_path_px"] / wet["reference_silhouette_length_px"],
        "digital_path_silh": model.get("path_silh"),
        "wet_net_displacement_silh": wet["projected_net_displacement_px"] / wet["reference_silhouette_length_px"],
        "digital_net_displacement_silh": model.get("net_silh"),
        "wet_straightness": wet["path_straightness"], "digital_straightness": model.get("straightness"),
        "digital_walking_gate": bool(model), "model_trials": 1,
    }


def compare_tap(wet, model):
    latency = model["takeoff_latency_ms"] if model["eligible"] else None
    lo, hi, cens = (wet[k] for k in ("latency_lower_ms", "latency_upper_ms", "no_departure_observation_ms"))
    gap = None
    if not model["eligible"]:
        relation = "MODEL INELIGIBLE"
    elif latency is None:
        relation = "NO MODEL DEPARTURE"
    elif cens is not None:
        relation = "DIFFERS: MODEL DEPARTS" if latency <= cens else "OUTSIDE FOLLOW-UP"
        gap = latency - cens
    else:
        relation = "WITHIN WET INTERVAL" if lo <= latency <= hi else "OUTSIDE WET INTERVAL"
        gap = latency - hi if latency > hi else latency - lo if latency < lo else 0.
    return {
        "clip_id": wet["clip_id"], "event_id": wet["event_id"], "condition": wet["condition"],
        "time_since_first_recording_s": wet["capture_elapsed_s"], "recorded_at_cest": wet["recorded_at_cest"],
        "wet_departure_lower_ms": lo, "wet_departure_upper_ms": hi, "wet_no_departure_through_ms": cens,
        "digital_support_loss_latency_ms": latency, "digital_GF_latency_ms": model["gf_latency_ms"],
        "digital_eligible": model["eligible"], "digital_flight_gate": model["successor_retention_pass"],
        "digital_inverted": model["inversion_after_input"], "relation": relation,
        "departure_boundary_difference_ms": gap,
        "boundary_difference_scope": "one-sided followup boundary" if cens is not None else "signed distance to measured interval; zero is inside",
        "comparison_scope": "hand-tap departure versus simulated antennal-input support loss; stimulus and endpoint differ",
    }


def wetlab(out):
    from projection import project
    source = read(ROOT / "experiments/wetlab-25.json")
    records = source["records"]
    movement = [m for r in records for m in r["movement"]]
    taps = [t for r in records for t in r["taps"]]
    assert len(records) == len({r["clip_id"] for r in records}) == 25
    assert len(movement) == 21 and len(taps) == 11
    projected, responses = {}, {}
    for condition, arm in ARM.items():
        intervals = [300] + [m["frame_intervals"] for m in movement if m["condition"] == condition]
        projected[condition] = project(out / "runs" / f"{arm}-walk", intervals)
        responses[condition] = read(out / "runs" / f"{arm}-puff/readout.json")
    walking = [compare_movement(m, projected[m["condition"]]["windows"].get(str(m["frame_intervals"]), {})) for m in movement]
    escape = [compare_tap(t, responses[t["condition"]]) for t in taps]
    summary = []
    for condition in ARM:
        eligible = [r for r in walking if r["condition"] == condition and r["walking_summary_eligible"]]
        valid = [r for r in eligible if r["digital_walking_gate"]]
        summary.append({"condition": condition, "walking_windows": len(eligible), "model_trials": 1,
            "wet_median_silh_s": float(np.median([r["wet_speed_silh_s"] for r in eligible])),
            "digital_median_silh_s": float(np.median([r["digital_speed_silh_s"] for r in valid])) if valid else None,
            "median_wet_over_digital_speed": float(np.median([r["wet_over_digital_speed"] for r in valid])) if valid else None,
            "median_digital_percent_slower": float(np.median([r["digital_percent_slower"] for r in valid])) if valid else None,
            "digital_standing_pass": bool(valid),
            "scope": "descriptive selected clips; repeated windows share one simulated trial; no treatment-effect inference"})
    all_records = []
    for r in records:
        m = next((x for x in walking if x["clip_id"] == r["clip_id"]), {})
        ts = [x for x in escape if x["clip_id"] == r["clip_id"]]
        # One row per clip; event-level values remain in taps.csv (one clip has two taps).
        all_records.append({"clip_id": r["clip_id"], "condition": r["condition"],
            "time_since_first_recording_s": r["capture_elapsed_s"], "recorded_at_cest": r["recorded_at_cest"],
            "movement_windows": len(r["movement"]), "tap_events": len(ts),
            "walking_summary_eligible": m.get("walking_summary_eligible"),
            "wet_speed_silh_s": m.get("wet_speed_silh_s"), "digital_speed_silh_s": m.get("digital_speed_silh_s"),
            "wet_over_digital_speed": m.get("wet_over_digital_speed"), "digital_percent_slower": m.get("digital_percent_slower"),
            "wet_path_silh": m.get("wet_path_silh"), "digital_path_silh": m.get("digital_path_silh"),
            "wet_straightness": m.get("wet_straightness"), "digital_straightness": m.get("digital_straightness"),
            "tap_event_ids": "; ".join(t["event_id"] for t in ts),
            "tap_relations": "; ".join(t["relation"] for t in ts)})
    result = {"measurement_id": source["measurement_id"], "measurement_status": source["status"],
              "records": all_records, "movement": walking, "taps": escape, "summary": summary,
              "projection": projected, "wet_input_sha256": sha(ROOT / "experiments/wetlab-25.json"),
              "independent_model_runs": 6, "time_model": "none: elapsed capture time is a record label, not simulated drug exposure"}
    (out / "statistics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name, rows in (("records-25", all_records), ("movement", walking), ("taps", escape), ("summary", summary)):
        write(out, name, rows)


def aging(out, days):
    from aging_analysis import case
    cases, rows = [], []
    for day in days:
        got = {p: case(day, p, out / "runs" / f"day-{day:02d}-{p}") for p in ("walk", "puff", "sham")}
        if any(v is None for v in got.values()):
            raise ValueError(f"Incomplete age: {day}")
        cases.extend(got.values())
        w, e, s = (got[p] for p in ("walk", "puff", "sham"))
        rows.append({"age_days": day, "synaptic_efficacy": 2 ** (-day / 45),
            "walking_status": w["status"], "walking_cases": 1, "standing_cases": int(w["standing"]),
            "walk_speed_mm_s": w["speed_mm_s"], "path_mm": w["path_mm"],
            "net_displacement_mm": w["net_mm"], "straightness": w["straightness"],
            "internal_body_speed_mm_s": w["internal_speed_mm_s"],
            "within_trial_window_speed_sd_mm_s": w["window_speed_sd_mm_s"],
            "escape_status": e["status"], "escape_cases": 1, "eligible_escape_cases": int(e["eligible"]),
            "departure_cases": int(e["departure_by_416_7ms"]), "native_departure_latency_ms": e["departure_latency_ms"],
            "neural_GF_latency_ms": e["gf_latency_ms"], "no_departure_through_ms": e["no_departure_through_ms"],
            "sham_status": s["status"], "sham_departure": s["departure_by_416_7ms"],
            "sham_GF_latency_ms": s["gf_latency_ms"], "simulation_seed": 0,
            "between_trial_uncertainty": "Unavailable: one selected development seed",
            "age_mapping": "ASSUMED 45-day efficacy half-life; uncalibrated to biological days"})
    result = {"complete": len(cases) == 3 * len(days), "days": days, "rows": rows, "cases": cases,
              "analysis_rule": "120 Hz analysis clock, 5-frame median, stride 3; full interval includes pauses",
              "trial_scope": "one paired development seed per age/protocol; windows and ages are not independent animals"}
    (out / "statistics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write(out, "aging", rows)
