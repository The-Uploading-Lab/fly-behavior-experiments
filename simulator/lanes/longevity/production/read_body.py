"""Read body replay archives with continuous posture and support checks."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analyze import (analyze_escape, digest, distribution, movement_windows,
                     wilson, write_csv)

METRICS = ("moving_fraction", "moving_speed_mm_s", "window_path_mm",
           "net_displacement_mm", "straightness", "moving_abs_turn_rate_deg_s")
GROUP_FIELDS = ("model_id", "protocol", "behavior", "arm", "cohort")


def validate_state(state):
    b = np.asarray(state, float)
    if b.ndim != 2 or b.shape[1] != 8 or len(b) < 3:
        raise ValueError("expected at least three eight-column body-state samples")
    if not np.isfinite(b).all() or not (np.diff(b[:, 0]) > 0).all():
        raise ValueError("invalid body-state values or times")
    if np.max(np.diff(b[:, 0])) > .001001:
        raise ValueError("body-state coverage has a gap beyond the 1 ms sampling interval")
    return b


def state_fraction(b, start, end, mask):
    """Integrate the last recorded state, clipped to the requested interval."""
    if end <= start or start < b[0, 0] or end > b[-1, 0] + 1e-8:
        raise ValueError("interval outside recorded body state")
    dt = np.maximum(0, np.minimum(b[1:, 0], end) - np.maximum(b[:-1, 0], start))
    if not np.isclose(dt.sum(), end - start):
        raise ValueError("incomplete body-state interval")
    return float(dt[np.asarray(mask, bool)[:-1]].sum() / dt.sum())


def trim_track(track, start):
    t = np.asarray(track, float)
    if not t[0, 0] <= start < t[-1, 0]:
        raise ValueError("walking initialization cutoff outside trajectory")
    first = [start, *[float(np.interp(start, t[:, 0], t[:, i])) for i in (1, 2)]]
    return np.vstack([first, t[t[:, 0] > start]])


def supported_movement(track, state, width=.25, threshold=.5):
    r = movement_windows(track, width, threshold)
    windows = r["windows"]
    valid = np.array([state_fraction(state, w["start_s"], w["end_s"], state[:, 3] <= .15) >= .95
                      for w in windows])
    duration = np.array([w["end_s"] - w["start_s"] for w in windows])
    speed = np.array([w["speed_mm_s"] for w in windows])
    moving = np.array([w["moving"] for w in windows]) & valid
    eligible_s = float(duration[valid].sum())
    r["supported_window_duration_s"] = eligible_s
    r["eligible_windows"], r["total_windows"] = int(valid.sum()), len(windows)
    r["moving_fraction"] = float(duration[moving].sum() / eligible_s) if eligible_s else None
    r["moving_speed_mm_s"] = float((speed[moving] * duration[moving]).sum() / duration[moving].sum()) if moving.any() else None
    r["window_path_mm"] = float((speed[valid] * duration[valid]).sum()) if valid.any() else None
    if not valid.all():
        r["net_displacement_mm"] = r["straightness"] = None
        r["moving_abs_turn_rate_deg_s"] = None
    r["bouts"] = []
    begin = 0
    while begin < len(windows):
        if not valid[begin]:
            begin += 1
            continue
        end = begin + 1
        while end < len(windows) and valid[end] and moving[end] == moving[begin]:
            end += 1
        r["bouts"].append({"moving": bool(moving[begin]), "start_s": windows[begin]["start_s"],
                           "end_s": windows[end - 1]["end_s"], "duration_s": float(duration[begin:end].sum()),
                           "boundary_censored": bool(begin == 0 or end == len(windows) or
                                                      not valid[begin - 1] or not valid[end])})
        begin = end
    for w, ok in zip(windows, valid):
        w["supported"] = bool(ok)
        if not ok:
            w["speed_mm_s"] = w["moving"] = None
    return r


def walking_readout(receipt, state):
    b = validate_state(state)
    track = trim_track(receipt["track"], .3)
    start, end = track[0, 0], track[-1, 0]
    q = b[b[:, 0] >= start]
    r = supported_movement(track, b)
    r.update(analysis_start_s=float(start), analysis_end_s=float(end), posture_end_s=float(b[-1, 0]),
             minimum_upright=float(q[:, 2].min()), final_upright=float(b[-1, 2]),
             upright_fraction=state_fraction(b, start, b[-1, 0], b[:, 2] >= .5),
             supported_fraction=state_fraction(b, start, b[-1, 0], b[:, 3] <= .15),
             retention_pass=bool((q[:, 2] >= .5).all()), posture_coverage="continuous_1ms_samples",
             target_walk_fraction_input=receipt["walk_fraction_input"],
             realized_walk_fraction_input=sum(max(0, min(end, z) - max(start, a))
                                              for a, z in receipt["schedule_s"]) / (end - start))
    r["sensitivity"] = [{k: v for k, v in supported_movement(track, b, width, threshold).items()
                         if k not in ("windows", "bouts")}
                        for width in (.125, .25, .5) for threshold in (.25, .5, 1.0)]
    if not r["retention_pass"]:
        r["censor_reason"] = "failed_continuous_upright_retention"
        for key in METRICS:
            r[key] = None
        r["windows"], r["bouts"], r["sensitivity"] = [], [], []
    return r


def escape_readout(receipt, state, land_s):
    b = validate_state(state)
    r = analyze_escape({**receipt, "raster": receipt.get("raster", [])})
    onset = receipt["flash_ms"] * .001
    pre = b[(b[:, 0] >= onset - .02) & (b[:, 0] < onset)]
    if len(pre) < 19:
        raise ValueError("incomplete 20 ms pre-stimulus posture interval")
    supported = bool((pre[:, 3] <= .15).all())
    upright = bool((pre[:, 2] >= .5).all())
    predeparture = receipt["t_off_ms"] is not None and receipt["t_off_ms"] < receipt["flash_ms"]
    r.update(eligible=supported and upright and not predeparture,
             eligibility="continuous_preinput_posture_and_support", preinput_supported=supported,
             preinput_upright=upright, preinput_minimum_upright=float(pre[:, 2].min()),
             pretrigger_seen=any(t < receipt["flash_ms"] for t in receipt["gf_spikes_ms"]),
             minimum_upright_after_input=float(b[b[:, 0] >= onset, 2].min()),
             inversion_after_input=bool((b[b[:, 0] >= onset, 2] < 0).any()))
    tail_start = max(onset, float(b[-1, 0] - .1))
    tail = b[b[:, 0] >= tail_start]
    r["final_100ms_supported_fraction"] = state_fraction(b, tail_start, b[-1, 0], b[:, 3] <= .15)
    r["final_100ms_minimum_upright"] = float(tail[:, 2].min())
    r["upright_supported_finish"] = bool((tail[:, 2] >= .5).all() and r["final_100ms_supported_fraction"] >= .95)
    returned = b[(b[:, 0] >= land_s + .05) & (b[:, 3] <= .15)] if land_s is not None else np.empty((0, 8))
    r["first_supported_after_land_s"] = float(returned[0, 0]) if len(returned) and r["body_takeoff"] else None
    if not r["eligible"]:
        r["censor_reason"] = "predeparture" if predeparture else "failed_preinput_posture_or_support"
    return r


def load_capture(path):
    m = json.loads((path / "manifest.json").read_text())
    if m["schema"] != "longevity-production-capture-v1":
        raise ValueError("unsupported capture schema")
    if "candidate" in m:
        if m.get("review_required") or not m.get("scientific_parity", {}).get("exact"):
            raise ValueError("candidate capture does not pass scientific parity")
        if digest(path / "integrated-receipt.json") != m["integrated_receipt_sha256"]:
            raise ValueError("integrated receipt hash mismatch")
    for name, field in (("capture.npz", "capture_sha256"), ("receipt.json", "receipt_sha256")):
        if digest(path / name) != m[field]:
            raise ValueError(f"capture hash mismatch: {path / name}")
    receipt = json.loads((path / "receipt.json").read_text())
    if (receipt["arm"], receipt["seed"]) != (m["arm"], m["seed"]):
        raise ValueError("capture identity mismatch")
    with np.load(path / "capture.npz", allow_pickle=False) as arrays:
        b = arrays["body_state"]
    r = (walking_readout(receipt, b) if m["behavior"] == "walking" else
         escape_readout(receipt, b, m["land_s"]))
    r.update(model_id=m.get("candidate", {}).get("id", "original_baseline"),
             candidate_sha256=m.get("candidate_sha256"),
             runtime_theta_sha256=m.get("runtime_theta_sha256"),
             protocol=m.get("integrated_protocol", m["behavior"]),
             behavior=m["behavior"], arm=m["arm"], seed=m["seed"],
             cohort="delivered_candidate_development_seed" if "candidate" in m else
                    "matched_assumption" if m["arm"] == "ibuprofen" and m["seed"] < 4 else
                    "illustrative" if m["arm"] == "pre_death" else "original_development_seeds",
             capture=str(path), capture_sha256=m["capture_sha256"], receipt_sha256=m["receipt_sha256"])
    if "candidate" in m:
        integrated = json.loads((path / "integrated-receipt.json").read_text())
        r.update(recorded_gf_active_bins_ms=receipt["gf_spikes_ms"],
                 gf_event_list_capped_at=10,
                 ttm_spikes_by_side_ms=integrated["ttm_spikes_by_side_ms"],
                 scientific_parity=m["scientific_parity"],
                 integrated_receipt_sha256=m["integrated_receipt_sha256"])
        if m["behavior"] == "walking":
            r.update(schedule_is_imposed=receipt["schedule_is_imposed"],
                     internal_body_regression_speed_mm_s=integrated["walking_regression"]["speed"],
                     internal_body_regression_cadence_hz=integrated["walking_regression"]["freq_tip_h"],
                     body_metric_scope="primary movement metrics describe the displayed winged body; internal-body metrics are separate")
    return m, r


def summaries(rows):
    groups = dict.fromkeys(tuple(r[k] for k in GROUP_FIELDS) for r in rows)
    out = []
    for key in groups:
        group = [r for r in rows if tuple(r[k] for k in GROUP_FIELDS) == key]
        s = dict(zip(GROUP_FIELDS, key))
        s.update(n_trials=len(group), seeds=[r["seed"] for r in group])
        if s["behavior"] == "walking":
            s["retention_pass_n"] = sum(r["retention_pass"] for r in group)
            for key in (*METRICS, "realized_walk_fraction_input"):
                s[key] = distribution(r[key] for r in group)
        else:
            eligible = [r for r in group if r["eligible"]]
            s["eligible_n"] = len(eligible)
            for key in ("neural_trigger", "body_takeoff"):
                n, k = len(eligible), sum(r[key] for r in eligible)
                s[key] = {"n": n, "successes": k, "fraction": k / n if n else None, "wilson95": wilson(k, n)}
            s["takeoff_latency_ms_responders"] = distribution(r["takeoff_latency_ms"] for r in eligible)
            s["upright_supported_finish_n"] = sum(r["upright_supported_finish"] for r in group)
            s["inversions_n"] = sum(r["inversion_after_input"] for r in group)
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("captures", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    manifests, rows, seen = [], [], set()
    for path in a.captures:
        m, r = load_capture(path)
        key = (r["model_id"], r["protocol"], r["behavior"], r["arm"], r["seed"])
        if key in seen:
            raise ValueError("duplicate body trial")
        if manifests:
            for field in ("source_sha256", "packages", "engine"):
                if m[field] != manifests[0][field]:
                    raise ValueError(f"model identities differ: {field}")
            for field in ("candidate_sha256", "runtime_theta_sha256"):
                if m.get(field) != manifests[0].get(field):
                    raise ValueError(f"model identities differ: {field}")
            if {k: v["sha256"] for k, v in m["data"].items()} != {k: v["sha256"] for k, v in manifests[0]["data"].items()}:
                raise ValueError("data identities differ")
        seen.add(key)
        manifests.append(m)
        rows.append(r)
    a.out.mkdir(parents=True)
    result = {"schema": "longevity-production-body-analysis-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "reader_sha256": digest(Path(__file__)), "protocol_sha256": digest(Path(__file__).with_name("BODY-COHORT.md")),
              "biological_n": 0, "inference": "fixed_model_development_replays; retrospective_analysis",
              "foot_position_time_offset_ms": -.2, "summary": summaries(rows), "trials": rows,
              "source_manifests": [{"path": str(p / "manifest.json"), "sha256": digest(p / "manifest.json")}
                                   for p in a.captures]}
    (a.out / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for behavior in ("walking", "escape"):
        subset = [r for r in rows if r["behavior"] == behavior]
        if subset:
            write_csv(a.out / (behavior + "-trials.csv"), subset)
    print(json.dumps({"out": str(a.out), "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
