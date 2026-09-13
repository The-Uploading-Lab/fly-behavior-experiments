"""Apply the registered walking bands to existing integrated replay receipts.

This is a single-replay constraint read, not a population or title verdict.
Unrecorded constraints remain unrecorded. No gait value is read after a
standing failure. The registry supplies the bands; this file maps instruments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "CONSTRAINT-REGISTRY.json"
FIELDS = {
    "gait.cadence": "freq_tip_h",
    "gait.swing_ms": "swing_tip_h_ms",
    "gait.duty": "duty_tip_h",
    "gait.contra_phase": "contra_tip_h",
    "gait.speed": "speed",
    "gait.all_feet_lifts": "lifts_h",
    "posture.nonfoot_force": "nonfoot_force",
    "rom.front_fti_deg": "rom.fFTi",
    "rom.front_ctr_deg": "rom.fCTr",
    "rom.middle_fti_deg": "rom.mFTi",
    "rom.middle_ctr_deg": "rom.mCTr",
    "rom.hind_fti_deg": "rom.hFTi",
    "rom.hind_ctr_deg": "rom.hCTr",
    "gait.phase_lf_rf_deg": "phase_lf_rf_deg",
    "gait.phase_lm_rm_deg": "phase_lm_rm_deg",
    "gait.phase_lh_rh_deg": "phase_lh_rh_deg",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assess(receipt, registry):
    walk = receipt.get("walking_regression") or {}
    internal_standing = bool(walk.get("standing") or walk.get("standing_v32"))
    body = receipt["result"]
    sampled_body = [frame for frame in receipt.get("winged_track", []) if frame[0] >= .3]
    displayed_standing = (bool(sampled_body) and all(frame[4] > 0. for frame in sampled_body)
                          and body["t_off_ms"] is None and body["upright_final"] > .9
                          and body["z_final"] > 1.)
    standing = internal_standing and displayed_standing
    constraints = {row["id"]: row for row in registry["constraints"]}
    rows = {}
    for name, field in FIELDS.items():
        band = constraints[name]["band"]
        if (not isinstance(band, list) or len(band) != 2
                or any(v is not None and not isinstance(v, (int, float)) for v in band)):
            raise ValueError(f"Registered band changed shape: {name}")
        value = walk
        for key in field.split("."):
            value = value.get(key) if isinstance(value, dict) else None
        if field == "lifts_h":
            value = min(value) if isinstance(value, list) and len(value) == 6 else None
        finite = isinstance(value, (int, float)) and math.isfinite(value)
        if not standing:
            status, value = "censored_after_standing_failure", None
        elif name.startswith("gait.phase_"):
            status, value = "requires_population_circular_mean", None
        elif not finite:
            status, value = "missing_readout", None
        else:
            inside = ((band[0] is None or value >= band[0]) and
                      (band[1] is None or value <= band[1]))
            status = "within_band_on_this_replay" if inside else "outside_band_on_this_replay"
        rows[name] = {"instrument": field, "value": value, "band": band, "status": status,
                      "units": constraints[name].get("units")}
    return {"standing": standing, "internal_body_standing": internal_standing,
        "displayed_body_standing_without_takeoff": displayed_standing, "rows": rows,
        "constraints_without_a_mapped_readout": sorted(set(constraints)-set(FIELDS)),
        "additional_observations": {
            "uncued_gf_spikes": len(receipt["result"].get("gf_spikes_ms", [])),
            "whole_window_heave_fraction": walk.get("heave_frac") if standing else None,
            "heave_scope": "Whole-window p95-p5 / median height; the registry's ceiling is per stride, so no biological band verdict is assigned here",
            "nonfoot_contact_fraction": walk.get("nonfoot"),
            "mean_absolute_quaternion_w": receipt["result"].get("walking_body", {}).get("up"),
            "physical_final_tilt_deg": walk.get("final_tilt_deg")}}


def compare(receipt, baseline, registry):
    for row in (receipt, baseline):
        if row.get("seed") != 0 or row.get("protocol") != "walk" or row.get("duration_ms") != 6000.:
            raise ValueError("This reader expects the six-second development-seed-zero walk")
        if row.get("review_required"):
            raise ValueError("Resolve receipt review before interpreting this battery")
    current, previous = assess(receipt, registry), assess(baseline, registry)
    inside = "within_band_on_this_replay"
    lost = [name for name, row in previous["rows"].items()
            if row["status"] == inside and current["rows"][name]["status"] != inside]
    gained = [name for name, row in current["rows"].items()
              if row["status"] == inside and previous["rows"][name]["status"] == "outside_band_on_this_replay"]
    gates = {"both_bodies_standing": current["standing"] and previous["standing"],
             "retains_observed_bands": current["standing"] and previous["standing"] and not lost,
             "no_uncued_gf": not receipt["result"]["gf_spikes_ms"],
             "no_uncued_ttm": not any(receipt["ttm_spikes_by_side_ms"].values())}
    for name, field, direction in (("speed_retained", "speed", 1),
                                   ("heave_not_increased", "heave_frac", -1),
                                   ("physical_tilt_not_increased", "final_tilt_deg", -1)):
        a, b = receipt["walking_regression"].get(field), baseline["walking_regression"].get(field)
        gates[name] = bool(gates["both_bodies_standing"] and a is not None and b is not None
                           and math.isfinite(a) and math.isfinite(b) and direction*a >= direction*b)
    return {"scope": "same development replay; no population or full-fly acceptance",
            "candidate": receipt["candidate"], "baseline": baseline["candidate"],
            "candidate_read": current, "baseline_read": previous,
            "lost_in_band_rows": lost, "new_in_band_rows": gained,
            "quiet_joint_gates": gates, "quiet_joint_pass": all(gates.values()),
            "retains_observed_bands": current["standing"] and previous["standing"] and not lost}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", type=Path, nargs="+")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    registry, baseline = json.loads(REGISTRY.read_text()), json.loads(args.baseline.read_text())
    results = {str(path): compare(json.loads(path.read_text()), baseline, registry) for path in args.receipts}
    output = {"registry_sha256": digest(REGISTRY), "reader_sha256": digest(__file__),
              "receipt_sha256": {str(path): digest(path) for path in set(args.receipts+[args.baseline])},
              "results": results}
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    for name, result in results.items():
        print(json.dumps({"receipt": name, "standing": result["candidate_read"]["standing"],
                          "lost_in_band_rows": result["lost_in_band_rows"],
                          "new_in_band_rows": result["new_in_band_rows"]}))
