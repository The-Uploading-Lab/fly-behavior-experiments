"""Compile the registered hemilineage rule into the existing row-patch path.

Assignments: proposed measurement 7d69e5f1-945b-4e35-ada0-7ae41804a11c.
They are marker identities transferred to BANC by annotation, not measured
postsynaptic signs. Glutamate uses the model's existing sign convention.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np
import pandas as pd

import cns

MEASUREMENT = "7d69e5f1-945b-4e35-ada0-7ae41804a11c"
MAP = {**dict.fromkeys("00A 01B 03B 05B 06A 06B 11B 13A 19A".split(), "gaba"),
       **dict.fromkeys("02A 08A 09B 14A 16B 21A".split(), "glutamate"),
       **dict.fromkeys("01A 04B 07B 08B 11A 12A 17A 18B 19B 23B".split(), "acetylcholine")}
HISTORICAL_MAP = {**MAP, **dict.fromkeys("09A 12B 13B".split(), "gaba"),
                  **dict.fromkeys("15B 24A".split(), "glutamate"),
                  **dict.fromkeys("03A 10B 20A.22A".split(), "acetylcholine")}
CLASSES = ("ascending", "ventral_nerve_cord_intrinsic")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def rule(direction="both"):
    return {"mapping": MAP, "classes": list(CLASSES), "direction": direction,
            "measurement": MEASUREMENT}


def present(x):
    return not pd.isna(x) and str(x) not in ("", "nan", "None")


def resolve(meta, spec, transmitter_map):
    if not isinstance(spec, dict) or set(spec) != {"mapping", "classes", "direction", "measurement"}:
        raise ValueError("hemilineage_nt needs mapping, classes, direction, measurement")
    if (spec["mapping"] != MAP or sorted(spec["classes"]) != sorted(CLASSES)
            or spec["measurement"] != MEASUREMENT):
        raise ValueError("hemilineage_nt differs from the registered H01 rule")
    direction = spec["direction"]
    if direction not in ("both", "positive_to_negative", "negative_to_positive", "neutral"):
        raise ValueError("unknown hemilineage_nt sign direction")
    p = cns.SimParams(transmitter_map=transmitter_map)
    nt = cns._resolve_nt(meta, p)
    signs = cns._sign_of(nt, p)
    assigned = meta["hemilineage"].map(HISTORICAL_MAP)
    expected = cns._sign_of(assigned, p)
    # The inherited 557-row census counts opposite known fast signs only.
    # Zero-sign monoamines and unresolved transmitters are outside it.
    disagree = np.isfinite(expected) & (signs == -expected)
    patches, rows, excluded = {}, [], Counter()
    score_keys = [k for k in meta.columns if "neurotransmitter" in k and "score" in k]
    for idx in np.flatnonzero(disagree):
        cell = meta.iloc[idx]
        cls = None if pd.isna(cell["super_class"]) else str(cell["super_class"])
        verified = cell["neurotransmitter_verified"]
        token = assigned.iloc[idx]
        if cls == "motor":
            reason = "motor"
        elif cls == "descending":
            reason = "descending"
        elif present(verified) and str(verified) != token:
            reason = "verified_label_conflict"
        elif cls not in CLASSES:
            reason = cls or "missing_class"
        elif cell["hemilineage"] not in MAP:
            raise ValueError("eligible disagreement outside the registered 25-label map")
        else:
            reason = None
        group = "A" if signs[idx] > 0 else "B"
        selected = reason is None and (direction in ("both", "neutral") or
                    direction == ("positive_to_negative" if group == "A" else "negative_to_positive"))
        corrected = str(nt.iloc[idx]) if direction == "neutral" else str(token)
        if selected:
            if not present(cell["cell_type"]):
                raise ValueError("selected row has no cell type; row patch would relabel it")
            if present(verified):
                raise ValueError("selected row carries a verified label")
            patches[str(cell["banc_888_id"])] = {
                "cell_type": str(cell["cell_type"]), "neurotransmitter": corrected,
                "citation": MEASUREMENT}
        if reason:
            excluded[reason] += 1
        rows.append({"row": int(idx), "root_id": str(cell["banc_888_id"]),
                     "hemilineage": str(cell["hemilineage"]), "super_class": cls,
                     "cell_type": None if pd.isna(cell["cell_type"]) else str(cell["cell_type"]),
                     "predicted": None if pd.isna(cell["neurotransmitter_predicted"]) else str(cell["neurotransmitter_predicted"]),
                     "verified": str(verified) if present(verified) else None,
                     "resolved": str(nt.iloc[idx]), "assigned": str(token),
                     "corrected": corrected if selected else str(nt.iloc[idx]),
                     "original_sign": float(signs[idx]), "assigned_sign": float(expected[idx]),
                     "group": group, "selected": selected, "exclusion": reason,
                     "direction_excluded": reason is None and not selected,
                     "prediction_scores": {k: None if pd.isna(cell[k]) else float(cell[k]) for k in score_keys}})
    eligible = [r for r in rows if r["exclusion"] is None]
    outside = []
    for idx in np.flatnonzero(assigned.notna().to_numpy() & ~disagree):
        if str(nt.iloc[idx]) == str(assigned.iloc[idx]):
            continue
        cell = meta.iloc[idx]
        outside.append({"row":int(idx), "root_id":str(cell["banc_888_id"]),
                        "hemilineage":str(cell["hemilineage"]),
                        "super_class":None if pd.isna(cell["super_class"]) else str(cell["super_class"]),
                        "resolved":str(nt.iloc[idx]), "assigned":str(assigned.iloc[idx]),
                        "resolved_sign":float(signs[idx]) if np.isfinite(signs[idx]) else None,
                        "assigned_sign":float(expected[idx]), "selected":False,
                        "exclusion":"same_sign_transmitter_disagreement" if signs[idx] == expected[idx]
                                    else "neutral_or_unresolved_sign"})
    audit = {"rule": spec, "rule_sha256": digest(spec), "rows": rows,
             "disagreements": len(rows), "eligible": len(eligible),
             "eligible_groups": dict(Counter(r["group"] for r in eligible)),
             "eligible_classes": dict(Counter(r["super_class"] for r in eligible)),
             "exclusions": dict(excluded), "selected": len(patches),
             "outside_census_rows":outside,
             "outside_census_counts":dict(Counter(r["exclusion"] for r in outside)),
             "outside_census_neutral_or_unresolved": int((np.isfinite(expected) &
                 ((signs == 0) | ~np.isfinite(signs))).sum()),
             "resolved_mask_sha256": digest(sorted(patches))}
    return patches, audit


def qualify(audit):
    expected = {"motor": 93, "descending": 2, "verified_label_conflict": 1,
                "visceral_circulatory": 3, "missing_class": 7}
    if (audit["disagreements"] != 557 or audit["eligible"] != 451 or audit["exclusions"] != expected
            or audit["eligible_groups"] != {"A": 364, "B": 87}
            or audit["eligible_classes"] != {"ascending": 184, "ventral_nerve_cord_intrinsic": 267}
            or any(r["hemilineage"] == "13B" and r["exclusion"] is None for r in audit["rows"])):
        raise ValueError("registered hemilineage recount drifted")
