"""BANC leg motor neurons -> NeuroMechFly joint degrees of freedom.

This is the missing link in the chain. Everything upstream (connectome, neuron
sim) produces motor-neuron activity; everything downstream (body model, scorer)
consumes joint angles. This module is the only place the two vocabularies meet.

WHY THIS IS MOSTLY LOOKUP, NOT GUESSWORK
BANC annotates each motor neuron with the peripheral muscle it innervates
(`peripheral_target_type`), which leg it belongs to (`body_part_effector`) and
which side (`side`). Fly leg muscle -> joint anatomy is textbook. So the map is
a join, not a model, for 391 of 391 leg motor neurons. Two independent columns
agree on leg identity (`nerve` prefix, 383/383 where present) and on side
(391/391), which is why this is stated as measurement rather than hypothesis.

WHAT IS ACTUALLY UNCERTAIN, AND HOW IT IS HANDLED
Not *which* joint a muscle drives — that is anatomy — but the *sign convention*
of NeuroMechFly's joint angles, which is a property of someone else's URDF and
is not documented per-axis. So:

  - RELATIVE sign within a joint is asserted and is anatomically certain: a
    flexor and an extensor pull in opposite directions. That is the load-bearing
    claim here.
  - ABSOLUTE sign per DOF is NOT asserted. `SIGN_FLIP` is 42 free parameters,
    one per DOF, defaulting to +1. Fitting them is a 42-bit search, and
    `calibrate_signs_against_cpg` solves it directly against the known-good CPG
    trajectory instead.

Per AGENTS.md's parameter rule, nothing is deleted to make this tidy:
  - Each of the 17 muscles keeps its own gain even where three of them pull on
    the same joint in the same direction. They are not merged.
  - Muscles that plausibly span two joints get an entry on both, the second
    marked `hypothesised` with gain 0 — present in the search space, inert in
    the baseline, so a fit can turn it on without the map having to be rewritten.
  - The 414 non-leg motor neurons (wing, neck, haltere, abdomen, proboscis,
    pharynx, antenna...) are carried in the table with `dof=None` rather than
    dropped. The locomotion body model actuates 42 leg DOFs and nothing else, so
    they have no target today; they will when the head or wings are added.
"""
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# the 42 DOFs, in flygym's own order
# --------------------------------------------------------------------------

# flygym leg prefixes, in the order get_actuated_jointdofs_order() returns them.
LEGS = ["lf", "lm", "lh", "rf", "rm", "rh"]

# Per-leg DOFs in flygym's order. Names are anatomical:
#   ThC  thorax-coxa          (3 axes: yaw, pitch, roll)
#   CTr  coxa-trochanter      (2 axes: pitch, roll)  NeuroMechFly fuses
#                             trochanter+femur into one 'trochanterfemur'
#                             segment, so the TrF rotation the femur reductor
#                             drives lands on CTr roll.
#   FTi  femur-tibia          (1 axis: pitch)  the 'knee'
#   TiTa tibia-tarsus         (1 axis: pitch)
JOINT_AXES = [
    "ThC_yaw",
    "ThC_pitch",
    "ThC_roll",
    "CTr_pitch",
    "CTr_roll",
    "FTi_pitch",
    "TiTa_pitch",
]

DOF_NAMES = [f"{leg}_{ja}" for leg in LEGS for ja in JOINT_AXES]
DOF_INDEX = {name: i for i, name in enumerate(DOF_NAMES)}
N_DOF = len(DOF_NAMES)  # 42

# BANC body_part_effector + side -> flygym leg prefix
LEG_KEY = {
    ("front_leg", "left"): "lf",
    ("middle_leg", "left"): "lm",
    ("hind_leg", "left"): "lh",
    ("front_leg", "right"): "rf",
    ("middle_leg", "right"): "rm",
    ("hind_leg", "right"): "rh",
}

# --------------------------------------------------------------------------
# muscle -> joint, the anatomy
# --------------------------------------------------------------------------

# Each entry: muscle -> list of (joint_axis, relative_sign, hypothesised)
#
# relative_sign is +1/-1 WITHIN a joint only. Its job is to make antagonists
# oppose each other; its absolute polarity is fixed later by SIGN_FLIP.
#
# hypothesised=True means "this muscle plausibly also acts here, but the
# baseline does not claim it" — the entry exists so a parameter search can use
# it, and carries gain 0 until something turns it on. See AGENTS.md.
#
# ---------------------------------------------------------------------------
# THE ONE PLACE THIS MAP IS GENUINELY UNDERDETERMINED: which ThC axis
# ---------------------------------------------------------------------------
# Muscle -> JOINT is anatomy and is certain. Muscle -> which of NeuroMechFly's
# three thorax-coxa AXES is not, because pitch/roll/yaw at ThC are directions
# in the body model's local frame — a modelling choice in someone else's URDF —
# and the fly's promotor/remotor/rotator/adductor axes do not line up with them
# one-to-one. The geometry also differs by leg: the T1 coxa swings, while T2/T3
# coxae are more fixed and achieve protraction largely by rotating.
#
# Committing to a guess here was measurably harmful: assigning the 36 rotator
# MNs to yaw alone left rm_ThC_pitch with ZERO motor neurons and the other four
# middle/hind ThC_pitch DOFs with one each, despite ThC_pitch travelling ~31
# degrees per step cycle in the recorded kinematics. That is a DOF the
# connectome could not drive at all, at any parameter setting.
#
# So every ThC muscle gets an entry on all three ThC axes: the anatomically
# favoured axis at gain 1.0, the other two hypothesised at gain 0. The baseline
# is unchanged from the anatomical reading, no ThC DOF is structurally
# unreachable, and a fit can redistribute without the map being rewritten.
# This is the parameter rule applied literally: widen the space, do not guess.
MUSCLE_TO_DOF = {
    # --- thorax-coxa: swings the whole leg relative to the body ---
    "tergopleural_promotor_muscle": [
        ("ThC_pitch", +1, False),   # promotion, anatomically favoured
        ("ThC_yaw", +1, True),
        ("ThC_roll", +1, True),
    ],
    "pleural_remotor_and_abductor_muscle": [
        ("ThC_pitch", -1, False),   # remotion: swings the coxa backward
        ("ThC_roll", +1, False),    # ...and abducts it. Named for both actions.
        ("ThC_yaw", -1, True),
    ],
    "sternal_anterior_rotator_muscle": [
        ("ThC_yaw", +1, False),
        ("ThC_pitch", +1, True),    # rotation reads as protraction in T2/T3
        ("ThC_roll", +1, True),
    ],
    "sternal_posterior_rotator_muscle": [
        ("ThC_yaw", -1, False),
        ("ThC_pitch", -1, True),    # ...and as retraction
        ("ThC_roll", -1, True),
    ],
    "sternal_adductor_muscle": [
        ("ThC_roll", -1, False),
        ("ThC_pitch", -1, True),
        ("ThC_yaw", -1, True),
    ],

    # --- coxa-trochanter: levation/depression, the power stroke ---
    # Extension at CTr drives the leg down against the ground (stance);
    # flexion lifts it (swing). Three separate extensors are kept separate.
    # Sign convention here follows FANC Table S5 (extension positive).
    "trochanter_extensor_muscle": [("CTr_pitch", +1, False)],
    "tergotrochanter_extensor_muscle": [
        # TDT / tergotrochanteral, the giant-fibre jump-and-escape muscle. It
        # does extend CTr, but it is an escape effector rather than a walking
        # one, so expect it near-silent in a walking fit.
        ("CTr_pitch", +1, False),
        # Originates on the tergum, so it spans the thorax-coxa joint too.
        ("ThC_pitch", +1, True),
    ],
    "sternotrochanter_extensor_muscle": [
        ("CTr_pitch", +1, False),
        ("ThC_pitch", +1, True),   # sternal origin, same argument
    ],
    "trochanter_flexor_muscle": [("CTr_pitch", -1, False)],
    "accessory_trochanter_flexor_muscle": [("CTr_pitch", -1, False)],

    # --- trochanter-femur rotation ---
    # The TrF joint is FUSED in the adult fly, and NeuroMechFly fuses
    # trochanter and femur into one 'trochanterfemur' segment to match. FANC
    # lists the femur reductor's function as unknown. CTr roll is the only
    # rotational DOF left for it to act on, so that is where it goes, but this
    # is placement by elimination rather than by measured action.
    "femur_reductor_muscle": [("CTr_roll", +1, False)],

    # --- femur-tibia: the knee ---
    "tibia_extensor_muscle": [("FTi_pitch", +1, False)],
    "tibia_flexor_muscle": [("FTi_pitch", -1, False)],
    "accessory_tibia_flexor_muscle": [("FTi_pitch", -1, False)],

    # --- tibia-tarsus ---
    "tarsus_levator_muscle": [("TiTa_pitch", +1, False)],
    "tarsus_depressor_muscle": [("TiTa_pitch", -1, False)],
    # CONTESTED — see CONTESTED below. FANC Table S5 gives the long tendon
    # muscle's function as claw retraction, and the body model has no claw DOF.
    # Its tendon does run the length of the tibia and tarsus, so tarsal flexion
    # is a real secondary action, which is why it stays on TiTa at full gain.
    "long_tendon_muscle": [
        ("TiTa_pitch", -1, False),
        ("FTi_pitch", -1, True),   # ltm2 belly sits in the femur, crossing FTi
    ],
}

# CD2-HF (lane A, 2026-09-08): individual-muscle search coordinates.  The
# FlyMimic source stores fitted maximum-isometric-force values rather than the
# underlying physiological cross-sectional areas, so 0.3..3.0 is carried as a
# model-derived relative-gain box, not as a measurement.  It is the same fixed
# scale-factor interval used by Oezdil et al. during their muscle fit.  The five
# represented muscles absent from that model use it as an explicitly assumed
# physical-plausibility box.  Every default remains exactly 1.0.
MUSCLE_GAIN_MIN = 0.3
MUSCLE_GAIN_MAX = 3.0
MUSCLE_GAIN_DEFAULT = 1.0
MUSCLE_GAIN_NAMES = tuple(MUSCLE_TO_DOF)

# Muscles whose flygym DOF assignment a primary source disputes. They keep
# their baseline gain — deleting 48 of 391 leg motor neurons to tidy up the
# story is exactly what the parameter rule forbids — but any result that turns
# on TiTa needs to be read with this in view.
CONTESTED = {
    "long_tendon_muscle": (
        "FANC Table S5 gives its function as claw retraction. The body model "
        "has no claw/pretarsus DOF, so its true target is unrepresentable; it "
        "is carried on TiTa_pitch, its secondary action. 48 of 391 leg MNs."
    ),
    "femur_reductor_muscle": (
        "TrF is fused in the adult and FANC lists this muscle's function as "
        "unknown. Placed on CTr_roll by elimination. 34 of 391 leg MNs."
    ),
}

# 42 free parameters, one per DOF. Not fitted here; see
# calibrate_signs_against_cpg(). Default +1 = "assume flygym's positive
# direction matches the anatomical positive direction", which is a guess.
SIGN_FLIP = {name: +1.0 for name in DOF_NAMES}


def measured_sign_flip():
    """DEFAULT-OFF capability (re-freeze event 2026-08-31): the 23 signs
    CD1-MH-R1 measured from body geometry (judge-verified; six FTi_pitch
    inverted), with the 19 undeterminable bits at the +1 pin.
    walk_search.evaluate selects them only for signs="measured"; an absent
    selector keeps the historical +1 pins. Returns a dict shaped like
    SIGN_FLIP."""
    import json as _json
    import os as _os
    p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                      "lanes", "A", "results-cd1mh-r1-signs.json")
    with open(p, encoding="utf-8") as stream:
        art = _json.load(stream)
    measured = art["sign_flip_proposed"]
    out = {name: +1.0 for name in DOF_NAMES}
    out.update({k: float(v) for k, v in measured.items()})
    return out

LEG_EFFECTORS = ("front_leg", "middle_leg", "hind_leg")


# --------------------------------------------------------------------------
# building the map
# --------------------------------------------------------------------------

def _wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def load_meta(path="data/banc_888_meta.feather"):
    return pd.read_feather(path)


def build_map(meta):
    """One row per (motor neuron, DOF) assignment.

    Non-leg motor neurons appear once with dof=NaN so the table is a complete
    census of super_class=='motor', not just the part that currently has
    somewhere to go.
    """
    mn = meta[meta["super_class"] == "motor"]
    rows = []

    for r in mn.itertuples(index=False):
        eff, side = r.body_part_effector, r.side
        muscle = r.peripheral_target_type
        base = {
            "banc_888_id": r.banc_888_id,
            "cell_type": r.cell_type,
            "body_part_effector": eff,
            "side": side,
            "muscle": muscle,
            "neuromere": r.neuromere,
        }

        if eff not in LEG_EFFECTORS:
            rows.append({**base, "leg": None, "dof": None, "joint_axis": None,
                         "sign": np.nan, "gain": np.nan, "hypothesised": False,
                         "status": "non_leg_effector"})
            continue

        leg = LEG_KEY.get((eff, side))
        if leg is None:
            rows.append({**base, "leg": None, "dof": None, "joint_axis": None,
                         "sign": np.nan, "gain": np.nan, "hypothesised": False,
                         "status": "no_side"})
            continue

        targets = MUSCLE_TO_DOF.get(muscle)
        if not targets:
            # Kept, not dropped: a leg MN with an unrecognised or missing muscle
            # is a real neuron with a real effect we cannot place yet.
            rows.append({**base, "leg": leg, "dof": None, "joint_axis": None,
                         "sign": np.nan, "gain": np.nan, "hypothesised": False,
                         "status": "unmapped_muscle"})
            continue

        for joint_axis, sign, hypo in targets:
            dof = f"{leg}_{joint_axis}"
            rows.append({**base, "leg": leg, "dof": dof,
                         "joint_axis": joint_axis,
                         "sign": float(sign),
                         "gain": 0.0 if hypo else 1.0,
                         "hypothesised": hypo,
                         "status": "mapped"})

    return pd.DataFrame(rows)


def to_matrix(mapping, include_hypothesised=False, sign_flip=None):
    """Signed (n_mn, 42) matrix M. joint_drive = M.T @ mn_activity.

    Rows are indexed by the returned `mn_ids` order. Only motor neurons with at
    least one DOF assignment get a row; the rest have no effect on the body by
    construction, and carrying 414 all-zero rows into every matmul helps nobody.
    They remain in `mapping`.
    """
    sf = SIGN_FLIP if sign_flip is None else sign_flip
    use = mapping[mapping["dof"].notna()]
    if not include_hypothesised:
        use = use[~use["hypothesised"]]

    mn_ids = sorted(use["banc_888_id"].unique())
    row_of = {i: k for k, i in enumerate(mn_ids)}

    M = np.zeros((len(mn_ids), N_DOF), dtype=np.float64)
    for r in use.itertuples(index=False):
        M[row_of[r.banc_888_id], DOF_INDEX[r.dof]] += (
            r.sign * r.gain * sf[r.dof]
        )
    return M, mn_ids


def report(mapping):
    """Coverage numbers. Every claim in the docstring should be checkable here."""
    out = []
    A = out.append
    mn_total = mapping["banc_888_id"].nunique()
    leg = mapping[mapping["body_part_effector"].isin(LEG_EFFECTORS)]
    leg_ids = leg["banc_888_id"].nunique()
    mapped_ids = mapping[mapping["status"] == "mapped"]["banc_888_id"].nunique()

    A("=== MOTOR NEURON -> JOINT DOF MAP ===")
    A(f"motor neurons in BANC (super_class=='motor'): {mn_total}")
    A(f"  leg motor neurons:                          {leg_ids}")
    A(f"  mapped to >=1 DOF:                          {mapped_ids} "
      f"({100*mapped_ids/leg_ids:.1f}% of leg MNs)")
    for st, n in mapping.groupby("status")["banc_888_id"].nunique().items():
        A(f"    status={st:<20s} {n:5d}")

    A("")
    A("=== MOTOR NEURONS PER DOF ===")
    real = mapping[(mapping["status"] == "mapped") & (~mapping["hypothesised"])]
    piv = real.pivot_table(index="joint_axis", columns="leg",
                           values="banc_888_id", aggfunc="nunique",
                           fill_value=0)
    piv = piv.reindex(index=JOINT_AXES, columns=LEGS, fill_value=0)
    A(piv.to_string())
    allmapped = mapping[mapping["status"] == "mapped"]
    A(f"\ndistinct DOFs innervated, baseline (gain>0):  "
      f"{real['dof'].nunique()} / {N_DOF}")
    A(f"distinct DOFs reachable, incl. hypothesised: "
      f"{allmapped['dof'].nunique()} / {N_DOF}")
    empty = sorted(set(DOF_NAMES) - set(real["dof"].dropna()))
    if empty:
        A(f"DOFs with no baseline motor neuron: {len(empty)}")
        for e in empty:
            n_h = allmapped[allmapped["dof"] == e]["banc_888_id"].nunique()
            A(f"    {e}  (reachable via {n_h} hypothesised)")
    unreachable = sorted(set(DOF_NAMES) - set(allmapped["dof"].dropna()))
    A(f"DOFs unreachable at ANY parameter setting: {len(unreachable)}")
    for e in unreachable:
        A(f"    {e}")

    A("")
    A("=== AGONIST / ANTAGONIST BALANCE PER DOF ===")
    A("(a DOF driven in only one direction cannot be controlled bidirectionally)")
    A(f"{'dof':<16} {'n+':>4} {'n-':>4}  note")
    for dof in DOF_NAMES:
        d = real[real["dof"] == dof]
        npos = d[d["sign"] > 0]["banc_888_id"].nunique()
        nneg = d[d["sign"] < 0]["banc_888_id"].nunique()
        note = ""
        if npos and not nneg:
            note = "PUSH ONLY"
        elif nneg and not npos:
            note = "PULL ONLY"
        elif not npos and not nneg:
            note = "no innervation"
        A(f"{dof:<16} {npos:>4} {nneg:>4}  {note}")

    A("")
    A("=== HYPOTHESISED (in the search space, gain 0 in the baseline) ===")
    hyp = mapping[mapping["hypothesised"]]
    if len(hyp):
        A(hyp.groupby(["muscle", "joint_axis"])["banc_888_id"]
          .nunique().to_string())
    A("")
    A("=== CONTESTED ASSIGNMENTS (kept at baseline gain, read results with care) ===")
    for mus, why in CONTESTED.items():
        n = mapping[mapping["muscle"] == mus]["banc_888_id"].nunique()
        A(f"  {mus}  ({n} MNs)")
        for line in _wrap(why, 72):
            A(f"      {line}")
    A("")
    A("=== NON-LEG MOTOR NEURONS (carried, no effector in the body model) ===")
    nonleg = mapping[mapping["status"] == "non_leg_effector"]
    A(nonleg.groupby("body_part_effector")["banc_888_id"].nunique().to_string())
    return "\n".join(out)


if __name__ == "__main__":
    m = load_meta()
    mp = build_map(m)
    print(report(mp))
    M, ids = to_matrix(mp)
    print(f"\nmatrix: {M.shape}  nonzero entries {int((M != 0).sum())}")
    print(f"per-DOF |weight| sum: min {np.abs(M).sum(0).min():.0f} "
          f"max {np.abs(M).sum(0).max():.0f}")
