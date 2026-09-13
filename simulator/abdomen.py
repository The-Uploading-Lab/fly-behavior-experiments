"""Abdomen adapter: BANC abdominal motor neurons -> NeuroMechFly abdomen joints.

Step 2 of `gap.abdomen_actuation` (judge-owned, 2026-09-03). The walker's body
is flygym's NeuroMechFly built with `JointPreset.LEGS_ONLY`, so its five
abdomen links (abdomen12, abdomen3, abdomen4, abdomen5, abdomen6) are welded
to the thorax and the 190 abdominal motor neurons in BANC drive nothing.
flyscore's posture_variability reads the abdomen-thorax vector, which is why
that metric was a CONFLICT on every connectome body: the vector never moved.

What this module does, all of it gated on `th["abd_gain"] > 0` (absent or 0 =
the adapter is not built, the fly is `make_locomotion_fly`'s, bit-identical):

- `make_fly`: the same fly as `make_locomotion_fly` plus the five chain joints
  thorax->abdomen12->...->abdomen6, each with two bending axes, and ten more
  position actuators (kp 45, forcerange +-65, the leg values) appended AFTER
  the 42 leg actuators. Probed 2026-09-03 on flygym 2.1.0: in the abdomen
  joint frame flygym's PITCH (axis y) bends the abdomen dorsoventrally and its
  ROLL (axis z) bends it laterally; YAW (axis x) twists about the long axis
  and moves the tip by 0.05 mm at 0.3 rad on every joint, so it is left out.
  The neutral pose carries no abdomen entries, so neutral is 0 rad on all ten.
- `AbdomenAdapter`: the motor join and the muscle model. The join is a GUESS
  (no-inert-element ranking 3, a principled guess from what the cell type
  does): abdominal muscles are segmental and bilaterally paired, so for each
  joint the dorsoventral drive is the summed activity of both sides' motor
  neurons in that segment and the lateral drive is left minus right. Segments
  fold onto the body's five links as A1+A2 (+ the 18 thoracic_abdominal and
  T3's two abdomen-labelled cells, lane B's guess) -> joint 0, A3 -> 1,
  A4 -> 2, A5 -> 3, A6..A9 -> 4; the 16 cells with no neuromere label join
  joint 0 (the largest group) rather than sit inert. The 4 uterus cells are
  excluded: the body has no uterus, which is the structural reason `zero` is
  allowed here. Each joint's drive matrix column is normalised by its summed
  absolute weight, as the leg matrix is, so the drive is a weighted mean spike
  count per 5 ms times `abd_gain`. `abd_signs` (ten +-1, default +1) is the
  searched polarity per DOF; `abd_range_deg` (default 30) the half-range of
  each joint; tau_joint, drive_scale, tau_act and the filter mode are the leg
  body's. The joints are `body.Body`, first-order, 10 DOFs.

Nothing here is measured. Membership (cell_class) is BANC's; segment and
axis assignment and every sign are hypotheses to be searched, per the
parameter rule, and are labelled so wherever a result is quoted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import body as B

ABD_LINKS = ["abdomen12", "abdomen3", "abdomen4", "abdomen5", "abdomen6"]
# joint k connects CHAIN[k] -> CHAIN[k+1]
CHAIN = ["thorax", *ABD_LINKS]
AXES = ("pitch", "roll")           # flygym names; dorsoventral, lateral
N_JOINT = len(ABD_LINKS)
N_DOF = N_JOINT * len(AXES)
DOF_NAMES = [f"{CHAIN[k]}-{CHAIN[k + 1]}-{ax}"
             for k in range(N_JOINT) for ax in AXES]

ABD_CLASSES = ("abdomen_motor_neuron",
               "thoracic_abdominal_segmental_motor_neuron",
               "unknown_thoracic_abdominal_motor_neuron")
# neuromere -> joint index (lane B's 2026-09-03 guess folded onto five links)
SEGMENT_JOINT = {"T1": 0, "T2": 0, "T3": 0, "A1": 0, "A2": 0, "A3": 1,
                 "A4": 2, "A5": 3, "A6": 4, "A7": 4, "A8": 4, "A9": 4}


def make_fly(name="nmf", *, joint_stiffness=0.05, joint_damping=0.06,
             passive_tarsus_stiffness=7.5, passive_tarsus_damping=1e-2,
             actuator_gain=45.0, actuator_forcerange=(-65.0, 65.0),
             add_adhesion=True, adhesion_gain=40.0):
    """`make_locomotion_fly` plus ten actuated abdomen DOFs after the legs."""
    from flygym.anatomy import (ActuatedDOFPreset, AnatomicalJoint,
                                AxesSet, AxisOrder, BodySegment,
                                JointPreset, PASSIVE_TARSAL_LINKS,
                                RotationAxis, Skeleton)
    from flygym.compose import ActuatorType, KinematicPosePreset, NeuroMechFly
    neutral = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(
        AxisOrder.YAW_PITCH_ROLL)
    axes = AxesSet([RotationAxis(a) for a in AXES])
    joints = JointPreset.LEGS_ONLY.to_joint_list() + [
        AnatomicalJoint(BodySegment(f"c_{CHAIN[k]}"),
                        BodySegment(f"c_{CHAIN[k + 1]}"), axes)
        for k in range(N_JOINT)]
    skeleton = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL,
                        anatomical_joints=joints)
    fly = NeuroMechFly(name=name)
    built = fly.add_joints(skeleton, neutral_pose=neutral,
                           stiffness=joint_stiffness, damping=joint_damping)
    for jointdof, joint in built.items():
        if jointdof.child.link in PASSIVE_TARSAL_LINKS:
            joint.stiffness[0] = passive_tarsus_stiffness
            joint.damping[0] = passive_tarsus_damping
    leg_dofs = skeleton.get_actuated_dofs_from_preset(
        ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
    abd_dofs = [jd for jd in built if jd.child.is_abdomen()]
    if len(leg_dofs) != 42 or len(abd_dofs) != N_DOF:
        raise ValueError((len(leg_dofs), len(abd_dofs)))
    fly.add_actuators(leg_dofs + abd_dofs, ActuatorType.POSITION,
                      neutral_input=neutral, kp=actuator_gain,
                      forcerange=actuator_forcerange)
    if add_adhesion:
        fly.add_leg_adhesion(gain=adhesion_gain)
    order = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
    got = [f"{o.parent.link}-{o.child.link}-{o.axis.value}"
           for o in order[42:]]
    if got != DOF_NAMES:
        raise ValueError(f"abdomen actuator order {got} != {DOF_NAMES}")
    return fly


def abdomen_rows(meta):
    """(banc ids, joint index, side sign) for every abdominal motor neuron
    the adapter drives; the uterus cells are returned separately."""
    mn = meta[(meta["super_class"] == "motor")
              & meta["cell_class"].isin(ABD_CLASSES)]
    uterus = mn[mn["body_part_effector"] == "uterus"]
    mn = mn[mn["body_part_effector"] != "uterus"]
    joint = mn["neuromere"].map(SEGMENT_JOINT).fillna(0).astype(int)
    side = mn["side"].map({"left": 1.0, "right": -1.0}).fillna(0.0)
    return (mn["banc_888_id"].to_numpy(), joint.to_numpy(),
            side.to_numpy(dtype=float), uterus["banc_888_id"].to_numpy())


class AbdomenAdapter:
    def __init__(self, th, net):
        gain = float(th.get("abd_gain", 0.0) or 0.0)
        if not np.isfinite(gain) or gain <= 0.0:
            raise ValueError("abd_gain must be finite and > 0")
        self.gain = gain
        signs = np.asarray(th.get("abd_signs", [1.0] * N_DOF), dtype=float)
        if signs.shape != (N_DOF,) or not np.all(np.isin(signs, (-1.0, 1.0))):
            raise ValueError(f"abd_signs must be {N_DOF} entries of +-1")
        self.signs = signs
        half = np.radians(float(th.get("abd_range_deg", 30.0)))
        if not np.isfinite(half) or half <= 0.0:
            raise ValueError("abd_range_deg must be finite and > 0")
        ids, joint, side, uterus = abdomen_rows(net.meta)
        pos = pd.Series(np.arange(net.N), index=net.ids)
        rows = pos.reindex(ids).to_numpy()
        present = ~np.isnan(rows)
        self.n_missing = int((~present).sum())     # cut away by the caller's net
        self.n_uterus = int(len(uterus))
        self.ids = ids[present]
        self.rows = rows[present].astype(int)
        joint, side = joint[present], side[present]
        M = np.zeros((len(self.rows), N_DOF))
        for k in range(len(self.rows)):
            M[k, 2 * joint[k]] = 1.0            # pitch: both sides add
            M[k, 2 * joint[k] + 1] = side[k]    # roll: left minus right
        self.M = M / np.maximum(np.abs(M).sum(axis=0, keepdims=True), 1.0)
        self.n_rows = len(self.rows)
        lo = np.full(N_DOF, -half)
        hi = np.full(N_DOF, half)
        tau_act = th.get("abd_tau_act_ms")
        self.bod = B.Body(
            tau_joint_ms=float(th.get("abd_tau_joint_ms", th["tau_joint"])),
            drive_scale=float(th["drive_scale"]),
            tau_act_ms=(B.TAU_ACT_MS if tau_act is None else float(tau_act)),
            ranges=(lo, hi), exact_filters=(th.get("filt") == "exact"))

    def pad_mask(self, mask):
        """A leg-DOF mask extended over the ten abdomen actuators (False)."""
        return np.concatenate([np.asarray(mask, bool),
                               np.zeros(N_DOF, dtype=bool)])

    def joint_angles(self, leg_theta):
        return np.concatenate([np.asarray(leg_theta, float), self.bod.theta])

    def step(self, counts, cscale, tick_ms):
        d = self.gain * self.signs * (
            self.M.T @ (np.asarray(counts, dtype=float) * cscale))
        self.bod.step(d, tick_ms)
        return d

    def describe(self):
        return {"n_rows": self.n_rows, "n_missing_from_net": self.n_missing,
                "n_uterus_excluded": self.n_uterus, "gain": self.gain,
                "signs": self.signs.tolist(), "dofs": DOF_NAMES,
                "cells_per_dof": np.abs(np.sign(self.M)).sum(axis=0).tolist()}
