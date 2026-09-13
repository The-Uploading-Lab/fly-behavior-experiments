#!/usr/bin/env python3
"""FL2: can the flybody body lift its own weight with the animal's wing kinematics?

Robin (2026-09-09, live): flight is Lane A's goal, and the work runs top-down from
the target. Step one of that order is the body: what wing motion must the motor
layer deliver, and does the body model turn that motion into weight-supporting
lift? This runner asks exactly that, with no nervous system in the loop. It is
the labelled body-calibration control the 14:49 goal names: prescribed wing
motion, no connectome, no flight claim.

Body: flygym 2.1's FlyBody (Vaxenburg et al. 2025, Nature 2025,
doi:10.1038/s41586-025-09029-4; source TuragaLab/flybody, Apache-2.0) composed
in Lane A's namespace with the released FLIGHT configuration of flybody's
`Flying` task (tasks/base.py, tasks/constants.py, fruitfly/fruitfly.py), which
flygym does not ship:
  - the wing fluid ellipsoids are re-added (flygym's parser drops every geom whose
    name contains "fluid" or "inertial"), with fluidshape=ellipsoid and
    fluidcoef=[1.0, 0.5, 1.5, 1.7, 1.0], flybody's flight values;
  - the 8 ug wing mass sits on the inertial box and the membrane mesh is massless,
    as in the source XML (flygym moved the mass onto the mesh);
  - the wing bodies are re-framed so the stroke plane is horizontal at a 47.5 deg
    body pitch (flybody `change_body_frame` with the `hover_up_dir` site, body
    pitch 47.5, stroke plane 0). This replaces flygym's 90 deg yaw correction of
    the wing bodies, which rotates the hinge axes out of flybody's frame;
  - air in mm-g-s units: density 1.28e-6 g/mm3, viscosity 1.85e-5 g/(mm s).
    flygym's mujoco_globals.yaml carries the cgs numbers 0.00128 / 0.000185
    unscaled, which in a mm model is water, not air;
  - wing joints: stiffness 1.0, damping 0.777, armature 1e-4 (flybody Flying
    values in mm-g-s), ranges from flybody; position actuators kp 1800 with the
    torque limit 1800, the same PD as flybody's `action += (target - qpos)` with
    gain 18 in cm units;
  - physics dt 5e-5 s, control dt 2e-4 s (flybody flight constants).
Wing kinematics: flybody's base wingbeat pattern `wing_pattern_fmech.npy`
(TuragaLab/FlySuite data; one cycle, 500 x [yaw, roll, pitch] rad, both wings
identical), which follows the hovering Drosophila melanogaster wing pattern of
Dickson et al. 2008, played at 218 Hz (Fry et al. 2005). No policy, no feedback.

Arms (same body, same kinematics):
  ellipsoid    flybody's released flight air model (ellipsoid fluid on the wings).
  inertia_box  ablation: ellipsoid model off, so the wing body falls back to
               MuJoCo's inertia-based box drag. Labels how much of the lift is
               the Kutta/Magnus/added-mass terms.

Protocol per arm: 30 wingbeats tethered (root pose reset after every physics
step; the total aerodynamic force and torque on the fly are read from
`qfrc_fluid` on the free joint's six DoFs, force in g mm/s2 = uN, torque
about the centre of mass in g mm2/s2 = nN m), then 50 wingbeats released from
rest under gravity. Mechanical power at the wing hinges and power lost to the
air are read every step. `--dt-scan` repeats the tethered lift at 1e-4, 5e-5
and 2.5e-5 s to show convergence in the time step.

Pre-set reading, written before the run:
  LIFTS    mean vertical fluid force over the last 20 tethered cycles within 30%
           of the weight (0.985 mg x 9810 mm/s2 = 9.66 uN) AND the released
           body's centre of mass is higher after 20 cycles than at release.
  PARTIAL  one of the two holds.
  FALLS    neither holds. Then the body blocker is the result, and FL2b scans
           frequency and amplitude before any blade-element fallback.
No seed: the model and the drive are deterministic. No CNS run: this is a body
test, so there is no Metal engine identity; the engine is MuJoCo, reported by
version. No shared code changes; everything lives in lanes/A.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import mujoco as mj

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ASSETS = HERE / "fl2_assets"
PATTERN_PATH = ASSETS / "wing_pattern_fmech.npy"
PATTERN_SHA256 = "f97b975ef1b5adbe42c208ea9665f3c37fa432914f56f3943f1697cdb090d1ce"
SOURCES = {
    "flybody_repo": "https://github.com/TuragaLab/flybody (main, fetched 2026-09-09)",
    "flybody_license": "Apache-2.0 (LICENSE in the repository root)",
    "flybody_paper": "Vaxenburg et al. 2025, Nature, doi:10.1038/s41586-025-09029-4",
    "flight_config_files": ["flybody/tasks/base.py (class Flying)",
                            "flybody/tasks/constants.py (_WING_PARAMS, _FLY_* timesteps)",
                            "flybody/fruitfly/fruitfly.py (flight frames, change_body_frame)"],
    "pattern_repo": "https://github.com/TuragaLab/FlySuite (data/wing_pattern_fmech.npy)",
    "pattern_license": "MPL-2.0 (FlySuite LICENSE); the 6 KB file is redistributed unchanged in lanes/A/fl2_assets with this attribution",
    "pattern_url": "https://github.com/TuragaLab/FlySuite/raw/main/data/wing_pattern_fmech.npy",
    "pattern_provenance": "flybody methods: baseline WPG pattern follows Dickson et al. 2008 hovering D. melanogaster",
    "wingbeat_frequency": "218 Hz, flybody _WING_PARAMS base_freq (Fry et al. 2005 D. melanogaster)",
    "flygym": "flygym 2.1.0 pip package, flygym/compose/fly/flybody.py and flygym/flybody/parse_flybody.py",
}

import flygym  # noqa: E402
from flygym.compose.fly import FlyBody  # noqa: E402
from flygym.compose.world import FlatGroundWorld  # noqa: E402
from flygym.compose.pose import KinematicPose  # noqa: E402
from flygym.flybody.anatomy_flybody import (  # noqa: E402
    FlyBodyAnatomicalJoint, FlyBodyAxisOrder, FlyBodyBodySegment, FlyBodyJointPreset,
    FlyBodySkeleton, WingFlyBodyAxesSet, WingFlyBodyRotationAxis)
from flygym.utils.math import Rotation3D  # noqa: E402

# ---- flybody flight constants, cm-g-s -> mm-g-s -------------------------------
CM = 10.0                        # length x10
TORQUE = 100.0                   # M L^2 quantities x100 when L x10
AIR_DENSITY = 1.28e-6            # g/mm3  (1.28 kg/m3)
AIR_VISCOSITY = 1.85e-5          # g/(mm s) (1.85e-5 Pa s)
GRAVITY = 9810.0                 # mm/s2
PHYSICS_DT = 5e-5                # s, flybody _FLY_PHYSICS_TIMESTEP
CONTROL_DT = 2e-4                # s, flybody _FLY_CONTROL_TIMESTEP
BASE_FREQ_HZ = 218.0             # flybody _WING_PARAMS base_freq (Fry et al. 2005)
BODY_PITCH_DEG = 47.5            # flybody _BODY_PITCH_ANGLE (Muijres et al. 2014)
STROKE_PLANE_DEG = 0.0
FLUIDCOEF = (1.0, 0.5, 1.5, 1.7, 1.0)      # blunt, slender, angular, Kutta, Magnus
WING_GAIN = 18.0 * TORQUE                  # 1800 g mm2/s2 per rad, and the torque limit
WING_STIFFNESS = 0.01 * TORQUE             # 1.0
WING_DAMPING = 0.007769230 * TORQUE        # 0.777
WING_ARMATURE = 1e-6 * TORQUE              # 1e-4 g mm2
WING_SPRINGREF = {"yaw": 1.5, "roll": 0.7, "pitch": -1.0}   # folded rest pose
HOVER_UP_DIR_QUAT = (0.915, 0.0, 0.403, 0.0)               # thorax site, fruitfly.xml
WING_MASS_G = 8e-6
# fruitfly.xml wing fluid / inertial geoms, cm: identical size, pos, quat per side.
WING_GEOM = {
    "l": dict(size=(0.0005, 0.0551, 0.114), pos=(0.0263, -0.148, -0.0289),
              quat=(-0.685, -0.634, 0.265, -0.243)),
    "r": dict(size=(0.0005, 0.0551, 0.114), pos=(-0.0263, 0.148, 0.0289),
              quat=(0.243, 0.265, 0.634, -0.685)),
}
SIDE_QUAT = {"l": (0.0, 0.0, 0.0, 1.0), "r": (0.0, -1.0, 0.0, 0.0)}  # fruitfly.py
AXES = ("yaw", "roll", "pitch")
SPAWN_Z = 12.0                   # mm; the ground never matters
TETHER_CYCLES, READ_CYCLES, FREE_CYCLES = 30, 20, 50
LIFT_TOLERANCE = 0.30
CLIMB_CYCLES = 20
ARMS = {"ellipsoid": True, "inertia_box": False}
DT_SCAN = (1e-4, 5e-5, 2.5e-5)
UW_PER_UNIT = 1e-3               # g mm2/s3 = 1e-9 W = 1e-3 uW

FLYGYM_DIR = Path(flygym.__file__).resolve().parent

# tethered record columns
T_T, T_F, T_TAU, T_Q, T_CTRL, T_FORCE, T_PS, T_PA, T_TIP, T_PD = (
    0, slice(1, 4), slice(4, 7), slice(7, 13), slice(13, 19), slice(19, 25), 25, 26, slice(27, 30), 30)
T_COLS = 31
# free record columns
R_T, R_F, R_TAU, R_COM, R_TZ, R_HZ, R_PS, R_PA, R_PD = (
    0, slice(1, 4), slice(4, 7), slice(7, 10), 10, 11, 12, 13, 14)
R_COLS = 15


# ---- small helpers -----------------------------------------------------------------
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def qunit(a):
    """Unit quaternion. flybody's XML quats carry 3 decimals (|q| = 0.9998); MuJoCo
    normalises them at compile, so all frame arithmetic here does the same first."""
    a = np.asarray(a, float)
    return a / np.linalg.norm(a)


def qmul(a, b):
    r = np.zeros(4)
    mj.mju_mulQuat(r, qunit(a), qunit(b))
    return r


def qneg(a):
    r = np.zeros(4)
    mj.mju_negQuat(r, qunit(a))
    return r


def qrot(v, q):
    r = np.zeros(3)
    mj.mju_rotVecQuat(r, np.asarray(v, float), qunit(q))
    return r


def load_pattern() -> np.ndarray:
    require(sha256(PATTERN_PATH) == PATTERN_SHA256,
            f"wing pattern hash mismatch: {PATTERN_PATH}")
    pattern = np.load(PATTERN_PATH).astype(float)
    require(pattern.shape == (500, 3), f"pattern shape {pattern.shape}")
    return pattern


def pattern_at(pattern: np.ndarray, phase: float) -> np.ndarray:
    """Wing angles [yaw, roll, pitch] at a cycle phase, periodic."""
    n = pattern.shape[0]
    grid = np.arange(n + 1) / n
    wrapped = np.vstack([pattern, pattern[:1]])
    p = phase % 1.0
    return np.array([np.interp(p, grid, wrapped[:, i]) for i in range(3)])


# ---- body ----------------------------------------------------------------------------
class FlightFlyBody(FlyBody):
    """flybody in flygym's composer, keeping flybody's wing frames.

    flygym rotates each wing body 90 deg about the thorax vertical so that zero
    joint angle looks like the folded rest pose. That moves the hinge axes; the
    flight rig below sets the frames flybody's flight tasks use instead.
    """

    def _correct_wing_default_pose(self) -> None:  # noqa: D401
        return


def wing_body(fly: FlyBody, side: str):
    return fly.bodyseg_to_mjcfbody[FlyBodyBodySegment(f"{side}_wing")]


def add_wing_fluid_geoms(fly: FlyBody, ellipsoid: bool) -> None:
    for side, g in WING_GEOM.items():
        body = wing_body(fly, side)
        membrane = [geom for geom in body.geoms if geom.name.endswith("_membrane")]
        require(len(membrane) == 1, f"{side} wing membrane geom not found")
        membrane[0].mass = 0.0
        size = np.array(g["size"]) * CM
        pos = np.array(g["pos"]) * CM
        quat = np.array(g["quat"], float)
        body.add_geom(name=f"{side}_wing_inertial", type=mj.mjtGeom.mjGEOM_BOX,
                      size=size, pos=pos, quat=quat, mass=WING_MASS_G,
                      contype=0, conaffinity=0, group=3, rgba=(0, 0, 0, 0))
        fluid = body.add_geom(name=f"{side}_wing_fluid",
                              type=mj.mjtGeom.mjGEOM_ELLIPSOID,
                              size=size, pos=pos, quat=quat, mass=0.0,
                              contype=0, conaffinity=0, group=3,
                              rgba=(0.2, 0.55, 1.0, 0.35))
        fluid.fluid_ellipsoid = 1 if ellipsoid else 0
        fluid.fluid_coefs = np.array(FLUIDCOEF)


def change_body_frame(body, frame_quat) -> None:
    """Port of flybody fruitfly.py change_body_frame (frame_pos = body.pos)."""
    frame_quat = qunit(frame_quat)
    old_quat = qunit(body.quat)
    dquat = qmul(qneg(frame_quat), old_quat)
    body.quat = frame_quat
    children = list(body.geoms) + list(body.sites) + list(body.bodies) + list(body.cameras)
    for child in children:
        child.quat = qmul(dquat, np.array(child.quat, float))
        child.pos = qrot(qrot(np.array(child.pos, float), old_quat), qneg(frame_quat))


def hover_up_quat() -> np.ndarray:
    up = np.array(HOVER_UP_DIR_QUAT)
    delta = math.radians(BODY_PITCH_DEG) - 2.0 * math.acos(up[0])
    return qmul((math.cos(delta / 2), 0.0, math.sin(delta / 2), 0.0), up)


def set_flight_frames(fly: FlyBody) -> np.ndarray:
    """flybody fruitfly.py, use_wings=True: the wing yaw axis becomes hover-up."""
    up = hover_up_quat()
    spa = math.radians(STROKE_PLANE_DEG)
    stroke_plane_quat = (math.cos(spa / 2), 0.0, math.sin(spa / 2), 0.0)
    for side in "lr":
        dquat = qmul(qneg(stroke_plane_quat), SIDE_QUAT[side])
        change_body_frame(wing_body(fly, side), qmul(dquat, qneg(up)))
    return up


def wing_skeleton() -> FlyBodySkeleton:
    joints = [FlyBodyAnatomicalJoint(parent=FlyBodyBodySegment("c_thorax"),
                                     child=FlyBodyBodySegment(f"{side}_wing"),
                                     axes=WingFlyBodyAxesSet(WingFlyBodyRotationAxis))
              for side in "lr"]
    return FlyBodySkeleton(axis_order=FlyBodyAxisOrder.YAW_ROLL_PITCH,
                           anatomical_joints=joints)


# ---- leg pose (Robin, 2026-09-09: the clips showed the legs open; close them) -----------
# "spawn": fruitfly.xml zero pose, legs standing and spread (every FL2-FL3d clip).
# "flight": flybody fruitfly.py, use_legs=False: "Set orientation quaternions to retracted
#           leg position ... body.quat = body_quat_from_springrefs(body)", then the leg joints
#           are removed. The spring references are flybody's rest pose for the legs (and the
#           proboscis), shipped by flygym as pose/flight/yaw_roll_pitch.yaml. Programmatic:
#           the CNS's leg motor output does not yet fold the flight body's legs.
LEG_POSE = "spawn"
LEG_POSES = ("spawn", "flight")
FLIGHT_POSE_PATH = FLYGYM_DIR / "assets/model/flybody/pose/flight/yaw_roll_pitch.yaml"
LEG_LINKS = ("coxa", "trochanterfemur", "tibia", "tarsus1", "tarsus2", "tarsus3", "tarsus4", "tarsus5")
_FLIGHT_POSE_QUATS: dict | None = None


def qaxis(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    return np.concatenate([[math.cos(angle / 2.0)], math.sin(angle / 2.0) * a])


def flight_pose_quats() -> dict:
    """Body quaternions of flybody's retracted pose, keyed by flygym segment name.

    A scratch fly gets flygym's biological joints in yaw-roll-pitch order (the XML's
    abduct-twist-extend order) with the flight pose as spring reference; each leg or
    proboscis body's quaternion is composed with its joints' references in listed order,
    as MuJoCo composes hinge joints and as flybody's body_quat_from_springrefs does.
    """
    global _FLIGHT_POSE_QUATS
    if _FLIGHT_POSE_QUATS is None:
        scratch = FlightFlyBody(name="scratch")
        skeleton = FlyBodySkeleton(axis_order=FlyBodyAxisOrder.YAW_ROLL_PITCH,
                                   joint_preset=FlyBodyJointPreset.ALL_BIOLOGICAL)
        dofs = scratch.add_joints(skeleton, neutral_pose=KinematicPose(path=FLIGHT_POSE_PATH))
        quats, angles = {}, {}
        for dof in dofs:
            seg = dof.child
            if not (seg.is_leg() or seg.is_proboscis()) or seg.name in quats:
                continue
            body = scratch.bodyseg_to_mjcfbody[seg]
            q = qunit(np.array(body.quat, float))
            for joint in body.joints:
                q = qmul(q, qaxis(joint.axis, float(joint.springref)))
                angles[joint.name] = float(joint.springref)
            quats[seg.name] = q
        require(len(quats) == 6 * len(LEG_LINKS) + 4, f"flight pose covers {len(quats)} bodies")   # 48 leg links, rostrum, haustellum, two labra
        _FLIGHT_POSE_QUATS = {"quats": quats, "angles_rad": angles}
    return _FLIGHT_POSE_QUATS


def apply_leg_pose(fly: FlyBody) -> dict:
    require(LEG_POSE in LEG_POSES, f"unknown leg pose {LEG_POSE}")
    if LEG_POSE == "spawn":
        return {}
    pose = flight_pose_quats()
    for seg, q in pose["quats"].items():
        fly.bodyseg_to_mjcfbody[FlyBodyBodySegment(seg)].quat = np.array(q, float)
    return pose["quats"]


def leg_extent(rig) -> dict:
    """Where the legs are, in the thorax frame (mm): lowest point along hover-up, widest
    point sideways, and each tarsus tip. Body frame origins and geom centres, no meshes."""
    m, d = rig.model, rig.data
    R = d.xmat[rig.thorax].reshape(3, 3)
    origin = d.xpos[rig.thorax]
    up = R.T @ rig.hover_axis_world()
    pts, tips = [], {}
    for b in range(m.nbody):
        name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b) or ""
        seg = name.split("/")[-1]
        if not any(seg.endswith("_" + link) for link in LEG_LINKS):
            continue
        p = R.T @ (d.xpos[b] - origin)
        pts.append(p)
        if seg.endswith("_tarsus5"):
            tips[seg] = p.tolist()
        for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]):
            pts.append(R.T @ (d.geom_xpos[g] - origin))
    P = np.array(pts)
    along_up = P @ up
    return {"leg_pose": getattr(rig, "leg_pose", LEG_POSE), "points": int(len(P)),
            "lowest_mm_along_hover_up": float(along_up.min()), "widest_mm_lateral": float(np.abs(P[:, 1]).max()),
            "mean_abs_lateral_mm": float(np.abs(P[:, 1]).mean()), "mean_along_hover_up_mm": float(along_up.mean()),
            "tarsus5_tips_thorax_frame_mm": tips}


def build(ellipsoid: bool, dt: float = PHYSICS_DT):
    """Compose the flight rig; returns (world, fly, spawn_quat, model, data)."""
    fly = FlightFlyBody(name="fb")
    apply_leg_pose(fly)
    add_wing_fluid_geoms(fly, ellipsoid)
    up = set_flight_frames(fly)
    dofs = fly.add_joints(wing_skeleton(), stiffness=WING_STIFFNESS,
                          damping=WING_DAMPING, armature=WING_ARMATURE)
    require(len(dofs) == 6, f"expected 6 wing DoFs, got {len(dofs)}")
    for dof, joint in dofs.items():
        joint.springref = WING_SPRINGREF[dof.axis.value]
    fly.add_actuators(list(dofs.keys()), "position", kp=WING_GAIN,
                      forcelimited=True, forcerange=(-WING_GAIN, WING_GAIN))
    spawn_quat = qneg(up)          # maps the hover-up direction onto world +z
    world = FlatGroundWorld(name="fl2_world")
    world.add_fly(fly, spawn_position=(0.0, 0.0, SPAWN_Z),
                  spawn_rotation=Rotation3D("quat", tuple(float(x) for x in spawn_quat)),
                  bodysegs_with_ground_contact=[], add_ground_contact_sensors=False)
    opt = world.mjcf_root.option
    opt.density = AIR_DENSITY
    opt.viscosity = AIR_VISCOSITY
    opt.timestep = dt
    opt.gravity = np.array([0.0, 0.0, -GRAVITY])
    model, data = world.compile()
    return world, fly, spawn_quat, model, data


def name_id(model, kind, name: str) -> int:
    i = mj.mj_name2id(model, kind, name)
    require(i >= 0, f"{name} not in model")
    return i


def joint_name(side: str, axis: str) -> str:
    return f"fb/c_thorax-{side}_wing-{axis}"


class Rig:
    """Compiled model plus the indices the loop needs."""

    def __init__(self, arm: str, dt: float = PHYSICS_DT):
        self.arm = arm
        self.dt = dt
        self.leg_pose = LEG_POSE
        self.ellipsoid = ARMS[arm]
        self.world, self.fly, self.spawn_quat, self.model, self.data = build(self.ellipsoid, dt)
        m = self.model
        require(abs(m.opt.timestep - dt) < 1e-12, "timestep not applied")
        free = [j for j in range(m.njnt) if m.jnt_type[j] == mj.mjtJoint.mjJNT_FREE]
        require(len(free) == 1, f"expected one free joint, found {len(free)}")
        self.root_qpos = int(m.jnt_qposadr[free[0]])
        self.root_dof = int(m.jnt_dofadr[free[0]])
        self.thorax = name_id(m, mj.mjtObj.mjOBJ_BODY, "fb/c_thorax")
        self.wing_joints = [name_id(m, mj.mjtObj.mjOBJ_JOINT, joint_name(s, a))
                            for s in "lr" for a in AXES]
        self.wing_qpos = np.array([m.jnt_qposadr[j] for j in self.wing_joints])
        self.wing_dof = np.array([m.jnt_dofadr[j] for j in self.wing_joints])
        self.actuators = [name_id(m, mj.mjtObj.mjOBJ_ACTUATOR,
                                  joint_name(s, a) + "-position")
                          for s in "lr" for a in AXES]
        self.fluid_geoms = {s: name_id(m, mj.mjtObj.mjOBJ_GEOM, f"fb/{s}_wing_fluid")
                            for s in "lr"}
        self.wing_bodies = {s: name_id(m, mj.mjtObj.mjOBJ_BODY, f"fb/{s}_wing") for s in "lr"}
        self.mass_g = float(m.body_subtreemass[self.thorax])
        self.weight_uN = self.mass_g * GRAVITY
        self.spawn_qpos = np.concatenate([[0.0, 0.0, SPAWN_Z], self.spawn_quat])
        # exact reproduction of flybody's servo: gain kp, bias -kp*q, torque limit
        for a in self.actuators:
            require(abs(m.actuator_gainprm[a, 0] - WING_GAIN) < 1e-9
                    and abs(m.actuator_biasprm[a, 1] + WING_GAIN) < 1e-9
                    and m.actuator_forcelimited[a] == 1
                    and abs(m.actuator_forcerange[a, 1] - WING_GAIN) < 1e-9,
                    f"actuator {a} is not flybody's kp=1800 servo")
        for j in self.wing_joints:
            require(abs(m.dof_damping[m.jnt_dofadr[j]] - WING_DAMPING) < 1e-9
                    and abs(m.jnt_stiffness[j] - WING_STIFFNESS) < 1e-9,
                    "wing joint passive parameters are not flybody's")
        for s, g in self.fluid_geoms.items():
            flag = m.geom_fluid[g, 0]
            require((flag != 0) == self.ellipsoid, f"{s} wing fluid flag {flag} for {arm}")
        require(abs(m.opt.density - AIR_DENSITY) < 1e-15
                and abs(m.opt.viscosity - AIR_VISCOSITY) < 1e-15, "air is not air")
        # the far end of the fluid ellipsoid from the hinge is the wing tip
        mj.mj_forward(m, self.data)
        self.tip_sign = {}
        for s, g in self.fluid_geoms.items():
            centre = self.data.geom_xpos[g]
            span = self.data.geom_xmat[g].reshape(3, 3)[:, 2] * m.geom_size[g, 2]
            hinge = self.data.xpos[self.wing_bodies[s]]
            far = np.linalg.norm(centre + span - hinge) > np.linalg.norm(centre - span - hinge)
            self.tip_sign[s] = 1.0 if far else -1.0

    def tip(self, side: str) -> np.ndarray:
        g = self.fluid_geoms[side]
        centre = self.data.geom_xpos[g]
        span = self.data.geom_xmat[g].reshape(3, 3)[:, 2] * self.model.geom_size[g, 2]
        return centre + self.tip_sign[side] * span

    def hover_axis_world(self) -> np.ndarray:
        """World direction of the thorax hover-up axis (vertical in hover)."""
        quat = self.data.qpos[self.root_qpos + 3: self.root_qpos + 7]
        return qrot(qrot(np.array([0.0, 0.0, 1.0]), hover_up_quat()), quat)

    def fluid_force_torque(self):
        """Total air force (world, uN) and torque about the CoM (world, nN m).

        The free joint's rotational generalized force is the torque about the
        thorax frame origin expressed in the thorax frame (MuJoCo free-joint
        angular DoFs are body-local); it is moved to the centre of mass here.
        """
        d = self.data
        F = d.qfrc_fluid[self.root_dof: self.root_dof + 3].copy()
        tau_local = d.qfrc_fluid[self.root_dof + 3: self.root_dof + 6]
        R = d.xmat[self.thorax].reshape(3, 3)
        tau_world = R @ tau_local
        arm = d.subtree_com[self.thorax] - d.xpos[self.thorax]
        return F, tau_world - np.cross(arm, F)

    def powers(self):
        """Servo power at the six wing hinges, power into the air, power into hinge damping."""
        d = self.data
        qd = d.qvel[self.wing_dof]
        p_servo = float(np.dot(d.actuator_force[self.actuators], qd))
        p_aero = float(-np.dot(d.qfrc_fluid[self.wing_dof], qd))
        p_damp = float(WING_DAMPING * np.dot(qd, qd))
        return p_servo, p_aero, p_damp

    def reset(self, pattern: np.ndarray) -> None:
        d = self.data
        mj.mj_resetData(self.model, d)
        d.qpos[self.root_qpos: self.root_qpos + 7] = self.spawn_qpos
        angles0 = pattern_at(pattern, 0.0)
        angles1 = pattern_at(pattern, self.dt * BASE_FREQ_HZ)
        for k in range(6):
            d.qpos[self.wing_qpos[k]] = angles0[k % 3]
            d.qvel[self.wing_dof[k]] = (angles1[k % 3] - angles0[k % 3]) / self.dt
        d.ctrl[self.actuators] = np.tile(angles0, 2)
        mj.mj_forward(self.model, d)


# ---- the run -------------------------------------------------------------------------
def simulate(rig: Rig, pattern: np.ndarray, tether_cycles: int, free_cycles: int,
             frame_hook=None) -> dict:
    m, d = rig.model, rig.data
    period = 1.0 / BASE_FREQ_HZ
    steps_per_ctrl = max(1, int(round(CONTROL_DT / rig.dt)))
    rig.reset(pattern)
    rows, free_rows = [], []
    step = 0
    t_end_tether = tether_cycles * period
    t_end = t_end_tether + free_cycles * period
    released = False
    while d.time < t_end - 0.5 * rig.dt:
        if step % steps_per_ctrl == 0:
            d.ctrl[rig.actuators] = np.tile(pattern_at(pattern, d.time * BASE_FREQ_HZ), 2)
        mj.mj_step(m, d)
        step += 1
        F, tau = rig.fluid_force_torque()
        p_servo, p_aero, p_damp = rig.powers()
        if not released:
            row = np.empty(T_COLS)
            row[T_PD] = p_damp
            row[T_T] = d.time
            row[T_F] = F
            row[T_TAU] = tau
            row[T_Q] = d.qpos[rig.wing_qpos]
            row[T_CTRL] = d.ctrl[rig.actuators]
            row[T_FORCE] = d.actuator_force[rig.actuators]
            row[T_PS] = p_servo
            row[T_PA] = p_aero
            row[T_TIP] = rig.tip("l")
            rows.append(row)
            # kinematic tether: the body never moves, the wings integrate freely
            d.qpos[rig.root_qpos: rig.root_qpos + 7] = rig.spawn_qpos
            d.qvel[rig.root_dof: rig.root_dof + 6] = 0.0
            if d.time >= t_end_tether - 0.5 * rig.dt:
                released = True
                mj.mj_forward(m, d)
        else:
            row = np.empty(R_COLS)
            row[R_T] = d.time
            row[R_F] = F
            row[R_TAU] = tau
            row[R_COM] = d.subtree_com[rig.thorax]
            row[R_TZ] = d.xpos[rig.thorax, 2]
            row[R_HZ] = rig.hover_axis_world()[2]
            row[R_PS] = p_servo
            row[R_PA] = p_aero
            row[R_PD] = p_damp
            free_rows.append(row)
        if frame_hook is not None:
            frame_hook(rig, released)
    return {"tethered": np.array(rows), "free": np.array(free_rows).reshape(-1, R_COLS),
            "period_s": period, "steps": step}


def lag_ms(cmd: np.ndarray, got: np.ndarray, dt: float, max_lag_steps: int) -> float:
    """Lag of got behind cmd (ms) by the cross-correlation peak, both mean-removed."""
    c = cmd - cmd.mean()
    g = got - got.mean()
    best, best_lag = -np.inf, 0
    for lag in range(0, max_lag_steps + 1):
        v = float(np.dot(c[: len(c) - lag], g[lag:]))
        if v > best:
            best, best_lag = v, lag
    return best_lag * dt * 1e3


def vec3(v) -> dict:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def tethered_lift(rig: Rig, out: dict, read_cycles: int) -> dict:
    """Cycle-averaged force/torque/power over the last read_cycles tethered beats."""
    period = out["period_s"]
    T = out["tethered"]
    t = T[:, T_T]
    t_read = t[-1] - read_cycles * period
    w = t >= t_read
    per_cycle = []
    for k in range(read_cycles):
        sel = (t >= t_read + k * period) & (t < t_read + (k + 1) * period)
        per_cycle.append(float(T[sel, T_F][:, 2].mean()))
    F = T[w, T_F]
    tau = T[w, T_TAU]
    mean_F = F.mean(axis=0)
    ps, pa, pd = T[w, T_PS], T[w, T_PA], T[w, T_PD]
    return {
        "damping_power_uW": {"mean": float(pd.mean() * UW_PER_UNIT), "peak": float(pd.max() * UW_PER_UNIT)},
        "read_window_cycles": read_cycles,
        "mean_fluid_force_uN": vec3(mean_F),
        "lift_over_weight": float(mean_F[2] / rig.weight_uN),
        "per_cycle_Fz_uN": per_cycle,
        "per_cycle_Fz_sd_uN": float(np.std(per_cycle)),
        "peak_Fz_uN": float(F[:, 2].max()), "min_Fz_uN": float(F[:, 2].min()),
        "mean_torque_about_com_nNm_world": vec3(tau.mean(axis=0)),
        "peak_abs_torque_nNm": vec3(np.abs(tau).max(axis=0)),
        "servo_power_uW": {"mean": float(ps.mean() * UW_PER_UNIT),
                           "mean_positive": float(np.clip(ps, 0, None).mean() * UW_PER_UNIT),
                           "peak": float(ps.max() * UW_PER_UNIT)},
        "aero_power_uW": {"mean": float(pa.mean() * UW_PER_UNIT), "peak": float(pa.max() * UW_PER_UNIT)},
        "servo_power_W_per_kg_body": float(ps.mean() * UW_PER_UNIT / (rig.mass_g * 1e3)),
        "aero_power_W_per_kg_body": float(pa.mean() * UW_PER_UNIT / (rig.mass_g * 1e3)),
    }


def read(rig: Rig, out: dict) -> dict:
    """Lift, torque, power, tracking quality, release outcome, and the verdict."""
    period = out["period_s"]
    T = out["tethered"]
    t = T[:, T_T]
    t_read = t[-1] - READ_CYCLES * period
    w = t >= t_read
    tethered = tethered_lift(rig, out, READ_CYCLES)
    tip = T[w, T_TIP]
    tip_speed = np.linalg.norm(np.diff(tip, axis=0), axis=1) / rig.dt   # mm/s
    tethered["left_tip_speed_mean_m_s"] = float(tip_speed.mean() / 1e3)
    tethered["left_tip_speed_peak_m_s"] = float(tip_speed.max() / 1e3)
    q, ctrl, force = T[w, T_Q], T[w, T_CTRL], T[w, T_FORCE]
    tracking = {}
    for k, (s, a) in enumerate([(s, a) for s in "lr" for a in AXES]):
        err = q[:, k] - ctrl[:, k]
        tracking[f"{s}_{a}"] = {
            "commanded_amplitude_rad": float(ctrl[:, k].max() - ctrl[:, k].min()),
            "achieved_amplitude_rad": float(q[:, k].max() - q[:, k].min()),
            "rms_error_rad": float(np.sqrt(np.mean(err ** 2))),
            "lag_ms": lag_ms(ctrl[:, k], q[:, k], rig.dt, int(0.5 * period / rig.dt)),
            "saturated_fraction": float(np.mean(np.abs(force[:, k]) >= 0.999 * WING_GAIN)),
            "peak_torque": float(np.abs(force[:, k]).max()),
        }
    tethered["tracking"] = tracking
    lift_ratio = tethered["lift_over_weight"]
    within = abs(lift_ratio - 1.0) <= LIFT_TOLERANCE

    R = out["free"]
    free = {"cycles": FREE_CYCLES, "climb_window_cycles": CLIMB_CYCLES}
    climbs = False
    if len(R):
        tf = R[:, R_T] - R[0, R_T]
        com_z = R[:, R_COM][:, 2]
        n_climb = min(max(int(np.searchsorted(tf, CLIMB_CYCLES * period)), 2), len(tf))
        vz_mean = float((com_z[n_climb - 1] - com_z[0]) / (tf[n_climb - 1] - tf[0]))
        climbs = bool(com_z[n_climb - 1] > com_z[0] and vz_mean > 0)
        samples = {}
        for cyc in (0, 10, 20, 30, 40, 50):
            i = min(int(np.searchsorted(tf, cyc * period)), len(tf) - 1)
            samples[str(cyc)] = {"com_z_mm": float(com_z[i]), "thorax_z_mm": float(R[i, R_TZ]),
                                 "hover_axis_z": float(R[i, R_HZ]),
                                 "fluid_Fz_uN": float(R[i, R_F][2])}
        free.update({
            "com_vz_mean_mm_s_first_window": vz_mean,
            "mean_fluid_force_uN_first_window": vec3(R[:n_climb, R_F].mean(axis=0)),
            "mean_torque_about_com_nNm_first_window": vec3(R[:n_climb, R_TAU].mean(axis=0)),
            "max_com_rise_mm": float(com_z.max() - com_z[0]),
            "final_com_drop_mm": float(com_z[0] - com_z[-1]),
            "hover_axis_z_final": float(R[-1, R_HZ]),
            "samples_by_cycle": samples,
            "climbs": climbs,
        })
    verdict = "LIFTS" if (within and climbs) else ("PARTIAL" if (within or climbs) else "FALLS")
    return {"weight_uN": rig.weight_uN, "mass_mg": rig.mass_g * 1e3, "tethered": tethered,
            "free": free, "within_30pct": bool(within), "verdict": verdict, "steps": out["steps"]}


def dt_scan(pattern: np.ndarray, arm: str = "ellipsoid", cycles: int = 20, read_cycles: int = 10) -> dict:
    """Tethered lift at three time steps: convergence check the goal asks for."""
    result = {}
    for dt in DT_SCAN:
        t0 = time.time()
        rig = Rig(arm, dt=dt)
        out = simulate(rig, pattern, cycles, 0)
        lift = tethered_lift(rig, out, read_cycles)
        result[f"{dt:g}"] = {"lift_over_weight": lift["lift_over_weight"],
                             "mean_fluid_force_uN": lift["mean_fluid_force_uN"],
                             "mean_torque_about_com_nNm_world": lift["mean_torque_about_com_nNm_world"],
                             "aero_power_uW_mean": lift["aero_power_uW"]["mean"],
                             "steps": out["steps"], "wall_s": round(time.time() - t0, 2)}
        print(f"[dt-scan {arm}] dt {dt:g}: lift/weight {lift['lift_over_weight']:.4f}, "
              f"F {lift['mean_fluid_force_uN']}, {out['steps']} steps, {result[f'{dt:g}']['wall_s']} s")
    return result


# ---- footage -------------------------------------------------------------------------
class Recorder:
    """The last two tethered wingbeats in slow motion, then the release."""

    def __init__(self, rig: Rig, path: Path, tether_cycles: int):
        from PIL import Image, ImageDraw
        self.Image, self.ImageDraw = Image, ImageDraw
        self.rig, self.path = rig, path
        self.frames = []
        self.renderer = mj.Renderer(rig.model, height=720, width=960)
        self.cam = mj.MjvCamera()
        self.cam.type = mj.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = rig.data.xpos[rig.thorax]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 9.0, 150.0, -12.0
        self.opt = mj.MjvOption()
        self.opt.geomgroup[3] = 1
        self.period = 1.0 / BASE_FREQ_HZ
        self.tether_start = (tether_cycles - 2) * self.period
        self.every_tethered, self.every_free = 2, 40
        self.step = 0
        self.stamp = time.strftime("%Y-%m-%d %H:%M %Z")

    def tempo(self, every: int) -> str:
        sim_per_video_second = every * self.rig.dt * 30
        return f"1/{1 / sim_per_video_second:.0f} speed ({every * self.rig.dt * 1e3:.2f} ms per frame)"

    def __call__(self, rig: Rig, released: bool) -> None:
        d = rig.data
        self.step += 1
        if not released:
            if d.time < self.tether_start or self.step % self.every_tethered:
                return
            tempo = self.tempo(self.every_tethered)
        else:
            if self.step % self.every_free:
                return
            tempo = self.tempo(self.every_free)
            self.cam.lookat[:] = 0.7 * self.cam.lookat + 0.3 * d.xpos[rig.thorax]
        self.renderer.update_scene(d, self.cam, self.opt)
        img = self.Image.fromarray(self.renderer.render())
        draw = self.ImageDraw.Draw(img)
        Fz = d.qfrc_fluid[rig.root_dof + 2]
        lines = [
            f"FL2 flybody lift test, arm {rig.arm}, recorded {self.stamp}",
            f"{'TETHERED' if not released else 'RELEASED'}  t = {d.time * 1e3:7.2f} ms   {tempo}",
            f"vertical air force now {Fz:6.1f} uN   weight {rig.weight_uN:.1f} uN   "
            f"CoM z {d.subtree_com[rig.thorax, 2]:.2f} mm",
            "wings: flybody Dickson-2008 hover pattern at 218 Hz, prescribed; no nervous system",
        ]
        y = 8
        for line in lines:
            draw.text((10, y), line, fill=(20, 20, 20))
            y += 18
        self.frames.append(np.asarray(img))

    def close(self) -> dict:
        import imageio.v2 as imageio
        self.renderer.close()
        imageio.mimwrite(self.path, self.frames, fps=30, codec="libx264",
                         quality=8, macro_block_size=None)
        return {"path": str(self.path.relative_to(ROOT)), "frames": len(self.frames),
                "sha256": sha256(self.path)}


# ---- receipt -------------------------------------------------------------------------
def identity() -> dict:
    return {
        "git_head": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "script_sha256": sha256(Path(__file__)),
        "pattern": {"path": str(PATTERN_PATH.relative_to(ROOT)), "sha256": sha256(PATTERN_PATH)},
        "sources": SOURCES,
        "mujoco": mj.__version__,
        "flygym": {"version": "2.1.0", "dir": str(FLYGYM_DIR),
                   "flybody_py_sha256": sha256(FLYGYM_DIR / "compose/fly/flybody.py"),
                   "parse_flybody_py_sha256": sha256(FLYGYM_DIR / "flybody/parse_flybody.py"),
                   "fruitfly_xml_sha256": sha256(FLYGYM_DIR / "assets/model/flybody/fruitfly.xml"),
                   "mujoco_globals_yaml_sha256": sha256(FLYGYM_DIR / "assets/model/flybody/mujoco_globals.yaml")},
        "python": sys.version.split()[0], "platform": platform.platform(),
        "machine": platform.machine(),
    }


def parameters() -> dict:
    return {
        "air_density_g_mm3": AIR_DENSITY, "air_viscosity_g_mm_s": AIR_VISCOSITY,
        "gravity_mm_s2": GRAVITY, "physics_dt_s": PHYSICS_DT, "control_dt_s": CONTROL_DT,
        "wingbeat_hz": BASE_FREQ_HZ, "body_pitch_deg": BODY_PITCH_DEG,
        "stroke_plane_deg": STROKE_PLANE_DEG, "fluidcoef": list(FLUIDCOEF),
        "wing_servo_kp": WING_GAIN, "wing_torque_limit": WING_GAIN,
        "wing_stiffness": WING_STIFFNESS, "wing_damping": WING_DAMPING,
        "wing_armature": WING_ARMATURE, "wing_springref": WING_SPRINGREF,
        "wing_mass_g": WING_MASS_G, "wing_fluid_geom_cm": WING_GEOM,
        "tether_cycles": TETHER_CYCLES, "read_cycles": READ_CYCLES,
        "free_cycles": FREE_CYCLES, "climb_cycles": CLIMB_CYCLES,
        "lift_tolerance": LIFT_TOLERANCE, "spawn_z_mm": SPAWN_Z, "dt_scan_s": list(DT_SCAN),
        "units": {"force": "g mm/s2 = uN", "torque": "g mm2/s2 = nN m", "power": "uW; W/kg = uW per mg body"},
        "pre_set_reading": {
            "LIFTS": "mean Fz within 30% of weight over the last 20 tethered cycles AND CoM higher after 20 free cycles",
            "PARTIAL": "one of the two", "FALLS": "neither"},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--render", action="store_true", help="MP4 of the first arm listed")
    ap.add_argument("--dt-scan", action="store_true", help="tethered lift at three time steps")
    ap.add_argument("--smoke", action="store_true", help="3 + 3 cycles, no receipt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        require(a in ARMS, f"unknown arm {a}")
    tether_cycles, free_cycles = (3, 3) if args.smoke else (TETHER_CYCLES, FREE_CYCLES)
    pattern = load_pattern()
    started = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    receipt = {"experiment": "FL2 flybody lift", "started": started,
               "identity": identity(), "parameters": parameters(), "arms": {}}
    for i, arm in enumerate(arms):
        t0 = time.time()
        rig = Rig(arm)
        recorder = None
        if args.render and i == 0 and not args.smoke:
            recorder = Recorder(rig, HERE / f"fl2_flybody_lift_{arm}.mp4", tether_cycles)
        out = simulate(rig, pattern, tether_cycles, free_cycles, frame_hook=recorder)
        if args.smoke:
            n = int(out["period_s"] / rig.dt)
            result = {"weight_uN": rig.weight_uN, "mass_mg": rig.mass_g * 1e3,
                      "mean_F_uN_last_cycle": vec3(out["tethered"][-n:, T_F].mean(axis=0)),
                      "mean_tau_last_cycle": vec3(out["tethered"][-n:, T_TAU].mean(axis=0)),
                      "servo_power_uW": float(out["tethered"][-n:, T_PS].mean() * UW_PER_UNIT),
                      "free_com_z_start_end": [float(out["free"][0, R_COM][2]), float(out["free"][-1, R_COM][2])],
                      "steps": out["steps"]}
        else:
            result = read(rig, out)
        result["wall_s"] = round(time.time() - t0, 2)
        result["model"] = {"nbody": rig.model.nbody, "nv": rig.model.nv, "nu": rig.model.nu,
                           "ngeom": rig.model.ngeom, "signature": int(rig.model.signature)}
        if recorder is not None:
            result["clip"] = recorder.close()
        receipt["arms"][arm] = result
        if args.smoke:
            print(f"[{arm}] {json.dumps(result)}")
            continue
        teth, free = result["tethered"], result["free"]
        print(f"[{arm}] lift/weight {teth['lift_over_weight']:.3f}  mean F uN {teth['mean_fluid_force_uN']}  "
              f"per-cycle sd {teth['per_cycle_Fz_sd_uN']:.2f}  torque nNm {teth['mean_torque_about_com_nNm_world']}")
        print(f"[{arm}] power uW: servo {teth['servo_power_uW']}  aero {teth['aero_power_uW']}  "
              f"hinge damping {teth['damping_power_uW']}  "
              f"({teth['servo_power_W_per_kg_body']:.1f} / {teth['aero_power_W_per_kg_body']:.1f} W/kg)  "
              f"tip {teth['left_tip_speed_mean_m_s']:.2f} m/s")
        for k, v in teth["tracking"].items():
            print(f"[{arm}]   {k}: cmd {v['commanded_amplitude_rad']:.3f} got {v['achieved_amplitude_rad']:.3f} rad, "
                  f"rms {v['rms_error_rad']:.3f}, lag {v['lag_ms']:.2f} ms, sat {v['saturated_fraction']:.3f}, "
                  f"peak torque {v['peak_torque']:.0f}")
        print(f"[{arm}] free: vz {free['com_vz_mean_mm_s_first_window']:.1f} mm/s, rise {free['max_com_rise_mm']:.2f} mm, "
              f"drop {free['final_com_drop_mm']:.2f} mm, hover axis z {free['hover_axis_z_final']:.2f}, "
              f"climbs {free['climbs']}  -> {result['verdict']}  ({result['wall_s']} s)")
    if args.dt_scan and not args.smoke:
        receipt["dt_scan"] = dt_scan(pattern)
    receipt["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    if not args.smoke:
        out_path = Path(args.out) if args.out else HERE / f"{time.strftime('%Y-%m-%d')}-fl2-flybody-lift.json"
        out_path.write_text(json.dumps(receipt, indent=1))
        print(f"receipt {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
