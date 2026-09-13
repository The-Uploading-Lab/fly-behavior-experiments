"""One winged body that jumps and then flies (longevity lane, 2026-09-12, on
Robin's instruction: "make it fly after the jump with A's work").

Body: lane A's flight composition of flygym's FlyBody (lanes/A/fl2_flybody_lift.py:
wing fluid ellipsoids, hover frames, flybody's wing servos and air) with the
LEGS ADDED (flybody's biological leg joints, position servos mirroring the
walking body's) and a floor with leg contacts. Lane A ran this body legless in
the air; nobody has stood it on its legs before, so `standing_test` is the
first thing this module does.

Everything here is a demonstration rig: the flight half carries lane A's
labelled limits (prescribed wingbeat, external attitude hold, power modulated
by the CNS's flight motor neurons).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import mujoco as mj

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lanes.A import fl2_flybody_lift as FL2  # noqa: E402
from flygym.compose.world import FlatGroundWorld  # noqa: E402
from flygym.flybody import FlyBodySkeleton, FlyBodyAxisOrder, FlyBodyJointPreset, FlyBodyBodySegment  # noqa: E402
from flygym.utils.math import Rotation3D  # noqa: E402

LEG_LINKS = FL2.LEG_LINKS
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
CTR_JOINT = {"L": "lm_coxa-lm_trochanterfemur-pitch", "R": "rm_coxa-rm_trochanterfemur-pitch"}


def leg_wing_skeleton() -> FlyBodySkeleton:
    legs = FlyBodySkeleton(axis_order=FlyBodyAxisOrder.YAW_ROLL_PITCH,
                           joint_preset=FlyBodyJointPreset.ALL_BIOLOGICAL)
    return FlyBodySkeleton(axis_order=FlyBodyAxisOrder.YAW_ROLL_PITCH,
                           anatomical_joints=list(legs.anatomical_joints)
                           + list(FL2.wing_skeleton().anatomical_joints))


def build(leg_kp: float, leg_forcerange: float, spawn_z: float, dt: float = FL2.PHYSICS_DT,
          spawn_quat=(1.0, 0.0, 0.0, 0.0), ellipsoid: bool = True):
    """Compose the standing winged fly on a floor. Returns a dict of handles."""
    fly = FL2.FlightFlyBody(name="fb")
    FL2.add_wing_fluid_geoms(fly, ellipsoid)
    up = FL2.set_flight_frames(fly)
    dofs = fly.add_joints(leg_wing_skeleton())
    wing_dofs = {d: j for d, j in dofs.items() if "wing" in j.name}
    leg_dofs = {d: j for d, j in dofs.items() if "wing" not in j.name}
    for dof, joint in wing_dofs.items():
        # MuJoCo 3.7+: MjsJoint stiffness/damping are 3-element polynomials
        joint.stiffness[0], joint.damping[0] = FL2.WING_STIFFNESS, FL2.WING_DAMPING
        try:
            joint.armature = FL2.WING_ARMATURE
        except TypeError:
            joint.armature[0] = FL2.WING_ARMATURE
        joint.springref = FL2.WING_SPRINGREF[dof.axis.value]
    # leg joints: the walking body's passive values (make_locomotion_fly:
    # stiffness 0.05, damping 0.06; tarsus joints 7.5 / 1e-2)
    for dof, joint in leg_dofs.items():
        tarsal = "tarsus" in joint.name
        joint.stiffness[0] = 7.5 if tarsal else 0.05
        joint.damping[0] = 1e-2 if tarsal else 0.06
    fly.add_actuators(list(wing_dofs.keys()), "position", kp=FL2.WING_GAIN,
                      forcelimited=True, forcerange=(-FL2.WING_GAIN, FL2.WING_GAIN))
    fly.add_actuators(list(leg_dofs.keys()), "position", kp=leg_kp,
                      forcelimited=True, forcerange=(-leg_forcerange, leg_forcerange))
    contact_segs = [FlyBodyBodySegment(f"{leg}_{link}") for leg in LEGS for link in LEG_LINKS]
    contact_segs += [FlyBodyBodySegment(s) for s in ("c_thorax", "c_head", "c_abdomen") if s in {seg.name for seg in fly.bodyseg_to_mjcfbody}]
    world = FlatGroundWorld(name="jumpfly_world")
    world.add_fly(fly, spawn_position=(0.0, 0.0, spawn_z),
                  spawn_rotation=Rotation3D("quat", tuple(float(x) for x in spawn_quat)),
                  bodysegs_with_ground_contact=contact_segs, add_ground_contact_sensors=False)
    opt = world.mjcf_root.option
    opt.density, opt.viscosity, opt.timestep = FL2.AIR_DENSITY, FL2.AIR_VISCOSITY, dt
    opt.gravity = np.array([0.0, 0.0, -FL2.GRAVITY])
    model, data = world.compile()
    free = [j for j in range(model.njnt) if model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE]
    assert len(free) == 1
    h = {"world": world, "fly": fly, "model": model, "data": data, "up": up,
         "root_qpos": int(model.jnt_qposadr[free[0]]), "root_dof": int(model.jnt_dofadr[free[0]]),
         "thorax": FL2.name_id(model, mj.mjtObj.mjOBJ_BODY, "fb/c_thorax")}
    h["wing_act"] = [FL2.name_id(model, mj.mjtObj.mjOBJ_ACTUATOR, FL2.joint_name(s, a) + "-position") for s in "lr" for a in FL2.AXES]
    h["leg_act"] = [i for i in range(model.nu) if i not in h["wing_act"]]
    h["leg_act_joint"] = {i: int(model.actuator_trnid[i, 0]) for i in h["leg_act"]}
    h["ctr"] = {}
    for side, jn in CTR_JOINT.items():
        jid = FL2.name_id(model, mj.mjtObj.mjOBJ_JOINT, "fb/" + jn)
        acts = [i for i in h["leg_act"] if h["leg_act_joint"][i] == jid]
        h["ctr"][side] = {"jid": jid, "qadr": int(model.jnt_qposadr[jid]), "dofadr": int(model.jnt_dofadr[jid]), "acts": acts}
    h["mass_g"] = float(model.body_subtreemass[h["thorax"]])
    h["tarsi"] = [FL2.name_id(model, mj.mjtObj.mjOBJ_BODY, f"fb/{leg}_tarsus5") for leg in LEGS]
    return h


def body_z(h) -> float:
    return float(h["data"].xpos[h["thorax"]][2])


def upright(h) -> float:
    """cos(tilt): z-component of the body's up axis in world frame."""
    q = h["data"].qpos[h["root_qpos"] + 3: h["root_qpos"] + 7]
    R = np.zeros(9); mj.mju_quat2Mat(R, q)
    return float(R.reshape(3, 3)[2, 2])


def standing_test(leg_kp: float, leg_forcerange: float, spawn_z: float, seconds: float = 0.4) -> dict:
    h = build(leg_kp, leg_forcerange, spawn_z)
    m, d = h["model"], h["data"]
    mj.mj_forward(m, d)
    # hold the spawn pose: every leg servo targets its current angle
    for i in h["leg_act"]:
        d.ctrl[i] = d.qpos[m.jnt_qposadr[h["leg_act_joint"][i]]]
    for i in h["wing_act"]:
        d.ctrl[i] = d.qpos[m.jnt_qposadr[int(m.actuator_trnid[i, 0])]]
    z0 = body_z(h); zs, ups = [], []
    n = int(round(seconds / m.opt.timestep))
    for k in range(n):
        mj.mj_step(m, d)
        if k % 200 == 0:
            zs.append(body_z(h)); ups.append(upright(h))
    tz = [float(d.xpos[b][2]) for b in h["tarsi"]]
    return {"leg_kp": leg_kp, "leg_forcerange": leg_forcerange, "spawn_z": spawn_z, "mass_g": h["mass_g"],
            "z0": z0, "z_final": body_z(h), "z_min": min(zs), "upright_final": upright(h), "upright_min": min(ups),
            "tarsus_z_final": [round(x, 3) for x in tz], "n_leg_act": len(h["leg_act"]), "ctr": {s: v["jid"] for s, v in h["ctr"].items()}}


def ctr_lever_mm(h) -> dict:
    """Perpendicular distance from each middle-leg CTr hinge axis to its tarsus tip."""
    m, d = h["model"], h["data"]
    out = {}
    for side, c in h["ctr"].items():
        anchor, axis = d.xanchor[c["jid"]], d.xaxis[c["jid"]]
        tip = d.xpos[FL2.name_id(m, mj.mjtObj.mjOBJ_BODY, f"fb/{'lm' if side == 'L' else 'rm'}_tarsus5")]
        r = tip - anchor
        out[side] = float(np.linalg.norm(r - np.dot(r, axis) * axis))
    return out


def twitch(t_ms: float, tp_ms: float = 8.2) -> float:
    """Alpha twitch, peak 1 at tp (Zumstein 2004 time-to-peak)."""
    if t_ms <= 0.0:
        return 0.0
    x = t_ms / tp_ms
    return x * math.exp(1.0 - x)


def jump_test(leg_kp: float = 45.0, leg_forcerange: float = 65.0, spawn_z: float = 1.0,
              settle_s: float = 0.30, twitch_at_s: float = 0.30, F_uN: float = 101.0,
              ctr_stop_deg: float = 0.0, seconds: float = 0.45, log_every_s: float = 0.001) -> dict:
    """Stand, then one TTM twitch per middle leg (Zumstein 2004: 101 uN per leg,
    8.2 ms to peak) as hinge torque F x lever with the CTr servos released
    while the twitch acts and stopped at ctr_stop_deg. No nervous system."""
    h = build(leg_kp, leg_forcerange, spawn_z)
    m, d = h["model"], h["data"]
    mj.mj_forward(m, d)
    for i in h["leg_act"]:
        d.ctrl[i] = d.qpos[m.jnt_qposadr[h["leg_act_joint"][i]]]
    for i in h["wing_act"]:
        d.ctrl[i] = d.qpos[m.jnt_qposadr[int(m.actuator_trnid[i, 0])]]
    hold = {i: float(d.ctrl[i]) for i in h["leg_act"]}
    lever = ctr_lever_mm(h)
    stop = math.radians(ctr_stop_deg)
    rows = []
    n = int(round(seconds / m.opt.timestep)); every = int(round(log_every_s / m.opt.timestep))
    for k in range(n):
        t = k * m.opt.timestep
        tw = twitch((t - twitch_at_s) * 1e3) if t >= twitch_at_s else 0.0
        for side, c in h["ctr"].items():
            d.qfrc_applied[c["dofadr"]] = 0.0
            q = float(d.qpos[c["qadr"]])
            if tw > 0.0:
                if q < stop:
                    d.qfrc_applied[c["dofadr"]] = F_uN * tw * lever[side]   # uN * mm
                    for i in c["acts"]:
                        d.ctrl[i] = q                                       # servo compliant
                else:
                    for i in c["acts"]:
                        d.ctrl[i] = stop
            else:
                for i in c["acts"]:
                    d.ctrl[i] = hold[i]
        mj.mj_step(m, d)
        if k % every == 0:
            rows.append((t, body_z(h), upright(h), float(d.qvel[h["root_dof"] + 2]),
                         min(float(d.xpos[b][2]) for b in h["tarsi"]),
                         float(np.degrees(d.qpos[h["ctr"]["L"]["qadr"]])), float(np.degrees(d.qpos[h["ctr"]["R"]["qadr"]]))))
    R = np.array(rows)
    pre = R[(R[:, 0] > twitch_at_s - 0.1) & (R[:, 0] < twitch_at_s)]
    post = R[R[:, 0] >= twitch_at_s]
    return {"lever_mm": lever, "F_uN": F_uN, "ctr_stop_deg": ctr_stop_deg, "mass_g": h["mass_g"],
            "z_pre_mean": float(pre[:, 1].mean()), "z_peak": float(post[:, 1].max()), "z_peak_at_ms": float((post[np.argmax(post[:, 1]), 0] - twitch_at_s) * 1e3),
            "vz_peak_mm_s": float(post[:, 3].max()), "min_tarsus_z_max": float(post[:, 4].max()),
            "airborne_ms": float(((post[:, 4] > 0.15).sum()) * log_every_s * 1e3),
            "ctr_deg_pre": [float(pre[-1, 5]), float(pre[-1, 6])], "ctr_deg_max": [float(post[:, 5].max()), float(post[:, 6].max())],
            "upright_min_post": float(post[:, 2].min()), "z_final": float(R[-1, 1]), "rows": R.tolist()}


if __name__ == "__main__":
    import json
    if len(sys.argv) > 1 and sys.argv[1] == "--jump":
        r = jump_test(**{k: float(v) for k, v in (a.split("=") for a in sys.argv[2:])})
        r.pop("rows")
        print(json.dumps(r), flush=True)
        raise SystemExit(0)
    out = []
    for kp, fr, z in ((float(a), float(b), float(c)) for a, b, c in
                      (sys.argv[1:] and [tuple(x.split(",")) for x in sys.argv[1:]] or [("20", "65", "1.2")])):
        r = standing_test(kp, fr, z)
        print(json.dumps(r), flush=True)
        out.append(r)
