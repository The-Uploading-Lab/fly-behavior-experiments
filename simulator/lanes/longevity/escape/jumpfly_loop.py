"""Jump, then fly: one winged body on a floor (longevity lane, 2026-09-12).

Stand -> one TTM twitch per middle leg (Zumstein 2004, 101 uN, 8.2 ms to peak)
-> take-off -> wings beat on lane A's measured trim (FL2b), power from lane
A's power line at the 5 Hz anchor (or from the CNS when driven), attitude
held by lane A's labelled external scaffold (the animal's reflex gains,
Whitehead 2015 / Beatus 2015, plus FL2b's altitude loop) -> legs fold to
flybody's measured flight pose (FL2c) over 30 ms after take-off (PROGRAMMATIC,
labelled). Everything in the flight half carries lane A's limits: prescribed
wingbeat, external attitude hold.

    PYTHONPATH=. .venv/bin/python lanes/longevity/escape/jumpfly_loop.py [--render out.mp4]
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import mujoco as mj

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lanes.A import fl2_flybody_lift as FL2  # noqa: E402
from lanes.A import fl2b_control_derivatives as FL2B  # noqa: E402
from lanes.A import fl3_flight_muscle_layer as FL3  # noqa: E402
import jumpfly as JF  # noqa: E402

FL2B_RECEIPT = ROOT / "lanes/A/2026-09-09-fl2b-control-derivatives.json"
FL1_RECEIPT = ROOT / "lanes/A/2026-09-09-fl1-flight-command-wing-motor.json"
SERVO_FRONT, SERVO_ASYM = 0.9761368813113314, 0.9485600431636894   # lane A fl3b receipt, servo_compensation
CONTROL_DT = FL2.CONTROL_DT


class JumpFlyRig:
    """Lane A's Rig interface over the standing winged body."""

    hover_axis_world = FL2.Rig.hover_axis_world
    fluid_force_torque = FL2.Rig.fluid_force_torque
    powers = FL2.Rig.powers
    tip = FL2.Rig.tip

    def __init__(self, h: dict):
        self.h = h
        self.model, self.data, self.dt = h["model"], h["data"], float(h["model"].opt.timestep)
        self.root_qpos, self.root_dof, self.thorax = h["root_qpos"], h["root_dof"], h["thorax"]
        m = self.model
        self.wing_joints = [FL2.name_id(m, mj.mjtObj.mjOBJ_JOINT, FL2.joint_name(s, a)) for s in "lr" for a in FL2.AXES]
        self.wing_qpos = np.array([m.jnt_qposadr[j] for j in self.wing_joints])
        self.wing_dof = np.array([m.jnt_dofadr[j] for j in self.wing_joints])
        self.actuators = h["wing_act"]
        self.fluid_geoms = {s: FL2.name_id(m, mj.mjtObj.mjOBJ_GEOM, f"fb/{s}_wing_fluid") for s in "lr"}
        self.wing_bodies = {s: FL2.name_id(m, mj.mjtObj.mjOBJ_BODY, f"fb/{s}_wing") for s in "lr"}
        self.mass_g = h["mass_g"]
        self.weight_uN = self.mass_g * FL2.GRAVITY
        self.ellipsoid = True
        mj.mj_forward(m, self.data)
        self.tip_sign = {}
        for s, g in self.fluid_geoms.items():
            centre = self.data.geom_xpos[g]
            span = self.data.geom_xmat[g].reshape(3, 3)[:, 2] * m.geom_size[g, 2]
            hinge = self.data.xpos[self.wing_bodies[s]]
            self.tip_sign[s] = 1.0 if np.linalg.norm(centre + span - hinge) > np.linalg.norm(centre - span - hinge) else -1.0


def flight_context():
    fl2b = json.loads(FL2B_RECEIPT.read_text())
    fl1 = json.loads(FL1_RECEIPT.read_text())
    trim3 = fl2b["trim3"]
    pat = FL2B.Pattern(FL2.load_pattern())
    geom = FL3.StrokeGeometry(pat)
    line = FL3.PowerLine(fl2b)
    index = FL3.WingMotorIndex.from_fl1(fl1)
    return fl2b, trim3, pat, geom, line, index


def run(F_uN: float = 101.0, twitch_at_s: float = 0.30, seconds: float = 1.10, z_ref_mm: float = 6.0,
        anchor_hz: float = FL3.POWER_ANCHOR_HZ, fold_ms: float = 30.0, wing_delay_ms: float = 0.0,
        release: float = 1.0, wings_with_push: float = 0.0, wing_lead_ms: float = 0.67, stance: dict | None = None,
        land_at_s: float | None = None, unfold_ms: float = 40.0,
        render_path: str | None = None, playback: float = 0.05, fps: int = 30,
        power_fn=None, ttm_fn=None, log_every_s: float = 0.001) -> dict:
    fl2b, trim3, pat, geom, line, index = flight_context()
    FL2.STROKE_PLANE_DEG = float(trim3["stroke_plane_deg"])
    h = JF.build(45.0, 65.0, 1.0)
    rig = JumpFlyRig(h)
    m, d = h["model"], h["data"]
    s0 = float(trim3.get("shift_rad", math.radians(trim3["shift_deg"])))
    theta0 = float(trim3.get("theta0_rad", math.radians(trim3["theta0_deg"])))
    layer = FL3.FlightMotorLayer(rig, geom, line, index, s0, theta_ref=theta0, anchor_hz=anchor_hz,
                                 reflex=None, altitude_loop=None, servo_gain_front=SERVO_FRONT, servo_gain_amp=SERVO_ASYM)
    # steady power drive, as lane A's loop feeds it every tick when the CNS is
    # not applied: the anchor rate on every power motor neuron, zero elsewhere
    steady = np.array([anchor_hz if t_ in FL3.POWER_TYPES else 0.0 for t_ in index.types])
    layer.feed(steady)
    layer.set_power_steady(anchor_hz)
    reflex = FL3.AnimalReflex(theta_ref=theta0)
    alt = FL3.AltitudeLoop(rig.mass_g, z_ref_mm, float(trim3["local_derivatives_at_trim"]["dF_d_amp_uN"]))
    fold = FL2.flight_pose_quats()["angles_rad"]
    jname = {i: mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, h["leg_act_joint"][i]).split("/", 1)[1] for i in h["leg_act"]}

    if stance:
        # searched standing pose (jumpfly_stance_search.py), joint name -> rad
        for jn, val in stance.items():
            j = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "fb/" + jn)
            if j >= 0:
                lo, hi = m.jnt_range[j]
                d.qpos[m.jnt_qposadr[j]] = float(np.clip(val, lo, hi)) if m.jnt_limited[j] else float(val)
    mj.mj_forward(m, d)
    hold = {}
    for i in h["leg_act"]:
        hold[i] = float(d.qpos[m.jnt_qposadr[h["leg_act_joint"][i]]]); d.ctrl[i] = hold[i]
    wing_rest = np.array([FL2.WING_SPRINGREF[a] for s in "lr" for a in FL2.AXES], float)
    d.ctrl[h["wing_act"]] = wing_rest
    lever = JF.ctr_lever_mm(h)
    stop = float(m.jnt_range[h["ctr"]["L"]["jid"]][1])
    steps_per_ctrl = max(1, int(round(CONTROL_DT / m.opt.timestep)))
    n_ctrl = int(round(seconds / CONTROL_DT))
    t_off = None
    rows, frames = [], []
    renderer = cam = None
    if render_path:
        renderer = mj.Renderer(m, 480, 640)
        cam = mj.MjvCamera(); cam.type = mj.mjtCamera.mjCAMERA_TRACKING; cam.trackbodyid = h["thorax"]
        cam.distance, cam.azimuth, cam.elevation = 9.0, 135.0, -15.0
        frame_every = max(1, int(round((1.0 / fps) * playback / CONTROL_DT)))
    twitch_hist = []
    for k in range(n_ctrl):
        t = k * CONTROL_DT
        # ---- jump muscle: TTM twitch from the trigger (fixed time, or the CNS's TTMn spikes)
        if ttm_fn is not None:
            twitch_hist.append(float(ttm_fn(t)))
        tw = 0.0
        if ttm_fn is None:
            tw = JF.twitch((t - twitch_at_s) * 1e3) if t >= twitch_at_s else 0.0
        else:
            for j, u in enumerate(twitch_hist):
                if u > 0:
                    tw += u * JF.twitch((t - j * CONTROL_DT) * 1e3)
        for side, c in h["ctr"].items():
            d.qfrc_applied[c["dofadr"]] = 0.0
            q = float(d.qpos[c["qadr"]])
            if tw > 0.0 and t_off is None:
                if q < stop:
                    d.qfrc_applied[c["dofadr"]] = F_uN * tw * lever[side]
                    if release > 0:
                        for i in c["acts"]:
                            d.ctrl[i] = q
                else:
                    for i in c["acts"]:
                        d.ctrl[i] = stop
        # ---- take-off: every foot above 0.15 mm
        feet = [float(d.xpos[b][2]) for b in h["tarsi"]]
        if t_off is None and min(feet) > 0.15 and t > twitch_at_s:
            t_off = t
            layer.reflex, layer.altitude_loop = reflex, alt
        t_wing0 = None
        if wings_with_push > 0 and t >= twitch_at_s + wing_lead_ms * 1e-3:
            t_wing0 = twitch_at_s + wing_lead_ms * 1e-3
        if t_off is not None and t_wing0 is None:
            t_wing0 = t_off
        if t_wing0 is not None and t_off is None:
            # wings beating before the feet leave the ground: the layer runs, the
            # scaffold is not yet on (it is armed at take-off)
            tf0 = t - t_wing0
            layer.feed(power_fn(t) if power_fn is not None else steady)
            cmd = layer.update(FL2B.body_state(rig), tf0)
            d.ctrl[h["wing_act"]] = pat.angles(tf0 * cmd.freq_hz, cmd)
        landing = land_at_s is not None and t >= land_at_s
        if t_off is not None:
            tf = t - t_off
            # legs fold to the measured flight pose over fold_ms (programmatic);
            # on landing they unfold back to the stance over unfold_ms and the
            # power drive stops (the animal's landing response extends the legs
            # before touchdown; PROGRAMMATIC, labelled)
            if landing:
                a = 1.0 - min(1.0, (t - land_at_s) * 1e3 / unfold_ms)
            else:
                a = min(1.0, tf * 1e3 / fold_ms)
            for i in h["leg_act"]:
                tgt = fold.get(jname[i])
                if tgt is not None:
                    d.ctrl[i] = (1 - a) * hold[i] + a * tgt
            # wings: lane A's layer from the trim, external scaffold on
            if tf * 1e3 >= wing_delay_ms:
                layer.feed(np.zeros_like(steady) if landing else (power_fn(t) if power_fn is not None else steady))
                if landing:
                    layer.altitude_loop = None
                cmd = layer.update(FL2B.body_state(rig), tf)
                d.ctrl[h["wing_act"]] = pat.angles((t - (t_wing0 if wings_with_push > 0 else t_off + wing_delay_ms * 1e-3)) * cmd.freq_hz, cmd)
        for _ in range(steps_per_ctrl):
            mj.mj_step(m, d)
        if k % max(1, int(round(log_every_s / CONTROL_DT))) == 0:
            st = FL2B.body_state(rig)
            rows.append((t, JF.body_z(h), JF.upright(h), float(d.qvel[h["root_dof"] + 2]), min(feet),
                         st.body_pitch_deg, st.roll_deg, float(layer.cmd.amp[0]) if t_off else 0.0, tw))
        if renderer is not None and k % frame_every == 0:
            renderer.update_scene(d, cam); frames.append(renderer.render().copy())
    R = np.array(rows)
    post = R[R[:, 0] >= (t_off if t_off is not None else twitch_at_s)]
    out = {"F_uN": F_uN, "t_off_s": t_off, "z_peak": float(R[:, 1].max()), "z_peak_at_s": float(R[np.argmax(R[:, 1]), 0]),
           "z_final": float(R[-1, 1]), "upright_final": float(R[-1, 2]), "upright_min_after_off": float(post[:, 2].min()) if len(post) else None,
           "pitch_final_deg": float(R[-1, 5]), "roll_final_deg": float(R[-1, 6]),
           "airborne_s_after_off": float((post[:, 4] > 0.15).sum() * log_every_s) if len(post) else 0.0,
           "still_airborne_at_end": bool(R[-1, 4] > 0.15), "amp_final": float(R[-1, 7]), "seconds": seconds, "z_ref_mm": z_ref_mm}
    if render_path and frames:
        import imageio.v2 as iio
        iio.mimwrite(render_path, frames, fps=fps, codec="libx264", quality=8)
        out["render"] = render_path; out["frames"] = len(frames)
    out["rows"] = R.tolist()
    return out


if __name__ == "__main__":
    args = dict(a.split("=") for a in sys.argv[1:] if "=" in a)
    render = args.pop("render", None)
    kw = {k: float(v) for k, v in args.items()}
    r = run(render_path=render, **kw)
    r.pop("rows")
    print(json.dumps(r), flush=True)
