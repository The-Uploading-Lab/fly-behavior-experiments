"""Jump-then-fly driven by the nervous system (longevity lane, 2026-09-12).

Lane A's two-body loop shape (fl3b): the whole CNS runs in the walking rig for
its sensory loop (standing, command 1 Hz, one forced giant-fibre spike at
600 ms; the walking body's own jump element is OFF, it is the sensory puppet,
a labelled limit), and the winged body is stepped inside every 0.1 ms CNS
tick from the same spikes:
  * the CNS's two TTMn spikes fire the winged body's twitch (measured Zumstein
    force and time-to-peak; the fixed clock of jumpfly_loop is gone);
  * the 64 wing motor neurons' rates over a 10 ms window feed lane A's flight
    layer (power -> stroke amplitude, steering -> stroke shape), or the steady
    5 Hz anchor in the control arm;
  * the winged body's rotation drives the CNS's 439 haltere afferents
    (lane A's declared rate code);
  * at take-off the leg load afferents (campaniform sensilla) are silenced in
    the walking drive, since the walking body's feet never unload, and the
    cord's leg motor output is read before and after: the neural version of
    the leg fold (Robin, 2026-09-12). The fold itself stays programmatic.
Flight half: lane A's limits (prescribed wingbeat, external attitude hold).

    PYTHONPATH=. .venv/bin/python lanes/longevity/escape/jumpfly_cns.py --arm steady|cns [--no-silence] [--render out.mp4]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import mujoco as mj
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cns  # noqa: E402
import walk_search as W  # noqa: E402
from lanes.A import fl2_flybody_lift as FL2  # noqa: E402
from lanes.A import fl2b_control_derivatives as FL2B  # noqa: E402
from lanes.A import fl3_flight_muscle_layer as FL3  # noqa: E402
from lanes.A import fl1_flight_command_wing_motor as FL1  # noqa: E402
import jumpfly as JF  # noqa: E402
import jumpfly_loop as L  # noqa: E402

TICK_MS = 0.1
MN_WINDOW_TICKS = 100                     # 10 ms, about two wingbeats (lane A)
HAL_BASE_HZ, HAL_GAIN, HAL_MAX_HZ = FL2.BASE_FREQ_HZ, 0.1, 2.0 * FL2.BASE_FREQ_HZ   # lane A fl3b, rate code HYPOTHESISED
TTMN_IDS = ("720575941476434996", "720575941500579913")
HERE = Path(__file__).resolve().parent


def haltere_rows(meta):
    sc = meta["super_class"].astype(str).to_numpy(); bps = meta["body_part_sensory"].astype(str).to_numpy(); side = meta["side"].astype(str).to_numpy()
    rows = np.flatnonzero(np.isin(sc, ["sensory", "sensory_ascending"]) & (bps == "haltere"))
    return rows, np.where(side[rows] == "left", 1.0, np.where(side[rows] == "right", -1.0, 0.0))


def leg_camp_rows(meta):
    cc = meta["cell_class"].astype(str).to_numpy(); bps = meta["body_part_sensory"].astype(str).to_numpy()
    return np.flatnonzero((cc == "campaniform_sensillum_neuron") & ~np.isin(bps, ["haltere", "wing_base"]))


class JumpFlyBody:
    """The winged body stepped to CNS time; twitch from TTMn spikes, power from the layer."""

    def __init__(self, stance: dict, anchor_hz: float, z_ref_mm: float, land_at_s: float | None, render_path=None, playback=0.1, fps=30):
        fl2b, trim3, self.pat, geom, line, self.index = L.flight_context()
        FL2.STROKE_PLANE_DEG = float(trim3["stroke_plane_deg"])
        self.h = JF.build(45.0, 65.0, 1.0)
        self.rig = L.JumpFlyRig(self.h); m, d = self.h["model"], self.h["data"]
        s0 = float(trim3.get("shift_rad", math.radians(trim3["shift_deg"]))); theta0 = float(trim3.get("theta0_rad", math.radians(trim3["theta0_deg"])))
        self.layer = FL3.FlightMotorLayer(self.rig, geom, line, self.index, s0, theta_ref=theta0, anchor_hz=anchor_hz, reflex=None, altitude_loop=None,
                                          servo_gain_front=L.SERVO_FRONT, servo_gain_amp=L.SERVO_ASYM)
        self.steady = np.array([anchor_hz if t_ in FL3.POWER_TYPES else 0.0 for t_ in self.index.types])
        self.layer.feed(self.steady); self.layer.set_power_steady(anchor_hz)
        self.reflex = FL3.AnimalReflex(theta_ref=theta0)
        self.alt = FL3.AltitudeLoop(self.rig.mass_g, z_ref_mm, float(trim3["local_derivatives_at_trim"]["dF_d_amp_uN"]))
        self.fold = FL2.flight_pose_quats()["angles_rad"]
        self.jname = {i: mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, self.h["leg_act_joint"][i]).split("/", 1)[1] for i in self.h["leg_act"]}
        for jn, val in stance.items():
            j = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "fb/" + jn)
            if j >= 0:
                lo, hi = m.jnt_range[j]; d.qpos[m.jnt_qposadr[j]] = float(np.clip(val, lo, hi))
        mj.mj_forward(m, d)
        self.hold = {}
        for i in self.h["leg_act"]:
            self.hold[i] = float(d.qpos[m.jnt_qposadr[self.h["leg_act_joint"][i]]]); d.ctrl[i] = self.hold[i]
        d.ctrl[self.h["wing_act"]] = np.array([FL2.WING_SPRINGREF[a] for s in "lr" for a in FL2.AXES], float)
        self.lever = JF.ctr_lever_mm(self.h); self.stop = float(m.jnt_range[self.h["ctr"]["L"]["jid"]][1])
        self.steps_per_ctrl = max(1, int(round(FL2.CONTROL_DT / m.opt.timestep)))
        self.land_at_s = land_at_s
        self.t_off = None; self.t_wing0 = None; self.t_twitch0 = None
        self.spikes = []                  # (t_s, count) TTMn spikes seen
        self.omega_acc, self.omega_n = np.zeros(3), 0
        self.rows = []; self.frames = []; self.renderer = None
        if render_path:
            self.renderer = mj.Renderer(m, 480, 640); self.cam = mj.MjvCamera(); self.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
            self.cam.trackbodyid = self.h["thorax"]; self.cam.distance, self.cam.azimuth, self.cam.elevation = 9.0, 135.0, -15.0
            self.frame_every = max(1, int(round((1.0 / fps) * playback / FL2.CONTROL_DT))); self.render_path, self.fps = render_path, fps
        self.ctrl_ticks = 0

    def ttm(self, t_s: float, count_l: float, count_r: float = 0.0) -> None:
        # one TTMn per side drives its own leg's muscle (King and Wyman 1980):
        # spikes are kept per side, never summed across sides
        if count_l > 0 or count_r > 0:
            self.spikes.append((t_s, float(count_l), float(count_r)))
            if self.t_twitch0 is None:
                self.t_twitch0 = t_s

    def feed(self, rates_hz) -> None:
        self.layer.feed(rates_hz)

    def twitch_now(self, t_s: float) -> dict:
        return {"L": sum(cl * JF.twitch((t_s - ts) * 1e3) for ts, cl, cr in self.spikes if t_s >= ts),
                "R": sum(cr * JF.twitch((t_s - ts) * 1e3) for ts, cl, cr in self.spikes if t_s >= ts)}

    def advance_to(self, t_s: float) -> None:
        m, d, h = self.h["model"], self.h["data"], self.h
        while d.time + FL2.CONTROL_DT <= t_s + 1e-9:
            t = d.time
            tws = self.twitch_now(t); tw = max(tws.values())
            for side, c in h["ctr"].items():
                d.qfrc_applied[c["dofadr"]] = 0.0
                q = float(d.qpos[c["qadr"]])
                if tws[side] > 0.0 and self.t_off is None:
                    if q < self.stop:
                        d.qfrc_applied[c["dofadr"]] = 101.0 * tws[side] * self.lever[side]
                        for i in c["acts"]:
                            d.ctrl[i] = q
                    else:
                        for i in c["acts"]:
                            d.ctrl[i] = self.stop
            feet = [float(d.xpos[b][2]) for b in h["tarsi"]]
            if self.t_off is None and self.t_twitch0 is not None and t > self.t_twitch0 and min(feet) > 0.15:
                self.t_off = t; self.layer.reflex, self.layer.altitude_loop = self.reflex, self.alt
            # wings start with the push (Card and Dickinson 2008: first downstroke 0.67 ms after extension begins)
            if self.t_wing0 is None and self.t_twitch0 is not None and t >= self.t_twitch0 + 0.67e-3:
                self.t_wing0 = t
            landing = self.land_at_s is not None and t >= self.land_at_s
            # controlled landing: legs unfold, the attitude reflex stays on, the
            # power drive ramps down over 300 ms so lift falls below weight and
            # the body sinks; the wings stop only once a foot touches (the
            # animal extends its legs before touchdown and keeps flying to it;
            # the ramp is PROGRAMMATIC, labelled)
            if landing and self.t_off is not None and getattr(self, "t_touch", None) is None and min(feet) < 0.15 and t > self.land_at_s + 0.05:
                self.t_touch = t
            if self.t_off is not None:
                a = (1.0 - min(1.0, (t - self.land_at_s) * 1e3 / 40.0)) if landing else min(1.0, (t - self.t_off) * 1e3 / 30.0)
                for i in h["leg_act"]:
                    tgt = self.fold.get(self.jname[i])
                    if tgt is not None:
                        d.ctrl[i] = (1 - a) * self.hold[i] + a * tgt
            if self.t_wing0 is not None:
                if landing:
                    self.layer.altitude_loop = None
                    touched = getattr(self, "t_touch", None) is not None
                    ramp = 0.0 if touched else max(0.6, 1.0 - (t - self.land_at_s) / 0.3)
                    self.layer.feed(self.steady * ramp)
                cmd = self.layer.update(FL2B.body_state(self.rig), max(0.0, t - (self.t_off if self.t_off is not None else self.t_wing0)))
                if getattr(self, "t_touch", None) is not None:
                    d.ctrl[h["wing_act"]] = np.array([FL2.WING_SPRINGREF[a_] for s_ in "lr" for a_ in FL2.AXES], float)
                else:
                    d.ctrl[h["wing_act"]] = self.pat.angles((t - self.t_wing0) * cmd.freq_hz, cmd)
            for _ in range(self.steps_per_ctrl):
                mj.mj_step(m, d)
            self.omega_acc += np.degrees(d.qvel[h["root_dof"] + 3: h["root_dof"] + 6]); self.omega_n += 1
            if self.ctrl_ticks % 5 == 0:
                st = FL2B.body_state(self.rig)
                self.rows.append((d.time, JF.body_z(h), JF.upright(h), min(feet), st.body_pitch_deg, st.roll_deg, float(self.layer.cmd.amp[0]) if self.t_wing0 else 0.0, tw))
            if self.renderer is not None and self.ctrl_ticks % self.frame_every == 0:
                self.renderer.update_scene(d, self.cam); self.frames.append(self.renderer.render().copy())
            self.ctrl_ticks += 1

    def mean_omega(self) -> np.ndarray:
        w = self.omega_acc / max(1, self.omega_n); self.omega_acc[:] = 0.0; self.omega_n = 0; return w

    @property
    def beating(self) -> bool:
        return self.t_wing0 is not None and self.layer.cmd is not None and min(self.layer.cmd.amp) > 0.0


def run(arm: str, silence_load: bool, render_path=None, dur_ms: float = 3000.0, gf_ms: float = 600.0, land_at_s: float = 2.5, gate_ms: float = 3.0) -> dict:
    meta = pd.read_feather(ROOT / "data/banc_888_meta.feather"); edges = pd.read_feather(ROOT / "data/banc_888_edgelist_simple_v3.feather")
    th = json.loads((HERE / "theta-wj1-jump-film.json").read_text())
    th["esc_ttm"] = 0; th["force_spikes_ms"] = {"DNp01": [gf_ms]}
    for k in ("esc_ttm_from_ms", "esc_ttm_to_ms", "esc_ttm_release", "esc_ttm_twitch_ms", "esc_ttm_ctr_max_deg", "esc_ttm_F_uN", "esc_ttm_r_mm", "esc_nmj_ms"):
        th.pop(k, None)
    stance = json.loads((HERE / "jumpfly-stance-v2-best.json").read_text())
    rows = FL1.motor_rows(meta); wing_rows = np.asarray(rows["wing"], dtype=np.int64)
    hal_rows, hal_side = haltere_rows(meta); camp = leg_camp_rows(meta)
    ids = meta["banc_888_id"].astype(str).to_numpy(); ttmn_rows = np.array([int(np.flatnonzero(ids == s)[0]) for s in TTMN_IDS])
    body = JumpFlyBody(stance, FL3.POWER_ANCHOR_HZ, 8.0, land_at_s, render_path=render_path)
    index_rows = np.asarray(body.index.rows, dtype=np.int64)
    print(f"wing rows {len(wing_rows)} (layer index rows {len(index_rows)}), haltere rows {len(hal_rows)}, leg campaniform rows {len(camp)}, TTMn rows {ttmn_rows}", flush=True)
    window = deque(maxlen=MN_WINDOW_TICKS)
    state = {"calls": 0, "ttm_spikes": [], "legmn_pre": [], "legmn_post": [], "hal_mean": [], "silenced": 0}
    orig_run = cns.CNS.run

    def patched_run(net, *args, **kwargs):
        orig_fb = np.asarray(kwargs["feedback_rows"], dtype=np.int64); n_orig = len(orig_fb)
        assert not np.isin(index_rows, orig_fb).any(), "wing rows already in the walking feedback set"
        assert not np.isin(hal_rows, orig_fb).any()
        ttm_pos = [int(np.flatnonzero(orig_fb == r)[0]) for r in ttmn_rows]
        leg_pos = np.arange(min(391, n_orig))
        ct = meta["cell_type"].astype(str).to_numpy()[orig_fb[leg_pos]]
        flex_pos = leg_pos[np.array([("flex" in x.lower() or "reductor" in x.lower() or "levator" in x.lower()) for x in ct])]
        ext_pos = leg_pos[np.array([("ext" in x.lower() or "depressor" in x.lower()) for x in ct])]
        state["pools"] = {"flexor_like": int(len(flex_pos)), "extensor_like": int(len(ext_pos))}
        orig_fn = kwargs["drive_fn"]

        def fn(t_ms, fb):
            idx, rates = orig_fn(t_ms, None if fb is None else fb[:n_orig])
            idx, rates = np.asarray(idx), np.asarray(rates, float).copy()
            t_s = t_ms * 1e-3
            if fb is not None:
                c = np.asarray(fb)
                n_ttm = float(c[ttm_pos[0]] + c[ttm_pos[1]])
                if n_ttm > 0:
                    state["ttm_spikes"].append(t_ms)
                # only the giant-fibre-driven TTMn spike fires the muscle: the
                # model's TTMn also fires outside escape (CONFLICT, 2026-09-12),
                # the animal's does not; same instrument as esc_ttm_from_ms
                if gf_ms - 1.0 <= t_ms <= gf_ms + gate_ms:
                    body.ttm(t_s, float(c[ttm_pos[0]]), float(c[ttm_pos[1]]))
                window.append(np.asarray(c[n_orig:], float))
                leg_total = (float(c[leg_pos].sum()), float(c[flex_pos].sum()), float(c[ext_pos].sum()))
                if body.t_off is None and t_ms > gf_ms - 100.0:
                    state["legmn_pre"].append(leg_total)
                elif body.t_off is not None and t_s > body.t_off + 0.02 and t_s < body.t_off + 0.12:
                    state["legmn_post"].append(leg_total)
            mn_hz = np.sum(window, axis=0) / (len(window) * TICK_MS * 1e-3) if window else np.zeros(len(index_rows))
            body.feed(mn_hz if arm == "cns" else body.steady)
            body.advance_to(t_s)
            hal = np.clip(HAL_BASE_HZ + HAL_GAIN * (lambda w: w[1] + hal_side * (w[0] + w[2]))(body.mean_omega()), 0.0, HAL_MAX_HZ) if body.beating else np.zeros(len(hal_rows))
            if silence_load and body.t_off is not None:
                mask = np.isin(idx, camp); rates[mask] = 0.0; state["silenced"] += int(mask.sum())
            state["calls"] += 1
            if state["calls"] % 1000 == 0:
                state["hal_mean"].append(float(hal.mean()))
            return np.concatenate([idx, hal_rows]), np.concatenate([rates, hal])

        kwargs["feedback_rows"] = np.concatenate([orig_fb, index_rows])
        kwargs["drive_fn"] = fn
        return orig_run(net, *args, **kwargs)

    cns.CNS.run = patched_run
    t0 = time.time()
    try:
        score, raw, up, z = W.evaluate({}, meta, edges, th, seed=0, dur_ms=dur_ms)
    finally:
        cns.CNS.run = orig_run
    R = np.array(body.rows)
    out = {"arm": arm, "silence_load": silence_load, "wall_s": round(time.time() - t0, 1), "gf_ms": gf_ms,
           "ttm_spike_ms": state["ttm_spikes"][:10], "t_twitch0_ms": None if body.t_twitch0 is None else round(body.t_twitch0 * 1e3, 1),
           "t_off_ms": None if body.t_off is None else round(body.t_off * 1e3, 1), "z_peak": float(R[:, 1].max()), "z_final": float(R[-1, 1]),
           "upright_final": float(R[-1, 2]), "upright_min_after_off": float(R[R[:, 0] >= (body.t_off or 0), 2].min()) if body.t_off else None,
           "airborne_s": float(((R[:, 3] > 0.15).sum()) * 5 * FL2.CONTROL_DT), "pitch_final": float(R[-1, 4]),
           "amp_mean_flight": float(R[(R[:, 6] > 0), 6].mean()) if (R[:, 6] > 0).any() else None,
           "legmn_rate_pre_hz": [float(x) for x in (np.mean(state["legmn_pre"], axis=0) / (TICK_MS * 1e-3))] if state["legmn_pre"] else None,
           "legmn_rate_post_hz": [float(x) for x in (np.mean(state["legmn_post"], axis=0) / (TICK_MS * 1e-3))] if state["legmn_post"] else None,
           "legmn_pools": state.get("pools"), "legmn_rate_order": ["all", "flexor_like", "extensor_like"],
           "load_rows_silenced_ticks": state["silenced"], "haltere_mean_hz_samples": state["hal_mean"][:12], "walking_body": {"disp": float(raw), "up": float(up), "z": float(z)}}
    if render_path and body.frames:
        import imageio.v2 as iio
        iio.mimwrite(render_path, body.frames, fps=body.fps, codec="libx264", quality=8); out["render"] = render_path; out["frames"] = len(body.frames)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", default="steady", choices=("steady", "cns")); ap.add_argument("--no-silence", action="store_true")
    ap.add_argument("--render", default=None); ap.add_argument("--dur", type=float, default=3000.0); ap.add_argument("--land", type=float, default=2.5)
    a = ap.parse_args()
    r = run(a.arm, not a.no_silence, render_path=a.render, dur_ms=a.dur, land_at_s=a.land)
    print(json.dumps(r), flush=True)
    (HERE / f"2026-09-12-jumpfly-cns-{a.arm}{'' if not a.no_silence else '-nosilence'}.json").write_text(json.dumps(r, indent=1))
