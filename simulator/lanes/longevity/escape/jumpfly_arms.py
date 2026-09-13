"""The three arms in the body: a 20 ms light-off flash drives LC4 and LPLC2
inside the whole-CNS loop, the giant fibre fires or does not, the winged body
jumps and flies or stands (longevity lane, 2026-09-12; Robin: "I want video
once you have results").

Built on jumpfly_cns.run (lane A's loop shape) with the giant fibre no longer
forced: the flash is Poisson drive on LC4 at LIGHTOFF_HZ and LPLC2 at 0.2 of it
for STIM_MS (escape_visual's frozen light-off stimulus), the junction rules
rectify (gap_rectify) and the GF carries escape_visual's input resistance
factor and balance exemption, all applied through the same theta and the same
cns hooks. The twitch gate is the flash window instead of a fixed spike time.

Arms: control (LC4 -> GF at the connectome's counts); starved (the pre-death
state, Gaitanidis 2025: LC4 -> GF transmission lost, counts x0; refeeding
restores it within an hour); ibuprofen, young wild type (no measured change
on any nervous-system readout, Setzu 2026: counts x1, the run differs from
control only by its seed). Escape probability over 24 trials is read from
escape_visual's receipts; each body run is one trial at its own seed.

    PYTHONPATH=. .venv/bin/python lanes/longevity/escape/jumpfly_arms.py --arm control --seed 0 --render out.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cns  # noqa: E402
import walk_search as W  # noqa: E402
import jumpfly_cns as JC  # noqa: E402
from jumpfly_cns import FL1, FL2, FL3, JumpFlyBody, haltere_rows, leg_camp_rows, HAL_BASE_HZ, HAL_GAIN, HAL_MAX_HZ, TICK_MS, MN_WINDOW_TICKS, TTMN_IDS, ROOT, HERE  # noqa: E402
import escape_visual as EV  # noqa: E402
from escape_test import find_cells  # noqa: E402

import mujoco as mj  # noqa: E402
import jumpfly as JF  # noqa: E402


def add_scene(spec) -> None:
    """Gradient sky, warm floor, one shadow-casting light (the interventions
    lane's render_slap.py scene, taken verbatim, 2026-09-12)."""
    for t in spec.textures:
        if t.name == "skybox":
            t.builtin = mj.mjtBuiltin.mjBUILTIN_GRADIENT; t.rgb1 = [0.55, 0.75, 0.95]; t.rgb2 = [0.93, 0.96, 1.0]
        if t.name == "checker":
            t.rgb1 = [0.86, 0.78, 0.62]; t.rgb2 = [0.79, 0.71, 0.56]
    spec.worldbody.add_light(pos=[10, -10, 30], dir=[-0.3, 0.3, -1], diffuse=[0.9, 0.9, 0.85], specular=[0.3, 0.3, 0.3], castshadow=True)


def colour_fly(m) -> None:
    for g in range(m.ngeom):
        n = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) or ""
        if not n.startswith("fb/"):
            continue
        if "_red" in n: m.geom_rgba[g] = [0.85, 0.15, 0.1, 1]
        elif "_black" in n or "ocelli" in n: m.geom_rgba[g] = [0.15, 0.12, 0.1, 1]
        elif "membrane" in n: m.geom_rgba[g] = [0.85, 0.9, 1.0, 0.35]
        elif "_brown" in n: m.geom_rgba[g] = [0.45, 0.3, 0.15, 1]
        elif "fluid" in n or "inertial" in n: m.geom_rgba[g] = [0, 0, 0, 0]
        else: m.geom_rgba[g] = [0.82, 0.6, 0.28, 1]
    gid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "ground_plane")
    m.geom_matid[gid] = mj.mj_name2id(m, mj.mjtObj.mjOBJ_MATERIAL, "grid"); m.mat_texrepeat[m.geom_matid[gid]] = [60, 60]


AXIS_MAP = {"ThC_yaw": "c_thorax-{leg}_coxa-yaw", "ThC_pitch": "c_thorax-{leg}_coxa-pitch", "ThC_roll": "c_thorax-{leg}_coxa-roll",
            "CTr_pitch": "{leg}_coxa-{leg}_trochanterfemur-pitch", "CTr_roll": "{leg}_coxa-{leg}_trochanterfemur-roll",
            "FTi_pitch": "{leg}_trochanterfemur-{leg}_tibia-pitch", "TiTa_pitch": "{leg}_tibia-{leg}_tarsus1-pitch"}
WALK_SIGN = {ax: 1.0 for ax in AXIS_MAP}; WALK_SIGN["CTr_pitch"] = -1.0
WALK_GAIN = 0.75   # excursion scale onto FlyBody's legs: 1.0 falls at 1.7 s, 0.75 upright over 4 s (2.3 mm), 0.5 upright (1.2 mm); screened 2026-09-13. The jump films were made at 1.0 with a 1 s walk
# Signs screened 2026-09-12 (arms/2026-09-12-puppet-sign-screen.json): travel
# over 1.5 s at the champion's 104 Hz, all + 0.45 mm; CTr pitch flipped 2.39
# mm upright (0.96); coxa roll flipped 3.66 mm but by tumbling (upright
# -0.66); the other five flips 0.2 to 0.8 mm. The walking body travels 3.7.


class ColourBody(JumpFlyBody):
    """The lane's body, coloured, and with its legs PUPPETED from the walking
    body's joint angles (2026-09-12, Robin: the fly should be able to walk
    slowly before the flash). Each leg servo target is the stance angle plus
    the walking body's excursion from its own standing angle on the same
    joint role (motormap axis names), so the validated muscle and joint
    chain drives both bodies. Off once the twitch fires or the fly is
    airborne (the jump and the fold keep their own servo logic)."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        if self.renderer is not None:
            m = self.h["model"]; colour_fly(m)
            self.renderer = mj.Renderer(m, 480, 640)
            self.cam.distance, self.cam.azimuth, self.cam.elevation = 13.0, 150.0, -14.0
        import motormap as MM
        m = self.h["model"]
        self.puppet = []                   # (actuator, dof index, sign, lo, hi)
        for i in self.h["leg_act"]:
            jn = self.jname[i]
            for leg in MM.LEGS:
                for ax, tmpl in AXIS_MAP.items():
                    if tmpl.format(leg=leg) == jn:
                        j = self.h["leg_act_joint"][i]; lo, hi = m.jnt_range[j]
                        self.puppet.append((i, MM.DOF_INDEX[f"{leg}_{ax}"], WALK_SIGN[ax], float(lo), float(hi)))
        self.walk_theta = None; self.walk_ref = None
        self.x0 = np.asarray(self.h["data"].xpos[self.h["thorax"]], float).copy()
        # frame capture on a schedule: dense (every DENSE_MS of simulated time)
        # around the flash so the reflex can be shown at 50x slow motion, sparse
        # (SPARSE_MS) elsewhere; the parent's fixed capture is disabled
        self.frame_every = 10 ** 9; self.frame_t = []; self.next_cap = 0.0
        self.dense_window = None   # (t0_s, t1_s), set by run()

    def _capture(self):
        d = self.h["data"]
        if self.renderer is None:
            return
        dense = self.dense_window is not None and self.dense_window[0] <= d.time <= self.dense_window[1]
        if d.time + 1e-9 >= self.next_cap:
            self.renderer.update_scene(d, self.cam); self.frames.append(self.renderer.render().copy()); self.frame_t.append(float(d.time))
            self.next_cap = d.time + (DENSE_MS if dense else SPARSE_MS) * 1e-3

    def advance_to(self, t_s: float) -> None:
        if self.walk_theta is not None and self.t_twitch0 is None and self.t_off is None:
            if self.walk_ref is None:
                self.walk_ref = np.asarray(self.walk_theta, float).copy()
            d = self.h["data"]; exc = np.asarray(self.walk_theta, float) - self.walk_ref
            if PUPPET_GATE is not None and not PUPPET_GATE(float(d.time)):
                exc = exc * 0.0
            for i, k, sgn, lo, hi in self.puppet:
                d.ctrl[i] = float(np.clip(self.hold[i] + WALK_GAIN * sgn * exc[k], lo, hi))
        m, d = self.h["model"], self.h["data"]
        while d.time + FL2.CONTROL_DT <= t_s + 1e-9:
            super().advance_to(d.time + FL2.CONTROL_DT)   # one control tick at a time so capture sees every tick
            self._capture()


_orig_compile = JF.FlatGroundWorld.compile
def _compile_with_scene(self, *aa, **kk):
    add_scene(self.mjcf_root); return _orig_compile(self, *aa, **kk)
JF.FlatGroundWorld.compile = _compile_with_scene

COMMAND_GATE = None   # optional fn(t_ms, idx, rates) -> (idx, rates), used by walk_arms.py to gate the walking command
PUPPET_GATE = None    # optional fn(t_s) -> bool: False holds the legs at the stance (walk_arms.py rest bouts; the command rate is inert for walking, 2026-09-13 sweep)
DENSE_MS, SPARSE_MS = 0.5, 8.333   # capture spacing in simulated ms (dense around the flash, sparse elsewhere)
LIGHTOFF_HZ, LPLC2_W, STIM_MS = 80.0, 0.2, 30.0     # the loop's own frozen light-off rate (2026-09-12-jumpfly-arms-loop-calibration.json: 40 Hz 0 of 1, 80 Hz 2 of 2 at 32 ms); the runner's is 40 Hz, the walking network's background inhibits the GF more
ARMS = {"control": 1.0, "starved": 0.75, "half_starved": 0.5, "pre_death": 0.0, "ibuprofen": 1.0}   # starved = mid-course (Robin: the demo arm moves); half and pre-death kept for the record


def run(arm: str, seed: int, render_path=None, dur_ms: float = 2200.0, flash_ms: float = 600.0, land_at_s: float = 1.7, gate_ms: float = 100.0, playback: float = 0.25, cmd_hz: float | None = None, track: list | None = None) -> dict:
    p_lc4 = ARMS[arm]
    meta = pd.read_feather(ROOT / "data/banc_888_meta.feather"); edges = pd.read_feather(ROOT / "data/banc_888_edgelist_simple_v3.feather")
    ct = meta["cell_type"].astype(str); ids = meta["banc_888_id"].astype(str).to_numpy()
    gf, ttm, dlm, psi = find_cells(meta)
    lc4 = np.flatnonzero(ct.str.fullmatch("LC4") | ct.str.startswith("LC4_")); lplc2 = np.flatnonzero(ct.str.fullmatch("LPLC2") | ct.str.startswith("LPLC2_"))
    gf_ids = set(ids[gf]); lc4_ids = set(ids[lc4])
    edges, transfer_log = EV.correct_gf_visual_input(edges, meta); print("GF visual counts TRANSFERRED:", transfer_log, flush=True)
    cable = meta["l2_cable_length_um"]; gf_rin = {ids[i]: float(cable.median() / cable.iloc[i]) for i in gf}   # derived, as escape_visual
    if p_lc4 != 1.0:
        mask = edges["pre"].astype(str).isin(lc4_ids) & edges["post"].astype(str).isin(gf_ids)
        edges = edges.copy(); edges.loc[mask, "count"] = np.maximum(0, np.round(edges.loc[mask, "count"] * p_lc4)).astype(edges["count"].dtype)
        print(f"LC4 -> GF edges scaled x{p_lc4}: {int(mask.sum())} rows", flush=True)
    th = json.loads((HERE / "theta-wj1-jump-film.json").read_text())
    th["esc_ttm"] = 0; th.pop("force_spikes_ms", None); th["gap_rectify"] = True
    if cmd_hz is not None:
        th["cmd_hz"] = float(cmd_hz)   # the walking command (DNg100); the jump film held it at 1 Hz (standing), the champion walks at 104
    for k in ("esc_ttm_from_ms", "esc_ttm_to_ms", "esc_ttm_release", "esc_ttm_twitch_ms", "esc_ttm_ctr_max_deg", "esc_ttm_F_uN", "esc_ttm_r_mm", "esc_nmj_ms"):
        th.pop(k, None)
    stance = json.loads((HERE / "jumpfly-stance-v2-best.json").read_text())
    rows = FL1.motor_rows(meta); wing_rows = np.asarray(rows["wing"], dtype=np.int64)
    hal_rows, hal_side = haltere_rows(meta); camp = leg_camp_rows(meta)
    ttmn_rows = np.array([int(np.flatnonzero(ids == s)[0]) for s in TTMN_IDS])
    body = ColourBody(stance, FL3.POWER_ANCHOR_HZ, 8.0, land_at_s, render_path=render_path, playback=playback)
    index_rows = np.asarray(body.index.rows, dtype=np.int64)
    body.dense_window = ((flash_ms - 10.0) * 1e-3, (flash_ms + 160.0) * 1e-3)
    window = deque(maxlen=MN_WINDOW_TICKS)
    state = {"ttm_spikes": [], "gf_spikes": [], "raster": [], "silenced": 0}
    orig_run = cns.CNS.run; orig_cns = cns.CNS
    orig_body = W.B.Body; captured = {}

    def body_capture(*a, **k):
        b = orig_body(*a, **k)
        if len(np.asarray(b.theta)) == 42:   # the leg body, not the 10-joint abdomen adapter
            captured["bod"] = b
        return b
    gf_sid = [ids[i] for i in gf]

    def cns_with_gf(m, e, p):
        # escape_visual's GF: input resistance factor (searched) and balance exemption
        p.r_in_override = {**(p.r_in_override or {}), **gf_rin}
        ex = np.zeros(len(m), dtype=bool) if p.balance_exempt is None else np.asarray(p.balance_exempt, dtype=bool).copy()
        ex[gf] = True; p.balance_exempt = ex
        # the flash is a Poisson burst on the visual pools (k 1): under the
        # measured afferent regularity (k 44) a 16 Hz cell's first spike comes
        # a full interval (~62 ms) after onset, past the 30 ms flash (measured
        # 2026-09-12: LPLC2 0 spikes in the window, LC4 226)
        base = p.spike_reg_per_neuron
        reg = np.full(len(m), float(p.spike_reg), dtype=np.float64) if base is None else np.asarray(base, dtype=np.float64).copy()
        reg[lc4] = 1.0; reg[lplc2] = 1.0; p.spike_reg_per_neuron = reg
        return orig_cns(m, e, p)

    def patched_run(net, *args, **kwargs):
        orig_fb = np.asarray(kwargs["feedback_rows"], dtype=np.int64); n_orig = len(orig_fb)
        extra = np.concatenate([index_rows, gf, lc4, lplc2]); extra = extra[~np.isin(extra, orig_fb)]
        extra = pd.unique(extra)
        fb_all = np.concatenate([orig_fb, extra])
        pos = {name: np.array([int(np.flatnonzero(fb_all == r)[0]) for r in rr]) for name, rr in (("gf", gf), ("lc4", lc4), ("lplc2", lplc2), ("ttm", ttmn_rows), ("dlm", dlm))}
        idx_pos = np.array([int(np.flatnonzero(fb_all == r)[0]) for r in index_rows])
        orig_fn = kwargs["drive_fn"]
        vis_idx = np.concatenate([lc4, lplc2]); vis_rates = np.concatenate([np.full(len(lc4), LIGHTOFF_HZ), np.full(len(lplc2), LIGHTOFF_HZ * LPLC2_W)])

        def fn(t_ms, fb):
            idx, rates = orig_fn(t_ms, None if fb is None else fb[:n_orig])
            idx, rates = np.asarray(idx), np.asarray(rates, float).copy()
            if COMMAND_GATE is not None:
                idx, rates = COMMAND_GATE(t_ms, idx, rates)
            t_s = t_ms * 1e-3
            if track is not None and int(round(t_ms * 10)) % 100 == 0:   # every 10 ms
                xy = body.h["data"].xpos[body.h["thorax"]]; track.append((t_s, float(xy[0]), float(xy[1])))
            if fb is not None:
                c = np.asarray(fb)
                n_gf = float(c[pos["gf"]].sum())
                if n_gf > 0:
                    state["gf_spikes"].append(t_ms)
                if float(c[pos["ttm"]].sum()) > 0:
                    state["ttm_spikes"].append(t_ms)
                if flash_ms <= t_ms <= flash_ms + gate_ms:
                    body.ttm(t_s, float(c[pos["ttm"][0]]), float(c[pos["ttm"][1]]))
                if flash_ms - 10.0 <= t_ms <= flash_ms + 80.0:
                    state["raster"].append((round(t_ms, 1), int(c[pos["lc4"]].sum()), int(c[pos["lplc2"]].sum()), int(n_gf), int(c[pos["ttm"]].sum()), int(c[pos["dlm"]].sum())))
                window.append(np.asarray(c[idx_pos], float))
            mn_hz = np.sum(window, axis=0) / (len(window) * TICK_MS * 1e-3) if window else np.zeros(len(index_rows))
            body.feed(body.steady)   # steady power (the CNS-power arm is a separate receipt)
            if "bod" in captured:
                body.walk_theta = captured["bod"].theta
            body.advance_to(t_s)
            hal = np.clip(HAL_BASE_HZ + HAL_GAIN * (lambda w: w[1] + hal_side * (w[0] + w[2]))(body.mean_omega()), 0.0, HAL_MAX_HZ) if body.beating else np.zeros(len(hal_rows))
            if body.t_off is not None:
                mask = np.isin(idx, camp); rates[mask] = 0.0; state["silenced"] += int(mask.sum())
            if flash_ms <= t_ms < flash_ms + STIM_MS:
                idx, rates = np.concatenate([idx, vis_idx]), np.concatenate([rates, vis_rates])
            return np.concatenate([idx, hal_rows]), np.concatenate([rates, hal])

        kwargs["feedback_rows"] = fb_all
        kwargs["drive_fn"] = fn
        return orig_run(net, *args, **kwargs)

    orig_cns.run = patched_run; cns.CNS = cns_with_gf; W.cns.CNS = cns_with_gf; W.B.Body = body_capture
    t0 = time.time()
    try:
        score, raw, up, z = W.evaluate({}, meta, edges, th, seed=seed, dur_ms=dur_ms)
    finally:
        orig_cns.run = orig_run; cns.CNS = orig_cns; W.cns.CNS = orig_cns; W.B.Body = orig_body
    R = np.array(body.rows)
    gf_in = [t for t in state["gf_spikes"] if flash_ms <= t <= flash_ms + 100.0]   # light-off jump latency 60-80 ms measured (Trimarchi and Schneiderman 1995)
    out = {"arm": arm, "p_lc4": p_lc4, "seed": seed, "wall_s": round(time.time() - t0, 1), "flash_ms": flash_ms, "stim": {"lc4_hz": LIGHTOFF_HZ, "lplc2_hz": LIGHTOFF_HZ * LPLC2_W, "ms": STIM_MS}, "gf_rin": gf_rin, "transfer": transfer_log, "cmd_hz": th["cmd_hz"], "puppet_joints": len(body.puppet), "winged_body_disp_mm": float(np.linalg.norm(np.asarray(body.h["data"].xpos[body.h["thorax"]])[:2] - body.x0[:2])),
           "escape": bool(gf_in), "gf_latency_ms": (round(gf_in[0] - flash_ms, 1) if gf_in else None), "gf_spikes_ms": state["gf_spikes"][:10], "ttm_spike_ms": state["ttm_spikes"][:10],
           "t_twitch0_ms": None if body.t_twitch0 is None else round(body.t_twitch0 * 1e3, 1), "t_off_ms": None if body.t_off is None else round(body.t_off * 1e3, 1),
           "z_peak": float(R[:, 1].max()), "z_final": float(R[-1, 1]), "upright_final": float(R[-1, 2]),
           "airborne_s": float(((R[:, 3] > 0.15).sum()) * 5 * FL2.CONTROL_DT), "raster": state["raster"], "walking_body": {"disp": float(raw), "up": float(up), "z": float(z)}}
    if render_path and body.frames:
        import imageio.v2 as iio
        iio.mimwrite(render_path, body.frames, fps=body.fps, codec="libx264", quality=8); out["render"] = render_path; out["frames"] = len(body.frames); out["frame_t_ms"] = [round(t * 1e3, 2) for t in body.frame_t]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", default="control", choices=tuple(ARMS)); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", default=None); ap.add_argument("--dur", type=float, default=2200.0); ap.add_argument("--cmd-hz", type=float, default=None); ap.add_argument("--flash", type=float, default=600.0); ap.add_argument("--land", type=float, default=1.7); a = ap.parse_args()
    r = run(a.arm, a.seed, render_path=a.render, dur_ms=a.dur, cmd_hz=a.cmd_hz, flash_ms=a.flash, land_at_s=a.land)
    print(json.dumps(r))
    (HERE / f"2026-09-12-jumpfly-arms-{a.arm}-s{a.seed}.json").write_text(json.dumps(r, indent=1))
