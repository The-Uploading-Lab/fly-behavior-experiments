"""regression_walk.py -- the walking-lab constraint battery.

AGENTS.md: "every satisfied constraint becomes a test that re-runs
forever." This is that, for behaviour. Takes ANY theta and re-measures
the full accumulated checklist on fresh seeds; every chamber winner
runs through here BEFORE it gets any status (Robin's question,
2026-08-14: structural choices ratchet forward, so old truths must be
re-measured, not assumed).

Bands are FIXED here (sources: CONSTRAINTS.md / research-constraint-
hunt.md; never widen one to make a conflict go away):
  step_freq   5-16 Hz      standing median (Wosnitza/DeAngelis band)
  swing_ms    15-60 ms     median swing episode duration (~31 typical)
  duty        0.50-0.83    stance fraction per leg
  contra      0.35-0.65    L-R phase offset per segment (anti-phase 0.5)
  speed       2-45 mm/s    standing displacement rate
  ROM         Haustein mean +/- 2 sd per joint pair (deg):
              fFTi 96.6+/-7.7  mCTr 22.5+/-3.4  hFTi 84.1+/-9.9
              fCTr 91.2+/-9.5  mFTi 21.5+/-5.4  hCTr 56.3+/-9.0
  lifts       all 6 feet >= 5 lifts (>0.15 mm) per standing 5.5 s run
              (2026-08-14: load oscillation is not lift)
  R1i/R1c     swing-overlap avoidance vs circular null, majority of
              standing runs (fly-measured anti-phase, Mendes 2013)
  R5          load prolongs stance (certified match 2026-08-14)
  nonfoot     <=5% of ticks with proximal-leg (coxa/femur/tibia) or
              body ground contact -- "stands on its feet" (posture,
              registered 2026-08-18 from Robin's video-17 catch)

BATTERY v2 (2026-08-16): swing/tempo/duty/contra/R1 are now measured
from TIP HEIGHT (swing = clearance > 0.15 mm -- the same signal the
lift bar already uses), because the load signal chatters at the tick
scale: t025's load-swing read 7.5 ms = exactly 3 ticks, physically
impossible, while the same runs showed 13-42 clean tip lifts. The
BANDS ARE UNCHANGED; only the measure moved to the honest signal.
Load-based numbers are still computed and printed "(load, legacy)"
forever, so a measure migration can never quietly flip a verdict.
R5 stays load-native (it is ABOUT load). --reuse re-runs specific
seeds for measure re-analysis ONLY, never selection.

BATTERY v3.4 (2026-09-03, judge, from lane A's CD1-PX): the tip signal
is ALSO read with hysteresis, in the air once clearance exceeds 0.15 mm
and on the ground again only once it drops below 0.05 mm. The plain
threshold counts every crossing of the bar, and on the champion nc1 46%
of its swing episodes lasted one 2.5 ms tick: an unloaded foot hovering
within 0.1 mm of the bar, crossing it repeatedly, read as 5.7 Hz where
the hysteresis read gives 3.5 Hz. The `_h` fields (freq_tip_h,
swing_tip_h_ms, duty_tip_h, contra_tip_h, lifts_h) carry the hysteresis
read; every earlier field is computed exactly as before and printed
forever, as the v2 migration did for the load signal. The title clause
(AGENTS.md v3.4) reads the `_h` fields; the bands are unchanged.
Usage: .venv/bin/python3 regression_walk.py --theta results-round12.json
"""
import argparse
import json
import multiprocessing as mp
import os

import numpy as np

from cruse1 import (CONTACT_N, r1_stat, r5_stat, PAIRS_IPSI, PAIRS_CONTRA,
                    episodes, battery_ok, used_seeds)

# 2026-09-06 15:1x (judge, on lane E's degree table of 14:47 and its R430
# packet): the three per-segment left-right phase rows, computed by the same
# module the card instrument uses (lanes/A/phasemeasures.py, on main since
# 2026-09-03), so there is one implementation and not two. They are the
# project's best-measured coordination quantities -- 28,059 tracked sequences
# of Schulz et al. 2024, coherence-cut to 9,353, cross-spectral phase at each
# sequence's own step line -- and until now no title instrument computed them,
# so v3.8's floor veto could not read them on paired runs. Additive: a failure
# to load or to compute leaves the fields None and every other field is
# computed exactly as before.
try:
    import importlib.util as _ilu
    _pm_spec = _ilu.spec_from_file_location(
        "phasemeasures", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "lanes", "A", "phasemeasures.py"))
    _PM = _ilu.module_from_spec(_pm_spec); _pm_spec.loader.exec_module(_PM)
except Exception:
    _PM = None
# Lane E, 2026-09-06 15:50: the capture is built over sorted(LEG_KEYS.values())
# = [lf, lh, lm, rf, rh, rm] and lanes/A/phasemeasures.py labels the same arrays
# [lf, lm, lh, rf, rm, rh], so PAIRS[7]=(1,4) is the HIND pair and PAIRS[11]=(2,5)
# the MIDDLE one, the opposite of the names both files give them. Verified here
# against SEG_PAIRS two lines below, which uses the capture's own order. The root
# defect is in from_capture's tip read and is lane A's to fix; these two labels
# are corrected in the consumers so no receipt written from here is mislabelled.
_PHASE_IDX = {"lf_rf": 2, "lm_rm": 7, "lh_rh": 11}

LEGS = ["lf", "lh", "lm", "rf", "rh", "rm"]
SEG_PAIRS = {"front": (0, 3), "hind": (1, 4), "mid": (2, 5)}
HAUSTEIN = {("f", "FTi_pitch"): (96.6, 7.7), ("f", "CTr_pitch"): (91.2, 9.5),
            ("m", "FTi_pitch"): (21.5, 5.4), ("m", "CTr_pitch"): (22.5, 3.4),
            ("h", "FTi_pitch"): (84.1, 9.9), ("h", "CTr_pitch"): (56.3, 9.0)}
DUR = 6000.0
CLEAR_MM = 0.15     # foot leaves the ground above this clearance
LAND_MM = 0.05      # v3.4: and is down again only below this one
BF_RISE_MM = 0.15   # 2026-09-04 (judge): body-frame tip rise that makes a 10 ms
                    # episode a lift the leg made (lane A CD1-SS anatomy, lane D
                    # body-frame table: corner legs 0.27-0.59 mm, rocked middle
                    # feet 0.01-0.09; the middle legs are bimodal at this value)

_G = {}


def hysteresis_swing(clear, up=CLEAR_MM, down=LAND_MM):
    """(T,6) foot-in-air boolean from a (T,6) clearance matrix: in the air
    once clearance exceeds `up`, on the ground again only once it drops
    below `down`. Same semantics as lanes/A/swingmeasures._hyst (CD1-PX)."""
    out = np.zeros(clear.shape, dtype=bool)
    for j in range(clear.shape[1]):
        air = False
        col = clear[:, j]
        for t in range(len(col)):
            if air and col[t] < down:
                air = False
            elif not air and col[t] > up:
                air = True
            out[t, j] = air
    return out


def _cut_mask(e, keep, ids, cut):
    """Edges with count >= cut and both endpoints in `keep`. The membership
    test used pandas isin on 13.6M id strings, 1.9 s per process; the ids
    are decimal 64-bit integers, so they are cast in Arrow and joined as
    integers, 1.15 s, and the frame that results is equal row for row
    (verified with DataFrame.equals). Falls back to isin if a cast fails.
    (compute seat, 2026-09-02)"""
    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.compute as pc
        idx = pd.Index(pc.cast(keep["banc_888_id"].array._pa_array, pa.int64()).to_numpy())
        pre_ok = idx.get_indexer(pc.cast(e["pre"].array._pa_array, pa.int64()).to_numpy()) >= 0
        post_ok = idx.get_indexer(pc.cast(e["post"].array._pa_array, pa.int64()).to_numpy()) >= 0
        return (e["count"].to_numpy() >= cut) & pre_ok & post_ok
    except Exception:
        return ((e["count"] >= cut) & e["pre"].isin(ids) & e["post"].isin(ids)).to_numpy()


def _init(th, shuffle_seed=None, meta_path="data/banc_888_meta.feather",
          edges_path="data/banc_888_edgelist_simple_v3.feather", cut=10):
    os.environ["OMP_NUM_THREADS"] = "2"
    import pandas as pd
    m = pd.read_feather(meta_path)
    keep = m[(m["region"] == "ventral_nerve_cord")
             | (m["super_class"] == "descending")].reset_index(drop=True)
    ids = set(keep["banc_888_id"])
    e = pd.read_feather(edges_path)
    e = e[_cut_mask(e, keep, ids, cut)].reset_index(drop=True)
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        e["post"] = e["post"].to_numpy()[rng.permutation(len(e))]
    _G["meta"] = keep
    _G["edges"] = e
    _G["cache"] = {}
    _G["th"] = th


def _run(seed):
    from walk_search import evaluate
    th = dict(_G["th"])
    tick = float(th.get("tick_ms", 5.0))
    cap = []
    try:
        _, raw, up, z = evaluate(_G["cache"], _G["meta"], _G["edges"], th,
                                 seed=seed, dur_ms=DUR, capture=cap)
        # v3.2 (re-freeze event 2026-08-31): record the posture angle the
        # tilt-based standing clause grades on. Additive field; every
        # pre-existing field is computed exactly as before.
        import walk_search as _ws
        _tilt = _ws.LAST_EXTRAS.get("final_tilt_deg")
    except Exception as e:
        return {
            "seed": seed,
            "error": type(e).__name__,
            "error_message": str(e),
        }
    trans = int(round(500.0 / tick))
    loads = np.array([c[4] for c in cap])[trans:]
    tipz = np.array([c[5] for c in cap])[trans:]
    theta = np.array([c[3] for c in cap])[trans:]
    nonfoot = np.array([c[8] for c in cap], dtype=bool)[trans:]
    # force-weighted posture (2026-08-18): share of ground force borne
    # by non-foot segments. The tick measure above cannot tell a graze
    # from bearing weight; this can. Both are reported forever.
    _nf_f = np.array([c[8] for c in cap], dtype=float)[trans:]
    _tot_f = np.array([c[9] for c in cap], dtype=float)[trans:]
    contact = loads > CONTACT_N
    standing = bool(up > 0.90 and z > 1.0 and raw >= 0.3)
    # v3.3 (re-freeze 2026-09-02): record whether step timing is a model
    # output. A commanded external step tape (measured_step_hz) makes the
    # cadence rows an echo of the command; the title clause needs the
    # distinction machine-checked. Additive fields; nothing else changes.
    _cmd_hz = th.get("measured_step_hz")
    rec = {"seed": seed, "raw": round(raw, 2), "up": round(up, 2),
           "standing": standing,
           "final_tilt_deg": (round(_tilt, 2) if _tilt is not None else None),
           "standing_v32": bool(_tilt is not None and _tilt < 51.68
                                and z > 1.0 and raw >= 0.3),
           "commanded_step_hz": (float(_cmd_hz) if _cmd_hz is not None
                                 else None),
           "endogenous_timing": bool(_cmd_hz is None),
           "nonfoot": round(float(np.mean(nonfoot)), 4),
           "nonfoot_force": round(float(_nf_f.sum()
                                        / max(_tot_f.sum(), 1e-9)), 4)}
    # v3.3 re-freeze: gait fields are also computed for runs that stand by
    # the tilt clause but fail the legacy |qw| test (the heading-turn class
    # v3.2 exists for). Legacy standing runs are computed exactly as before,
    # and the seven-metric summary still filters on the legacy flag.
    if not (standing or rec["standing_v32"]):
        return rec
    span_s = contact.shape[0] * tick / 1000.0
    # v2: tip-height swing signal -- clearance above each foot's own
    # 5th-percentile height, same 0.15 mm bar as the lift count
    clear = tipz - np.percentile(tipz, 5, axis=0, keepdims=True)
    tip_swing = clear > CLEAR_MM                # True = foot in the air
    # v3.4: the same signal with hysteresis (see the module docstring).
    tip_swing_h = hysteresis_swing(clear)

    def gait_measures(down):
        """(tempo Hz, median swing ms, median duty) from a (T,6)
        foot-down boolean matrix."""
        fs, swings, duties = [], [], []
        for j in range(6):
            on = down[:, j]
            ed = np.flatnonzero(on[1:] & ~on[:-1])
            if len(ed) >= 2:
                fs.append(len(ed) / span_s)
            for s_, e_ in episodes(~on):
                swings.append((e_ - s_) * tick)
            duties.append(float(np.mean(on)))
        return (float(np.nanmedian(fs)) if fs else 0.0,
                float(np.median(swings)) if swings else np.nan,
                float(np.median(duties)))

    def contra_phase(down):
        """nearest R swing-onset after each L swing-onset, as a
        fraction of the L step period, per segment pair."""
        phases = []
        for a, b in SEG_PAIRS.values():
            la = np.flatnonzero(~down[1:, a] & down[:-1, a]) + 1
            lb = np.flatnonzero(~down[1:, b] & down[:-1, b]) + 1
            if len(la) >= 3 and len(lb) >= 3:
                per = np.median(np.diff(la))
                for t in la:
                    nxt = lb[lb > t]
                    if len(nxt) and per > 0:
                        phases.append(((nxt[0] - t) / per) % 1.0)
        return float(np.median(phases)) if phases else np.nan

    rec["freq"], rec["swing_ms"], rec["duty"] = gait_measures(contact)
    rec["contra"] = contra_phase(contact)
    (rec["freq_tip"], rec["swing_tip_ms"],
     rec["duty_tip"]) = gait_measures(~tip_swing)
    rec["contra_tip"] = contra_phase(~tip_swing)
    rec["speed"] = raw / (DUR / 1000.0 - 0.5)
    # achieved ROM per Haustein joint pair (deg, max over both sides)
    import motormap as MM
    rom = {}
    for (seg, jnt), _ in HAUSTEIN.items():
        vals = []
        for side in ("l", "r"):
            nm = f"{side}{seg}_{jnt}"
            if nm in MM.DOF_NAMES:
                k = MM.DOF_NAMES.index(nm)
                vals.append(float(np.degrees(theta[:, k].max()
                                             - theta[:, k].min())))
        rom[f"{seg}{jnt.split('_')[0]}"] = max(vals) if vals else np.nan
    rec["rom"] = rom
    # per-leg real lifts (onsets of the tip-swing signal)
    rec["lifts"] = [int(np.sum(tip_swing[1:, j] & ~tip_swing[:-1, j]))
                    for j in range(6)]
    # v3.4: the hysteresis read of the same five quantities, plus the
    # fraction of plain-threshold swing episodes that last one tick (the
    # flicker CD1-PX measured; 0.46 on nc1, a step signal reads near 0).
    (rec["freq_tip_h"], rec["swing_tip_h_ms"],
     rec["duty_tip_h"]) = gait_measures(~tip_swing_h)
    rec["contra_tip_h"] = contra_phase(~tip_swing_h)
    rec["lifts_h"] = [int(np.sum(tip_swing_h[1:, j] & ~tip_swing_h[:-1, j]))
                      for j in range(6)]
    # 2026-09-04 (judge, on lane A's CD1-QZ): the hysteresis read with a
    # minimum swing episode of 10 ms. On nc1 the two reads agreed (v3.4);
    # on qh1 32% of hysteresis episodes last one tick (hover taps) and the
    # reads diverge (6.0 against 4.3 Hz). Episodes shorter than 10 ms are
    # counted as ground; every existing field above is untouched.
    _min_ticks = max(1, int(round(10.0 / tick)))
    tip_swing_h10 = tip_swing_h.copy()
    for j in range(6):
        for s_, e_ in episodes(tip_swing_h[:, j]):
            if e_ - s_ < _min_ticks:
                tip_swing_h10[s_:e_, j] = False
    (rec["freq_tip_h10"], rec["swing_tip_h10_ms"],
     rec["duty_tip_h10"]) = gait_measures(~tip_swing_h10)
    rec["contra_tip_h10"] = contra_phase(~tip_swing_h10)
    rec["lifts_h10"] = [int(np.sum(tip_swing_h10[1:, j] & ~tip_swing_h10[:-1, j]))
                        for j in range(6)]
    rec["one_tick_frac_h"] = (round(float(np.mean(np.array(
        [e_ - s_ for j in range(6) for s_, e_ in episodes(tip_swing_h[:, j])]
        or [0]) <= 1)), 4))
    # 2026-09-04 07:5x (judge, on lane A's lift anatomy and lane D's
    # body-frame rise table): the body-frame read of each 10 ms episode.
    # qh1's middle feet are lifted by the body rolling 8-11 degrees onto the
    # other side with no joint moving more than 2.4 degrees, and the world
    # clearance counts that as a lift; the tip's rise in the BODY frame is
    # what the leg itself contributes. For each hysteresis-10 ms episode the
    # rise is the body-frame tip height at the tick of peak world clearance
    # minus its value four ticks before lift-off (lane A's CD1-SS definition);
    # an episode is a body-frame lift when that rise is at least BF_RISE_MM.
    # Additive fields only; every field above is computed exactly as before.
    # Capture element 0 is the body position, 1 the body quaternion (MuJoCo
    # w, x, y, z), 10 the tarsus-tip world xyz per leg in the same leg order
    # as the tip-z field; a capture without element 10 leaves the fields None.
    try:
        _pos = np.array([c[0] for c in cap], dtype=float)[trans:]
        _quat = np.array([c[1] for c in cap], dtype=float)[trans:]
        _tipxyz = np.array([c[10] for c in cap], dtype=float)[trans:]
        _bf = np.zeros(tipz.shape, dtype=float)
        for t in range(_bf.shape[0]):
            w_, x_, y_, z_ = _quat[t]
            _R = np.array([
                [1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ - z_ * w_), 2 * (x_ * z_ + y_ * w_)],
                [2 * (x_ * y_ + z_ * w_), 1 - 2 * (x_ * x_ + z_ * z_), 2 * (y_ * z_ - x_ * w_)],
                [2 * (x_ * z_ - y_ * w_), 2 * (y_ * z_ + x_ * w_), 1 - 2 * (x_ * x_ + y_ * y_)]])
            _bf[t] = (_R.T @ (_tipxyz[t] - _pos[t]).T)[2]
        tip_swing_bf10 = np.zeros_like(tip_swing_h10)
        _rises = [[] for _ in range(6)]
        for j in range(6):
            for s_, e_ in episodes(tip_swing_h10[:, j]):
                _pk = s_ + int(np.argmax(clear[s_:e_, j]))
                _rise = float(_bf[_pk, j] - _bf[max(s_ - 4, 0), j])
                _rises[j].append(_rise)
                if _rise >= BF_RISE_MM:
                    tip_swing_bf10[s_:e_, j] = True
        (rec["freq_tip_bf10"], rec["swing_tip_bf10_ms"],
         rec["duty_tip_bf10"]) = gait_measures(~tip_swing_bf10)
        rec["contra_tip_bf10"] = contra_phase(~tip_swing_bf10)
        rec["lifts_bf10"] = [int(np.sum(tip_swing_bf10[1:, j] & ~tip_swing_bf10[:-1, j]))
                             for j in range(6)]
        rec["bf_rise_median_mm"] = [(round(float(np.median(r_)), 4) if r_ else None)
                                    for r_ in _rises]
        rec["bf_frac_ge_thresh"] = [(round(float(np.mean(np.array(r_) >= BF_RISE_MM)), 3)
                                     if r_ else None) for r_ in _rises]
        rec["bf_rises_mm"] = [[round(v, 4) for v in r_] for r_ in _rises]
    except (IndexError, TypeError, ValueError):
        rec["freq_tip_bf10"] = None
        rec["contra_tip_bf10"] = None
        rec["lifts_bf10"] = None
        rec["bf_rise_median_mm"] = None
        rec["bf_frac_ge_thresh"] = None
        rec["bf_rises_mm"] = None
    # heave: body height oscillation (body.height_oscillation CONFLICT, lane A
    # CD1-SW/SX, confirmed lane D 2026-09-04 at ~45% on qh1). p95 minus p5 of
    # body z over the scoring window, normalised by median body z. Real ceiling
    # ~10% (Chun 2021, derived). From capture element 0 (root xyz), z index 2.
    # Lane D's additive diff (lanes/D/2026-09-04-heave_frac.patch), ported by
    # the judge; every existing field is untouched.
    _bz = np.array([c[0] for c in cap], dtype=float)[trans:, 2]
    rec["heave_frac"] = (
        round(float((np.percentile(_bz, 95) - np.percentile(_bz, 5))
                    / np.median(_bz)), 4)
        if _bz.size and float(np.median(_bz)) != 0.0 else None)
    # mean total foot load per run (lane E R394, 2026-09-06 01:12): the
    # 171-degree inversion has no precursor in tilt or in the number of feet
    # down, and has a large one in LOAD. Both falls on lane E's record carried
    # about half the champion's foot load for a second before departure (41.4
    # and 34.4 units in the last 200 ms against wb4's steady 65.8 and 66.2)
    # while never being airborne for a single tick. The capture holds 2,400
    # ticks of six foot loads and the receipt held none of it, so a precursor
    # nobody could read was the reason the inversion looked mechanism-less.
    # Added by the judge under the probe; every existing field is untouched.
    # `loads` is capture element 4, per-leg ground load, already read above.
    rec["foot_load_mean"] = (round(float(np.mean(np.sum(loads, axis=1))), 4)
                             if loads.ndim == 2 and loads.size else None)
    rec["foot_load_min_200ms"] = (
        round(float(np.min(np.convolve(np.sum(loads, axis=1),
                                       np.ones(int(round(200.0 / tick))) / int(round(200.0 / tick)),
                                       mode="valid"))), 4)
        if loads.ndim == 2 and loads.shape[0] >= int(round(200.0 / tick)) else None)
    # body-frame forward/lateral speed (registry speed conflict; lane A found the
    # 07B/12A/18B between-segment seam drifts one-signed sideways, so the SIGN of
    # speed_lat is the quantity that catches it). Net world xy displacement per
    # tick, rotated into the body frame by the per-tick heading, summed over the
    # scoring window, divided by window seconds. Capture element 0 (root xyz),
    # element 1 (root quaternion w,x,y,z); yaw as walk_search._body_yaw. Lane D's
    # additive diff (lanes/D/2026-09-04-speed_fwd_lat.patch), ported by the judge.
    _sp = np.array([c[0] for c in cap], dtype=float)[trans:]
    _sq = np.array([c[1] for c in cap], dtype=float)[trans:]
    if _sp.shape[0] >= 2:
        _dxy = np.diff(_sp[:, :2], axis=0)
        _yw = np.arctan2(2.0 * (_sq[:-1, 0] * _sq[:-1, 3] + _sq[:-1, 1] * _sq[:-1, 2]),
                         1.0 - 2.0 * (_sq[:-1, 2] ** 2 + _sq[:-1, 3] ** 2))
        _fwd = np.cos(_yw) * _dxy[:, 0] + np.sin(_yw) * _dxy[:, 1]
        _lat = -np.sin(_yw) * _dxy[:, 0] + np.cos(_yw) * _dxy[:, 1]
        _win_s = _dxy.shape[0] * tick / 1000.0
        rec["speed_fwd"] = round(float(_fwd.sum() / _win_s), 4) if _win_s > 0 else None
        rec["speed_lat"] = round(float(_lat.sum() / _win_s), 4) if _win_s > 0 else None
    else:
        rec["speed_fwd"] = None
        rec["speed_lat"] = None
    # net heading change over the scoring window (judge, 2026-09-05, for the
    # vision grade: does the grating's direction turn the fly). Unwrapped body
    # yaw from capture element 1, last tick minus first, radians, positive
    # anticlockwise seen from above; yaw_trace_rad samples the same unwrapped
    # yaw every 250 ms from the window start so a stimulus window can be read
    # without a rerun. Additive; every existing field is computed as before.
    if _sq.shape[0] >= 2:
        _yaw_all = np.unwrap(np.arctan2(
            2.0 * (_sq[:, 0] * _sq[:, 3] + _sq[:, 1] * _sq[:, 2]),
            1.0 - 2.0 * (_sq[:, 2] ** 2 + _sq[:, 3] ** 2)))
        rec["yaw_net_rad"] = round(float(_yaw_all[-1] - _yaw_all[0]), 4)
        _every = max(1, int(round(250.0 / tick)))
        rec["yaw_trace_rad"] = [round(float(v), 4) for v in _yaw_all[::_every]]
    else:
        rec["yaw_net_rad"] = None
        rec["yaw_trace_rad"] = None
    _ep = [e_ - s_ for j in range(6) for s_, e_ in episodes(tip_swing[:, j])]
    rec["flicker_frac"] = (round(float(np.mean(np.array(_ep) <= 1)), 4)
                           if _ep else None)
    # Cruse statistics; R1 on both swing signals, R5 load-native
    rng = np.random.default_rng(seed + 999000)
    for name, (o, nl, p) in (
            ("R1i", r1_stat(~contact, PAIRS_IPSI, rng)),
            ("R1c", r1_stat(~contact, PAIRS_CONTRA, rng)),
            ("R1i_tip", r1_stat(tip_swing, PAIRS_IPSI, rng)),
            ("R1c_tip", r1_stat(tip_swing, PAIRS_CONTRA, rng)),
            ("R5", r5_stat(loads, contact, rng))):
        rec[name] = None if (isinstance(p, float) and np.isnan(p)) \
            else round(float(p), 4)
        # The effect size behind that p-value. r1_stat and r5_stat each return
        # (observed statistic, null mean, p) and this loop unpacked all three
        # and recorded only the last, so every R1 and R5 column in the record
        # is a permutation p-value floored at 1/(1 + NSHUF) = 0.005. Measured
        # 2026-09-04 (lane E, 22 runs on qh1): R1c took four distinct values
        # and 17 of 22 sat on that floor, so the column cannot grade a change
        # in degree, only detect one in kind. Additive: every pre-existing
        # field is computed exactly as before and the rng draw is unchanged.
        rec[name + "_obs"] = None if (isinstance(o, float) and np.isnan(o)) \
            else round(float(o), 6)
        rec[name + "_null"] = None if (isinstance(nl, float) and np.isnan(nl)) \
            else round(float(nl), 6)
    # The per-segment phase rows. from_capture takes the same 500 ms transient
    # this function already uses, so the battery and the card read one number.
    for _k in _PHASE_IDX:
        rec["phase_" + _k + "_deg"] = None
    rec["phase_line_hz"] = None
    if _PM is not None:
        try:
            _pm = _PM.from_capture(cap, tick, transient_ms=500.0)
            _ph = _pm["phase_feet_deg"]
            for _k, _i in _PHASE_IDX.items():
                rec["phase_" + _k + "_deg"] = round(float(abs(_ph[_i])), 4)
            rec["phase_line_hz"] = (None if _pm["line_hz"] is None
                                    else round(float(_pm["line_hz"]), 4))
        except Exception:
            pass
    return rec


def band(x, lo, hi):
    return "PASS" if (x is not None and not np.isnan(x) and lo <= x <= hi) \
        else "CONFLICT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", required=True)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--start", type=int, default=386)
    ap.add_argument("--shuffle", type=int, default=None,
                    help="post-permute the core with this rng seed "
                         "(shuffled-connectome comparison arm)")
    ap.add_argument("--reuse", default=None,
                    help="comma-separated seeds to re-run verbatim; "
                         "MEASURE RE-ANALYSIS ONLY, never selection")
    ap.add_argument("--out", default="results-regression-walk.json")
    ap.add_argument("--meta", default="data/banc_888_meta.feather")
    ap.add_argument("--edges",
                    default="data/banc_888_edgelist_simple_v3.feather")
    ap.add_argument("--cut", type=int, default=10,
                    help="edge count threshold (male: winner's own cut)")
    args = ap.parse_args()
    d = json.load(open(args.theta))
    th = d["winner"] if isinstance(d, dict) and "winner" in d else d
    assert battery_ok(), "battery gate"
    if args.reuse:
        seeds = [int(s) for s in args.reuse.split(",")]
        print("!! REUSED SEEDS: measure re-analysis only -- this run "
              "must never pick a winner", flush=True)
    else:
        u = used_seeds()
        seeds, k = [], args.start
        while len(seeds) < args.seeds:
            if k not in u:
                seeds.append(k)
            k += 1
    print(f"battery on {args.theta}"
          + (f" [SHUFFLED core, seed {args.shuffle}]"
             if args.shuffle is not None else " [real core]")
          + f", seeds {seeds}", flush=True)
    mp.set_start_method("spawn")
    with mp.Pool(6, initializer=_init,
                 initargs=(th, args.shuffle, args.meta, args.edges,
                           args.cut)) as pool:
        res = list(pool.imap_unordered(_run, seeds))
    json.dump(res, open(args.out, "w"), indent=1)
    errs = [r for r in res if "error" in r]
    if errs:
        print(f"!! {len(errs)}/{len(res)} runs ERRORED "
              f"({errs[0]['error']}) -- battery invalid, fix before "
              f"reading any line below", flush=True)
    st = [r for r in res if r.get("standing")]
    n = len(res)
    print(f"\nstanding {len(st)}/{n} (real ~1.0; open conflict, report "
          f"only)", flush=True)
    if not st:
        print("no standing runs -- battery cannot score; CONFLICT by "
              "default", flush=True)
        return

    def med(key):
        v = [r[key] for r in st if r.get(key) is not None
             and not (isinstance(r[key], float) and np.isnan(r[key]))]
        return float(np.median(v)) if v else float("nan")

    rows = [("step_freq Hz", med("freq_tip"), 5.0, 16.0, med("freq")),
            ("swing ms", med("swing_tip_ms"), 15.0, 60.0,
             med("swing_ms")),
            ("duty", med("duty_tip"), 0.50, 0.83, med("duty")),
            ("contra phase", med("contra_tip"), 0.35, 0.65,
             med("contra")),
            ("speed mm/s", med("speed"), 2.0, 45.0, None)]
    for name, v, lo, hi, legacy in rows:
        extra = "" if legacy is None else f"   (load, legacy: {legacy:.2f})"
        print(f"{name:<14} {v:7.2f}  band [{lo}, {hi}]  "
              f"{band(v, lo, hi)}{extra}", flush=True)
    # v3.4: the hysteresis read (the title clause's), same bands
    if st and st[0].get("freq_tip_h") is not None:
        for name, key, lo, hi in (("step_freq Hz", "freq_tip_h", 5.0, 16.0),
                                  ("swing ms", "swing_tip_h_ms", 15.0, 60.0),
                                  ("duty", "duty_tip_h", 0.50, 0.83),
                                  ("contra phase", "contra_tip_h", 0.35,
                                   0.65)):
            v = med(key)
            print(f"{name:<14} {v:7.2f}  band [{lo}, {hi}]  "
                  f"{band(v, lo, hi)}   (v3.4 hysteresis read)", flush=True)
        print(f"{'flicker frac':<14} {med('flicker_frac'):7.3f}  "
              f"(one-tick share of threshold swings; step signal ~0)",
              flush=True)
    # posture (registered 2026-08-18): fraction of post-transient ticks
    # with proximal-leg or body ground contact. Real walking flies bear
    # on the tarsi with body and proximal segments clear; 5% allows
    # incidental grazes. Absent from pre-instrument batteries (--reuse).
    nf = med("nonfoot")
    if not np.isnan(nf):
        print(f"{'nonfoot ticks':<14} {nf:7.3f}  band [0.0, 0.05]  "
              f"{band(nf, 0.0, 0.05)}   (any non-foot contact)", flush=True)
    nff = med("nonfoot_force")
    if not np.isnan(nff):
        print(f"{'nonfoot FORCE':<14} {nff:7.3f}  band [0.0, 0.05]  "
              f"{band(nff, 0.0, 0.05)}   (weight actually borne off "
              f"the feet)", flush=True)
    for key, (mu, sd) in HAUSTEIN.items():
        tag = f"{key[0]}{key[1].split('_')[0]}"
        v = float(np.median([r["rom"][tag] for r in st
                             if tag in r.get("rom", {})]))
        print(f"ROM {tag:<10} {v:7.1f}  band [{mu - 2 * sd:.1f}, "
              f"{mu + 2 * sd:.1f}]  {band(v, mu - 2 * sd, mu + 2 * sd)}",
              flush=True)
    minlift = min(min(r["lifts"]) for r in st)
    print(f"all-feet lifts  min {minlift:>3}  bar >= 5   "
          f"{'PASS' if minlift >= 5 else 'CONFLICT'}", flush=True)
    if st and st[0].get("lifts_h") is not None:
        minlift_h = min(min(r["lifts_h"]) for r in st)
        print(f"all-feet lifts  min {minlift_h:>3}  bar >= 5   "
              f"{'PASS' if minlift_h >= 5 else 'CONFLICT'}   "
              f"(v3.4 hysteresis read)", flush=True)
    for rule in ("R1i_tip", "R1c_tip", "R5"):
        sig = sum(1 for r in st if r.get(rule) is not None
                  and r[rule] < 0.05)
        ok = sig * 2 > len(st)
        leg = rule.replace("_tip", "")
        lsig = sum(1 for r in st if r.get(leg) is not None
                   and r[leg] < 0.05) if rule != "R5" else None
        extra = "" if lsig is None else f"   (load, legacy: {lsig}/{len(st)})"
        print(f"{rule:<14} {sig}/{len(st)} runs p<0.05  "
              f"{'PASS' if ok else 'CONFLICT' if rule != 'R5' else 'GAP'}"
              f"{extra}", flush=True)


if __name__ == "__main__":
    main()
