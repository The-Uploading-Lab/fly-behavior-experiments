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

LEGS = ["lf", "lh", "lm", "rf", "rh", "rm"]
SEG_PAIRS = {"front": (0, 3), "hind": (1, 4), "mid": (2, 5)}
HAUSTEIN = {("f", "FTi_pitch"): (96.6, 7.7), ("f", "CTr_pitch"): (91.2, 9.5),
            ("m", "FTi_pitch"): (21.5, 5.4), ("m", "CTr_pitch"): (22.5, 3.4),
            ("h", "FTi_pitch"): (84.1, 9.9), ("h", "CTr_pitch"): (56.3, 9.0)}
DUR = 6000.0
CLEAR_MM = 0.15     # foot leaves the ground above this clearance
LAND_MM = 0.05      # v3.4: and is down again only below this one

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
    rec["lifts_h10"] = [int(np.sum(tip_swing_h10[1:, j] & ~tip_swing_h10[:-1, j]))
                        for j in range(6)]
    rec["one_tick_frac_h"] = (round(float(np.mean(np.array(
        [e_ - s_ for j in range(6) for s_, e_ in episodes(tip_swing_h[:, j])]
        or [0]) <= 1)), 4))
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
