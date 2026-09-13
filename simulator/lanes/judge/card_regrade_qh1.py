"""Re-grade the champion's card on the champion, qh1 (judge goal, Robin 2026-09-03).
card_regrade_nc1.py's rows (battery measures through the title path, footfall
separation, body height, ROM, reflex rules) plus the rows the programme has
carried unread onto the champion: the flyscore metrics including posture
variability from the abdomen capture channel (element 18), the three
contralateral step-phase rows (lane A's phasemeasures at the run's step line,
CD1-QP read), per-leg hysteresis lifts and the hind left-right cadence
asymmetry, and path straightness over 500-5500 ms. Metal by default under the
exclusive lock (the engine rule), --engine numpy for the one confirmation.
From 2026-09-07 the three phase rows carry a fixed-heading twin
(lanes/judge/phase_fixed_heading.py, lane A's CD2-FD: the body-frame read folds
body yaw over planted feet into left-right phase), with per-leg stance fractions
and the yaw swing beside it. Reported only: no band, no grade, no existing field
changed.
Reproduce:
    PYTHONPATH=. .venv/bin/python lanes/judge/card_regrade_qh1.py \
        --theta results-qh1-theta.json --seeds 30146-30157 --out lanes/judge/2026-09-03-card-qh1-metal.json
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import regression_walk as RW  # noqa: E402
import walk_search  # noqa: E402
from footfall1 import BL_MM, LEGS, PAIR, quat_to_R  # noqa: E402
from flyscore import layout as L  # noqa: E402
from flyscore import metrics as M  # noqa: E402
from adapt_and_score import resample_to_fps  # noqa: E402
_spec = importlib.util.spec_from_file_location("phasemeasures", ROOT / "lanes/A/phasemeasures.py")
PM = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(PM)
# lane A's canonical line estimator, named by lane A at 06:40 so the judge merges
# one function rather than guessing among six files. REPORTED ONLY: it is not in
# BANDS and not in title_match_v38.CARD_ROWS, so it grades nothing and no verdict
# moves. It exists because gait.cord_rhythm_frequency is open as a CONFLICT and
# the card's own phase_line_hz takes exactly two values across all 69 judge card
# runs (5.47 and 6.25 Hz), which cannot resolve a 4.2 Hz gap or any movement in it.
_lspec = importlib.util.spec_from_file_location("line_estimator", ROOT / "lanes/A/line_estimator.py")
LE = importlib.util.module_from_spec(_lspec); _lspec.loader.exec_module(LE)
sys.path.insert(0, str(ROOT / "lanes/judge"))
from card_regrade_nc1 import BANDS as BANDS0, RULES, footfall, sha, git  # noqa: E402
import phase_fixed_heading as PFH  # noqa: E402
import capture_conflict_rows as CCR  # noqa: E402

BANDS = dict(BANDS0)
BANDS.update({
    "fs_posture_variability_rad": (0.0256, 0.0859),   # flyscore.posture_variability (metrics key posture_variability_rad)
    "fs_tripod_contrast": (-0.091, 0.179),         # flyscore.tripod_contrast
    "fs_gait_signal_quality": (0.14, 0.297),       # flyscore.gait_signal_quality
    "fs_lr_asymmetry": (0.008, 0.323),             # flyscore.lr_asymmetry
    "straightness_5p5s": (0.50, 0.98),             # gait.straightness_5p5s (real p5-p95)
    # body.height_oscillation, added 2026-09-05 23:5x by the judge after the row
    # was found carrying qh1's value across three title changes because nothing
    # graded it. Chun/Biswas/Bhandawat 2021: the bob is at most about 10% of body
    # height. The field is (p95 - p5) of body z over the scoring window divided by
    # median height, which is the registry row's own stated read.
    "heave_frac": (0.0, 0.10),                     # body.height_oscillation
    # Four registry rows predate the current champion because the card did not
    # retain the capture channels needed to re-read them.  The first two fit the
    # card's ordinary scalar shape.  The two per-leg knee rows are summarised
    # separately below because every leg must satisfy its own condition.
    "abdomen_lateral_excursion_deg": CCR.ABDOMEN_BAND_DEG,
    "conduction_delay_mean_ms": CCR.CONDUCTION_BAND_MS,
    # gait.leg_excursion_body_lengths was added here 2026-09-06 02:07 and
    # REMOVED at 02:17 on lane A's CD2-BK: the two body_length landmark pairs
    # do not span the same share of the animal (reference 0.360 of AP extent,
    # model 0.551, a factor of 1.531 all in the denominator), so the ratio is
    # NOT scale-free between the sides and a band on it grades an artefact.
    # The comparison that works is excursion over AP EXTENT on both sides,
    # which the card does not compute; the row returns when it does.
})
PHASE_REAL = {"lf_rf": (179.1, 177.8, 180.0), "lm_rm": (179.9, 178.3, 180.0), "lh_rh": (179.6, 178.4, 180.0)}
# swapped 2026-09-06 15:5x on lane E's finding; see regression_walk.py _PHASE_IDX
PHASE_IDX = {"lf_rf": 2, "lm_rm": 7, "lh_rh": 11}
TIPZ_ORDER = ("lf", "lh", "lm", "rf", "rh", "rm")   # capture element 10 leg order
LEG_MAP = {"lf": "L1", "lh": "L3", "lm": "L2", "rf": "R1", "rh": "R3", "rm": "R2"}
WINDOW_MS = (500.0, 5500.0)

def forward_xy(quat):
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    f = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z)], axis=1)
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)

def egocentric_tips(cap):
    root = np.asarray([r[0] for r in cap], dtype=float)[:, :2]
    quat = np.asarray([r[1] for r in cap], dtype=float)
    tips = np.asarray([r[10] for r in cap], dtype=float)
    fwd = forward_xy(quat); right = np.stack([fwd[:, 1], -fwd[:, 0]], axis=1)
    pts = np.full((len(cap), 32, 2), np.nan)
    for i, lg in enumerate(TIPZ_ORDER):
        rel = tips[:, i, :2] - root
        pts[:, L.LEG_TIPS[LEG_MAP[lg]], :] = np.stack([np.sum(rel * right, 1), -np.sum(rel * fwd, 1)], 1)
    pts[:, L.THORAX, :] = 0.0
    if len(cap) and len(cap[0]) > 18 and len(cap[0][18]) == 3:
        abd = np.asarray([r[18] for r in cap], dtype=float)[:, :2]; rel = abd - root
        pts[:, L.ABDOMEN, :] = np.stack([np.sum(rel * right, 1), -np.sum(rel * fwd, 1)], 1)
    return pts

def flyscore_rows(cap, tick):
    t = np.arange(len(cap)) * tick
    a, b = int(np.searchsorted(t, WINDOW_MS[0])), int(np.searchsorted(t, WINDOW_MS[1]))
    pts = resample_to_fps(egocentric_tips(cap[a:b]), tick / 1000.0, L.FPS)
    sc = M.score_sequence(pts, fps=L.FPS)
    out = {f"fs_{k}": (None if (isinstance(v, float) and np.isnan(v)) else float(v)) for k, v in sc.items()}
    # straightness as gait.straightness_5p5s was read (lane A CD1-PV, Katsov 2017 at 30 fps):
    # net displacement over path length with the body position resampled to 30 Hz first,
    # so per-tick sway does not inflate the path.
    body = np.asarray([r[0] for r in cap[a:b]], dtype=float)[:, :2]
    tt = np.arange(len(body)) * tick / 1000.0
    grid = np.arange(0.0, tt[-1], 1.0 / 30.0)
    b30 = np.stack([np.interp(grid, tt, body[:, 0]), np.interp(grid, tt, body[:, 1])], axis=1)
    path = float(np.sum(np.linalg.norm(np.diff(b30, axis=0), axis=1)))
    out["straightness_5p5s"] = float(np.linalg.norm(b30[-1] - b30[0]) / path) if path > 0 else None
    return out

def phase_rows(cap, tick):
    pm = PM.from_capture(cap, tick, transient_ms=500.0)
    ph = pm["phase_feet_deg"]
    out = {"phase_line_hz": pm["line_hz"], "phase_line_coh": pm["line_coh"]}
    for name, idx in PHASE_IDX.items():
        out[f"phase_{name}_deg"] = float(abs(ph[idx]))
    return out

def phase_fixed_rows(cap, tick, line_hz):
    """The fixed-heading twin of the three rows above (2026-09-07, lane A's
    CD2-FD): same capture, same 500 ms transient, same tick, and the step line
    located by the same function on the same net drive. `line_hz` is the
    body-frame rows' phase_line_hz; a twin read at a different bin is refused
    rather than reported beside rows it does not pair with."""
    fx = PFH.fixed_heading_rows(cap, tick, transient_ms=500.0)
    if line_hz is not None and round(fx["line_hz"], 2) != line_hz:
        raise ValueError(f"fixed-heading twin located the step line at {fx['line_hz']:.4f} Hz, "
                         f"the body-frame rows at {line_hz}")
    out = {k: v for k, v in fx.items() if k not in ("line_hz", "line_coh", "line_floor")}
    out["phase_line_floor"] = round(fx["line_floor"], 4)   # the readable flags' own gate
    return out

def circ_abs_mean(deg):
    z = np.exp(1j * np.radians(deg)); return float(abs(np.degrees(np.angle(z.mean()))))

def boot_interval(deg, n=2000, seed=20260903):
    rng = np.random.default_rng(seed); deg = np.asarray(deg, float)
    if len(deg) < 2: return [None, None]
    means = [circ_abs_mean(rng.choice(deg, len(deg), replace=True)) for _ in range(n)]
    return [round(float(np.percentile(means, 2.5)), 1), round(float(np.percentile(means, 97.5)), 1)]

def card_fan_out(args, seeds):
    """Run the card's seeds over `args.workers` processes, in seed order.

    Each worker is this same file with its own subset, so one code path
    produces every row, and the card aggregate below is computed by the
    parent from all twelve exactly as it is in the serial path.
    """
    import tempfile
    n = min(args.workers, len(seeds))
    groups = [seeds[i::n] for i in range(n)]
    tmp = Path(tempfile.mkdtemp(prefix="card-"))
    procs = []
    for i, group in enumerate(groups):
        out = tmp / f"w{i}.json"
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--theta", args.theta, "--seeds", args.seeds, "--out", args.out,
               "--engine", args.engine, "--workers", str(args.workers),
               "--worker-seeds", ",".join(str(s) for s in group),
               "--worker-out", str(out)]
        procs.append((subprocess.Popen(cmd, cwd=ROOT), group, out))
    by_seed, failed = {}, []
    for proc, group, out in procs:
        if proc.wait() != 0:
            failed.append(f"worker on seeds {group} exited {proc.returncode}")
            continue
        by_seed.update(dict(zip(group, json.loads(out.read_text()))))
    if failed:
        raise RuntimeError("card invalid: " + "; ".join(failed))
    return [by_seed[s] for s in seeds]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", default="results-qh1-theta.json")
    ap.add_argument("--seeds", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--engine", default="metal", choices=("metal", "numpy"))
    ap.add_argument("--workers", type=int, default=1,
                    help="seed processes at once; 1 (default) is the serial path "
                         "unchanged under the exclusive Metal lock. Above 1 the lock "
                         "is shared, permitted for title receipts from 2026-09-07.")
    ap.add_argument("--worker-seeds", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--worker-out", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.workers < 1: raise SystemExit("--workers is at least 1")
    lo, hi = (int(x) for x in args.seeds.split("-")); seeds = list(range(lo, hi + 1))
    jlo, jhi = json.loads((ROOT / "seed_blocks.json").read_text())["blocks"]["judge"]
    if not all(jlo <= s <= jhi for s in seeds): raise SystemExit("card seeds come from the judge block")
    # Card evidence is decision-grade whatever the layout; walk_search read
    # that off the Metal lock mode, which stops being a proxy once the card
    # runs several workers (they cannot each hold LOCK_EX), so state it.
    os.environ["FLY_WBE_DECISION_GRADE"] = "1"
    if args.engine == "metal":
        # Shared at every worker count: the exclusive lock stopped being a
        # correctness measure on 2026-09-07 08:20 and costs a median
        # 13.46 s of wait per decision-grade seed. See title_match_v36.
        os.environ["FLY_WBE_METAL_LOCK_MODE"] = "shared"
    my_seeds = ([int(x) for x in args.worker_seeds.split(",")]
                if args.worker_seeds else seeds)
    prov = {"execution_commit": git("rev-parse", "HEAD"), "engine_sha256": sha(ROOT / "walk_search.py"),
            "cns_sha256": sha(ROOT / "cns.py"), "battery_sha256": sha(ROOT / "regression_walk.py"),
            "footfall_sha256": sha(ROOT / "footfall1.py"), "phasemeasures_sha256": sha(ROOT / "lanes/A/phasemeasures.py"),
            "phase_fixed_heading_sha256": sha(ROOT / "lanes/judge/phase_fixed_heading.py"),
            "capture_conflict_rows_sha256": sha(ROOT / "lanes/judge/capture_conflict_rows.py"),
            "abdomen_reference_reader_sha256": sha(ROOT / "lanes/D/abd_real_lateral_output.py"),
            "knee_source_reader_sha256": sha(ROOT / "lanes/A/cd2af_knee_stance_direction.py"),
            "flyscore_metrics_sha256": sha(ROOT / "flyscore/metrics.py"), "runner_sha256": sha(Path(__file__).resolve()),
            "metal_lock_mode": os.environ.get("FLY_WBE_METAL_LOCK_MODE")}
    theta_p = ROOT / args.theta; theta_doc = json.loads(theta_p.read_text())
    th = dict(theta_doc["winner"] if "winner" in theta_doc else theta_doc)
    th["spike_reg"] = 44.0; th["backend"] = args.engine
    tick = float(th.get("tick_ms", 5.0))
    started = time.time()
    # The parent of a parallel card does no simulating itself, so it never
    # pays the graph load; the workers each do, and the aggregate below is a
    # pure function of the rows either way.
    fans_out = args.workers > 1 and not args.worker_seeds
    rows = card_fan_out(args, seeds) if fans_out else []
    if not fans_out:
        meta = pd.read_feather(ROOT / "data/banc_888_meta.feather")
        edges = pd.read_feather(ROOT / "data/banc_888_edgelist_simple_v3.feather")
        RW._G.update({"th": th, "meta": meta, "edges": edges, "cache": {}})
        held = {}; _evaluate = walk_search.evaluate
        def evaluate(*a, **k):
            held["cap"] = k.get("capture"); return _evaluate(*a, **k)
        walk_search.evaluate = evaluate
    for s in (my_seeds if not fans_out else []):
        t0 = time.time(); rec = RW._run(s)
        if "error" not in rec and held.get("cap"):
            cap = held["cap"]
            rec.update(footfall(cap, tick))
            try: rec.update(flyscore_rows(cap, tick))
            except Exception as e: rec["flyscore_error"] = repr(e)
            try: rec.update(phase_rows(cap, tick))
            except Exception as e: rec["phase_error"] = repr(e)
            try: rec.update(phase_fixed_rows(cap, tick, rec.get("phase_line_hz")))
            except Exception as e: rec["phase_fixed_error"] = repr(e)
            try: rec.update(CCR.capture_rows(cap, tick))
            except Exception as e: rec["capture_conflict_error"] = repr(e)
            try: rec["conduction_delay_mean_ms"] = CCR.built_delay_mean_ms(RW._G["cache"])
            except Exception as e: rec["conduction_delay_error"] = repr(e)
            # lane A's caller's contract: standing runs only, since a fallen body's
            # height trace is dominated by the fall and its peak is meaningless.
            if rec.get("standing_v32"):
                try: rec["heave_line_hz"] = round(float(LE.heave_line_hz(
                        [r[0] for r in cap], tick_ms=tick)), 4)
                except Exception as e: rec["heave_line_error"] = repr(e)
            if rec.get("lifts_h"):
                win = 5.5
                rec["per_leg_cadence_h_hz"] = dict(zip(TIPZ_ORDER, [round(x / win, 3) for x in rec["lifts_h"]]))
        rec["wall_s"] = round(time.time() - t0, 1); rows.append(rec)
        print(f"seed {s}: standing={rec.get('standing_v32')} freq_h={rec.get('freq_tip_h')} speed={rec.get('speed')} "
              f"pv={rec.get('fs_posture_variability_rad')} str={rec.get('straightness_5p5s')} "
              f"ph(lf,lm,lh)={rec.get('phase_lf_rf_deg')},{rec.get('phase_lm_rm_deg')},{rec.get('phase_lh_rh_deg')} "
              f"fixed={rec.get('phase_lf_rf_fixed_deg')},{rec.get('phase_lm_rm_fixed_deg')},{rec.get('phase_lh_rh_fixed_deg')} "
              f"readable={int(bool(rec.get('phase_lf_rf_readable')))}{int(bool(rec.get('phase_lm_rm_readable')))}{int(bool(rec.get('phase_lh_rh_readable')))} "
              f"yaw_range={rec.get('yaw_range_deg')} "
              f"sep={rec.get('footfall_sep_bl')} h={rec.get('body_height_mm')} err={rec.get('error')}", flush=True)
    if args.worker_seeds:
        Path(args.worker_out).write_text(json.dumps(rows, default=float))
        return 0
    st = [r for r in rows if r.get("standing_v32")]
    def col(key):
        # 2026-09-05 23:5x (judge): these four read the v3.4 hysteresis fields
        # until now, which v3.6 retired for the title. A card grading a
        # challenger on a superseded instrument is the heave failure in another
        # place, so they move to the 10 ms fields the bar itself reads. On the
        # holder the in-band counts are identical on all four rows (medians
        # 5.8182->5.5455, 51.25->53.75, 0.6766->0.6782, 0.2687->0.2577), so no
        # grade and no veto margin moves on wb4; the reads diverge on bodies
        # with one-tick taps, which is the case v3.6 exists for.
        m = {"step_freq_hz": "freq_tip_h10", "swing_ms": "swing_tip_h10_ms", "duty": "duty_tip_h10", "contra_phase": "contra_tip_h10", "speed_mm_s": "speed"}
        if key in m: v = [r.get(m[key]) for r in st]
        elif key.startswith("rom_"): v = [(r.get("rom") or {}).get(key[4:-4]) for r in st]
        elif key == "leg_excursion_bl":
            v = [((r.get("fs_leg_excursion_px") / r["fs_body_length_px"])
                  if r.get("fs_leg_excursion_px") is not None and r.get("fs_body_length_px") else None)
                 for r in st]
        else: v = [r.get(key) for r in st]
        return [float(x) for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
    card = {}
    complete_capture_fields = {
        "abdomen_lateral_excursion_deg", "conduction_delay_mean_ms"}
    for key, (blo, bhi) in BANDS.items():
        v = col(key)
        if not v or (key in complete_capture_fields and len(v) != len(st)):
            card[key] = {"median": None, "band": [blo, bhi],
                         "in_band": f"{len(v)}/{len(st)}", "grade": "NO DATA",
                         "reason": ("incomplete standing-run read"
                                    if v else "no standing-run read")}
            continue
        med = float(np.median(v)); n_in = sum(blo <= x <= bhi for x in v)
        card[key] = {"median": round(med, 4), "band": [blo, bhi], "min": round(min(v), 4), "max": round(max(v), 4),
                     "in_band": f"{n_in}/{len(v)}", "grade": "PASS" if blo <= med <= bhi else "CONFLICT"}
    for rule in RULES:
        sig = sum(1 for r in st if r.get(rule) is not None and r[rule] < 0.05)
        card[rule] = {"runs_p_lt_0.05": f"{sig}/{len(st)}", "grade": "PASS" if sig * 2 > len(st) else "CONFLICT"}
    for name, (real, rlo, rhi) in PHASE_REAL.items():
        v = col(f"phase_{name}_deg")
        if not v: card[f"phase_{name}_deg"] = {"grade": "NO DATA"}; continue
        m = circ_abs_mean(v); iv = boot_interval(v)
        gap = max(0.0, rlo - (iv[1] if iv[1] is not None else m))
        grade = "CONFLICT" if gap >= 90 else ("CONSISTENT" if (iv[1] is not None and iv[1] >= rlo) else "BETWEEN")
        card[f"phase_{name}_deg"] = {"qh1_abs_circ_mean": round(m, 1), "qh1_95pct_over_runs": iv, "per_run": [round(x, 0) for x in v],
                                     "real_mean": real, "real_95pct": [rlo, rhi], "nearest_edge_gap_deg": round(gap, 1), "grade": grade}
    # The fixed-heading twin of the three rows above (2026-09-07): medians over
    # the standing runs beside the body-frame median of the same runs, the
    # count of runs the pair is readable on (coherence at the line at or above
    # the run's own floor, and not both feet planted more than 0.85 of the
    # window), and the median over those readable runs. The twin has no animal
    # band of its own. full_conflict_count uses it only as the causal screen
    # required before accepting closure on the body-frame phase row.
    for name in PHASE_REAL:
        v = col(f"phase_{name}_fixed_deg"); vb = col(f"phase_{name}_deg")
        rd = [r for r in st if r.get(f"phase_{name}_readable")]
        vr = [float(r[f"phase_{name}_fixed_deg"]) for r in rd if r.get(f"phase_{name}_fixed_deg") is not None]
        card[f"phase_{name}_fixed_deg"] = {
            "reported_only": True,
            "median": (round(float(np.median(v)), 1) if v else None),
            "body_frame_median": (round(float(np.median(vb)), 1) if vb else None),
            "per_run": [round(x, 0) for x in v],
            "readable": f"{len(rd)}/{len(st)}",
            "median_readable": (round(float(np.median(vr)), 1) if vr else None)}
    knee_rows = [r.get("knee_stance") for r in st if r.get("knee_stance")]
    direction_detail, excursion_detail = {}, {}
    for leg in CCR.EXPECTED_KNEE_SIGN:
        vals = [float(row[leg]["interior_angle_change_deg"])
                for row in knee_rows
                if row.get(leg, {}).get("interior_angle_change_deg") is not None]
        expected = CCR.EXPECTED_KNEE_SIGN[leg]
        med = float(np.median(vals)) if vals else None
        complete = len(vals) == len(st)
        direction_detail[leg] = {
            "median_interior_change_deg": (round(med, 4) if med is not None else None),
            "expected": "CLOSE" if expected < 0 else "OPEN",
            "matches": (None if med is None or not complete
                        else bool(np.sign(med) == expected)),
            "runs_read": len(vals),
            "standing_runs": len(st),
        }
        excursion_detail[leg] = {
            "median_abs_interior_change_deg": (round(abs(med), 4) if med is not None else None),
            "band": list(CCR.KNEE_EXCURSION_BAND_DEG),
            "in_band": (None if med is None or not complete
                        else bool(CCR.KNEE_EXCURSION_BAND_DEG[0]
                                  <= abs(med)
                                  <= CCR.KNEE_EXCURSION_BAND_DEG[1])),
            "runs_read": len(vals),
            "standing_runs": len(st),
        }
    direction_values = [row["matches"] for row in direction_detail.values()]
    excursion_values = [row["in_band"] for row in excursion_detail.values()]
    card["knee_stance_direction"] = {
        "per_leg": direction_detail,
        "all_legs_match": (bool(all(direction_values))
                           if all(value is not None for value in direction_values) else None),
        "grade": ("PASS" if all(direction_values) else "CONFLICT")
        if all(value is not None for value in direction_values) else "NO DATA",
        "sign_convention": "+FTi is flexion at the occupied posture; interior-angle change has the opposite sign",
    }
    card["knee_stance_excursion_deg"] = {
        "per_leg": excursion_detail,
        "all_legs_in_band": (bool(all(excursion_values))
                             if all(value is not None for value in excursion_values) else None),
        "grade": ("PASS" if all(excursion_values) else "CONFLICT")
        if all(value is not None for value in excursion_values) else "NO DATA",
    }
    if st:
        lh = [r["per_leg_cadence_h_hz"]["lh"] for r in st if r.get("per_leg_cadence_h_hz")]
        rh = [r["per_leg_cadence_h_hz"]["rh"] for r in st if r.get("per_leg_cadence_h_hz")]
        card["hind_lr_cadence_h"] = {"lh_median_hz": round(float(np.median(lh)), 3), "rh_median_hz": round(float(np.median(rh)), 3),
                                     "rh_faster_on": f"{sum(1 for a, b in zip(lh, rh) if b > a)}/{len(lh)}",
                                     "lh_in_band": f"{sum(1 for x in lh if 5 <= x <= 16)}/{len(lh)}", "rh_in_band": f"{sum(1 for x in rh if 5 <= x <= 16)}/{len(rh)}"}
        minlift = min(min(r["lifts_h"]) for r in st)
        card["lifts_h_min_all_feet"] = {"min": minlift, "bar": 5, "grade": "PASS" if minlift >= 5 else "CONFLICT"}
    out = {"instrument": "regression_walk._run through the title path (spike_reg 44) + footfall1 + flyscore (window 500-5500 ms, 80 fps) + lanes/A/phasemeasures at the step line + lanes/judge/phase_fixed_heading (fixed-heading twin and stance/readability context) + capture_conflict_rows (abdomen same-window and corrected knee-stance reads) + the built network's delivered delay vector, all on one capture per seed",
           "theta": args.theta, "theta_sha256": sha(theta_p), "spike_reg": 44.0, "backend": args.engine, "seeds": seeds,
           "standing_v32": f"{len(st)}/{len(rows)}", "card": card, "provenance": prov, "runs": rows,
           "wall_seconds": round(time.time() - started, 1)}
    Path(args.out).write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"\nstanding {len(st)}/{len(rows)}")
    for k, v in card.items(): print(f"  {k:<26} {json.dumps(v, default=float)[:160]}")
    print(f"receipt -> {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
