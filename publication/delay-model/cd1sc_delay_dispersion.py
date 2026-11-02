"""CD1-SC runner (from the CD1-SB runner): which part of qh1's length-derived
delay dispersion carries the cord's 5.8 Hz line. Six arms: base, flat 6.24,
flat 15, short delays raised to 6.24, long delays lowered to 6.24, the base
vector permuted across neurons. Registration:
lanes/A/2026-09-04-0407-cd1sc-delay-dispersion-preregister.md.

    PYTHONPATH=. FLY_WBE_METAL_LOCK_MODE=shared .venv/bin/python3 lanes/A/cd1sc_delay_dispersion.py \
        --theta results-qh1-theta.json --seeds 201303-201308 --procs 3 \
        --out lanes/A/2026-09-04-cd1sc-qh1-whole.json
"""


import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import regression_walk as RW

LEGS = ["T1", "T2", "T3"]
_CL = {"joint_int": "exact"}
ARMS = {"base": dict(_CL)}
ARMS["flat6"] = dict(_CL, delay_mode="by_type", delay_type_ms={"NO_SUCH_TYPE": 6.24})
ARMS["flat15"] = dict(_CL, delay_mode="by_type", delay_type_ms={"NO_SUCH_TYPE": 15.0},
                      delay_mean_ms=15.0)
# The three vector arms keep delay_mode length; delay_cap_ms is a TAG that
# separates the network cache keys and is never used as a cap (the patched
# length_delay_vector below always computes the base vector at 20.0).
_TAGS = {20.01: "noshort", 20.02: "nolong", 20.03: "perm"}
ARMS["noshort"] = dict(_CL, delay_cap_ms=20.01)
ARMS["nolong"] = dict(_CL, delay_cap_ms=20.02)
ARMS["perm"] = dict(_CL, delay_cap_ms=20.03)
TRACE_DIR = os.environ.get("CD1SC_TRACE_DIR", "lanes/A/traces_sc")
_BASE = {}


def _init(theta_path, engine):
    import walk_search as W
    d = json.load(open(theta_path))
    th = dict(d["winner"] if "winner" in d else d)
    th["spike_reg"] = 44.0
    th["backend"] = engine
    meta = pd.read_feather("data/banc_888_meta.feather")
    edges = pd.read_feather("data/banc_888_edgelist_simple_v3.feather")
    RW._G.update({"meta": meta, "edges": edges, "cache": {}})
    _BASE["th"] = th
    leg = meta["neuromere"].isin(LEGS).to_numpy()
    vncin = (meta["super_class"] == "ventral_nerve_cord_intrinsic").to_numpy()
    dng = (meta["cell_type"] == "DNg100").to_numpy()
    _BASE["dng_rows"] = np.flatnonzero(dng)
    _BASE["dng_ids"] = meta["banc_888_id"].astype(str).to_numpy()[_BASE["dng_rows"]]
    _BASE["rows"] = np.sort(np.flatnonzero((leg & vncin) | dng))
    _BASE["caches"] = {}
    _orig_ldv = W.length_delay_vector

    def _ldv(meta, mean_ms, cap_ms, groups):
        v = _orig_ldv(meta, mean_ms, 20.0, groups)
        tag = _TAGS.get(round(float(cap_ms), 2))
        if tag == "noshort":
            v = np.maximum(v, 6.24)
        elif tag == "nolong":
            v = np.minimum(v, 6.24)
        elif tag == "perm":
            v = v[np.random.default_rng(201303).permutation(len(v))]
        elif float(cap_ms) != 20.0:
            raise ValueError(f"untagged cap {cap_ms}")
        _BASE.setdefault("delay_vec_stats", {})[tag or "base"] = {
            "mean": round(float(v.mean()), 4), "min": round(float(v.min()), 3),
            "max": round(float(v.max()), 3), "n_over_6.24": int((v > 6.24).sum())}
        return v
    W.length_delay_vector = _ldv
    _orig = W.evaluate

    def _wrapped(cache, meta, edges, th, **kw):
        tr = []
        out = _orig(cache, meta, edges, th, record_rows=_BASE["rows"],
                    trace_out=tr, **kw)
        _BASE["last_trace"] = tr[0] if tr else None
        _BASE["cap"] = kw.get("capture") or []
        return out
    W.evaluate = _wrapped
    import motormap as MM
    _orig_bm = MM.build_map

    def _bm_wrapped(meta):
        rel = _BASE.get("relabel") or {}
        if rel:
            meta = meta.copy()
            ids = meta["banc_888_id"].astype(str)
            n = 0
            for i, muscle in rel.items():
                hit = ids == i
                n += int(hit.sum())
                meta.loc[hit, "peripheral_target_type"] = muscle
            _BASE["relabel_applied"] = n
        return _orig_bm(meta)
    MM.build_map = _bm_wrapped


def _job(arg):
    arm, seed = arg
    th = dict(_BASE["th"])
    th.update(ARMS[arm])
    if "cmd2_hz" in th and th["cmd2_hz"] is None:
        th["cmd2_hz"] = float(th["cmd_hz"])
    RW._G["th"] = th
    _BASE["relabel"] = th.get("mn_relabel")
    _BASE["relabel_applied"] = 0
    # size_factors is memoised on the frames' content, and the relabel acts
    # inside build_map, so the memo is cleared per job to keep arms apart
    # (both relabelled cells are singleton pools either way, so the factors
    # are the same; the clear is a guard, not a fix).
    import walk_search as W
    W._SIZE_FACTORS_CACHE.clear()
    # One network per worker: the seam's cache key carries the gain hash, so
    # the cache is swapped per arm to hold RAM rather than to force a rebuild.
    if arm not in _BASE["caches"]:
        _BASE["caches"] = {arm: {}}
    RW._G["cache"] = _BASE["caches"][arm]
    _BASE["last_trace"] = None
    _BASE["cap"] = []
    t0 = time.time()
    rec = RW._run(seed)
    if "error" in rec:
        raise RuntimeError(f"{arm} seed {seed}: {rec['error']}: {rec.get('error_message')}")
    cap = _BASE.get("cap") or []
    if cap:
        np.savez_compressed(
            f"{TRACE_DIR}/{arm}-{seed}.npz",
            body=np.array([c[0] for c in cap]), quat=np.array([c[1] for c in cap]),
            jtheta=np.array([c[3] for c in cap]), loads=np.array([c[4] for c in cap]),
            tipz=np.array([c[5] for c in cap]), a_flex=np.array([c[6] for c in cap]),
            a_ext=np.array([c[7] for c in cap]), tips=np.array([c[10] for c in cap]))
    tr = _BASE.get("last_trace")
    if tr is not None and tr[1] is not None:
        rows, trace = tr
        net = list(RW._G["cache"].values())[-1]
        np.savez_compressed(f"{TRACE_DIR}/trace-{arm}-{seed}.npz",
                            rows=np.asarray(rows), trace=np.asarray(trace, dtype=np.int16),
                            sign=np.asarray(net.sign)[np.asarray(rows)],
                            ids=np.asarray(net.ids)[np.asarray(rows)])
        rec["trace_rows"] = int(len(rows))
        pos = np.searchsorted(np.asarray(rows), _BASE["dng_rows"])
        trace = np.asarray(trace)
        rec["dng100_spikes"] = {str(i): int(trace[:, p].sum())
                                for i, p in zip(_BASE["dng_ids"], pos)}
    lh_ = rec.get("lifts_h")
    fh_ = rec.get("freq_tip_h")
    rec["qualifies_v35"] = bool(
        rec.get("standing_v32") and lh_ and min(lh_) >= 5 and fh_ is not None
        and 5.0 <= fh_ <= 16.0 and rec.get("endogenous_timing"))
    rec["arm"] = arm
    rec["delay_vec"] = _BASE.get("delay_vec_stats", {}).get(
        _TAGS.get(round(float(th.get("delay_cap_ms", 20.0)), 2), "base")
        if th.get("delay_mode") == "length" else None)
    rec["n_intersegmental_edges"] = [
        int(getattr(n, "n_intersegmental_edges", -1))
        for n in RW._G["cache"].values()]
    rec["wall_seconds"] = round(time.time() - t0, 1)
    return rec


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", default="results-qh1-theta.json")
    ap.add_argument("--seeds", default="201258-201260")
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", default="metal", choices=("metal", "numpy"))
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()
    os.makedirs(TRACE_DIR, exist_ok=True)
    a, b = (int(x) for x in args.seeds.split("-"))
    seeds = list(range(a, b + 1))
    arms = args.arms.split(",")
    # Arm-major so each worker keeps one network for a stretch of seeds.
    jobs = [(arm, s) for arm in arms for s in seeds]
    t0 = time.time()
    mp.set_start_method("spawn")
    rows = []
    with mp.Pool(args.procs, initializer=_init, initargs=(args.theta, args.engine)) as pool:
        for rec in pool.imap_unordered(_job, jobs):
            rows.append(rec)
            print(json.dumps({k: v for k, v in rec.items() if k not in ("stroke",)}), flush=True)
    rows.sort(key=lambda r: (r["seed"], arms.index(r["arm"])))
    import cns_metal
    out = {
        "experiment": "cd1sc-delay-dispersion-qh1",
        "registration": "lanes/A/2026-09-04-0407-cd1sc-delay-dispersion-preregister.md",
        "arms": ARMS,
        "theta": args.theta, "theta_sha256_16": sha16(args.theta),
        "backend": args.engine,
        "metal_lock_mode": os.environ.get("FLY_WBE_METAL_LOCK_MODE", "shared"),
        "procs": args.procs,
        "dylib_sha256_16": (sha16("gpu/libcnsengine.dylib")
                            if args.engine == "metal" else None),
        "metal_source_pair_sha256_16": (cns_metal._source_sha256()[:16]
                                        if args.engine == "metal" else None),
        "substrate": "whole BANC graph, unfiltered edges (title build)",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        "engine_sha256_16": {p: sha16(p) for p in (
            "walk_search.py", "cns.py", "body.py", "regression_walk.py")},
        "seeds": seeds, "dur_ms": RW.DUR,
        "wall_seconds": round(time.time() - t0, 1), "rows": rows}
    json.dump(out, open(args.out, "w"), indent=1)
    print("WROTE", args.out, "wall", out["wall_seconds"])


if __name__ == "__main__":
    sys.exit(main())
