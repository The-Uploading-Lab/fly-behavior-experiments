"""Cruse dissection 1, pre-registered: does the simulated VNC already
express the stick-insect inter-leg coordination rules?

WHY (Robin, 2026-08-14): the single-leg literature (Bassler/Buschges stick
insect; Cruse's rules; Walknet) says insect gaits emerge from a few LOCAL
neighbor-to-neighbor influences -- no central gait controller.  Our model
has the ingredients (per-leg load sensing, position sensors, load-closed
stiffness) but implements none of the rules explicitly.  If the connectome
contains the fly's version of them, their statistical signatures should be
measurable in captures we already collect.  Each rule found present is a
match; each absent is a named gap.  For R1 the fly itself is measured
(Mendes 2013, Wosnitza 2013: adjacent ipsilateral AND contralateral legs
step in anti-phase at tripod), so an R1 fail is a real-fly CONFLICT, not
just a stick-insect mismatch.  R2/R5 are conserved-mechanism predictions:
absent = gap, not conflict (not directly measured in walking flies).

ARMS (VNC-core, sanctioned by the tier-3 equivalence pass; 6000 ms;
capture on; same 8 fresh seeds per arm, first-unused >= 340 harvested
against ALL results-*.json seed fields):
  champion   CHAMPION.json theta      (MDN   @ 98.4 Hz, first-order leg)
  clock      results-round10 winner   (DNp09 @ 71.6 Hz, tuned leg f0 18)

MEASUREMENT (standing runs only: up > 0.90, z > 1.0, raw >= 0.3; first
100 ticks = 500 ms transient dropped; contact = leg load > 1.0, the
walk_search11 threshold; leg order lf lh lm rf rh rm):

  R1i swing-overlap avoidance, ipsilateral adjacent pairs (lf-lm, lm-lh,
      rf-rm, rm-rh): mean P(both legs in swing) vs a 200-draw circular-
      shift null (independent shift >= 50 ticks per pair per draw).
      One-sided: coordination -> observed BELOW null.
  R1c same, contralateral pairs (lf-rf, lm-rm, lh-rh).
  R2  touchdown of the posterior leg promotes liftoff of its anterior
      ipsilateral neighbor (Cruse rule 2): mean latency from each
      posterior touchdown to the next anterior liftoff, pooled over the
      4 pairs, vs circular-shift null.  One-sided: coupling -> SHORTER.
  R5  load prolongs stance: per leg with >= 5 stance episodes, Spearman
      rho of (mean episode load, episode duration); statistic = median
      rho across legs, vs 200 within-leg permutations of durations.
      One-sided: rule -> rho ABOVE null.

PRE-REGISTERED CLAIMS, fixed before running (per arm, standing runs
only, needing >= 3 standing runs else 'insufficient'):
  a rule is PRESENT if its one-sided p < 0.05 in a strict majority of
  standing runs; ABSENT if p >= 0.05 in a strict majority; else mixed.
LADDER: R1i or R1c ABSENT -> new open conflict (fly-measured anti-phase
violated).  R2/R5 PRESENT -> the connectome expresses a Cruse rule
nobody wired in.  R2/R5 ABSENT -> named gap for the mechanism ledger.
"""
import glob as _glob
import json
import multiprocessing as mp
import os
import subprocess

import numpy as np

DUR = 6000.0
OUT = "results-cruse1.json"
CONTACT_N = 1.0
NSHUF = 200
TRANSIENT = 100

LEGS = ["lf", "lh", "lm", "rf", "rh", "rm"]
I = {l: i for i, l in enumerate(LEGS)}
PAIRS_IPSI = [(I["lf"], I["lm"]), (I["lm"], I["lh"]),
              (I["rf"], I["rm"]), (I["rm"], I["rh"])]
PAIRS_CONTRA = [(I["lf"], I["rf"]), (I["lm"], I["rm"]), (I["lh"], I["rh"])]
# R2: (posterior, anterior) -- posterior touchdown -> anterior liftoff
PAIRS_R2 = [(I["lm"], I["lf"]), (I["lh"], I["lm"]),
            (I["rm"], I["rf"]), (I["rh"], I["rm"])]

ARMS = {"champion": json.load(open("CHAMPION.json"))["theta"],
        "clock": json.load(open("results-round10.json"))["winner"]}


def used_seeds():
    u = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("seeds", "search_seeds") and isinstance(v, list):
                    # search_seeds added 2026-08-23 (Lane A's 34th defect:
                    # every search script writes this key and the guard
                    # dropped it, so burnt seeds were drawable as fresh)
                    u.update(int(x) for x in v if isinstance(x, (int, float)))
                elif k == "seed" and isinstance(v, (int, float)):
                    u.add(int(v))
                else:
                    walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)

    # lanes/ glob added 2026-08-21 (Lane A's Result 0: both guard
    # families were blind to all lane work). Lane results also mirror
    # into root results-walk-search-lane<X>-seeds.json so the frozen
    # per-chamber copies of this function see them too.
    # Prose consumption sources (re-freeze event 2026-08-31, mirrors
    # seedguard): .log files and result-named .md files record execution;
    # packets and registrations narrate seeds in any modality and are
    # excluded. Overcounting is safe for a consumption guard. Standing
    # practice keeps JSON mirrors primary; this is defense in depth.
    import re as _re
    _pat_list = _re.compile(r"seeds?\s*[\[(]([0-9,\s.\u2026]+)")
    _pat_range = _re.compile(r"seeds?\s+(\d{4,6})\s*[-\u2013]\s*(\d{4,6})")
    _pat_reuse = _re.compile(r"--reuse[= ]([0-9,\s]+)")
    _prose = (_glob.glob("lanes/*/*.log")
              + [x for x in _glob.glob("lanes/*/*.md") if "result" in x.lower()]
              + [x for x in _glob.glob("exchange/*.md") if "result" in x.lower()]
              + _glob.glob("results-*.md"))
    for f in _prose:
        try:
            txt = open(f, errors="replace").read()
        except Exception:
            continue
        for m in _pat_list.finditer(txt):
            for tok in _re.findall(r"\d{4,6}", m.group(1)):
                u.add(int(tok))
        for m in _pat_range.finditer(txt):
            a, b = int(m.group(1)), int(m.group(2))
            if 0 < b - a <= 2000:
                u.update(range(a, b + 1))
        for m in _pat_reuse.finditer(txt):
            for tok in _re.findall(r"\d{4,6}", m.group(1)):
                u.add(int(tok))
    for f in _glob.glob("results-*.json") + _glob.glob("lanes/*/*.json"):
        try:
            walk(json.load(open(f)))
        except Exception:
            pass
    return u


def fresh_seeds(n=8, start=340):
    u = used_seeds()
    out, k = [], start
    while len(out) < n:
        if k not in u:
            out.append(k)
        k += 1
    return out


# SEEDS is computed on first use, not at import. fresh_seeds() walks every
# results-*.json, lane result and log for consumed seeds (3.1 s measured
# 2026-09-05), and 557 files import this module for its constants and
# statistics, so the whole fleet paid that scan in every process that never
# drew a seed. `cruse1.SEEDS` and `from cruse1 import SEEDS` still work,
# through the module __getattr__ below, and return the same list, computed
# when first asked for rather than when imported. (compute seat)
_SEEDS = None


def _seeds():
    global _SEEDS
    if _SEEDS is None:
        _SEEDS = fresh_seeds()
    return _SEEDS


def __getattr__(name):
    if name == "SEEDS":
        return _seeds()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_G = {}


def _init():
    os.environ["OMP_NUM_THREADS"] = "2"
    import pandas as pd
    m = pd.read_feather("data/banc_888_meta.feather")
    keep = m[(m["region"] == "ventral_nerve_cord")
             | (m["super_class"] == "descending")].reset_index(drop=True)
    ids = set(keep["banc_888_id"])
    e = pd.read_feather("data/banc_888_edgelist_simple_v3.feather")
    _G["meta"] = keep
    _G["edges"] = e[(e["count"] >= 10) & e["pre"].isin(ids)
                    & e["post"].isin(ids)].reset_index(drop=True)
    _G["cache"] = {}


def episodes(on):
    d = np.diff(on.astype(int))
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1
    if on[0]:
        starts = np.r_[0, starts]
    if on[-1]:
        ends = np.r_[ends, len(on)]
    return list(zip(starts, ends))


def r1_stat(swing, pairs, rng):
    T = swing.shape[0]
    obs = float(np.mean([np.mean(swing[:, a] & swing[:, b])
                         for a, b in pairs]))
    null = np.empty(NSHUF)
    for i in range(NSHUF):
        vals = [np.mean(swing[:, a]
                        & np.roll(swing[:, b], int(rng.integers(50, T - 50))))
                for a, b in pairs]
        null[i] = np.mean(vals)
    p = (1 + int(np.sum(null <= obs))) / (1 + NSHUF)
    return obs, float(np.mean(null)), p


def _mean_latency(contact, lifts_by_leg):
    lats = []
    for post, ant in PAIRS_R2:
        tds = np.flatnonzero(contact[1:, post] & ~contact[:-1, post]) + 1
        lifts = lifts_by_leg[ant]
        pair = []
        for t in tds:
            nxt = lifts[lifts > t]
            if len(nxt):
                pair.append(nxt[0] - t)
        if pair:
            lats.append(np.mean(pair))
    return float(np.mean(lats)) if lats else np.nan


def r2_stat(contact, rng):
    T = contact.shape[0]
    lifts = {j: np.flatnonzero(~contact[1:, j] & contact[:-1, j]) + 1
             for j in range(6)}
    obs = _mean_latency(contact, lifts)
    if np.isnan(obs):
        return np.nan, np.nan, np.nan
    null = np.empty(NSHUF)
    for i in range(NSHUF):
        sh = {}
        for j in range(6):
            k = int(rng.integers(50, T - 50))
            tr = np.zeros(T, dtype=bool)
            tr[lifts[j]] = True
            sh[j] = np.flatnonzero(np.roll(tr, k))
        null[i] = _mean_latency(contact, sh)
    null = null[~np.isnan(null)]
    if len(null) < 50:
        return obs, np.nan, np.nan
    p = (1 + int(np.sum(null <= obs))) / (1 + len(null))
    return obs, float(np.mean(null)), p


def r5_stat(loads, contact, rng):
    from scipy import stats
    per_leg = []
    for j in range(6):
        eps = [(s, e) for s, e in episodes(contact[:, j]) if e - s >= 2]
        if len(eps) < 5:
            continue
        x = np.array([np.mean(loads[s:e, j]) for s, e in eps])
        y = np.array([e - s for s, e in eps], dtype=float)
        per_leg.append((x, y))
    if len(per_leg) < 3:
        return np.nan, np.nan, np.nan

    def med_rho(pairs_xy):
        rhos = []
        for x, y in pairs_xy:
            r = stats.spearmanr(x, y)[0]
            if not np.isnan(r):
                rhos.append(r)
        return float(np.median(rhos)) if rhos else np.nan

    obs = med_rho(per_leg)
    null = np.empty(NSHUF)
    for i in range(NSHUF):
        null[i] = med_rho([(x, rng.permutation(y)) for x, y in per_leg])
    null = null[~np.isnan(null)]
    p = (1 + int(np.sum(null >= obs))) / (1 + len(null))
    return obs, float(np.mean(null)), p


def _run(job):
    arm, seed = job
    from walk_search import evaluate
    th = dict(ARMS[arm])
    cap = []
    try:
        _, raw, up, z = evaluate(_G["cache"], _G["meta"], _G["edges"], th,
                                 seed=seed, dur_ms=DUR, capture=cap)
    except Exception as e:
        return {"arm": arm, "seed": seed, "error": type(e).__name__}
    loads = np.array([c[4] for c in cap])[TRANSIENT:]
    contact = loads > CONTACT_N
    swing = ~contact
    standing = bool(up > 0.90 and z > 1.0 and raw >= 0.3)
    rec = {"arm": arm, "seed": seed, "raw": round(raw, 2),
           "up": round(up, 2), "standing": standing}
    if standing:
        rng = np.random.default_rng(seed + 777000)
        for name, (o, nl, p) in (
                ("R1i", r1_stat(swing, PAIRS_IPSI, rng)),
                ("R1c", r1_stat(swing, PAIRS_CONTRA, rng)),
                ("R2", r2_stat(contact, rng)),
                ("R5", r5_stat(loads, contact, rng))):
            rec[name] = {"obs": None if np.isnan(o) else round(float(o), 4),
                         "null": None if np.isnan(nl) else round(nl, 4),
                         "p": None if (isinstance(p, float) and np.isnan(p))
                         else round(float(p), 4)}
        rec["n_steps"] = int(np.median(
            [len(episodes(contact[:, j])) for j in range(6)]))
    return rec


def battery_ok():
    """Chamber power gate. Robin, 2026-08-18: "don't worry about
    battery it'll be enough, remove the limit" -- so the gate is now
    OFF by default. Set FLYWBE_BATTERY_GATE=1 to restore the old
    behaviour (AC power, or charge above 50%)."""
    if os.environ.get("FLYWBE_BATTERY_GATE", "0") != "1":
        return True
    out = subprocess.run(["pmset", "-g", "batt"], capture_output=True,
                         text=True).stdout
    return "AC Power" in out or (
        "%" in out and int(out.split("\t")[1].split("%")[0]) > 50)


def verdict(runs, rule):
    st = [r for r in runs if r.get("standing") and rule in r
          and r[rule]["p"] is not None]
    if len(st) < 3:
        return f"insufficient ({len(st)} usable)"
    sig = sum(1 for r in st if r[rule]["p"] < 0.05)
    if sig * 2 > len(st):
        return f"PRESENT ({sig}/{len(st)} runs p<0.05)"
    if (len(st) - sig) * 2 > len(st):
        return f"ABSENT ({sig}/{len(st)} runs p<0.05)"
    return f"mixed ({sig}/{len(st)})"


def main():
    assert battery_ok(), "battery gate"
    mp.set_start_method("spawn")
    seeds = _seeds()
    print(f"seeds {seeds}", flush=True)
    jobs = [(a, s) for s in seeds for a in ARMS]
    res = []
    with mp.Pool(6, initializer=_init) as pool:
        for r in pool.imap_unordered(_run, jobs):
            res.append(r)
            if "error" in r:
                line = f"ERROR {r['error']}"
            elif not r["standing"]:
                line = f"fell (raw {r['raw']}, up {r['up']})"
            else:
                line = ("  ".join(
                    f"{k} {r[k]['obs']} vs {r[k]['null']} p={r[k]['p']}"
                    for k in ("R1i", "R1c", "R2", "R5"))
                    + f"  steps~{r['n_steps']}")
            print(f"{r['arm']:<9} seed {r['seed']}: {line}", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    for arm in ARMS:
        runs = [r for r in res if r.get("arm") == arm]
        n_st = sum(1 for r in runs if r.get("standing"))
        print(f"\n{arm}: {n_st}/{len(runs)} standing", flush=True)
        for rule in ("R1i", "R1c", "R2", "R5"):
            print(f"  {rule}: {verdict(runs, rule)}", flush=True)


if __name__ == "__main__":
    main()
