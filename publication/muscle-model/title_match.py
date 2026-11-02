"""THE TITLE MATCH, pre-registered before any of its seeds are run.

Motivation (2026-08-11 audit): the round-2 champion's "3/4 standing" -- the
reference every adoption bar was calibrated against -- does not survive the
strict gate its challengers are judged by. Its verified record on seeds 0-3
is 2/4 (seed 3 ended upright 0.86 < 0.90), and seed 0 was the round-2
search's own evaluation seed, so its clean never-selected record is 1/3.
Round 5's challenger (cand 5: load-closed stiffness, k_gain 62, MDN command)
was REJECTED under the pre-registered bar -- that verdict STANDS, bars do not
move after results -- but on the identical seeds 0-3 it matches the champion
exactly (stands 0 and 2, falls 1 and 3) and carries 9/12 standing across all
never-selected seeds. The title is therefore decided by direct measurement,
not by the miscounted reference.

DESIGN: both flies run the SAME 30 reserved seeds 46-75 (never used by any
search, verification, or certification; 16-45 stay banked for certification).
Paired seeds cancel seed-difficulty noise in the comparison.

DECISION RULE (fixed here, before running):
  1. Strictly more standing seeds (gate: upright > 0.90 AND z_end > 1.0)
     takes the title.
  2. Tie: higher median gated displacement (fallen = 0) across all 30.
  3. If that also ties, the incumbent retains.
Also reported, for information not decision: each fly's one-sided 95% lower
bound on p(stand), and the sign test over discordant seeds.
"""
import json
import multiprocessing as mp
import os

import numpy as np

SEEDS = list(range(46, 76))
DUR = 6000.0
OUT = "results-title-match.json"

CHAMP = {"camp_gain": 281.29702940426563, "F_ref": 35.27037237257666,
         "drive_scale": 0.004150791502200652,
         "tau_joint": 10.614568046789337, "adapt_b": 1.1112776014348629,
         "cmd_hz": 45.167836591096595, "claw": -1, "hook": 1,
         "cmd": "DNg100"}
CAND5 = {"camp_gain": 25.28130242399116, "F_ref": 29.182760614694125,
         "drive_scale": 0.0076557652751406205,
         "tau_joint": 15.83088498017543, "adapt_b": 0.8594976627844031,
         "cmd_hz": 98.36508077732753, "claw": -1, "hook": 1, "cmd": "MDN",
         "k_min": 38.08440817634772, "k_gain": 62.0252839575318,
         "F_ref_k": 64.56562397006705, "k_mode": "load"}

_G = {}


def _init():
    os.environ["OMP_NUM_THREADS"] = "2"
    import pandas as pd
    _G["meta"] = pd.read_feather("data/banc_888_meta.feather")
    _G["edges"] = pd.read_feather("data/banc_888_edgelist_simple_v3.feather")
    _G["cache"] = {}


def _run(job):
    who, th, seed = job
    from walk_search import evaluate
    try:
        score, raw, up, z = evaluate(_G["cache"], _G["meta"], _G["edges"],
                                     th, seed=seed, dur_ms=DUR)
    except Exception as e:
        return {"who": who, "seed": seed, "score": 0.0, "raw": 0.0,
                "upright": 0.0, "z_end": 0.0, "error": type(e).__name__}
    return {"who": who, "seed": seed, "score": score, "raw": raw,
            "upright": up, "z_end": z}


def main():
    mp.set_start_method("spawn")
    jobs = []
    for s in SEEDS:                      # interleaved so pairs finish together
        jobs.append(("champ", CHAMP, s))
        jobs.append(("cand5", CAND5, s))
    res = []
    with mp.Pool(6, initializer=_init) as pool:
        for r in pool.imap_unordered(_run, jobs):
            res.append(r)
            print(f"{r['who']:>5} seed {r['seed']:>2}: raw {r['raw']:5.2f}  "
                  f"up {r['upright']:.2f}  z {r['z_end']:.2f}", flush=True)
    json.dump({"champ": CHAMP, "cand5": CAND5, "results": res},
              open(OUT, "w"), indent=1)

    from scipy import stats
    def stands(who):
        return {r["seed"] for r in res if r["who"] == who
                and r["upright"] > 0.90 and r["z_end"] > 1.0}
    def gated(who):
        return [r["raw"] if (r["upright"] > 0.90 and r["z_end"] > 1.0)
                else 0.0 for r in sorted((x for x in res if x["who"] == who),
                                         key=lambda x: x["seed"])]
    sc, s5 = stands("champ"), stands("cand5")
    n = len(SEEDS)
    for name, s in (("champion", sc), ("cand5", s5)):
        lo = stats.beta.ppf(0.05, len(s), n - len(s) + 1) if s else 0.0
        print(f"\n{name}: {len(s)}/{n} standing; one-sided 95% lower bound "
              f"p > {lo:.3f}; median gated disp "
              f"{np.median(gated('champ' if name == 'champion' else 'cand5')):.2f}",
              flush=True)
    only_c, only_5 = len(sc - s5), len(s5 - sc)
    if only_c + only_5:
        p = stats.binomtest(only_5, only_c + only_5, 0.5).pvalue
        print(f"discordant seeds: cand5-only {only_5}, champion-only {only_c}"
              f"; sign test p = {p:.4f}", flush=True)
    if len(s5) > len(sc):
        print("TITLE: CAND5 TAKES THE TITLE (rule 1)", flush=True)
    elif len(sc) > len(s5):
        print("TITLE: CHAMPION RETAINS (rule 1)", flush=True)
    else:
        m5, mc = np.median(gated("cand5")), np.median(gated("champ"))
        if m5 != mc:
            w = "CAND5 TAKES THE TITLE" if m5 > mc else "CHAMPION RETAINS"
            print(f"TITLE: {w} (rule 2: medians {m5:.2f} vs {mc:.2f})",
                  flush=True)
        else:
            print("TITLE: CHAMPION RETAINS (rule 3, incumbent)", flush=True)


if __name__ == "__main__":
    main()
