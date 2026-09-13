"""Re-grade the champion's card on the champion.

Robin's judge goal, 2026-09-03: the front-leg ROM, duty and Cruse closures on
the registry were measured on the C-V41 taped body and never on nc1. This
runner grades one theta on fresh judge seeds with the battery's own measures
(`regression_walk._run`, unchanged) through the title instrument's exact path
(full meta and edge list, spike_reg 44), and adds the two posture rows the
battery record does not carry, footfall separation and body height, computed
as `footfall1.py` computes them from the same capture.

Every row is graded per seed and at the median, so a conflict names both the
number and how many seeds sit in the band.

Reproduce:
    PYTHONPATH=. .venv/bin/python lanes/judge/card_regrade_nc1.py \
        --theta results-nc1-theta.json --seeds 30101-30106 \
        --out lanes/judge/2026-09-03-card-regrade-nc1-a.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import regression_walk as RW  # noqa: E402
import walk_search  # noqa: E402
from footfall1 import BL_MM, LEGS, PAIR, quat_to_R  # noqa: E402

# Bands as the registry and the battery carry them.
BANDS = {
    "step_freq_hz": (5.0, 16.0),       # row 0, v3.3 clause
    "swing_ms": (15.0, 60.0),          # row 1, Wosnitza 2013
    "duty": (0.50, 0.83),              # row 2
    "contra_phase": (0.35, 0.65),      # row 3, Mendes 2013
    "speed_mm_s": (2.0, 45.0),         # row 4
    "nonfoot_force": (0.0, 0.05),      # row 10
    "footfall_sep_bl": (1.02, 1.11),   # row 11, Mendes 2013 + Wosnitza 2013
    "body_height_mm": (0.80, 0.89),    # row 12, Chun 2021
}
for (seg, jnt), (mu, sd) in RW.HAUSTEIN.items():
    BANDS[f"rom_{seg}{jnt.split('_')[0]}_deg"] = (mu - 2 * sd, mu + 2 * sd)
RULES = ("R1i_tip", "R1c_tip", "R5", "R1i", "R1c")   # rows 7, 8, 9 (+legacy)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def footfall(cap, tick):
    """footfall1.py's measures on the battery's capture, post-transient."""
    trans = int(round(500.0 / tick))
    xyz = np.array([c[0] for c in cap])[trans:]
    quat = np.array([c[1] for c in cap])[trans:]
    tips = np.array([c[10] for c in cap])[trans:]
    tipz = np.array([c[5] for c in cap])[trans:]
    clear = tipz - np.percentile(tipz, 5, axis=0, keepdims=True)
    stance = clear <= 0.15
    body = np.zeros((len(xyz), 6, 3))
    for t in range(len(xyz)):
        body[t] = (tips[t] - xyz[t]) @ quat_to_R(quat[t])
    fore = body[:, :, 0] / BL_MM
    lat = np.abs(body[:, :, 1]) / BL_MM
    out = {}
    for pair, legs in PAIR.items():
        idx = [LEGS.index(l) for l in legs]
        st = stance[:, idx]
        vals, lats = fore[:, idx][st], lat[:, idx][st]
        out[f"mid_{pair}_bl"] = float(np.median(vals)) if len(vals) else None
        out[f"lat_{pair}_bl"] = float(np.median(lats)) if len(lats) else None
    if out["mid_front_bl"] is not None and out["mid_hind_bl"] is not None:
        out["footfall_sep_bl"] = out["mid_front_bl"] - out["mid_hind_bl"]
    else:
        out["footfall_sep_bl"] = None
    out["body_height_mm"] = float(np.median(xyz[:, 2]))
    out["body_pitch_deg"] = float(np.median([
        np.degrees(np.arcsin(np.clip(-quat_to_R(q)[2, 0], -1, 1)))
        for q in quat]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", default="results-nc1-theta.json")
    ap.add_argument("--seeds", required=True, help="e.g. 30101-30106")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    blocks = json.loads((ROOT / "seed_blocks.json").read_text())["blocks"]
    jlo, jhi = blocks["judge"]
    if not all(jlo <= s <= jhi for s in seeds):
        raise SystemExit("card seeds come from the judge block")

    # Hash the engine files NOW, before the run: the working tree can change
    # under a two-hour process (it did on 2026-09-03, receipts a and b).
    prov = {
        "execution_commit": git("rev-parse", "HEAD"),
        "engine_sha256": sha(ROOT / "walk_search.py"),
        "cns_sha256": sha(ROOT / "cns.py"),
        "battery_sha256": sha(ROOT / "regression_walk.py"),
        "footfall_sha256": sha(ROOT / "footfall1.py"),
    }
    theta_p = ROOT / args.theta
    theta_doc = json.loads(theta_p.read_text())
    th = dict(theta_doc["winner"] if "winner" in theta_doc else theta_doc)
    th["spike_reg"] = 44.0
    tick = float(th.get("tick_ms", 5.0))

    meta = pd.read_feather(ROOT / "data/banc_888_meta.feather")
    edges = pd.read_feather(ROOT / "data/banc_888_edgelist_simple_v3.feather")
    RW._G.update({"th": th, "meta": meta, "edges": edges, "cache": {}})

    # RW._run discards its capture; keep a reference to the list it passes.
    held = {}
    _evaluate = walk_search.evaluate

    def evaluate(*a, **k):
        held["cap"] = k.get("capture")
        return _evaluate(*a, **k)
    walk_search.evaluate = evaluate

    started = time.time()
    rows = []
    for s in seeds:
        t0 = time.time()
        rec = RW._run(s)
        if "error" not in rec and held.get("cap"):
            rec.update(footfall(held["cap"], tick))
        rec["wall_s"] = round(time.time() - t0, 1)
        rows.append(rec)
        print(f"seed {s}: standing_v32={rec.get('standing_v32')} "
              f"tip={rec.get('freq_tip')} swing_tip={rec.get('swing_tip_ms')} "
              f"duty_tip={rec.get('duty_tip')} R1i_tip={rec.get('R1i_tip')} "
              f"fFTi={(rec.get('rom') or {}).get('fFTi')} "
              f"sep={rec.get('footfall_sep_bl')} "
              f"h={rec.get('body_height_mm')}", flush=True)

    st = [r for r in rows if r.get("standing_v32")]

    def col(key):
        if key == "step_freq_hz":
            v = [r.get("freq_tip") for r in st]
        elif key == "swing_ms":
            v = [r.get("swing_tip_ms") for r in st]
        elif key == "duty":
            v = [r.get("duty_tip") for r in st]
        elif key == "contra_phase":
            v = [r.get("contra_tip") for r in st]
        elif key == "speed_mm_s":
            v = [r.get("speed") for r in st]
        elif key.startswith("rom_"):
            tag = key[len("rom_"):-len("_deg")]
            v = [(r.get("rom") or {}).get(tag) for r in st]
        else:
            v = [r.get(key) for r in st]
        return [float(x) for x in v
                if x is not None and not (isinstance(x, float) and np.isnan(x))]

    card = {}
    for key, (blo, bhi) in BANDS.items():
        v = col(key)
        if not v:
            card[key] = {"median": None, "band": [blo, bhi],
                         "in_band": f"0/{len(st)}", "grade": "NO DATA"}
            continue
        med = float(np.median(v))
        n_in = sum(blo <= x <= bhi for x in v)
        card[key] = {"median": round(med, 4), "band": [blo, bhi],
                     "min": round(min(v), 4), "max": round(max(v), 4),
                     "in_band": f"{n_in}/{len(v)}",
                     "grade": "PASS" if blo <= med <= bhi else "CONFLICT"}
    for rule in RULES:
        sig = sum(1 for r in st if r.get(rule) is not None
                  and r[rule] < 0.05)
        ok = sig * 2 > len(st)
        card[rule] = {"runs_p_lt_0.05": f"{sig}/{len(st)}",
                      "grade": "PASS" if ok else "CONFLICT"}
    minlift = min(min(r["lifts"]) for r in st) if st else None
    card["lifts_min_all_feet"] = {"min": minlift, "bar": 5,
                                  "grade": ("PASS" if minlift is not None
                                            and minlift >= 5 else "CONFLICT")}

    out = {
        "instrument": "regression_walk._run through the title path "
                      "(full meta and edges, spike_reg 44) + footfall1 "
                      "measures on the same capture",
        "theta": args.theta, "theta_sha256": sha(theta_p),
        "spike_reg": 44.0, "backend": "numpy", "seeds": seeds,
        "standing_v32": f"{len(st)}/{len(rows)}",
        "card": card,
        "provenance": prov,
        "runs": rows,
        "wall_seconds": round(time.time() - started, 1),
    }
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nstanding {len(st)}/{len(rows)}")
    for k, v in card.items():
        print(f"  {k:<20} {json.dumps(v)}")
    print(f"receipt -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
