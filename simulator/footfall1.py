"""footfall1.py -- WHERE THE FEET LAND, against Mendes 2013.

The four constraints registered 2026-08-18 (frozen before this script
existed). Converts our tarsus-tip positions into Mendes coordinates:
origin at the body, +y anterior, both axes divided by body length, and
reports per leg pair the stance midpoint, AEP/PEP, lateral offset, and
the origin-free front-minus-hind separation. Also reports body height
in mm against Chun 2021.

Bands (FIXED, from the ledger):
  stance midpoint fore-aft   front +0.50, middle -0.03, hind -0.60 BL
  front-minus-hind sep       1.02-1.11 BL   (origin-free, most robust)
  lateral |x|                0.25 / 0.63 / 0.36 BL
  body height                0.80-0.89 mm at 5-15 mm/s
"""
import json
import sys

import numpy as np
import pandas as pd

from walk_search import evaluate

BL_MM = 2.5          # Mendes' body length normaliser (Drosophila ~2.5 mm)
LEGS = ["lf", "lh", "lm", "rf", "rh", "rm"]     # sorted(LEG_KEYS.values())
PAIR = {"front": ("lf", "rf"), "middle": ("lm", "rm"), "hind": ("lh", "rh")}
TARGET_MID = {"front": 0.50, "middle": -0.03, "hind": -0.60}
TARGET_LAT = {"front": 0.25, "middle": 0.63, "hind": 0.36}


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]])


def main():
    theta_path = sys.argv[1] if len(sys.argv) > 1 else "results-f23-theta.json"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 2209
    male = len(sys.argv) > 3 and sys.argv[3].startswith("male")
    meta_override = sys.argv[4] if len(sys.argv) > 4 else None
    th = json.load(open(theta_path))["winner"]
    mp = meta_override or ("data/manc-as-banc-meta.feather" if male
          else "data/banc_888_meta.feather")
    ep = ("data/manc-as-banc-edges.feather" if male
          else "data/banc_888_edgelist_simple_v3.feather")
    m = pd.read_feather(mp)
    keep = (m if male else
            m[(m["region"] == "ventral_nerve_cord")
              | (m["super_class"] == "descending")]).reset_index(drop=True)
    ids = set(keep["banc_888_id"])
    e = pd.read_feather(ep)
    e = e[(e["count"] >= 10) & e["pre"].isin(ids)
          & e["post"].isin(ids)].reset_index(drop=True)
    cap = []
    _, raw, up, z = evaluate({}, keep, e, th, seed=seed, dur_ms=6000.0,
                             capture=cap)
    tick = float(th.get("tick_ms", 5.0))
    trans = int(round(500.0 / tick))
    xyz = np.array([c[0] for c in cap])[trans:]          # body xyz
    quat = np.array([c[1] for c in cap])[trans:]
    tips = np.array([c[10] for c in cap])[trans:]        # (T, 6, 3)
    tipz = np.array([c[5] for c in cap])[trans:]
    print(f"{theta_path} seed {seed}: disp {raw:.2f} mm, upright {up:.2f}, "
          f"speed {raw / 5.5:.2f} mm/s", flush=True)

    # stance = foot near its own floor (the battery's own swing signal
    # inverted: clearance <= 0.15 mm above the leg's 5th-percentile height)
    clear = tipz - np.percentile(tipz, 5, axis=0, keepdims=True)
    stance = clear <= 0.15

    # body-frame foot positions: rotate (tip - body) into body axes
    body = np.zeros((len(xyz), 6, 3))
    for t in range(len(xyz)):
        R = quat_to_R(quat[t])
        body[t] = (tips[t] - xyz[t]) @ R          # R^T applied on the right
    fore = body[:, :, 0] / BL_MM                  # +x = forward in flygym
    lat = np.abs(body[:, :, 1]) / BL_MM

    print(f"\n{'pair':<8} {'stance mid':>11} {'target':>7} {'|lat|':>7} "
          f"{'target':>7}", flush=True)
    mids = {}
    for pair, legs in PAIR.items():
        idx = [LEGS.index(l) for l in legs]
        st = stance[:, idx]
        vals = fore[:, idx][st]
        lats = lat[:, idx][st]
        mids[pair] = float(np.median(vals)) if len(vals) else np.nan
        lm_ = float(np.median(lats)) if len(lats) else np.nan
        print(f"{pair:<8} {mids[pair]:11.3f} {TARGET_MID[pair]:7.2f} "
              f"{lm_:7.3f} {TARGET_LAT[pair]:7.2f}", flush=True)
    sep = mids["front"] - mids["hind"]
    ok = "PASS" if 1.02 <= sep <= 1.11 else "CONFLICT"
    print(f"\nfront-minus-hind separation {sep:.3f} BL  "
          f"band [1.02, 1.11]  {ok}   (origin-free, the robust test)",
          flush=True)
    # body pitch: the abdomen can graze a HIGH body if it is nose-up.
    pitches = []
    for t in range(len(quat)):
        R = quat_to_R(quat[t])
        pitches.append(np.degrees(np.arcsin(np.clip(-R[2, 0], -1, 1))))
    print(f"body pitch median {np.median(pitches):+.1f} deg "
          f"(+ = nose up)   [no measured fly reference found]", flush=True)
    h = float(np.median(xyz[:, 2]))
    hok = "PASS" if 0.80 <= h <= 0.89 else "CONFLICT"
    print(f"body height median {h:.3f} mm  band [0.80, 0.89] at 5-15 mm/s "
          f"{hok}   (Chun 2021)", flush=True)


if __name__ == "__main__":
    main()
