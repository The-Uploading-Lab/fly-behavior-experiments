"""The walking demo: the winged body walking on its leg motor neurons, three
arms, with starvation as a walk/rest bout schedule (longevity lane, 2026-09-13;
Robin: "we're going to film walking video from the wet lab so please do a
walking demo as well").

What the model can and cannot say. The champion's speed does not follow the
walking command (104 to 260 Hz all walk at about 1.8 mm/s,
2026-09-13-walk-cmd-sweep.json): the cord's rhythm sets it. The measured
starvation effect is mostly more time walking (Bisen 2025: median forward
speed 0.3 mm/s fed, 1.9 at 24 h, 3.6 at 20 h, on bouts of about 2 mm/s
diluted by rest; the hyperactivity needs octopamine, Yang 2015), and the
model has no rest/walk state and no octopamine. The walking command is inert
for walking here (the walking body walks about 1 mm/s at a 1 Hz command), so
a rest cannot come from the command: the arms gate the leg drive onto the
body on a bout schedule (rest = legs held at the stance) whose walking
FRACTION is the arm's one input, an ASSUMED interpolation of Bisen's curve
(median speed over a 1.8 mm/s bout speed, capped at 1): 0 h 0.17, 5 h 0.30,
10 h 0.55, 15 h 0.80, 20 h 1.00. The switching is a labelled input, the
walking itself is the connectome's. Ibuprofen, young wild type: no measured
locomotor change attributable to it (Proshkina 2016 pools it with two
flavonoids), so its schedule is the control's at another seed. Bout lengths:
walking bouts exponential with mean 1.0 s, rests sized to the fraction.

Readings declared: fraction of time moving (thorax speed above 0.5 mm/s) rises
with the starvation hours; bout speed does not; a starved fly at 20 h walks
throughout. Falsified by the wet-lab film if starved flies walk no more of the
time than fed ones, or if ibuprofen flies change.

    PYTHONPATH=. .venv/bin/python lanes/longevity/escape/walk_arms.py --arm starved_20h --seed 0 --render out.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jumpfly_arms as JA  # noqa: E402

HERE = Path(__file__).resolve().parent
WALK_FRACTION = {"control": 0.17, "starved_5h": 0.30, "starved_10h": 0.55, "starved_15h": 0.80, "starved_20h": 1.00, "ibuprofen": 0.17}
CMD_HZ, BOUT_MEAN_S, SPEED_MOVING = 104.0, 1.0, 0.5


def bout_schedule(fraction: float, dur_s: float, seed: int) -> list:
    """[(t_on, t_off), ...] walking intervals over dur_s at the given fraction."""
    rng = np.random.default_rng(1000 + seed); t = 0.0; out = []
    rest_mean = BOUT_MEAN_S * (1.0 - fraction) / max(fraction, 1e-6)
    walking = rng.random() < fraction
    while t < dur_s:
        d = rng.exponential(BOUT_MEAN_S if walking else rest_mean) if fraction < 1.0 else dur_s
        d = max(0.3, min(d, dur_s - t)) if fraction < 1.0 else dur_s
        if walking:
            out.append((t, t + d))
        t += d; walking = not walking
    return out


def run(arm: str, seed: int, render_path=None, dur_ms: float = 8000.0) -> dict:
    frac = WALK_FRACTION[arm]; sched = bout_schedule(frac, dur_ms * 1e-3, seed)
    state = {"track": []}

    # rest = the legs held at the stance. The command rate is inert for walking
    # in this champion (2026-09-13 sweep) and the walking body walks about 1
    # mm/s at 1 Hz command, so a rest cannot come from the command; the
    # schedule gates the leg drive onto the body instead (labelled input).
    JA.PUPPET_GATE = lambda t_s: any(a <= t_s < b for a, b in sched)
    try:
        r = JA.run(arm if arm in JA.ARMS else "control", seed, render_path=render_path, dur_ms=dur_ms, flash_ms=1e9, land_at_s=None, cmd_hz=CMD_HZ, track=state["track"])
    finally:
        JA.PUPPET_GATE = None
    T = np.asarray(state["track"])  # (t_s, x, y) every 10 ms; the thorax sways, so motion is NET travel over 250 ms windows
    if len(T) > 30:
        W = 25; n = len(T) // W
        net = np.array([np.linalg.norm(T[(k + 1) * W - 1, 1:3] - T[k * W, 1:3]) / (T[(k + 1) * W - 1, 0] - T[k * W, 0]) for k in range(n)])
        moving = net > SPEED_MOVING
        frac_moving = float(moving.mean()); speed_bouts = float(net[moving].mean()) if moving.any() else 0.0
        dist = float(np.sum(net * 0.25))
    else:
        frac_moving = speed_bouts = dist = float("nan")
    out = {"arm": arm, "seed": seed, "walk_fraction_input": frac, "rest": "legs held at the stance (leg drive gated), labelled input", "schedule_s": [(round(a, 2), round(b, 2)) for a, b in sched], "frac_time_moving": frac_moving,
           "bout_speed_mm_s": speed_bouts, "distance_mm": dist, "upright_final": r["upright_final"], "frame_t_ms": r.get("frame_t_ms"), "track": [(round(a, 3), round(b, 3), round(c, 3)) for a, b, c in state["track"]],
           "render": r.get("render"), "wall_s": r["wall_s"]}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", default="control", choices=tuple(WALK_FRACTION)); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", default=None); ap.add_argument("--dur", type=float, default=8000.0); a = ap.parse_args()
    r = run(a.arm, a.seed, render_path=a.render, dur_ms=a.dur)
    print(json.dumps({k: v for k, v in r.items() if k not in ("track", "frame_t_ms")}))
    (HERE / f"2026-09-13-walk-arms-{a.arm}-s{a.seed}.json").write_text(json.dumps(r, indent=1))
