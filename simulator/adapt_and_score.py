"""Adapt a NeuroMechFly (world-frame, 10kHz, 69-segment) trajectory onto the
32-landmark egocentric layout flyscore expects, then score it.

WHY THIS IS SIMPLER THAN IT LOOKED
The Zenodo reference data is egocentric pixel space from one specific camera
rig. Matching a simulation to it sounds like it needs photometric calibration.
It doesn't: five of the seven scorer metrics only ever read the antero-
posterior projection of leg-tip motion (detrended), or a scale-free ratio.
Only body_length_px is genuinely pixel-scaled, and it's a sanity check, not a
behavioural result. So the adapter's real job is TWO things: (1) build a
proper egocentric frame (rotate into the fly's own heading each frame, so a
straight walk doesn't accidentally look like a strafe), and (2) resample from
the simulator's 10kHz physics rate down to the reference's 80Hz, matching the
frequency bands the scorer's Welch analysis was tuned for.

MAPPING, keyed to flyscore/layout.py's documented indices:
  LEG_TIPS   -> NeuroMechFly's most distal tarsus segment (tarsus5) per leg.
                Layout's 4-point proximal->distal chain maps to
                coxa, trochanterfemur, tibia, tarsus5.
  THORAX     -> c_thorax
  ABDOMEN    -> c_abdomen6 (most posterior abdominal segment)
  HEAD/WING  -> filled for completeness; no current metric reads these OUTPUT
                slots (verified against flyscore/metrics.py, which reads only
                LEG_TIPS, THORAX and ABDOMEN). The eye BODY positions are still
                load-bearing because their midpoint defines the frame below.

HEADING SOURCE
FlyGym 2.1's compiled MuJoCo model fuses the static ``c_head`` body away, but
``Simulation.get_body_positions`` still leaves that row in the documented
body-segment order with internal body ID -1.  NumPy then aliases the last body,
``rh_tarsus5``, and makes right-hind motion rotate every simulated landmark.
The midpoint of the two retained eye bodies supplies the same anterior heading
without referencing the fused body.  A regression test poisons the ``c_head``
row and requires exact output invariance.
"""
import numpy as np
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from flyscore import layout as L
from flyscore import metrics as M

TRAJ = str(REPO / "cpg_walk_trajectory.npz")
REFERENCE = str(REPO / "flyscore" / "reference_real_flies.json")
TARGET_FPS = L.FPS  # 80.0, matches the Zenodo reference data

# NeuroMechFly segment name -> layout.py leg key (R1=front..L3=hind, per L.LEGS)
NMF_LEG_PREFIX = {"R1": "rf", "R2": "rm", "R3": "rh", "L1": "lf", "L2": "lm", "L3": "lh"}
NMF_LEG_CHAIN = ["coxa", "trochanterfemur", "tibia", "tarsus5"]  # proximal -> distal


def build_egocentric_32(all_body_pos, names):
    """all_body_pos: (n_frames, n_segs, 3) world positions.
    Returns pts: (n_frames, 32, 2), matching layout.py's index scheme."""
    idx = {n: i for i, n in enumerate(names)}

    def seg(name):
        key = f"BodySegment(name='{name}')"
        return all_body_pos[:, idx[key], :2]  # drop z; walking fly, near-planar

    thorax = seg("c_thorax")
    # c_head is fused out of FlyGym's compiled MuJoCo model.  Both eye bodies
    # remain named and symmetric, so their midpoint is the retained head axis.
    head = 0.5 * (seg("l_eye") + seg("r_eye"))

    # Egocentric frame per frame: origin at thorax, y-axis along the fly's own
    # heading (thorax -> head), x-axis perpendicular. This is what "egocentric"
    # in the reference data means — without it, a straight walk in world-frame
    # X would only look right if the fly happened to face exactly +X, which a
    # tripod CPG walker approximately does but should not be ASSUMED.
    fwd = head - thorax
    fwd_length = np.linalg.norm(fwd, axis=1, keepdims=True)
    if not np.isfinite(fwd_length).all() or np.any(fwd_length <= 1e-9):
        raise ValueError("thorax-to-eye-midpoint heading is non-finite or zero")
    fwd_norm = fwd / fwd_length
    right = np.stack([fwd_norm[:, 1], -fwd_norm[:, 0]], axis=1)  # rotate -90deg

    def to_local(world_xy):
        rel = world_xy - thorax
        # NEGATIVE forward-projection: layout.py's convention is LOW y = anterior
        # (head at y~30) and HIGH y = posterior (abdomen at y~96). Head sits in
        # the +forward direction from thorax, so its forward-projection must be
        # sign-flipped to come out low.
        local_y = -np.sum(rel * fwd_norm, axis=1)
        local_x = np.sum(rel * right, axis=1)
        return np.stack([local_x, local_y], axis=1)

    n_frames = all_body_pos.shape[0]
    pts = np.full((n_frames, 32, 2), np.nan, dtype=np.float64)

    for leg_key, prefix in NMF_LEG_PREFIX.items():
        for j, part in enumerate(NMF_LEG_CHAIN):
            pts[:, L.LEGS[leg_key][j], :] = to_local(seg(f"{prefix}_{part}"))

    pts[:, L.HEAD_PAIR_ANTERIOR[0], :] = to_local(seg("l_eye"))
    pts[:, L.HEAD_PAIR_ANTERIOR[1], :] = to_local(seg("r_eye"))
    pts[:, L.HEAD_PAIR_POSTERIOR[0], :] = to_local(seg("l_pedicel"))
    pts[:, L.HEAD_PAIR_POSTERIOR[1], :] = to_local(seg("r_pedicel"))
    pts[:, L.WING_TIPS[0], :] = to_local(seg("l_wing"))
    pts[:, L.WING_TIPS[1], :] = to_local(seg("r_wing"))
    pts[:, L.THORAX, :] = to_local(thorax)  # = origin, always (0,0)
    pts[:, L.ABDOMEN, :] = to_local(seg("c_abdomen6"))

    return pts


def resample_to_fps(pts, src_dt, target_fps):
    """Decimate from the simulator's physics rate to the reference's 80Hz.
    Simple stride-based resampling: at 10kHz -> 80Hz the anti-alias margin is
    huge (125x oversampled going in) relative to the 1-30Hz band the scorer's
    Welch analysis reads, so no low-pass filter is needed."""
    src_fps = 1.0 / src_dt
    stride = round(src_fps / target_fps)
    return pts[::stride]


def main():
    d = np.load(TRAJ)
    names = d["body_seg_names"]
    dt = float(d["timestep"])

    print(f"loaded trajectory: {d['all_body_pos'].shape[0]} frames @ "
          f"{1/dt:.0f} Hz, {d['run_time_s']}s sim time")

    pts_full = build_egocentric_32(d["all_body_pos"], names)
    pts = resample_to_fps(pts_full, dt, TARGET_FPS)
    print(f"resampled to {pts.shape[0]} frames @ {TARGET_FPS} Hz "
          f"(matches flyscore/layout.py FPS)")

    assert not np.isnan(pts).any(), "NaN in adapted trajectory - mapping bug"

    scores = M.score_sequence(pts, fps=TARGET_FPS)

    import json
    with open(REFERENCE) as f:
        ref = json.load(f)["reference"]

    print("\n" + "=" * 78)
    print("SIMULATED FLY (NeuroMechFly, flygym's own tripod CPG controller,")
    print("NO connectome involved) vs REAL-FLY REFERENCE DISTRIBUTION")
    print("=" * 78)
    print(f"{'metric':<26} {'sim value':>12} {'real p5':>10} {'real median':>12} "
          f"{'real p95':>10}  in range?")
    n_in_range = 0
    n_scored = 0
    for name, val in scores.items():
        r = ref[name]
        if np.isnan(val):
            print(f"{name:<26} {'NaN':>12}  (metric could not be computed)")
            continue
        n_scored += 1
        in_range = r["p5"] <= val <= r["p95"]
        n_in_range += in_range
        flag = "YES" if in_range else "NO <-- outside real-fly 5-95th pctile"
        print(f"{name:<26} {val:12.4f} {r['p5']:10.4f} {r['median']:12.4f} "
              f"{r['p95']:10.4f}  {flag}")

    print(f"\n{n_in_range}/{n_scored} metrics land inside the real-fly "
          f"5th-95th percentile range.")
    print("\nWhat this number means: this is a KNOWN-GOOD walking controller,")
    print("no connectome. If most metrics land in-range, the SCORER and")
    print("ADAPTER are trustworthy and any later connectome result can be")
    print("attributed to the connectome. If they don't, the scorer/adapter")
    print("needs fixing before any connectome comparison means anything.")


if __name__ == "__main__":
    main()
