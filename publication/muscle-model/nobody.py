"""DROP THE BODY. Does 14 Hz survive with no loop, no muscle, no joints?

WHAT FORCED THIS. loop_necessity.py showed that freezing every proprioceptor at
its closed-loop mean rate -- same sensory drive, zero timing information -- gives
2/10 seeds in the 12-16 Hz band against the closed loop's 4/10 (Fisher p=0.628),
and that the claw_polarity manipulation that decides whether the rhythm appears
works purely by setting the afferents' TONIC RATE (96.1 Hz at +1 vs 44.6 Hz at
-1). Freezing the claw at a constant 96 Hz is indistinguishable from letting it
report position (p=0.324), and high tonic rates give 6/20 in band against 0/16
for low ones (p=0.020).

So the working claim is that the connectome rings when leg proprioceptors are
TONICALLY DRIVEN, and the body is incidental. This tests that directly.

THE DESIGN. No body, no muscle, no joints, no proprioceptor encoding, no
drive_fn. Plain net.run(drive={leg proprioceptors: R, DNg100: 50 Hz}) -- the
ordinary open-loop Poisson path. The 1,035 proprioceptors are stage0's
selection (PROPRIOCEPTOR_CLASSES x LEG_PARTS), not body.Proprioceptors', so the
sensory periphery is a flat Poisson source with no structure of any kind.

THE READOUT IS DELIBERATELY UNCHANGED. Leg motor-neuron spikes are binned at
1 ms, projected through JointReadout's signed matrix M (column-normalised
exactly as closed_loop.run_loop does it), and passed through body.Body's
dynamics OFFLINE before closed_loop.score_signal.

⚠️ Running the body offline is not a re-introduction of the body, and it is not
an approximation: with the loop open, Body is a pure feedforward filter of the
joint-drive sequence -- activation low-pass, tanh, joint low-pass, no state that
depends on anything the network hears. Stepping it after the run therefore gives
the SAME theta, to the bit, that stepping it during the run would have. It is
done because the detector and its seven guards (amplitude guard in radians,
drift guard, block-shuffle null) were tuned on theta, and scoring a differently
scaled signal would silently move the thresholds. The raw joint drive is scored
too and reported alongside, so both readouts are on the record.

⚠️ THE ONE-BIN OFFSET is preserved on purpose: in the closed loop, drive_fn at
t = i ms steps the body with the spikes of bin i-1, so theta[0] is a step with
zero drive. Reproduced here by prepending a zero row.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import body as B
import closed_loop as CL
import motormap as MM
from stage0 import JointReadout, LEG_PARTS, PROPRIOCEPTOR_CLASSES

BAND = (12.0, 16.0)
DRIVE_SCALE = 0.003
TAU_JOINT = 15.0
CMD_HZ = 50.0


def offline_body(jd, drive_scale=DRIVE_SCALE, tau_joint=TAU_JOINT, dt=CL.BIN):
    """Exactly body.Body.step, applied to a recorded joint-drive sequence."""
    bod = B.Body(tau_joint_ms=tau_joint, drive_scale=drive_scale)
    n = len(jd)
    theta = np.zeros((n, 42))
    prev = np.zeros(42)                      # bin i is stepped with bin i-1
    for i in range(n):
        bod.step(prev, dt)
        theta[i] = bod.theta
        prev = jd[i]
    return theta


class Readout:
    """MN spike trace -> 42 joint-drive channels, identical to the loop's."""

    def __init__(self, net):
        self.ro = JointReadout(net)
        self.rows = self.ro.rows                     # M-row order, unique
        self.rec = np.sort(self.rows)                # what net.run records
        self.unsort = np.argsort(np.argsort(self.rows))
        self.M = self.ro.M / np.maximum(
            np.abs(self.ro.M).sum(axis=0, keepdims=True), 1.0)

    def joint_drive(self, trace):
        return trace[:, self.unsort].astype(float) @ self.M


def run_nobody(net, ro, prop, cmd, rate, seed, dur=CL.DUR):
    d = {tuple(prop): float(rate)}
    if cmd is not None and len(cmd):
        d[tuple(cmd)] = CMD_HZ
    r = net.run(dur, drive=d, seed=seed, record=ro.rec,
                record_trace_every=CL.BIN)
    jd = ro.joint_drive(r["trace"])
    s0 = int(CL.SETTLE / CL.BIN)
    theta = offline_body(jd)
    return {"theta": theta[s0:], "jd": jd[s0:],
            "mn_spikes": int(r["trace"][s0:].sum()),
            "n_spikes": int(r["n_spikes"])}


def score(theta, jd):
    f, pk, z = CL.score_signal(theta, CL.DUR - CL.SETTLE, seed=3)
    sig = np.nan_to_num(z, nan=-9) > 3
    fj, _, zj = CL.score_signal(jd * 1e3, CL.DUR - CL.SETTLE, seed=3)
    sigj = np.nan_to_num(zj, nan=-9) > 3
    return {
        "n_sig": int(sig.sum()),
        "freqs": [round(float(x), 3) for x in f[sig]],
        "in_band": [round(float(x), 3) for x in f[sig]
                    if BAND[0] <= x <= BAND[1]],
        "z_max": float(np.nanmax(z)) if np.isfinite(z).any() else float("nan"),
        "swing_deg": float(np.median(theta.max(0) - theta.min(0))) * 180 / np.pi,
        "jd_n_sig": int(sigj.sum()),
        "jd_freqs": [round(float(x), 3) for x in fj[sigj]],
        "jd_in_band": [round(float(x), 3) for x in fj[sigj]
                       if BAND[0] <= x <= BAND[1]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", default="20,40,65,96,130,180")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--shuffled", action="store_true")
    ap.add_argument("--w-syn", type=float, default=0.5)
    ap.add_argument("--tag", default="real")
    ap.add_argument("--out", default="results-nobody.json")
    args = ap.parse_args()

    t0 = time.time()
    meta = pd.read_feather("data/banc_888_meta.feather")
    edges = pd.read_feather("data/banc_888_edgelist_simple_v3.feather")
    net = CL.build_net(meta, edges, args.shuffled, 0.35, args.w_syn, 1.8)
    ro = Readout(net)
    prop = net.select(cell_class=PROPRIOCEPTOR_CLASSES,
                      body_part_sensory=LEG_PARTS)
    cmd = net.select(cell_type="DNg100")
    print(f"{'SHUFFLED' if args.shuffled else 'REAL'} w_syn {args.w_syn}  "
          f"{len(prop)} leg proprioceptors driven, {len(cmd)} DNg100 at "
          f"{CMD_HZ} Hz, {len(ro.rows)} MNs read out")
    print(f"{CL.DUR/1000:.0f}s per run, first {CL.SETTLE/1000:.1f}s discarded, "
          f"NO body in the loop\n")

    prev = []
    if os.path.exists(args.out):
        prev = json.load(open(args.out))
    print(f"{'tag':>10} {'rate':>6} {'seed':>5} {'MN spk':>9} {'swing':>7} "
          f"{'z>3':>7} {'in band':>18}  {'all sig freqs'}")
    rows = list(prev)
    for rate in [float(x) for x in args.rates.split(",")]:
        hits = 0
        for seed in range(args.seed0, args.seed0 + args.seeds):
            res = run_nobody(net, ro, prop, cmd, rate, seed)
            sc = score(res["theta"], res["jd"])
            sc.update({"tag": args.tag, "shuffled": bool(args.shuffled),
                       "w_syn": args.w_syn, "rate": rate, "seed": seed,
                       "mn_spikes": res["mn_spikes"],
                       "n_spikes": res["n_spikes"]})
            rows.append(sc)
            hits += bool(sc["in_band"])
            fb = "  ".join(f"{x:.2f}" for x in sc["in_band"]) or "—"
            al = "  ".join(f"{x:.2f}" for x in sc["freqs"][:6]) or "—"
            print(f"{args.tag:>10} {rate:>6.0f} {seed:>5} "
                  f"{sc['mn_spikes']:>9,} {sc['swing_deg']:>6.2f}° "
                  f"{sc['n_sig']:>3d}/42 {fb:>18}  {al}", flush=True)
            json.dump(rows, open(args.out, "w"), indent=1)
        print(f"    -> {hits}/{args.seeds} seeds with a 12-16 Hz channel "
              f"at {rate:.0f} Hz\n", flush=True)
    print(f"wall {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
