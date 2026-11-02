"""The closed sensorimotor loop, and whether it produces a rhythm.

    motor neurons -> motormap -> joint drive -> muscle -> joint angle
          ^                                                    |
          +------------- proprioceptors <----------------------+

Three regimes have now been searched for a motor rhythm with the loop OPEN --
silent, latched, and the balanced asynchronous-irregular state -- and all were
empty. The remaining structural difference is that the proprioceptors were
seeing Poisson noise rather than the leg's own movement. This closes that.

THE THREE CONDITIONS, and why each is needed:

  CLOSED, real         the experiment
  OPEN, real           the same network with proprioceptors driven by Poisson
                       at the SAME mean rate the closed loop produced. This is
                       what isolates the effect of closing the loop from the
                       effect of merely changing how much sensory drive there
                       is -- without it, any difference could just be a drive
                       level, which has fooled this project twice already.
  CLOSED, shuffled     the connectome null. A closed loop through a shuffled
                       connectome still has a body, still has feedback, and
                       still has the same lags. If it oscillates just as well,
                       the rhythm is a property of the loop rather than of the
                       fly's wiring.

THE READOUT IS THE JOINT ANGLE, not the motor drive. Angles are what a body
does and what the scorer would consume; drive is one layer upstream. Both are
recorded.

⚠️ NOTHING IS FITTED. See body.py and MUSCLE.md. `tau_joint` and `drive_scale`
are SWEPT because they are the loop's lag and gain, and delay_law.py already
established that lag sets a feedback loop's frequency -- fixing them would
assume the answer.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

import body as B
import cns
import motormap as MM
import rhythm_detect as rd
from stage0 import JointReadout, LEG_PARTS

DUR = 6000.0
SETTLE = 1500.0
BIN = 1.0          # ms, also the body integration step


def build_net(meta, edges, shuffled, balance, w_syn, delay, seed_net=7):
    ed = edges
    if shuffled:
        ed = edges.copy()
        ed["post"] = np.random.default_rng(seed_net).permutation(
            ed["post"].to_numpy())
    p = cns.SimParams()
    p.balance_target, p.w_syn, p.delay = balance, w_syn, delay
    return cns.CNS(meta, ed, p)


def run_loop(net, meta, tau_joint, drive_scale, seed, closed=True,
             open_rates=None, cmd_hz=0.0, cmd_rows=None):
    """One simulation. Returns joint angles, joint drive and sensory rates."""
    ro = JointReadout(net)
    mn_rows = ro.rows                       # net row per mapped motor neuron
    prop = B.Proprioceptors(meta, net)
    bod = B.Body(tau_joint_ms=tau_joint, drive_scale=drive_scale)
    # NORMALISED per DOF, so the drive has physiological units. Raw M gives a
    # signed spike count per bin, whose scale depends on how many motor
    # neurons happen to target a joint (2 to 48 across the 42 DOFs) -- so an
    # arbitrary drive_scale was absorbing that. Dividing by the number of
    # contributing MNs makes the drive "mean spikes per motor neuron per ms",
    # and drive_scale then means the per-MN FIRING RATE that gives a full
    # joint excursion. 0.06 = 60 Hz, which is where fly leg motor neurons
    # actually work, so the parameter now has an anchor instead of a magnitude.
    M = ro.M / np.maximum(np.abs(ro.M).sum(axis=0, keepdims=True), 1.0)

    n_bins = int(DUR / BIN)
    theta = np.zeros((n_bins, 42))
    drive_rec = np.zeros((n_bins, 42))
    sens_rec = np.zeros(n_bins)
    rng = np.random.default_rng(1234 + seed)
    step = {"i": 0}

    def drive_fn(t_ms, counts):
        i = step["i"]
        jd = M.T @ counts.astype(float)     # signed joint drive
        bod.step(jd, BIN)
        if i < n_bins:
            theta[i] = bod.theta
            drive_rec[i] = jd
        if closed:
            idx, rates = prop.rates(bod)
        else:
            # OPEN control: same neurons, same MEAN rate, no timing structure.
            idx = prop.all_rows
            rates = np.full(len(idx), open_rates if open_rates else 20.0)
        if i < n_bins:
            sens_rec[i] = float(np.mean(rates))
        step["i"] = i + 1
        if cmd_hz > 0 and cmd_rows is not None and len(cmd_rows):
            idx = np.concatenate([idx, cmd_rows])
            rates = np.concatenate([rates, np.full(len(cmd_rows), cmd_hz)])
        return idx, rates

    r = net.run(DUR, seed=seed, record=np.sort(mn_rows),
                record_trace_every=BIN, drive_fn=drive_fn,
                feedback_rows=mn_rows, drive_every_ms=BIN)
    s0 = int(SETTLE / BIN)
    return {"theta": theta[s0:], "drive": drive_rec[s0:],
            "sens_mean": float(np.mean(sens_rec[s0:])),
            "n_spikes": int(r["n_spikes"]),
            "mn_spikes": int(r["trace"][s0:].sum()),
            "assumptions": prop.assumptions, "counts": prop.counts()}


def score_signal(X, dur_ms, seed=0, n_null=60):
    """Rhythm of each of the 42 channels, against a phase-preserving null.

    The null here has to differ from the spike-train one: these are continuous
    joint angles, not spike counts. Circularly shifting each DOF independently
    destroys any relationship BETWEEN joints while preserving each one's own
    autocorrelation exactly -- so it tests coordination. To test periodicity
    within a channel, the channel is also compared against a block-shuffled
    version of itself, which keeps short-timescale structure and destroys
    long-range order.
    """
    rng = np.random.default_rng(seed)
    n, k = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    ac = rd._acorr_rows(Xc.T)
    freqs, peaks = [], []
    for a in ac:
        f, p = rd._peak(a, dur_ms, rd.BAND, BIN)
        freqs.append(f)
        peaks.append(p)
    freqs, peaks = np.array(freqs), np.array(peaks)

    blk = max(4, int(60.0 / BIN))          # 60 ms blocks: shorter than a step
    null = np.empty((n_null, k))
    nb = n // blk
    for i in range(n_null):
        Y = Xc[:nb * blk].reshape(nb, blk, k)
        Y = Y[rng.permutation(nb)].reshape(nb * blk, k)
        acn = rd._acorr_rows(Y.T)
        null[i] = [rd._peak(a, dur_ms, rd.BAND, BIN)[1] for a in acn]
    mu, sd = np.nanmean(null, axis=0), np.nanstd(null, axis=0)
    valid = np.isfinite(null).mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (peaks - mu) / (sd + 1e-12)
    z = np.where((valid >= 0.5) & (sd > 1e-4), z, np.nan)
    # AMPLITUDE GUARD. A joint that does not move still has an autocorrelation
    # -- of numerical noise and any slow drift -- and block-shuffling destroys
    # the drift, so the real signal beats its own null and scores. Measured
    # 2026-08-09: a SHUFFLED run whose joints moved 0.00 degrees scored
    # z = +4.6, higher than the real network's best. A channel that does not
    # move cannot be rhythmic, whatever the statistic says.
    amp = X.max(axis=0) - X.min(axis=0)          # radians
    z = np.where(amp > np.deg2rad(0.5), z, np.nan)
    drift = np.zeros(k)
    for b in [(rd.BAND[0], rd.BAND[1] * 1.8),
              (max(1.0, rd.BAND[0] * 0.5), rd.BAND[1] * 0.7)]:
        fa = np.array([rd._peak(a, dur_ms, b, BIN)[0] for a in ac])
        with np.errstate(invalid="ignore", divide="ignore"):
            rr = np.stack([fa / freqs * m for m in (1., 2., 3., .5, 1 / 3.)])
            d = np.nanmin(np.abs(rr - 1.0), axis=0)
        drift = np.maximum(drift, np.where(np.isfinite(d), d, 1.0))
    z = np.where(drift <= 0.25, z, np.nan)
    return freqs, peaks, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau-joint", default="8,15,25,40")
    ap.add_argument("--drive-scale", type=float, default=0.06,
                    help="per-MN spikes/ms for full excursion; "
                         "0.06 = 60 Hz")
    ap.add_argument("--balance", type=float, default=0.35)
    ap.add_argument("--w-syn", type=float, default=0.5)
    ap.add_argument("--delay", type=float, default=1.8)
    ap.add_argument("--cmd", type=float, default=0.0,
                    help="tonic descending drive to DNg100, Hz")
    ap.add_argument("--out", default="results-closed-loop")
    args = ap.parse_args()

    t0 = time.time()
    meta = pd.read_feather("data/banc_888_meta.feather")
    edges = pd.read_feather("data/banc_888_edgelist_simple_v3.feather")

    print(f"balance {args.balance}, w_syn {args.w_syn}, delay {args.delay} ms, "
          f"drive_scale {args.drive_scale}, command {args.cmd} Hz")
    print(f"{DUR/1000:.0f}s runs, first {SETTLE/1000:.1f}s discarded, "
          f"body stepped every {BIN} ms\n")
    print(f"  {'condition':<18} {'tau_j':>6} {'MN spk':>8} {'sens Hz':>8} "
          f"{'angle range':>12} {'z>3':>8} {'median f':>10} {'max z':>7}")

    rows = []
    for tj in [float(x) for x in args.tau_joint.split(",")]:
        for tag, shuffled, closed in [("CLOSED real", False, True),
                                      ("OPEN real", False, False),
                                      ("CLOSED shuffled", True, True)]:
            net = build_net(meta, edges, shuffled, args.balance, args.w_syn,
                            args.delay)
            cmd_rows = net.select(cell_type="DNg100") if args.cmd > 0 else None
            if not closed:
                prev = [r for r in rows
                        if r["tau_joint"] == tj and r["cond"] == "CLOSED real"]
                open_rate = prev[0]["sens_mean"] if prev else 20.0
            else:
                open_rate = None
            res = run_loop(net, meta, tj, args.drive_scale, seed=0,
                           closed=closed, open_rates=open_rate,
                           cmd_hz=args.cmd, cmd_rows=cmd_rows)
            f, pk, z = score_signal(res["theta"], DUR - SETTLE, seed=3)
            sig = np.nan_to_num(z, nan=-9) > 3
            rng_deg = float(np.median(res["theta"].max(axis=0)
                                      - res["theta"].min(axis=0))) * 180 / np.pi
            rec = {"cond": tag, "tau_joint": tj, "shuffled": shuffled,
                   "closed": closed, "mn_spikes": res["mn_spikes"],
                   "sens_mean": res["sens_mean"], "angle_range_deg": rng_deg,
                   "n_sig": int(sig.sum()),
                   "f_sig": float(np.median(f[sig])) if sig.any() else None,
                   "z_max": float(np.nanmax(z)) if np.isfinite(z).any()
                   else float("nan")}
            rows.append(rec)
            fs = f"{rec['f_sig']:.2f} Hz" if rec["f_sig"] else "none"
            print(f"  {tag:<18} {tj:>6.0f} {rec['mn_spikes']:>8,} "
                  f"{rec['sens_mean']:>8.1f} {rng_deg:>10.2f}° "
                  f"{rec['n_sig']:>3d}/42  {fs:>10} {rec['z_max']:>+7.1f}",
                  flush=True)
            del net

    print("\nRead the angle-range column first: if the joints barely move, the")
    print("body is not being driven and the rhythm columns are about nothing.")
    with open(f"{args.out}.json", "w") as f_:
        json.dump(rows, f_, indent=1, default=float)
    print(f"\nwall {time.time()-t0:.0f}s -> {args.out}.json")


if __name__ == "__main__":
    main()
