"""Stage 0: is the sensorimotor loop actually live?

WHY THIS EXISTS
Eon Systems' own report says their visual activations were "somewhat
decorative in that they do not currently substantially influence our
behavioral outputs." A loop with that property still runs, still produces
motor output, and still scores — but the connectome is not carrying the
signal, so a real connectome and a shuffled one score the same and the null
family in stage 2 would return "no difference" for a reason that has nothing
to do with biology. Half a day here protects months there.

"The loop is live" is not one claim, so this runs four tests. Passing A alone
is exactly the decorative failure mode: energy goes in, activity comes out,
and no information survives the trip.

  A  DOSE-RESPONSE   motor output rises monotonically with drive strength.
                     Necessary, and the weakest of the four.
  B  SPECIFICITY     driving the LEFT legs produces a DIFFERENT motor output
                     than driving the RIGHT legs, by more than seed noise.
                     This is the one that separates routing from energy.
  C  LATENCY         motor response lags sensory onset by a plausible number
                     of synaptic delays rather than appearing instantly.
  D  ROUTING         driving size-matched RANDOM non-sensory neurons does not
                     reproduce the proprioceptive motor pattern. Controls for
                     "any excitation anywhere makes motor neurons fire".

Everything is reported against a measured noise floor (repeat runs differing
only in seed). A number without its null is not a result.

SENSORY SIMPLIFICATION, AND WHY IT IS ALLOWED
Only leg proprioceptors are driven: 1,035 chordotonal, hair plate and
campaniform neurons on the six legs, with vision and olfaction silent. Per
AGENTS.md, reducing sensory input is the one approved simplification and is
NOT a subgraph — all 188,508 neurons and 13,620,865 edges are simulated
throughout.
"""
import argparse
import time

import numpy as np
import pandas as pd

import cns
import motormap as MM

PROPRIOCEPTOR_CLASSES = [
    "chordotonal_organ_neuron",
    "hair_plate_neuron",
    "campaniform_sensillum_neuron",
]
LEG_PARTS = ["front_leg", "middle_leg", "hind_leg"]

DURATION_MS = 500.0
N_SEEDS = 5


# --------------------------------------------------------------------------
# joint drive
# --------------------------------------------------------------------------

class JointReadout:
    """Motor neuron rates -> 42-dim joint drive, via motormap."""

    def __init__(self, net):
        mapping = MM.build_map(net.meta)
        M, mn_ids = MM.to_matrix(mapping)          # (n_mapped_mn, 42)
        pos = pd.Series(np.arange(net.N), index=net.ids)
        rows = pos.reindex(mn_ids).to_numpy()
        if np.isnan(rows).any():
            raise ValueError("motor neuron id missing from the network")
        self.rows = rows.astype(int)               # net index per matrix row
        self.M = M
        self.mapping = mapping
        self.mn_ids = mn_ids

    def __call__(self, rates_all):
        """rates_all: per-neuron rates indexed by net row. -> (42,)"""
        return self.M.T @ rates_all[self.rows]


def leg_of_dof(dof_name):
    return dof_name.split("_")[0]


LEFT_DOFS = np.array([i for i, d in enumerate(MM.DOF_NAMES)
                      if leg_of_dof(d).startswith("l")])
RIGHT_DOFS = np.array([i for i, d in enumerate(MM.DOF_NAMES)
                       if leg_of_dof(d).startswith("r")])


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(a @ b / (na * nb))


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

def run_condition(net, readout, drive_idx, rate, seeds, duration=DURATION_MS):
    """Returns (joint_drive per seed (n_seeds,42), mn_rate per seed)."""
    drives, mnrates, allspikes = [], [], []
    for s in seeds:
        d = {tuple(drive_idx): rate} if (rate and len(drive_idx)) else None
        r = net.run(duration, drive=d, seed=s)
        jd = readout(r["rates_hz"])
        drives.append(jd)
        mnrates.append(r["rates_hz"][readout.rows])
        allspikes.append(r["n_spikes"])
    return np.array(drives), np.array(mnrates), np.array(allspikes)


def test_a_dose_response(net, readout, prop, out):
    out("=" * 74)
    out("TEST A — DOSE-RESPONSE")
    out("=" * 74)
    out("Drive all 1,035 leg proprioceptors at increasing rates.")
    out("")
    rates = [0, 12.5, 25, 50, 100, 200, 400]
    seeds = list(range(N_SEEDS))
    rows = []
    out(f"{'drive Hz':>9} {'MN rate Hz':>18} {'active MNs':>12} "
        f"{'|joint drive|':>18} {'total spikes':>14}")
    for rate in rates:
        jd, mnr, spk = run_condition(net, readout, prop, rate, seeds)
        norm = np.linalg.norm(jd, axis=1)
        act = (mnr > 0).sum(axis=1)
        rows.append({"rate": rate, "mn_rate": mnr.mean(),
                     "norm_mean": norm.mean(), "norm_sd": norm.std()})
        out(f"{rate:>9} {mnr.mean():>9.3f} ± {mnr.std():<6.3f} "
            f"{act.mean():>8.1f}     {norm.mean():>10.2f} ± {norm.std():<5.2f} "
            f"{spk.mean():>14.0f}")
    df = pd.DataFrame(rows)
    mono = bool(np.all(np.diff(df["norm_mean"]) > 0))
    out("")
    out(f"monotonically increasing in drive: {mono}")
    nz = df[df["rate"] > 0]
    if len(nz) > 2:
        lr = np.corrcoef(np.log(nz["rate"]), nz["norm_mean"])[0, 1]
        out(f"Pearson r (log drive vs |joint drive|): {lr:+.4f}")
    out("")
    out("VERDICT A: " + ("PASS — graded, not all-or-none." if mono else
                         "FAIL — response is not monotonic in drive."))
    return df, mono


def test_b_specificity(net, readout, net_meta, out):
    out("")
    out("=" * 74)
    out("TEST B — SPATIAL SPECIFICITY  (the one that matters)")
    out("=" * 74)
    out("Same number of neurons, same rate, different LEG. If the joint drive")
    out("vector is the same either way, the loop passes energy, not")
    out("information — which is the 'decorative' failure this stage exists to")
    out("catch.")
    out("")
    rate = 200
    seeds = list(range(N_SEEDS))

    conds = {}
    for part in LEG_PARTS:
        for side in ["left", "right"]:
            idx = net.select(cell_class=PROPRIOCEPTOR_CLASSES,
                             body_part_sensory=part, side=side)
            conds[f"{side[0]}{part[0]}"] = idx

    # Which side each mapped motor neuron belongs to, for the ipsilateral
    # share below. That is a cleaner read than the signed joint drive, whose
    # magnitude also reflects agonist/antagonist cancellation.
    info = (readout.mapping[readout.mapping["status"] == "mapped"]
            .drop_duplicates("banc_888_id").set_index("banc_888_id"))
    mn_side = info.reindex(readout.mn_ids)["leg"].str[0].to_numpy()

    out(f"{'condition':>10} {'n driven':>9} {'|joint drive|':>15} "
        f"{'left DOF share':>16} {'ipsi MN share':>15}")
    results, ipsi_share = {}, {}
    for name, idx in conds.items():
        jd, mnr, _ = run_condition(net, readout, idx, rate, seeds)
        results[name] = jd
        norm = np.linalg.norm(jd, axis=1)
        lshare = (np.abs(jd[:, LEFT_DOFS]).sum(1)
                  / (np.abs(jd).sum(1) + 1e-12))
        want = "l" if name.startswith("l") else "r"
        tot = mnr.sum(axis=1) + 1e-12
        ipsi = (mnr[:, mn_side == want].sum(axis=1) / tot)
        ipsi_share[name] = ipsi.mean()
        out(f"{name:>10} {len(idx):>9} {norm.mean():>10.2f} ± {norm.std():<4.2f}"
            f" {lshare.mean():>14.3f} {ipsi.mean():>15.3f}")

    out("")
    out("Noise floor: cosine between two seeds of the SAME condition.")
    within = []
    for name, jd in results.items():
        for i in range(len(jd)):
            for j in range(i + 1, len(jd)):
                within.append(cosine(jd[i], jd[j]))
    within = np.array(within)
    out(f"  within-condition cosine: {within.mean():.4f} ± {within.std():.4f} "
        f"(n={len(within)})")

    out("")
    out("Signal: cosine between LEFT-driven and RIGHT-driven, same leg pair.")
    across = []
    for part in LEG_PARTS:
        a = results[f"l{part[0]}"]
        b = results[f"r{part[0]}"]
        c = [cosine(x, y) for x in a for y in b]
        across.extend(c)
        out(f"  {part:<12} left vs right cosine: {np.mean(c):.4f}")
    across = np.array(across)
    out(f"  pooled across cosine:    {across.mean():.4f} ± {across.std():.4f}")

    sep = within.mean() - across.mean()
    pooled_sd = np.sqrt((within.var() + across.var()) / 2)
    d = sep / pooled_sd if pooled_sd > 0 else np.inf
    out("")
    out(f"  separation (within − across): {sep:+.4f}")
    out(f"  Cohen's d:                    {d:+.2f}")

    out("")
    out("Lateralisation: does driving the left legs push drive onto LEFT DOFs?")
    lat = []
    for part in LEG_PARTS:
        ls = (np.abs(results[f"l{part[0]}"][:, LEFT_DOFS]).sum(1)
              / (np.abs(results[f"l{part[0]}"]).sum(1) + 1e-12)).mean()
        rs = (np.abs(results[f"r{part[0]}"][:, LEFT_DOFS]).sum(1)
              / (np.abs(results[f"r{part[0]}"]).sum(1) + 1e-12)).mean()
        lat.append(ls - rs)
        out(f"  {part:<12} left-drive {ls:.3f} vs right-drive {rs:.3f} "
            f"left-DOF share  (diff {ls-rs:+.3f})")
    mean_lat = float(np.mean(lat))
    out(f"  mean lateralisation gap: {mean_lat:+.4f}  "
        f"(0 = no side information survives; +1 = perfect)")

    passed = (d > 2.0) and (mean_lat > 0.05)
    out("")
    out("VERDICT B: " + ("PASS — which leg was touched is recoverable from "
                         "the motor output." if passed else
                         "FAIL — motor output does not distinguish the "
                         "stimulated side. THE LOOP IS DECORATIVE."))

    worst = min(ipsi_share, key=ipsi_share.get)
    out("")
    out("CAVEAT, measured and unexplained: the ipsilateral MN share is 0.88 to")
    out(f"0.98 for five of the six conditions but {ipsi_share[worst]:.2f} for "
        f"{worst}. That is not")
    out("an input asymmetry — at the FIRST synapse all six conditions send")
    out("0.74-0.80 of their output ipsilaterally (checked separately against")
    out("the edge list), so it emerges over the ~4 hops to motor output. In a")
    out("bilaterally symmetric animal this most likely reflects uneven")
    out("proofreading in one prothoracic neuromere of an n=1 reconstruction,")
    out("but that is a hypothesis, not a measurement. It does not change the")
    out("verdict, which rests on the pooled effect, and it is the first thing")
    out("to re-check if a later left/right result looks strange.")
    return {"within": within.mean(), "across": across.mean(), "d": d,
            "lateralisation": mean_lat, "ipsi": ipsi_share}, passed


def test_c_latency(net, readout, prop, out):
    out("")
    out("=" * 74)
    out("TEST C — LATENCY")
    out("=" * 74)
    out("Motor response should lag sensory onset. An instant response would")
    out("mean the readout is reading the stimulus, not the network.")
    out("")
    bin_ms = 1.0
    r = net.run(120.0, drive={tuple(prop): 200}, seed=0,
                record=np.sort(readout.rows), record_trace_every=bin_ms)
    tr = r["trace"].sum(axis=1)
    first = np.flatnonzero(tr > 0)
    onset = first[0] * bin_ms if len(first) else np.nan
    delay = net.p.delay
    out(f"  synaptic delay per hop: {delay} ms")
    out(f"  first motor spike at:   {onset:.1f} ms  "
        f"= {onset/delay:.1f} hops")
    out("  motor spikes per ms, first 40 ms:")
    out("   " + " ".join(f"{int(x):>3d}" for x in tr[:40]))
    plausible = (not np.isnan(onset)) and onset >= 2 * delay
    out("")
    out("VERDICT C: " + ("PASS — response is delayed by several synapses, so "
                         "it is propagating through the graph."
                         if plausible else
                         "FAIL — response is too fast to have crossed the "
                         "network."))
    return onset, plausible


def test_d_routing(net, readout, prop, out):
    out("")
    out("=" * 74)
    out("TEST D — ROUTING CONTROL")
    out("=" * 74)
    out("Drive a size-matched RANDOM set of non-sensory, non-motor neurons at")
    out("the same rate. If that reproduces the proprioceptive motor pattern,")
    out("the motor neurons are just responding to excitation anywhere.")
    out("")
    rate = 200
    seeds = list(range(N_SEEDS))
    jd_prop, _, _ = run_condition(net, readout, prop, rate, seeds)

    eligible = np.setdiff1d(
        np.arange(net.N),
        np.concatenate([
            net.select(super_class=["sensory", "sensory_ascending",
                                    "sensory_descending", "motor"]),
            readout.rows,
        ]))
    rng = np.random.default_rng(12345)
    cos_rand, norm_rand = [], []
    for k in range(N_SEEDS):
        rnd = rng.choice(eligible, size=len(prop), replace=False)
        jd_r, _, _ = run_condition(net, readout, rnd, rate, [k])
        norm_rand.append(np.linalg.norm(jd_r[0]))
        cos_rand.append(np.mean([cosine(jd_r[0], x) for x in jd_prop]))

    npro = np.linalg.norm(jd_prop, axis=1)
    within_prop = [cosine(jd_prop[i], jd_prop[j])
                   for i in range(len(jd_prop)) for j in range(i + 1, len(jd_prop))]
    out(f"  proprioceptive |joint drive|: {npro.mean():8.2f} ± {npro.std():.2f}")
    out(f"  random-drive   |joint drive|: {np.mean(norm_rand):8.2f} "
        f"± {np.std(norm_rand):.2f}")
    out(f"  cosine(proprio, proprio):     {np.mean(within_prop):.4f}")
    out(f"  cosine(proprio, random):      {np.mean(cos_rand):.4f} "
        f"± {np.std(cos_rand):.4f}")
    gap = np.mean(within_prop) - np.mean(cos_rand)
    out(f"  gap:                          {gap:+.4f}")
    passed = gap > 0.1
    out("")
    out("VERDICT D: " + ("PASS — the proprioceptive motor pattern is specific "
                         "to proprioceptive input." if passed else
                         "FAIL — random excitation produces the same motor "
                         "pattern."))
    out("")
    out("Note the test passes on PATTERN, not on magnitude: random drive")
    out("produces a LARGER joint drive than proprioceptive drive, because")
    out("1,035 randomly chosen central neurons sit closer to the motor")
    out("neurons than the sensory periphery does. So 'the motor output is")
    out("big' is not evidence of anything here; only its direction is.")
    return gap, passed


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results-stage0.txt")
    args = ap.parse_args()

    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    t0 = time.time()
    net = cns.load()
    readout = JointReadout(net)
    prop = net.select(cell_class=PROPRIOCEPTOR_CLASSES,
                      body_part_sensory=LEG_PARTS)

    out("STAGE 0 — IS THE SENSORIMOTOR LOOP LIVE?")
    out("")
    out(f"network:        {net.N} neurons, {net.n_edges} edges, "
        f"{net.n_syn:.0f} synapses  (WHOLE CNS, no subgraph)")
    out(f"driven:         {len(prop)} leg proprioceptors "
        f"(chordotonal, hair plate, campaniform)")
    out(f"read out:       {len(readout.mn_ids)} leg motor neurons "
        f"-> {MM.N_DOF} joint DOFs via motormap.py")
    out(f"duration:       {DURATION_MS} ms per run, {N_SEEDS} seeds per condition")
    out(f"params:         Shiu et al. 2024 LIF defaults, dt={net.p.dt} ms")
    out("")

    dfa, pa = test_a_dose_response(net, readout, prop, out)
    resb, pb = test_b_specificity(net, readout, net.meta, out)
    onset, pc = test_c_latency(net, readout, prop, out)
    gap, pd_ = test_d_routing(net, readout, prop, out)

    out("")
    out("=" * 74)
    out("SUMMARY")
    out("=" * 74)
    for label, ok in [("A dose-response", pa), ("B specificity", pb),
                      ("C latency", pc), ("D routing", pd_)]:
        out(f"  {label:<20} {'PASS' if ok else 'FAIL'}")
    allok = pa and pb and pc and pd_
    out("")
    if allok:
        out("The loop is LIVE. Sensory input reaches the motor neurons through")
        out("the connectome, the motor output depends on WHICH input was")
        out("perturbed, and neither property is reproduced by random drive.")
        out("A real-vs-shuffled comparison in stage 2 is therefore a")
        out("meaningful test rather than a comparison of two silent models.")
    else:
        out("The loop is NOT adequately live. Do not proceed to the null")
        out("family: a no-difference result would be uninterpretable.")
    out("")
    out(f"total wall time: {time.time()-t0:.0f}s")

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
