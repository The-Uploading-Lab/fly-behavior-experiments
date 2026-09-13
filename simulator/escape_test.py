"""THE FIRST SHARP LOCK: giant-fibre escape latency, in milliseconds.

WHY THIS BEHAVIOUR AND NOT WALKING. Walking is scored against a distribution
over 3,000 flies with ~480 free muscle parameters in the causal path, so a
hundred parameter sets fit it equally well. Escape is scored against a NUMBER:
one spike in the giant fibre, and the time until the muscle fires. Every
constant that sets how fast signals move is in that number and nothing else is.

THE CIRCUIT IS ENTIRELY INSIDE BANC, and it is textbook-correct as measured
from the edge list:

    GF -> PSI       13 syn / 4 edges (both GFs onto both PSIs)
    GF -> TTMn      17 syn, strictly ipsilateral 1:1     MONOSYNAPTIC
    GF -> DLMn       0 syn  <- CORRECT; the GF does not contact DLMn directly
    PSI -> DLMn    358 syn onto all 10 DLMn              DISYNAPTIC

That zero is the useful one: it means the two effectors sit at DIFFERENT PATH
LENGTHS from the same single spike, so their latency DIFFERENCE isolates the
per-synapse cost with the stimulus and the muscle cancelling out.

THE MEASUREMENT (Gaitanidis et al. 2025 PLOS Biology; eNeuro 2019
6(2):ENEURO.0423-18.2019), young flies 5-7 days:

    GF spike -> TTM   0.93 ms      monosynaptic
    GF spike -> DLM   1.44 ms      disynaptic
    difference        0.51 ms      = the cost of ONE more synapse

⚠️ THE ARITHMETIC ALREADY FALSIFIES THE MODEL BEFORE IT RUNS. `delay` is
1.8 ms per synapse. A monosynaptic path therefore cannot deliver anything in
under 1.8 ms, against a measured 0.93. The disynaptic path cannot beat 3.6 ms,
against a measured 1.44. And one extra synapse costs the model at least 1.8 ms
where the animal pays 0.51. The model is not slightly off; it is 1.9x, 2.5x and
3.5x too slow on three independent numbers.

THE DICHOTOMY THIS FORCES (regression.py, part D). Either

  (a) `delay` = 1.8 ms is simply wrong and should be ~0.3-0.5 ms, or
  (b) the MODEL CLASS is missing the mechanism -- the real GF circuit runs on
      GAP JUNCTIONS, not chemical synapses. ShakB RNAi removes ~90-100% of the
      JO->GF connection, GF->TTMn is only 0.50-0.82% of TTMn's total input, and
      the published biophysical model needed a 135 uS gap-junction conductance
      to reproduce these latencies. cns.py has no electrical synapses at all.

(b) is the more likely reading, and it is a RESULT rather than a bug: it says a
purely chemical-synapse LIF model cannot reproduce Drosophila escape timing.
This script measures which, by sweeping delay and asking whether ANY value
fits all three numbers at once. If no single delay does, (a) is dead and the
model class is the problem.

⚠️ A TRAP WORTH RECORDING. Of the 14 effector cells, ZERO has a correct
neurotransmitter prediction: PSI-right is called glutamate (which this model
signs -1), 9 of 10 DLMn are called gaba, and TTMn is gaba+octopamine. The
circuit is excitatory in the model only because SimParams.mn_transmitter_override
forces super_class=='motor' to +1 -- and it catches PSI because PSI is annotated
motor. The right answer for the wrong reason; if that override is ever narrowed,
this circuit inverts and the escape silently stops working.
"""
import argparse
import json

import numpy as np
import pandas as pd

import cns
import closed_loop as CL

MEASURED = {"TTM": 0.93, "DLM": 1.44, "diff": 0.51}
SIGMA = {"TTM": 0.10, "DLM": 0.12, "diff": 0.08}   # between-animal, young flies


def _col(meta, name):
    if name not in meta.columns:
        return pd.Series([""] * len(meta)).str.upper()
    return meta[name].astype(str).fillna("").str.upper()


def find_cells(meta):
    ct = _col(meta, "cell_type")
    mc = _col(meta, "manc_cell_type")
    fc = _col(meta, "fanc_cell_type")
    sc = meta["super_class"].astype(str)
    cc = meta["cell_class"].astype(str)
    gf = np.flatnonzero(ct.str.startswith("DNP01").to_numpy())
    if not len(gf):
        gf = np.flatnonzero(fc.str.contains("GIANT FIBER").to_numpy())
    ttm = np.flatnonzero(mc.str.contains("TTMN").to_numpy())
    dlm = np.flatnonzero(ct.str.startswith("DLM").to_numpy())
    psi = np.flatnonzero(((cc == "peripheral_intrinsic_neuron")
                          & (sc == "motor")).to_numpy())
    return gf, ttm, dlm, psi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delays", default="0.1,0.3,0.5,1.0,1.8")
    ap.add_argument("--w-syn", default="0.5")
    ap.add_argument("--gap", default="",
                    help="SPIKELET amplitudes (mV) to try; enables ELECTRICAL "
                         "coupling on GF<->TTMn and GF<->PSI")
    ap.add_argument("--out", default="results-escape.json")
    args = ap.parse_args()

    meta = pd.read_feather("data/banc_888_meta.feather")
    edges = pd.read_feather("data/banc_888_edgelist_simple_v3.feather")
    gf, ttm, dlm, psi = find_cells(meta)
    print(f"GF {len(gf)}   TTMn {len(ttm)}   DLMn {len(dlm)}   PSI {len(psi)}")
    if not (len(gf) and len(ttm) and len(dlm)):
        raise SystemExit("could not identify the circuit -- refusing to guess")

    rows = []
    print(f"\n{'delay':>7} {'w_syn':>5} {'g_gap':>6} {'TTM ms':>8} {'DLM ms':>8} {'diff':>7} "
          f"{'D_TTM':>7} {'D_DLM':>7} {'D_diff':>7}   verdict")
    # ⚠️ WHICH CONNECTIONS ARE ELECTRICAL IS NOT A FREE CHOICE. GF->TTMn and
    # GF->PSI are ShakB gap junctions; PSI->DLMn is CHOLINERGIC, i.e. chemical.
    # That asymmetry is the whole point: TTM is fast because its path is
    # electrical, DLM lags because one chemical synapse sits in its path, and
    # the measured 0.51 ms difference is therefore a direct measurement of what
    # ONE chemical synapse costs. Making PSI->DLMn electrical too would erase
    # the very quantity being measured.
    gaps = [float(x) for x in args.gap.split(",")] if args.gap else [0.0]
    pairs = ([(int(a), int(b)) for a in gf for b in ttm]
             + [(int(a), int(b)) for a in gf for b in psi]) if args.gap else []
    if pairs:
        print(f"electrical: {len(pairs)} GF<->TTMn / GF<->PSI pairs; "
              f"PSI->DLMn left CHEMICAL")

    combos = [(d, w, gg) for gg in gaps
              for w in [float(x) for x in args.w_syn.split(",")]
              for d in [float(x) for x in args.delays.split(",")]]
    for d, wsyn, gg in combos:
        p = cns.SimParams()
        p.balance_target, p.w_syn, p.delay = 0.35, wsyn, d
        p.gap_pairs, p.g_gap, p.gap_spikelet = tuple(pairs), 0.05, gg
        net = cns.CNS(meta, edges, p)
        # ONE spike in the giant fibres at t = 0, nothing else. This is the
        # published protocol: stimulate the brain, time the muscle.
        r = net.run(20.0, drive=None, seed=0, force_spikes={0.0: gf},
                    first_spike=True, record=np.arange(net.N))
        tf = r["t_first_ms"]
        t_ttm = np.nanmin(tf[ttm]) if np.isfinite(tf[ttm]).any() else np.nan
        t_dlm = np.nanmin(tf[dlm]) if np.isfinite(tf[dlm]).any() else np.nan
        diff = t_dlm - t_ttm
        got = {"TTM": t_ttm, "DLM": t_dlm, "diff": diff}
        D = {k: (got[k] - MEASURED[k]) / SIGMA[k] for k in MEASURED}
        # ⚠️ NOT `all(... for k in D if isfinite)` -- that is vacuously TRUE
        # when nothing fired, and the first version of this script duly
        # reported PASS for a circuit in which no muscle spiked at all. A test
        # that passes when the result is missing is worse than no test.
        ok = all(np.isfinite(D[k]) and abs(D[k]) <= 2 for k in MEASURED)
        rows.append({"delay": d, "w_syn": wsyn, "g_gap": gg, **{f"t_{k}": (None if not np.isfinite(v) else float(v))
                                    for k, v in got.items()},
                     **{f"D_{k}": (None if not np.isfinite(v) else float(v))
                        for k, v in D.items()}, "pass": bool(ok)})
        f = lambda x: "  never" if not np.isfinite(x) else f"{x:8.2f}"
        g = lambda x: "      —" if not np.isfinite(x) else f"{x:+7.1f}"
        print(f"{d:>7.2f} {wsyn:>5.2f} {gg:>6.2f} {f(t_ttm)} {f(t_dlm)} {f(diff)[2:]} "
              f"{g(D['TTM'])} {g(D['DLM'])} {g(D['diff'])}   "
              f"{'PASS' if ok else 'fail'}", flush=True)
        del net
        json.dump(rows, open(args.out, "w"), indent=1)

    # How much gain would the monosynaptic arm even NEED? A single GF spike
    # delivers n_syn * w_syn * PSP_FACTOR mV; firing needs the threshold gap.
    n_syn_gf_ttm, gap, psp = 17.0, 7.0, 0.1575
    need = gap / (n_syn_gf_ttm * psp)
    print(f"\nARITHMETIC OF THE MONOSYNAPTIC ARM")
    print(f"  GF->TTMn is {n_syn_gf_ttm:.0f} synapses. One GF spike delivers")
    print(f"  {n_syn_gf_ttm:.0f} x w_syn x {psp} mV against a {gap:.0f} mV "
          f"threshold gap.")
    print(f"  At w_syn = 0.5 that is "
          f"{n_syn_gf_ttm*0.5*psp:.2f} mV -- {gap/(n_syn_gf_ttm*0.5*psp):.1f}x "
          f"SUBTHRESHOLD.")
    print(f"  Firing would need w_syn >= {need:.2f}, i.e. {need/0.5:.1f}x the "
          f"value the rest of")
    print(f"  the network runs at -- where it latches at the refractory "
          f"ceiling.")

    good = [r for r in rows if r["pass"]]
    print()
    if good:
        print(f"A delay of {good[0]['delay']} ms fits all three measured "
              f"numbers. Reading (a): the delay parameter was wrong, and this")
        print("is the value escape licenses. Lock it -- but only the parameters")
        print("this measurement can actually carry (regression.py: dof).")
    else:
        print("NO single delay fits all three. Reading (b): the MODEL CLASS is")
        print("the problem, not the parameter. The real circuit runs on GAP")
        print("JUNCTIONS -- ShakB RNAi removes ~90-100% of JO->GF, GF->TTMn is")
        print("0.50-0.82% of TTMn's input, and the published biophysical model")
        print("needed 135 uS of gap-junction conductance. cns.py has no")
        print("electrical synapses. That is a result about LIF-with-chemical-")
        print("synapses, and it is exactly the conclusion you cannot reach by")
        print("fitting harder.")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
