"""Whole-CNS leaky integrate-and-fire simulation over the BANC connectome.

WHOLE CNS, NOT A SUBGRAPH. All 188,508 annotated segments go in and all
13,620,865 edges go in. Per AGENTS.md the only approved simplification is
reducing *sensory input* — which neurons get driven — never cutting the graph.

WHY NOT BRIAN2
Brian2 on this graph is minutes per simulated second on CPU, and stage 0 needs
tens of runs at several drive levels. This is a clocked LIF with a ragged
column gather, which is the same model, ~40x faster here, and has no hidden
state. Cross-checking it against Brian2 on a small graph is worth doing before
any published number leans on it.

PARAMETERS
Defaults follow Shiu et al. 2024 (the published, behaviourally validated fly
LIF parameterisation) so the starting point is somebody's measured choice
rather than ours. Everything is a named field on SimParams: nothing is
hardcoded, nothing is deleted for being currently unused, and the fields that
are pure hypothesis (`sign_unknown`, `w_unknown_scale`) are called out as such.
Per the parameter rule these are search dimensions, not decisions.

SIGNS
Insect fast transmission: acetylcholine excitatory, GABA inhibitory, glutamate
inhibitory (GluCl), histamine inhibitory. Monoamines have no fast ionotropic
sign and are left at 0 by default — NOT because they do nothing, but because
what they do is modulatory and this model has no modulation yet. That is a
known missing mechanism, recorded in SimParams as `nt_sign` entries you can
set, not silently dropped.
"""
from dataclasses import dataclass, field
from fractions import Fraction

import sys
import numpy as np
import pandas as pd
from scipy import sparse


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------

def _default_nt_sign():
    return {
        "acetylcholine": +1.0,
        "glutamate": -1.0,     # GluCl; the single biggest sign assumption here
        "gaba": -1.0,
        "histamine": -1.0,     # photoreceptor transmitter, inhibitory
        # Monoamines: no fast ionotropic sign. Present and settable, 0 by
        # default because this model has no neuromodulation to carry them.
        "dopamine": 0.0,
        "serotonin": 0.0,
        "octopamine": 0.0,
        "tyramine": 0.0,
        "nitric_oxide": 0.0,
    }


@dataclass
class SimParams:

    # N1 (2026-08-21, registered): afferent spike REGULARITY. Real
    # proprioceptive afferents fire far more regularly than Poisson;
    # ours were pure Bernoulli-per-step (CV = 1.0). spike_reg is the
    # shape of a renewal process: 1 = Poisson = today's code path,
    # bit-identical and drawing the identical RNG stream; k > 1 fires
    # each driven ENTRY on an integrate-to-threshold clock with
    # Erlang-k thresholds (CV = 1/sqrt(k)), rate-faithful because the
    # accumulator integrates the same per-step probability. The value
    # used in experiments is pinned to the MEASURED literature anchor
    # per the N1 registration's anti-goalpost guard -- never tuned.
    spike_reg: float = 1.0
    # Optional per-neuron renewal shape. None preserves the scalar path.
    # Values <= 1 use Poisson; values > 1 use Erlang renewal. This prevents a
    # measured sensory-afferent prior from silently becoming universal on
    # driven descending or central neurons.
    spike_reg_per_neuron: object = None
    # --- membrane, Shiu et al. 2024 ---
    # Scalars are the default; all of these also accept an (N,) array.
    # stage1.py replaces tau_mem and v_rest with per-neuron arrays drawn per
    # cell class, because a single global value is itself an arbitrary choice
    # and a real-vs-shuffled comparison made at one arbitrary choice is not a
    # result.
    #
    # Heterogeneous excitability is not a nicety. With one global v_rest every
    # neuron in the animal sits exactly v_threshold - v_rest below threshold,
    # so as the gain rises they all cross at once -- which is a good way to
    # manufacture a discontinuous transition that no real nervous system
    # would have. Measured 2026-08-08: the silent-to-latched transition sits
    # between w_syn 0.254 and 0.256, a 0.8% window.
    v_rest: float = -52.0        # mV, scalar or (N,) array
    v_reset: float = -52.0       # mV, scalar or (N,) array
    v_threshold: float = -45.0   # mV, scalar or (N,) array
    tau_mem: float = 20.0        # ms, scalar or (N,) array
    # Scalar or (N,) postsynaptic decay constants. A vector lets receptor /
    # membrane classes integrate incoming chemical conductance on different
    # time scales instead of imposing one 5 ms constant on the whole animal.
    tau_syn: object = 5.0        # ms
    # Optional transmitter-sign split for true conductance runs. These are
    # scalar receptor-class kinetics: excitatory (predominantly nAChR) and
    # inhibitory (GABA_A/GluCl collapsed because the current engine has two
    # conductance channels). Both must be supplied together, require split
    # conductances and charge normalization, and are default-off.
    tau_syn_exc: object = None
    tau_syn_inh: object = None
    # Scalar or (N,) array. Per-neuron refractoriness matters because this
    # one number sets every neuron's maximum firing rate identically --
    # 1000/2.2 = 455 Hz for all 188,508 of them -- and real neurons differ.
    # It was scalar-only until 2026-08-09 for no reason except that the code
    # computed n_refrac as an int.
    t_refrac: float = 2.2        # ms, scalar or (N,) array
    delay: float = 1.8           # ms, uniform axonal delay
    # MEASURED-STRUCTURE DIRECTIVE (Robin, 2026-09-01): optional per-neuron
    # presynaptic delay, ms, one value per neuron in meta row order. None =
    # the scalar `delay` above, bit-identical. A neuron's spikes and graded
    # release reach ALL its targets after ITS OWN delay -- axonal, so
    # presynaptic. Callers should quantise to a few distinct values: per-tick
    # delivery cost scales with the number of distinct delays among the
    # neurons that fired that tick, and ring memory with max(delay)/dt.
    delay_per_neuron: object = None
    # Delivery mechanism for the per-neuron delay. "bucket" is the canonical
    # engine's: at each tick, partition this tick's edges by the source's
    # delay slot and run one masked bincount per distinct slot, so the cost
    # carries a factor of the number of distinct slots that fired. "queue"
    # defers the presynaptic SPIKE instead and does one gather and one
    # bincount at the arrival tick however many distinct delays exist.
    # "bucket" is the default and the two must agree exactly; the flag exists
    # so the two can be compared on one input, which is what the 14:38 ruling
    # asked for before the queue may be ported.
    delay_delivery: str = "bucket"
    # Graded release historically uses delay buckets even when spike delivery
    # uses the queue. Keep that path unless a caller explicitly opts into the
    # mathematically equivalent pending-release queue and validates its
    # floating-point accumulation seam.
    graded_delay_delivery: str = "bucket"
    w_syn: float = 0.275         # mV per synapse
    # HYPOTHESISED PARAMETER, added deliberately under AGENTS.md's rule that
    # adding is cheap and encouraged. Scales inhibitory synapses relative to
    # excitatory ones, i.e. the excitation/inhibition balance. This is the
    # classic knob that determines whether a recurrent network oscillates at
    # all: too little inhibition and it fires steadily or saturates, too much
    # and it falls silent, and somewhere between there is a rhythmic regime.
    # 1.0 reproduces the previous behaviour exactly, so nothing already
    # measured changes.
    w_inh_scale: float = 1.0
    # F27 (2026-08-18, PARAMETER RULE: "adding a hypothesised parameter
    # is cheap and encouraged"). Gain on synapses FROM the leg
    # neuromere's own inhibitory premotor populations -- the cells the
    # lesion work identified as the cord's balance machinery, never
    # before searchable. 1.0 = bit-identical; the row set is empty by
    # default so the multiply is a no-op even at other gains.
    premotor_inh_gain: float = 1.0
    premotor_inh_rows: tuple = ()
    # LAYER 1 receptor seam (2026-09-02, lane A; classes named by lane D in
    # lanes/D/LAYER1-STARTSTOP.md). One entry per receptor class: the
    # presynaptic transmitter (a resolved nt token, or "*" for any) meeting
    # a postsynaptic BANC `cell_class`. Each entry multiplies the assembled
    # edge weight by sign * gain, where sign is RELATIVE to nt_sign (+1
    # agrees with the transmitter's fast sign, -1 reverses it) and gain is
    # a positive scalar. Empty = off and bit-identical; an entry at
    # (+1, 1.0) is also bit-identical, because the fold multiplies by
    # exactly 1.0 in float32. The fold happens before balance_target so
    # homeostatic scaling sees the classed synapse. Applies to both engines:
    # Metal reads W.data, so the fold reaches it with no kernel change. A
    # per-class synaptic tau is NOT here yet (it needs one conductance
    # channel per class in both engines); a "tau_ms" key is refused.
    receptor_classes: tuple = ()
    # COMMISSURAL GAIN (2026-09-03, lane A, CD1-QM). One multiplier on every
    # synapse whose presynaptic and postsynaptic neurons carry different BANC
    # `side` labels (left/right; all 188,508 neurons are labelled). Exists to
    # test one hypothesis: at the step line the two cords are locked IN phase
    # (feet lf-rf, lm-rm, lh-rh within 45 degrees of 0 on the champion) where
    # the tripod needs 180, and the midline-crossing synapses are the wiring
    # that could carry that lock. 0.0 silences every crossing synapse (a
    # lesion), 1.0 is the historical path and bit-identical (the branch is not
    # taken). Folded into W before homeostatic balancing, like the receptor
    # classes, so Metal reads it from W.data with no kernel change.
    commissural_gain: float = 1.0
    # Which crossing synapses the gain touches, by the presynaptic sign:
    # "all" (historical), "inh" (crossing inhibition only) or "exc"
    # (crossing excitation only). CD1-QN splits the CD1-QM dose by sign.
    commissural_sign: str = "all"
    # INTERSEGMENTAL GAIN (2026-09-04, lane A, CD1-RO). One scalar on every
    # synapse from a `ventral_nerve_cord_intrinsic` neuron whose soma
    # neuromere (T1, T2 or T3) differs from its target's (measured on the
    # edge list: 274,868 edges, 1.05M synapses, 18% of VNC-intrinsic output
    # inside the leg neuromeres). Exists to test one hypothesis: CD1-RN found
    # the six leg levator pools locked in phase at 5.8-6.5 Hz with the body
    # feedback held flat, so the lock is the cord's; cutting the between-
    # segment wiring says whether each segment carries its own oscillator
    # (the lock is the coupling's) or the rhythm is one intersegmental loop.
    # 0.0 silences every such synapse (a lesion), 1.0 is the historical path
    # and bit-identical (the branch is not taken). Folded into W like the
    # commissural gain, so Metal reads it from W.data with no kernel change.
    intersegmental_gain: float = 1.0
    # "all" (historical), "inh" or "exc", by the presynaptic sign.
    intersegmental_sign: str = "all"
    # CD1-RP (2026-09-04): the two halves of the same class under separate
    # gains, so the between-segment excitation (which CD1-RO found carries
    # the in-phase lock) can be weakened while the between-segment
    # inhibition is raised in one network. Each multiplies its half on top
    # of intersegmental_gain; 1.0 is bit-identical (branch not taken).
    intersegmental_gain_exc: float = 1.0
    intersegmental_gain_inh: float = 1.0
    # ROW-PAIR SEAMS (2026-09-04, lane A, CD1-TF). A gain on the edges from
    # one stated set of rows to another, applied after the balance step like
    # the commissural and intersegmental gains, so the balance rescale does
    # not return the target's inhibition to its budget (CD1-TE raised the
    # 13A<->19A reciprocal edges twentyfold in the edge frame, pre-balance,
    # and the sets stayed in phase, PLV 0.89). One tuple per seam:
    # (pre_rows, post_rows, gain). () is bit-identical (branch not taken).
    # Folded into W, so Metal reads it with no kernel change.
    row_pair_seams: tuple = ()
    # TYPE-OUTGOING GAIN (2026-09-08, lane B). One post-balance scale on
    # every outgoing synapse of named cell types, with optional per-id
    # overrides for exceptional cells. None is the historical path and
    # bit-identical (the branch is not taken). Applied after balance and
    # after row_pair_seams so a type gain is an acute perturbation, not a
    # homeostatic no-op. Folded into W, so Metal reads it with no kernel
    # change. A per-neuron vector of outgoing gains; unnamed rows stay 1.0.
    type_out_gain: object = None
    # DESCENDING LEFT-RIGHT NORMALISATION (2026-09-03, lane A, CD1-QY). BANC
    # delivers 4-5% more descending synapses onto the left T1 and T2 leg
    # neuromeres and 7% more onto the right T3 (measured on the edge list:
    # T1 228,767 left against 219,510 right, T2 177,486 against 168,497, T3
    # 71,526 against 76,711), and the champion's cord fires with exactly that
    # sign pattern (T1 and T2 left, T3 right, 9 of 9 runs on qg1 and qh1),
    # with the right hind stepping 3.3 Hz faster and the path turning one way.
    # The target is the average fly, which is bilaterally symmetric, so when
    # this is on the descending edges onto each leg neuromere side are scaled
    # so both sides receive the same total descending weight, the neuromere's
    # total conserved (six factors derived from the graph, none searched).
    # False is the historical path and bit-identical (the branch is not
    # taken). Folded into W before homeostatic balancing, so Metal reads it
    # from W.data with no kernel change.
    desc_lr_normalise: bool = False
    # Optional per-neuron form of the same presynaptic gain. This exists so
    # biologically distinct premotor hemilineages need not share one scalar.
    # None is the historical path and therefore bit-identical.
    premotor_inh_gain_per_neuron: object = None
    # Optional all-sign presynaptic strength by neuron. Cell types such as
    # cholinergic 19B premotor neurons need their own searchable output
    # strength. None is the historical path; this is folded into W before
    # homeostatic input balancing and therefore reaches NumPy and Metal alike.
    presyn_gain_per_neuron: object = None

    # PER-NEURON E/I BALANCING (homeostatic synaptic scaling). 0 = off, and
    # off is bit-identical to before.
    #
    # WHY, measured 2026-08-08. The network is bistable -- silent, or latched
    # at ceiling, nothing between -- and the arithmetic says why: a neuron has
    # ~224 incoming synapses, so ~62 mV of input at w_syn = 0.275 against a
    # 7 mV threshold gap. It needs only ~11% of its inputs to coincide, and
    # once anything fires the recurrent input is an order of magnitude over
    # threshold. Adaptation did not fix it (24 points). Spreading resting
    # potentials did not fix it (moved the cliff from w_syn 0.255 to 0.175,
    # did not broaden it).
    #
    # The classic fix is the BALANCED STATE: excitation and inhibition both
    # large and nearly cancelling, so the mean net input is ~0 and firing is
    # driven by fluctuations. Then the gain is self-limiting.
    #
    # WHY IT HAS TO BE PER NEURON. Measured: the fly connectome is already
    # nearly balanced ON AVERAGE (median E/I of incoming weight = 1.13,
    # geometric mean 1.09 -- which is a striking fact about the animal), but
    # the spread is 2.59x per standard deviation and 90% of neurons span a
    # factor of 19.5. At one global w_inh_scale only 25.7% of neurons are
    # within 25% of balance. And the spread is mostly WITHIN cell classes
    # (7.4x) rather than between them (4.0x), so per-class gains cannot fix it
    # either -- which corrects the recommendation written into AGENTS.md.
    #
    # So: scale each neuron's incoming inhibition by whatever IT needs to
    # reach E_i / I_i = balance_target. That is one parameter, applied per
    # neuron, and it is what homeostatic synaptic scaling actually does in
    # real nervous systems.
    balance_target: float = 0.0
    # Neurons with little or no inhibitory input would need an unbounded
    # scale. 3,338 neurons have no inhibitory input at all. Cap it and report
    # how many hit the cap rather than silently creating monsters.
    balance_max_scale: float = 20.0
    # The balance step above sets EVERY neuron to the same E/I ratio, which is
    # a universal constant where the animal varies. What it does not do is
    # flatten a neuron's input mix: the scale is one factor per postsynaptic
    # neuron, so the relative weights of that neuron's inhibitory sources are
    # untouched. The homogenisation is therefore of OPERATING POINT across
    # neurons, and identical operating points are one of the standard ways a
    # recurrent network ends up synchronised. That matters here because the
    # champion's cord runs at one phase (six registry rows, 2026-09-04) while
    # its wiring separates the antagonist motor pools about fifteen to one
    # (lane A CD1-UH). A sweep of balance_target alone cannot test this: every
    # setting still gives every neuron the SAME ratio. balance_spread draws
    # each neuron's target from a lognormal around balance_target instead, so
    # the mechanism keeps the 2026-08-09 cure for bistability and restores the
    # heterogeneity. 0.0 = every target equal = bit-identical. The draw uses a
    # fixed stream, not the run seed, because the heterogeneity is a property
    # of the model and must be the same fly on every seed. Judge, 2026-09-04.
    balance_spread: float = 0.0
    balance_spread_stream: int = 20260904
    # 2026-09-05, judge, on lane C's C-N36 and the arithmetic behind it: the
    # balance step holds every target's post-balance E/I ratio at
    # balance_target BY CONSTRUCTION, so cutting a target's excitatory input
    # scales its inhibition down in proportion and the ratio does not move at
    # all. Measured directly: removing 47% of excitatory input drops scaled
    # inhibition 46.0%, leaves the post-balance ratio identical to 1e-9 on
    # 96.1% of targets, and changes only the magnitude (net drive -45.2%).
    # Lane C measured the behavioural consequence on the vision cells: cutting
    # 46-48% of T4/T5's input synapses moves their firing 1.2% WITH this
    # homeostasis and 6.7% without, a 5.6-fold suppression of the ablation.
    # So an ablation run with balance recomputed measures the homeostasis as
    # much as the ablation. balance_scale_override takes the INTACT network's
    # scale vector so the cut is not silently compensated. None = recompute =
    # bit-identical.
    balance_scale_override: object = None
    # 2026-09-05, lane E R360-R362, ported by the judge under Robin's 10:45
    # goal. T4a's raw E/I is 1.3271 and the body falls at bal_target 0.85, so
    # a global relaxation cannot return the image's drive onto T4 to net
    # excitatory (vision.t4_image_drive_sign). The remaining route is to
    # exempt the cells the rule wrongs while the cord keeps it. balance_exempt
    # is a boolean mask, length N: those rows keep their raw E/I ratio (scale
    # exactly 1.0), every other row is balanced as before. R362 verified that
    # every synapse onto an exempted cell then carries the same weight as at
    # bal_target 0, identical into the scope and identical to the balanced
    # build outside it, over 108,764 optic-lobe rows. None = nothing exempt =
    # BIT-IDENTICAL.
    balance_exempt: object = None
    # balance_by_type (lane E R339-R342, ported by the judge 2026-09-05 as a
    # default-off SEARCH dial): blends each neuron's target from the flat
    # balance_target toward its own exact type's measured median E/I ratio:
    #     target_i = balance_target * exp(balance_by_type * centred_log_i)
    # 0.0 = flat = BIT-IDENTICAL. 1.0 = the type's own measured ratio. R339:
    # 72.1% of the log ratio's variance lies between exact cell types against
    # a 17.0% permutation null, so one constant flattens a cell-type property
    # on 159,472 neurons. The offset is chosen on the SCALE so the network's
    # mean inhibitory scale stays where the flat target put it (balance step
    # below). Measured on wa1 to cost qualifying linearly in the blend
    # (R341/R342: net -4 of 24 at 0.25, -8 at 0.50), so no wa1 theta carries
    # it; it is on main so a body can be searched with it (parameter rule).
    balance_by_type: float = 0.0
    # Centred log of each neuron's exact-type median E/I ratio, length N.
    # Supplied by the caller (walk_search computes it from the same edge list
    # the network is built from); None with balance_by_type > 0 is an error
    # rather than a silent zero.
    balance_type_log_ratio: object = None

    # ---------------------------------------------------------- GAP JUNCTIONS
    # ELECTRICAL synapses. Added 2026-08-10 because the giant-fibre escape
    # falsified the chemical-only model on three independent numbers at once:
    # a single GF spike left TTMn 5.2x SUBTHRESHOLD so nothing fired at all;
    # forcing it to fire needed w_syn 2.61, which latches the rest of the
    # network; and when it did fire, DLM led TTM by 0.7-2.8 ms where the animal
    # has TTM leading by 0.51. No delay fixes an inverted order.
    #
    # The real circuit is electrical: ShakB RNAi removes ~90-100% of the
    # JO->GF connection, GF->TTMn is only 0.50-0.82% of TTMn's chemical input,
    # and the published biophysical model needed 135 uS of gap-junction
    # conductance to reproduce the measured latencies. The BANC synapse counts
    # this model runs on are the MINORITY component of that connection.
    #
    # A gap junction is a resistor between two membranes, so it is CONTINUOUS
    # (no spike required), BIDIRECTIONAL, and essentially INSTANT -- none of
    # which the chemical path can express:
    #     I_gap(i) = g_gap * sum_j [ (v_j - v_i) ]  over gap partners j
    # which is a graph Laplacian applied to v. It costs one extra sparse
    # matvec per tick over the gap graph ONLY, which is tiny.
    #
    # gap_pairs: iterable of (i, j) row indices, undirected -- each pair is
    # added in both directions. Empty by default, so every existing result is
    # bit-identical. `gap_pair_weights`, when present, is one dimensionless
    # conductance per pair. It exists because imposing one electrical coupling
    # on every molecularly and anatomically distinct junction is the same
    # universal-parameter mistake as one threshold or delay for every neuron.
    # Historical scalar-g_gap runs remain exactly replayable, but a caller may
    # not combine the scalar and vector paths.
    gap_pairs: tuple = ()
    # COINCIDENCE NONLINEARITY (lane C, 2026-09-04, under the realism
    # directive's "a coincidence nonlinearity for T4/T5"). Designated target
    # cells receive an extra current proportional to the PRODUCT of two
    # source pools, one of them delayed: the Reichardt/Hassenstein form that
    # a linearly summing point neuron cannot express. C-M73/C-M74 measured
    # offline that this product carries the a/b direction signature in T5 on
    # the male's own wiring (12 of 12 arms) and its label-permuted null does
    # not (0 of 12). Empty targets = off = bit-identical.
    coincidence_targets: object = None    # (n,) target rows, or None
    coincidence_fast_rows: object = None  # (m,) undelayed-arm source rows
    coincidence_slow_rows: object = None  # (k,) delayed-arm source rows
    coincidence_gain: float = 0.0         # 0 = off
    coincidence_lag_ms: float = 60.0      # delay applied to the slow pool
    # C-M75 (2026-09-04) measured that multiplying instantaneous per-tick
    # spike vectors gives a product that is zero on almost every tick, so
    # the term vanished and T5 firing did not move. A Reichardt detector
    # multiplies FILTERED signals; the offline calculation supplied that
    # filter implicitly by binning at 5 ms. Each pool is low-passed with
    # this time constant before the product is taken.
    coincidence_tau_ms: float = 5.0
    g_gap: float = 0.0           # dimensionless coupling per pair, subthreshold
    gap_pair_weights: tuple = () # optional heterogeneous conductance per pair
    # RECTIFYING JUNCTIONS (longevity lane, 2026-09-12). The giant-fibre to
    # TTMn and PSI junctions pass current anterogradely only (ShakB(N+16) on
    # the GF, ShakB(L) on the motor side: Phelan 2008 Curr Biol, measured).
    # With the symmetric Laplacian every pair also LOADS the presynaptic cell
    # with the partner's leak, so a GF with five partners at 30x leak could not
    # be charged by its visual input (measured 2026-09-12: weakening the
    # junction x0.25 raised escape probability 0.83 -> 1.00). When True, each
    # pair (pre, post) couples post to pre and pre sees nothing. False =
    # bit-identical.
    gap_rectify: bool = False

    # ⚠️ THE SPIKELET, AND WHY THE RESISTOR ALONE IS NOT ENOUGH. A gap junction
    # transmits the presynaptic VOLTAGE WAVEFORM. A LIF spike has no waveform:
    # the model detects threshold crossing and immediately resets v BELOW rest.
    # So a pure resistive coupling transmits the reset -- a HYPERPOLARISATION --
    # and the partner is pushed away from firing. Measured here: with g_gap
    # swept 0.5 to 20 on the GF->TTMn pairs, TTMn never fired at all, which is
    # the resistor working correctly on a spike that is upside down.
    #
    # The standard fix is a SPIKELET: on firing, deliver a fixed depolarising
    # pulse to every gap-coupled partner, instantly. Physically it is the
    # coupling coefficient times the real spike amplitude -- c ~ 0.1-0.3 and a
    # ~60 mV action potential give ~6-18 mV, against this model's 7 mV
    # threshold gap, so the scale is right for the giant fibre to fire TTMn on
    # one spike the way the animal does.
    gap_spikelet: float = 0.0    # mV delivered to each partner on a spike
    # gap_delay_ms (longevity lane, 2026-09-12): conduction time from the
    # spiking cell to its electrical partners, in ms. The junction itself is
    # instantaneous; the axon is not. The giant fibre's own axon time is
    # MEASURED at 0.29 ms in adults (Kadas 2019, 0.6 mm at 2.07 m/s, from
    # GF-TTM minus TTMn-TTM latency), and the spikelet delivered in one
    # timestep (0.10 ms) is why TTMn fired 3x early on 2026-08-10. One value
    # for every gap pair; 0 keeps the instantaneous path and is bit-identical.
    gap_delay_ms: float = 0.0
    # gap_ap_mv / gap_ap_ms (longevity lane, 2026-09-12, Robin: "make it as
    # realistic as possible"): the presynaptic ACTION POTENTIAL as seen through
    # the junction. A LIF spike has no waveform (it resets below rest), which
    # is why the spikelet above exists. Here the firing cell's electrical
    # partners see, after gap_delay_ms, a brief triangular waveform (rise one
    # third, fall two thirds of gap_ap_ms) of amplitude gap_ap_mv added to the
    # cell's voltage INSIDE the existing conductance coupling term, so the
    # postsynaptic response rises through the target's own membrane time
    # constant and a weaker junction gives a later spike, as in the animal.
    # Amplitude 60 mV assumed from the measured 50-60 mV motor-neuron spikes
    # (Fayyazuddin 2006); duration 0.6 ms measured on the giant fibre
    # (Blagburn 2020). 0 = off, bit-identical; needs gap pairs and coupling.
    gap_ap_mv: float = 0.0
    gap_ap_ms: float = 0.6

    # ------------------------------------------------------- GRADED NEURONS
    # NON-SPIKING cells. A large part of the fly brain does not fire action
    # potentials at all -- photoreceptors and lamina monopolar cells certainly,
    # and much of the medulla -- and signals with GRADED potentials instead.
    # BANC has 1,846 photoreceptor_neuron and 6,539 lamina_monopolar inside a
    # 72,947-cell optic_lobe_intrinsic super_class. Forcing all of them through
    # an integrate-and-fire threshold is a model-class error, not a parameter
    # one: a cell that never reaches threshold is recorded as SILENT when the
    # animal's equivalent is continuously active and transmitting.
    #
    # A graded neuron is SIMPLER than LIF, not harder. Same membrane equation,
    # but no threshold, no reset and no refractory period -- the output is a
    # continuous release that rides on the voltage:
    #
    #     release = 1 / (1 + exp(-(v - graded_v50) / graded_slope))
    #
    # and every tick it injects release * w into its targets, where a spiking
    # cell would inject w only at spike times. graded_gain sets the exchange
    # rate: what continuous release is worth against one spike, so that a fully
    # depolarised graded cell delivers about what a spiking cell at its maximum
    # rate would.
    #
    # graded_rows empty by default, so every existing result is bit-identical.
    graded_rows: tuple = ()
    graded_gain: object = 0.05   # release units per tick, vs 1.0 for a spike
    # Scalars preserve historical replay. All three also accept an (N,) vector
    # so evidence-backed graded cells need not share a universal release
    # curve or a universal exchange rate: a set added beside a champion's set
    # keeps the champion's gain on the champion's rows (lane A, 2026-09-02).
    graded_v50: object = -48.0   # mV at half-maximal release
    graded_slope: object = 2.5   # mV; smaller = more switch-like
    graded_parameter_mode: str = "scalar"

    # ⚠️ "CONTINUOUS" TAKEN LITERALLY IS 884x MORE WORK THAN SPIKING. Measured:
    # 32,558 graded cells own 1,636,056 outgoing edges (12% of the graph), and
    # updating them every 0.1 ms tick is 16.4 BILLION propagation events per
    # simulated second, against 18.5 million for the spiking path -- a 4.3x
    # whole-model slowdown for +11% more spikes. But tau_mem is 20 ms, so a
    # graded cell's voltage physically CANNOT change fast: over 0.9 ms it moves
    # only 4.4% of the way to its target. Recomputing release every tick is
    # therefore mostly redundant arithmetic, not fidelity.
    graded_every: int = 1        # ticks between graded release updates
    # High-pass on graded release (default 0 = bit-identical). Continuous
    # analog wait writes neighbor turns and also a DC brake. This is the
    # same formula body.Proprioceptors uses for FeCO phasic: subtract a
    # running baseline and re-amplify the residual. Frozen binary tests
    # use 1.0; do not sweep it as a gain.
    graded_phasic: float = 0.0
    graded_phasic_alpha: float = 0.01  # ~250 ms at 2.5 ms ticks

    # ------------------------------------------- INPUT RESISTANCE, PER NEURON
    # THE MEASURED FIX FOR UNNORMALISED EXCITATION. Every neuron here is given
    # the same membrane resistance, so a cell with few incoming synapses simply
    # cannot be driven: measured, the median neuron needs 51 Hz sustained from
    # EVERY input to reach threshold, against a real fly's 1-10 Hz, and 91.5%
    # of neurons need inputs faster than any real neuron fires.
    #
    # The animal solves this with resistance, not with normalisation. A PSP is
    # I_syn * R_in, so a small, sparsely-innervated cell turns the same current
    # into a bigger voltage. Measured: 598 +/- 69 MOhm for central neurons
    # (Gouwens & Wilson 2009, n = 14) and 150 / 300 / 700 MOhm across fast /
    # intermediate / slow leg motor neurons (Azevedo 2020) -- and the fast ones
    # are the LARGEST cells. R falls with surface area, and surface area is
    # roughly what synapse count measures.
    #
    #     r_scale_i = (median_synapses / synapses_i) ** r_in_alpha
    #
    # alpha = 0 is today's model (one R for everyone); alpha = 1 is full
    # normalisation (every neuron needs the same presynaptic rate). The truth
    # is in between and it is a SEARCHABLE parameter, per the parameter rule --
    # not an arbitrary rescaling, because the quantity it stands for is
    # measured and its direction is not in doubt.
    r_in_alpha: float = 0.0
    r_in_max: float = 8.0        # cap, so a 1-synapse cell is not infinite
    # Per-neuron input-resistance OVERRIDE, dict banc_id -> multiplicative
    # factor on incoming weights, applied on top of r_in_alpha. Exists for
    # the measured MN size-principle gradient (150-700 MOhm across each
    # motor pool, Azevedo 2020) -- the global alpha=0.10 gives only ~1.5x
    # over the pool where the fly has ~4.7x, and the 2026-08-13 measurement
    # showed the pool INVERTED as a result (CONSTRAINTS.md). None = off,
    # bit-identical.
    r_in_override: dict | None = None
    # Per-neuron RESTING POTENTIAL override, dict banc_id -> mV. The second
    # half of the measured MN excitability gradient: slow MNs rest at -48 mV
    # (3 mV below the model threshold -- tonic-prone), fast at -68 (Azevedo
    # 2020). None = off, bit-identical.
    v_rest_override: dict | None = None

    # ------------------------------------- SHORT-TERM SYNAPTIC DEPRESSION
    # THE BRAKE THE SINGLE NEURON CANNOT PROVIDE. Every chemical synapse
    # releases vesicles from a finite readily-releasable pool; firing depletes
    # it and recovery takes hundreds of ms. Tsodyks & Markram 1997; Abbott et
    # al. 1997 ("synaptic depression and cortical gain control"); measured
    # directly in the fly at the ORN->PN synapse (Kazama & Wilson 2008), which
    # depresses strongly. The functional consequence is automatic loop-gain
    # control: effective synaptic weight falls as ~1/rate at high presynaptic
    # rates, so recurrent amplification cools exactly when the network heats.
    # This is the mechanism the stabilisation literature credits for turning
    # the dead/hot knife-edge into a plateau -- adaptation brakes the NEURON,
    # depression brakes the LOOP.
    #
    # Implementation is exact and cheap because U and tau_rec are uniform per
    # presynaptic neuron: every outgoing synapse of neuron j shares j's spike
    # history, so the resource x_j is ONE float per neuron, not one per
    # synapse. On j firing: delivered weight scales by x_j, then
    # x_j *= (1 - U). Between spikes x recovers toward 1 with tau_rec.
    # std_U = 0 is off and bit-identical.
    std_U: float = 0.0           # release fraction per spike (0 = off)
    std_tau_rec: float = 400.0   # ms, recovery of the vesicle pool

    # CONDUCTANCE-BASED SYNAPSES. Off by default so nothing already measured
    # changes silently; this is a different model, not a parameter tweak.
    #
    # WHY IT EXISTS. The current-based default subtracts a fixed amount per
    # inhibitory spike regardless of membrane potential, so inhibition is
    # UNBOUNDED. Measured 2026-08-08: 10% of neurons are driven below -70 mV,
    # 5.6% below -90 mV, worst case -169.8 mV. Real neurons essentially never
    # go below about -90 mV, because inhibition opens chloride channels that
    # pull voltage toward a reversal potential near -70 mV and cannot push
    # past it -- the driving force g*(V-E) goes to zero as V approaches E.
    #
    # WHY IT MAY MATTER FOR RHYTHM. The classic CPG mechanism is POST-
    # INHIBITORY REBOUND: a cell is inhibited, then fires as soon as the
    # inhibition lifts. A neuron buried at -170 mV needs ~125 mV of excitation
    # to reach threshold instead of 7 mV, so it cannot rebound at all. If the
    # E-E-I ring is failing to oscillate here, this is a prime suspect.
    conductance_based: bool = False
    # CB3: track excitatory and inhibitory conductances separately so
    # simultaneous E and I SHUNT instead of cancelling. False keeps the
    # older net-signed behaviour (bounded voltage, no shunting).
    split_conductances: bool = False
    # Numerical integration of the conductance membrane equation. ``euler``
    # preserves every historical trajectory. ``analytic`` holds incoming
    # conductances constant for one dt and integrates that linear ODE exactly;
    # it cannot acquire Euler's large-conductance sign-flip instability.
    conductance_integrator: str = "euler"
    e_exc: float = 0.0           # mV, cation reversal (ACh receptors)
    # Optional element-scoped tonic excitatory conductance, relative to the
    # leak conductance.  This is a membrane mechanism, not forced firing: the
    # row still integrates inhibition, adaptation, threshold, reset and
    # refractory state normally.  None is the historical path and is
    # bit-identical.  A vector is required because imposing one tonic drive on
    # every neuron would repeat the universal-parameter mistake this model's
    # per-neuron threshold, delay and input-resistance seams were built to
    # avoid.  NumPy conductance dynamics only until explicitly ported.
    tonic_exc_conductance_per_neuron: object = None
    # Scalar or postsynaptic (N,) vector. Adult fly measurements reject a
    # universal chloride reversal; the scalar default preserves old runs.
    e_inh: object = -70.0         # mV, chloride reversal (GABA-A, GluCl)
    # Evidence-scoped presynaptic shunting.  Externally driven sensory spikes
    # historically bypass terminal inhibition and always release full weight.
    # When rows are supplied, their already-modelled inhibitory conductance
    # attenuates outgoing chemical release by the parameter-free shunt factor
    # 1/(1+gi).  Empty by default and therefore bit-identical.
    presynaptic_inhibition_rows: object = None

    # ADAPTATION (AdEx / adaptive-LIF). One extra state variable per neuron.
    # Off by default -- adapt_a = adapt_b = 0 reduces the update EXACTLY to the
    # LIF above, so nothing already measured changes.
    #
    # WHY IT EXISTS. Plain LIF on a constant input fires at a perfectly
    # constant rate forever, and does nothing special when inhibition is
    # released. Measured 2026-08-08 in figure_neuron_models.py: LIF's last
    # inter-spike interval is 1.00x its first (no adaptation at all), and it
    # emits 0 spikes on release from inhibition. Real fly neurons do both --
    # and both are exactly the mechanisms the CPG literature names as rhythm
    # generators. w is a slow current that builds while the cell is
    # depolarised and jumps by adapt_b on every spike:
    #
    #     tau_w dw/dt = adapt_a (v - v_rest) - w      and  w += adapt_b on spike
    #     tau_m dv/dt = (v_rest - v) + I - w
    #
    # adapt_b alone gives spike-frequency adaptation: fire, accumulate a brake,
    # slow down. adapt_a alone gives post-inhibitory rebound: while a cell is
    # hyperpolarised (v < v_rest) w goes NEGATIVE, i.e. becomes a depolarising
    # current, so releasing the inhibition leaves a cell primed to fire.
    #
    # Scalars or (N,) arrays, per the parameter rule -- per-cell-class
    # adaptation is a search dimension, not a decision.
    adapt_a: object = 0.0        # dimensionless, subthreshold coupling
    adapt_b: object = 0.0        # mV, spike-triggered increment
    tau_w: object = 100.0        # ms, adaptation decay
    adaptation_parameter_mode: str = "scalar"

    # R135: receptor-scoped VNC serotonin walking path. Empty populations are
    # the historical path. When populated, source spikes charge one slow
    # population state (an EMA of mean source firing in Hz); only the selected
    # 5-HT7 target rows receive gain * state as membrane drive. The populations
    # are evidence-map memberships, not universal neuron classes, and neither
    # firing threshold nor presynaptic delay is changed.
    serotonin_source_rows: object = None
    serotonin_target_rows: object = None
    serotonin_tau_ms: float = 3000.0
    serotonin_gain_mv_per_hz: float = 0.0
    # Exponential spike initiation (the "Ex" in AdEx). 0 keeps the hard
    # threshold, which is much cheaper -- an exp() over 188,508 neurons every
    # step is the dominant cost when it is on. It sharpens spike onset; it is
    # not the part that generates rhythm, so it defaults off.
    delta_T: float = 0.0         # mV, slope factor
    v_peak: float = -30.0        # mV, cutoff when delta_T > 0

    dt: float = 0.1              # ms
    # Engine backend. "numpy" is the canonical path. "metal" routes run() to
    # cns_metal.run_metal(): the same per-tick arithmetic on the laptop GPU,
    # drive draws and bookkeeping still done here in Python. Default off,
    # bit-identical when off; see cns_metal.py for what it refuses.
    backend: str = "numpy"

    # --- sign assignment ---
    nt_sign: dict = field(default_factory=_default_nt_sign)
    # HYPOTHESIS, not a measurement: what to do with the 34,455 neurons that
    # have neither a verified nor a predicted transmitter. Acetylcholine is
    # ~55% of the annotated population so +1 is the maximum-likelihood guess,
    # and w_unknown_scale lets a search discount them without removing them.
    sign_unknown: float = +1.0
    w_unknown_scale: float = 1.0

    # Q3bm (2026-08-27, prospectively registered): target-specific signed
    # fast-output hypothesis for exact zero-fast ordered pairs. Membership is
    # expressed in stable BANC IDs, never row positions. Empty membership is
    # the default and is bit-identical. The signed gain is applied after the
    # incumbent balance and input-resistance transforms, so testing +1/-1
    # changes only the selected stored weights and does not silently rebalance
    # other inputs. This is an arbitrary model axis, not biological transmitter
    # or receptor evidence.
    fast_edge_override_ids: tuple = ()
    fast_edge_override_gain: float = 0.0
    # Q3bv source-wide analogue of the exact-pair override above. Membership
    # is by canonical BANC ID and is empty by default.  It is deliberately an
    # arbitrary model axis: the gain is not a transmitter or receptor claim.
    fast_source_override_ids: tuple = ()
    fast_source_override_gain: float = 0.0
    # C-V68: normalize a nominated set of EXISTING graded excitatory edges
    # against the target cells' conductance equations. Membership is in stable
    # BANC-ID pairs and empty is the historical, bit-identical path. There is
    # deliberately no gain dial: one derived per-contact efficacy makes the
    # population's mean steady conductance at saturated graded release equal
    # the summed conductance required to move the isolated targets from their
    # own rests to their own thresholds. Applying it to raw EM counts preserves
    # every within- and cross-target count ratio. This is a BUILD hypothesis for
    # unknown synaptic
    # efficacy, not evidence that EM synapse counts imply that normalization.
    graded_edge_equilibrium_ids: tuple = ()

    # NORMALISED SYNAPTIC KERNEL. Off by default; True changes no result at
    # tau_syn = 5 by construction.
    #
    # WHY. A spike adds w_syn to g, g decays with tau_syn, and v integrates g.
    # So the total voltage one spike delivers is w_syn * tau_syn / tau_mem --
    # tau_syn sets the synapse's STRENGTH as well as its timing. Measured
    # 2026-08-08: one spike moves the membrane 0.062 mV at tau_syn = 2 and
    # 0.752 mV at tau_syn = 25, a 12x range across a sweep that was supposed
    # to be about timing alone.
    #
    # That confound is not merely untidy, it broke an experiment: sweeping
    # tau_syn to test the frequency law gave R2 = 0.69 where sweeping delay --
    # which really is pure timing -- gave R2 = 0.98. With this on, w_syn is
    # divided by tau_syn/syn_ref_tau, so the kernel's AREA is held fixed and
    # tau_syn becomes the timing parameter it was meant to be.
    #
    # Both conventions are defensible and the literature uses both, so this is
    # a parameter rather than a correction. Which one is right is an empirical
    # question about whether a slower synapse in the fly also delivers more
    # charge, and nobody has told us the answer.
    syn_normalised: bool = False
    syn_ref_tau: float = 5.0     # ms; the tau_syn at which gain is exactly 1

    # Prefer verified transmitter over predicted where both exist.
    prefer_verified: bool = True

    # MEASURED OVERRIDE, not a fit. BANC has no verified transmitter for any
    # of the 391 leg motor neurons and its predictor calls 260 of them GABA.
    # Independent ground truth says that is wrong for all of them: all leg MNs
    # are glutamatergic (Sustar et al. 2026), and Drosophila has no inhibitory
    # leg motor neurons at all — holometabolous insects lost the common
    # inhibitors that locusts and stick insects have (Azevedo et al. 2024).
    # Left off, 271 of 391 motor neurons simulate as inhibitory. See
    # nt_check.py, which also shows the predictor is 92.4% accurate CNS-wide,
    # so this is a class-specific failure and not a reason to distrust it
    # elsewhere.
    mn_transmitter_override: bool = True
    # Motor neurons release glutamate, but at the NMJ that is EXCITATORY
    # (GluRIIA/B), whereas this model's `glutamate: -1` encodes the central
    # GluCl action. Their muscle target is outside the volume, so what needs a
    # sign here is only their central output — 0.4% of the input to leg motor
    # neurons. Sign genuinely unknown, hence a free parameter rather than a
    # silent choice.
    mn_cns_output_sign: float = +1.0

    # CITATION-GATED PER-TYPE TRANSMITTER OVERRIDES (judge, 2026-09-03, from
    # lane C's C-A9 and C-V90: BANC verifies acetylcholine on one LPi14 row
    # against its own gaba prediction and its four gaba-predicted siblings,
    # and that row carries 4.9% of all input to the six HS cells, so the
    # fly's null-direction inhibition onto HS arrives as excitation). The
    # motor-neuron override above is the same kind of correction hard-wired
    # for one class; this is the general seam. Keys are BANC cell_type
    # tokens, values a transmitter token from nt_sign; applied after
    # verified/predicted resolution and before the motor override. EMPTY =
    # bit-identical. Membership lives in transmitter_map.json with a
    # citation per type; a theta enables it with transmitter_map: "map".
    transmitter_map: dict = field(default_factory=dict)
    # Citation-gated per-ROW cell-type patch (judge, 2026-09-05 20:30): BANC
    # rows whose cell_type is blank but whose identity is established by
    # evidence (the first candidate: the CT1-shaped left giant
    # 720575941481834208, output-profile cosine 0.931 to the typed left CT1,
    # lane C C-N65 and the judge's read). Keys are banc_888_id strings,
    # values {"cell_type": token, "neurotransmitter": token or absent}.
    # Applied to a COPY of meta at build time before transmitter resolution,
    # so the verified/predicted rule and the per-type map then act on the
    # patched row. EMPTY = bit-identical (lanes/A/probe.py). Membership
    # lives in cell_type_patch.json with a citation per row; a theta enables
    # it with cell_type_patch: "patch".
    cell_type_patch: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# building the network
# --------------------------------------------------------------------------

def _resolve_nt(meta, p):
    """Per-neuron transmitter string, verified preferred over predicted,
    then the citation-gated per-type map (SimParams.transmitter_map)."""
    ver = meta["neurotransmitter_verified"]
    pred = meta["neurotransmitter_predicted"]
    nt = ver.where(ver.notna(), pred) if p.prefer_verified else pred
    if p.transmitter_map:
        ct = meta["cell_type"].astype(object)
        for cell_type, token in p.transmitter_map.items():
            if token not in p.nt_sign:
                raise ValueError(
                    f"transmitter_map[{cell_type!r}] = {token!r} is not a "
                    f"transmitter token in nt_sign")
            nt = nt.where(ct != cell_type, token)
    return nt


# Tokens that assert ignorance rather than naming a transmitter. Anything
# here goes to the unknown path; anything else unrecognised is muted at 0.0
# and reported. Extend deliberately, never silently.
_ANNOUNCED = []          # one selection line per process, not per build

NT_MEANS_UNKNOWN = {"unclear", "unknown", "not_known", "none", "na", "nan",
                    "undefined", "unassigned", "ambiguous"}


_RECEPTOR_CLASS_KEYS = {"pre_nt", "post_cell_class", "sign", "gain"}


def _receptor_class_multiplier(classes, pre, post, nt_series, cell_class):
    """Per-edge float32 multiplier from the receptor-class table.

    Returns (multiplier, audit). Entries are applied in order and compose
    multiplicatively where they overlap. Refuses any key outside the four
    understood ones, so a future "tau_ms" cannot be silently ignored."""
    nt_tokens = [set(t.strip() for t in s.split(",")) if s else set()
                 for s in nt_series.fillna("").to_numpy()]
    mult = np.ones(len(pre), dtype=np.float32)
    audit = []
    for entry in classes:
        extra = set(entry) - _RECEPTOR_CLASS_KEYS
        if extra:
            raise ValueError(f"receptor class has unsupported keys {sorted(extra)}")
        sgn = float(entry.get("sign", 1.0))
        gn = float(entry.get("gain", 1.0))
        if sgn not in (1.0, -1.0):
            raise ValueError("receptor class sign must be +1 or -1 (relative to nt_sign)")
        if not np.isfinite(gn) or gn <= 0.0:
            raise ValueError("receptor class gain must be finite and > 0")
        pre_nt = str(entry.get("pre_nt", "*"))
        pcc = str(entry["post_cell_class"])
        post_mask = cell_class[post] == pcc
        if pre_nt == "*":
            pre_mask = np.ones(len(pre), dtype=bool)
        else:
            has = np.fromiter((pre_nt in t for t in nt_tokens), dtype=bool,
                              count=len(nt_tokens))
            pre_mask = has[pre]
        m = post_mask & pre_mask
        mult[m] *= np.float32(sgn * gn)
        audit.append({"pre_nt": pre_nt, "post_cell_class": pcc,
                      "sign": sgn, "gain": gn, "raw_edges": int(m.sum())})
    return mult, audit


def _sign_of(nt_series, p):
    """Signed multiplier per neuron. Multi-transmitter strings ('gaba,
    nitric_oxide') take the first entry that has a nonzero sign."""
    signs = np.full(len(nt_series), np.nan)
    vals = nt_series.fillna("").to_numpy()
    for i, s in enumerate(vals):
        if not s:
            continue
        for part in s.split(","):
            v = p.nt_sign.get(part.strip())
            if v:
                signs[i] = v
                break
        else:
            # AUDIT FIX 2026-08-19. This branch used to assign 0.0
            # unconditionally, with the comment "known transmitter, no fast
            # sign (monoamine)" -- which is only true when the token is
            # actually IN the vocabulary. It was also silently deleting every
            # neuron whose transmitter is recorded as NOT KNOWN: MANC's
            # 'unknown' (99 core rows) and MCNS's 'unclear' (1294 core rows,
            # 905 of them non-motor) had their entire output multiplied by
            # zero. "We do not know" is not "we know it is a monoamine".
            # Unrecognised tokens now take the NaN path, i.e. the searched
            # sign_unknown / w_unknown_scale hypothesis, the same treatment
            # a blank transmitter already gets. BANC is unaffected -- all 631
            # of its sign-0 core rows are real monoamines (serotonin 299,
            # octopamine 239, dopamine 93).
            # An UNRECOGNISED token is not the same thing in both
            # directions, so it is not lumped:
            #   * a token that MEANS "we do not know" ('unclear' in MCNS,
            #     1294 core rows; 'unknown' in MANC, 99) must take the NaN
            #     path -- the searched sign_unknown / w_unknown_scale
            #     hypothesis, the same treatment a blank already gets.
            #     Before this fix they were multiplied by ZERO, silently
            #     deleting 905 MCNS non-motor neurons as synaptic sources.
            #   * a token that names a REAL transmitter our vocabulary does
            #     not carry (BANC has 16 'glycine') stays muted at 0.0, as
            #     published -- calling it excitatory at full weight would be
            #     a stronger claim than the data supports.
            # Either way it is now COUNTED and announced, never silent.
            signs[i] = np.nan if s.strip().lower() in NT_MEANS_UNKNOWN else 0.0
    return signs


def _endpoint_rows(idx, ids, edges):
    """Row index of each edge's pre and post neuron, and the mask of edges
    whose endpoints are both in meta.

    Same values as ``idx.reindex(edges["pre"]).to_numpy()`` on the id
    strings, which is what this did until 2026-09-02 and cost about five of
    the constructor's eight seconds (two string joins over 13.6M rows plus
    pandas' copies around them). BANC ids are decimal strings of 64-bit
    integers, so they are cast to int64 inside Arrow and joined as integers:
    0.3 s. Verified identical on the full graph; anything that does not cast
    cleanly falls back to the string join. (compute seat, 2026-09-02)
    """
    try:
        import pyarrow as pa
        import pyarrow.compute as pc

        def as_i64(series_or_array):
            arr = getattr(getattr(series_or_array, "array", None), "_pa_array", None)
            if arr is None:
                arr = pa.array(np.asarray(series_or_array))
            return pc.cast(arr, pa.int64()).to_numpy()

        ids_i = as_i64(ids)
        index = pd.Index(ids_i)
        if not index.is_unique or len(ids_i) != len(ids):
            raise ValueError("ids not unique after cast")
        pre = index.get_indexer(as_i64(edges["pre"]))
        post = index.get_indexer(as_i64(edges["post"]))
        keep = (pre >= 0) & (post >= 0)
        return pre[keep].astype(np.int32), post[keep].astype(np.int32), keep
    except (ValueError, TypeError, ArithmeticError, ImportError) as exc:  # noqa: F841
        pass
    except Exception as exc:  # Arrow raises its own hierarchy
        if "Arrow" not in type(exc).__name__:
            raise
    pre = idx.reindex(edges["pre"].to_numpy()).to_numpy()
    post = idx.reindex(edges["post"].to_numpy()).to_numpy()
    keep = ~(np.isnan(pre) | np.isnan(post))
    return pre[keep].astype(np.int32), post[keep].astype(np.int32), keep


class CNS:
    """Sparse LIF over the whole BANC CNS."""

    def __init__(self, meta, edges, params=None):
        self.p = params or SimParams()
        p = self.p

        self.meta = meta.reset_index(drop=True)
        # cell_type_patch (judge, 2026-09-05): per-row identity corrections
        # on a copy of meta, before anything reads cell_type or transmitter.
        self.n_cell_type_patched = 0
        if p.cell_type_patch:
            self.meta = self.meta.copy()
            for col in ("cell_type", "neurotransmitter_predicted"):
                if str(self.meta[col].dtype) == "category":
                    self.meta[col] = self.meta[col].astype(object)
            ids_s = self.meta["banc_888_id"].astype(str)
            for rid, spec in p.cell_type_patch.items():
                hit = (ids_s == str(rid)).to_numpy()
                if int(hit.sum()) != 1:
                    raise ValueError(
                        f"cell_type_patch: row {rid} matches {int(hit.sum())} "
                        f"meta rows, expected exactly 1")
                if not isinstance(spec, dict) or not spec.get("cell_type"):
                    raise ValueError(f"cell_type_patch[{rid}] needs a cell_type")
                self.meta.loc[hit, "cell_type"] = str(spec["cell_type"])
                nt_tok = spec.get("neurotransmitter")
                if nt_tok:
                    if nt_tok not in p.nt_sign:
                        raise ValueError(
                            f"cell_type_patch[{rid}] neurotransmitter {nt_tok!r} "
                            f"is not a transmitter token in nt_sign")
                    self.meta.loc[hit, "neurotransmitter_predicted"] = nt_tok
                self.n_cell_type_patched += 1
        self.ids = self.meta["banc_888_id"].to_numpy()
        self.N = len(self.ids)
        idx = pd.Series(np.arange(self.N), index=self.ids)

        # Edges whose endpoints are both in the meta table. BANC's edge list
        # references a few segments the meta table does not describe; those
        # cannot be simulated because they have no transmitter, so they are
        # reported rather than silently dropped.
        pre, post, keep = _endpoint_rows(idx, self.ids, edges)
        self.n_edges_dropped = int((~keep).sum())
        cnt = edges["count"].to_numpy()[keep].astype(np.float32)

        nt = _resolve_nt(self.meta, p)
        self.nt = nt
        sign = _sign_of(nt, p)
        self.n_unknown_nt = int(np.isnan(sign).sum())

        # Receipt for the per-type transmitter map: rows it touched and rows
        # whose fast sign it changed (a type predicted gaba and mapped to
        # glutamate keeps its -1, so the two counts differ).
        self.n_transmitter_overridden = 0
        self.n_transmitter_sign_flips = 0
        if p.transmitter_map:
            from dataclasses import replace as _dc_replace
            _ct = self.meta["cell_type"].astype(object)
            self.n_transmitter_overridden = int(
                _ct.isin(list(p.transmitter_map)).sum())
            _base = _sign_of(_resolve_nt(
                self.meta, _dc_replace(p, transmitter_map={})), p)
            _a = np.where(np.isnan(_base), p.sign_unknown, _base)
            _b = np.where(np.isnan(sign), p.sign_unknown, sign)
            self.n_transmitter_sign_flips = int((_a != _b).sum())

        # Measured override for motor neurons — see SimParams.
        self.n_mn_overridden = 0
        if p.mn_transmitter_override:
            is_mn = (self.meta["super_class"] == "motor").to_numpy()
            self.n_mn_overridden = int(is_mn.sum())
            sign = np.where(is_mn, p.mn_cns_output_sign, sign)
        scale = np.where(np.isnan(sign), p.w_unknown_scale, 1.0)
        sign = np.where(np.isnan(sign), p.sign_unknown, sign)
        self.sign = sign

        gain = np.where(sign < 0, p.w_inh_scale, 1.0)
        if p.premotor_inh_rows and p.premotor_inh_gain != 1.0:
            _pm = np.zeros(self.N, dtype=bool)
            _pm[np.asarray(p.premotor_inh_rows, dtype=int)] = True
            # only their INHIBITORY output is scaled; excitatory rows in
            # the set (rare, mixed families) are left untouched
            gain = np.where(_pm & (sign < 0), gain * p.premotor_inh_gain,
                            gain)
        _pm_per_neuron = p.premotor_inh_gain_per_neuron
        if _pm_per_neuron is not None:
            _pm_per_neuron = np.asarray(_pm_per_neuron, dtype=np.float64)
            if (_pm_per_neuron.shape != (self.N,)
                    or not np.all(np.isfinite(_pm_per_neuron))
                    or np.any(_pm_per_neuron <= 0.0)):
                raise ValueError(
                    "premotor_inh_gain_per_neuron must be a finite positive "
                    "(N,) array")
            gain = np.where(sign < 0, gain * _pm_per_neuron, gain)
        _presyn_gain = p.presyn_gain_per_neuron
        if _presyn_gain is not None:
            _presyn_gain = np.asarray(_presyn_gain, dtype=np.float64)
            if (_presyn_gain.shape != (self.N,)
                    or not np.all(np.isfinite(_presyn_gain))
                    or np.any(_presyn_gain <= 0.0)):
                raise ValueError(
                    "presyn_gain_per_neuron must be a finite positive "
                    "(N,) array")
            gain = gain * _presyn_gain
        tau_syn = np.asarray(p.tau_syn, dtype=np.float64)
        if tau_syn.ndim == 0:
            if not np.isfinite(tau_syn) or float(tau_syn) <= 0.0:
                raise ValueError("tau_syn must be finite and > 0")
        elif (tau_syn.shape != (self.N,)
              or not np.all(np.isfinite(tau_syn))
              or np.any(tau_syn <= 0.0)):
            raise ValueError("tau_syn must be finite scalar or positive (N,) array")
        split_tau = (p.tau_syn_exc is not None or p.tau_syn_inh is not None)
        if split_tau:
            if (p.tau_syn_exc is None or p.tau_syn_inh is None
                    or not p.conductance_based or not p.split_conductances
                    or not p.syn_normalised):
                raise ValueError(
                    "split synaptic decay requires both tau_syn_exc/tau_syn_inh, "
                    "true split conductances, and syn_normalised=True")
            tau_exc = float(p.tau_syn_exc)
            tau_inh = float(p.tau_syn_inh)
            if (not np.isfinite(tau_exc) or tau_exc <= 0.0
                    or not np.isfinite(tau_inh) or tau_inh <= 0.0):
                raise ValueError("split synaptic decay constants must be positive")
            kernel_gain = 1.0
            edge_kernel_gain = np.where(
                sign[pre] < 0.0, p.syn_ref_tau / tau_inh,
                p.syn_ref_tau / tau_exc)
        elif p.syn_normalised:
            kernel_gain = p.syn_ref_tau / tau_syn
            edge_kernel_gain = (kernel_gain if np.ndim(kernel_gain) == 0
                                else kernel_gain[post])
        else:
            kernel_gain = 1.0
            edge_kernel_gain = 1.0
        self.kernel_gain = kernel_gain
        w = (cnt * p.w_syn * edge_kernel_gain * sign[pre] * scale[pre]
             * gain[pre])
        self.receptor_class_audit = None
        if p.receptor_classes:
            w = self._fold_receptor_classes(w, pre, post, nt)
        self.desc_lr_factors = None
        if p.desc_lr_normalise:
            _sc = self.meta["super_class"].astype(object).to_numpy()
            _side = self.meta["side"].astype(object).to_numpy()
            _nm = self.meta["neuromere"].astype(object).to_numpy()
            _desc = _sc[pre] == "descending"
            self.desc_lr_factors = {}
            for _seg in ("T1", "T2", "T3"):
                _m = {s_: _desc & (_nm[post] == _seg) & (_side[post] == s_)
                      for s_ in ("left", "right")}
                _tot = {s_: float(np.abs(w[_m[s_]]).sum()) for s_ in _m}
                if min(_tot.values()) <= 0.0:
                    raise ValueError(f"desc_lr_normalise: no descending weight onto {_seg}")
                _target = 0.5 * (_tot["left"] + _tot["right"])
                for s_ in _m:
                    _f = _target / _tot[s_]
                    w = np.where(_m[s_], w * _f, w)
                    self.desc_lr_factors[f"{_seg}.{s_}"] = _f

        # Per-neuron E/I balancing. Applied to the assembled edge weights,
        # postsynaptically -- `post` is the receiving neuron, which is the
        # side homeostatic scaling acts on.
        self.n_balance_capped = 0
        self.balance_scale = None
        if p.balance_target > 0:
            E_in = np.bincount(post, weights=np.maximum(w, 0.0),
                               minlength=self.N)
            I_in = np.bincount(post, weights=np.maximum(-w, 0.0),
                               minlength=self.N)
            _tgt = p.balance_target
            if p.balance_by_type > 0:
                if p.balance_type_log_ratio is None:
                    raise ValueError(
                        "balance_by_type > 0 needs balance_type_log_ratio")
                _lr = np.asarray(p.balance_type_log_ratio, dtype=np.float64)
                if _lr.shape != (self.N,):
                    raise ValueError(
                        f"balance_type_log_ratio must have {self.N} entries, "
                        f"got {_lr.shape}")
                if not np.all(np.isfinite(_lr)):
                    raise ValueError("balance_type_log_ratio must be finite")
                # Normalised on the SCALE, not on the target. The balance
                # step multiplies each neuron's inhibition by s = E/(I*tgt),
                # so s scales as 1/tgt and a vector centred on the target
                # still moves the network's TOTAL inhibition by Jensen: at
                # blend 0.5 the mean scale rises 6.5%. That would confound the
                # redistribution under test with a global inhibition change,
                # which is the same trap cns.py already records for
                # postbalance_presyn_gain_per_neuron. Choosing the offset so
                # mean(exp(-b*(lr-c))) = 1 leaves the mean inhibitory scale
                # exactly where the flat target put it, so this dial moves
                # WHICH neurons are inhibited and not how much inhibition the
                # network carries.
                _b = float(p.balance_by_type)
                # exp(-b(x-c)) = exp(-bx)exp(bc), so exp(bc) = 1/mean and
                # c is NEGATIVE. The first version had this sign backwards and
                # doubled the shift it meant to cancel; the test below caught it.
                _c = -np.log(np.mean(np.exp(-_b * (_lr - _lr.mean())))) / _b
                _tgt = _tgt * np.exp(_b * (_lr - _lr.mean() - _c))
            if p.balance_spread > 0:
                if not np.isfinite(p.balance_spread):
                    raise ValueError("balance_spread must be finite")
                _rng = np.random.default_rng(int(p.balance_spread_stream))
                _tgt = p.balance_target * np.exp(
                    float(p.balance_spread) * _rng.standard_normal(self.N))
            with np.errstate(divide="ignore", invalid="ignore"):
                s = E_in / (I_in * _tgt)
            s[~np.isfinite(s)] = 1.0          # no inhibition to scale
            s[I_in <= 0] = 1.0
            self.n_balance_exempt = 0
            if p.balance_exempt is not None:
                _ex = np.asarray(p.balance_exempt)
                if _ex.shape != (self.N,) or _ex.dtype != bool:
                    raise ValueError(
                        f"balance_exempt must be a boolean mask of shape "
                        f"({self.N},), got {_ex.shape} {_ex.dtype}")
                s = np.where(_ex, 1.0, s)
                self.n_balance_exempt = int(_ex.sum())
            self.n_balance_capped = int((s > p.balance_max_scale).sum())
            s = np.clip(s, 0.0, p.balance_max_scale)
            if p.balance_scale_override is not None:
                _ov = np.asarray(p.balance_scale_override, dtype=s.dtype)
                if _ov.shape != s.shape:
                    raise ValueError(
                        f"balance_scale_override must have shape {s.shape}, "
                        f"got {_ov.shape}")
                s = _ov
                self.balance_scale_frozen = True
            self.balance_scale = s
            neg = w < 0
            w = w.copy()
            w[neg] *= s[post[neg]]

        # Seam gains (commissural, intersegmental) are applied AFTER the
        # balance step: they are acute perturbations of a balanced network.
        # Applied before it (as until 2026-09-04) the per-target rescale
        # returned every target's total inhibition to E_in/balance_target,
        # so an inhibitory seam gain only changed WHICH sources carried a
        # fixed amount of inhibition (lanes/A/cd1rx_balance_static.py).
        self.n_commissural_edges = 0
        if p.commissural_gain != 1.0:
            _cg = float(p.commissural_gain)
            if not np.isfinite(_cg) or _cg < 0.0:
                raise ValueError("commissural_gain must be finite and >= 0")
            _side = self.meta["side"].to_numpy()
            _known = pd.notna(_side)
            _cross = (_side[pre] != _side[post]) & _known[pre] & _known[post]
            if p.commissural_sign == "inh":
                _cross &= sign[pre] < 0.0
            elif p.commissural_sign == "exc":
                _cross &= sign[pre] > 0.0
            elif p.commissural_sign != "all":
                raise ValueError("commissural_sign must be all, inh or exc")
            self.n_commissural_edges = int(_cross.sum())
            w = np.where(_cross, w * _cg, w)
        self.n_intersegmental_edges = 0
        if (p.intersegmental_gain != 1.0 or p.intersegmental_gain_exc != 1.0
                or p.intersegmental_gain_inh != 1.0):
            for _nm_, _v in (("intersegmental_gain", p.intersegmental_gain),
                             ("intersegmental_gain_exc", p.intersegmental_gain_exc),
                             ("intersegmental_gain_inh", p.intersegmental_gain_inh)):
                if not np.isfinite(float(_v)) or float(_v) < 0.0:
                    raise ValueError(_nm_ + " must be finite and >= 0")
            _nm = self.meta["neuromere"].astype(object).to_numpy()
            _sc = self.meta["super_class"].astype(object).to_numpy()
            _leg = np.isin(_nm, ("T1", "T2", "T3"))
            _iseg_all = (np.isin(_sc[pre], ("ventral_nerve_cord_intrinsic", "vnc_intrinsic"))
                         & _leg[pre] & _leg[post] & (_nm[pre] != _nm[post]))
            _iseg = _iseg_all.copy()
            if p.intersegmental_sign == "inh":
                _iseg &= sign[pre] < 0.0
            elif p.intersegmental_sign == "exc":
                _iseg &= sign[pre] > 0.0
            elif p.intersegmental_sign != "all":
                raise ValueError("intersegmental_sign must be all, inh or exc")
            _touched = np.zeros_like(_iseg)
            if p.intersegmental_gain != 1.0:
                w = np.where(_iseg, w * float(p.intersegmental_gain), w)
                _touched |= _iseg
            if p.intersegmental_gain_exc != 1.0:
                _m = _iseg_all & (sign[pre] > 0.0)
                w = np.where(_m, w * float(p.intersegmental_gain_exc), w)
                _touched |= _m
            if p.intersegmental_gain_inh != 1.0:
                _m = _iseg_all & (sign[pre] < 0.0)
                w = np.where(_m, w * float(p.intersegmental_gain_inh), w)
                _touched |= _m
            self.n_intersegmental_edges = int(_touched.sum())
        self.n_row_pair_seam_edges = 0
        if p.row_pair_seams:
            _touched = np.zeros(len(pre), dtype=bool)
            for _pr, _po, _g in p.row_pair_seams:
                _g = float(_g)
                if not np.isfinite(_g) or _g < 0.0:
                    raise ValueError("row_pair_seams gain must be finite and >= 0")
                _ma = np.zeros(self.N, dtype=bool)
                _ma[np.asarray(_pr, dtype=np.int64)] = True
                _mb = np.zeros(self.N, dtype=bool)
                _mb[np.asarray(_po, dtype=np.int64)] = True
                _m = _ma[pre] & _mb[post]
                w = np.where(_m, w * _g, w)
                _touched |= _m
            self.n_row_pair_seam_edges = int(_touched.sum())
        self.n_type_out_gain_edges = 0
        if p.type_out_gain is not None:
            _tg = np.asarray(p.type_out_gain, dtype=np.float64)
            if _tg.shape != (self.N,):
                raise ValueError(
                    f"type_out_gain must have {self.N} entries, got {_tg.shape}")
            if not np.all(np.isfinite(_tg)) or np.any(_tg < 0.0):
                raise ValueError("type_out_gain must be finite and >= 0")
            _tm = _tg[pre] != 1.0
            if np.any(_tm):
                w = np.where(_tm, w * _tg[pre], w)
            self.n_type_out_gain_edges = int(_tm.sum())


        # Per-neuron input resistance, folded into the INCOMING weights at
        # build time so it costs nothing per tick.
        self.r_scale = None
        post_input_scale = np.ones(self.N, dtype=np.float32)
        if p.r_in_alpha:
            n_syn_in = np.bincount(post, weights=cnt, minlength=self.N)
            med = np.median(n_syn_in[n_syn_in > 0])
            with np.errstate(divide="ignore", invalid="ignore"):
                rs = (med / n_syn_in) ** p.r_in_alpha
            rs[~np.isfinite(rs)] = 1.0
            rs = np.clip(rs, 1.0 / p.r_in_max, p.r_in_max)
            self.r_scale = rs.astype(np.float32)
            post_input_scale *= self.r_scale
            w = w * self.r_scale[post]

        self.r_in_override_scale = None
        if p.r_in_override:
            pos_ov = pd.Series(np.arange(self.N), index=self.ids)
            ov = np.ones(self.N, dtype=np.float32)
            for bid, fac in p.r_in_override.items():
                r_ = pos_ov.get(bid)
                if r_ is not None and not np.isnan(r_):
                    ov[int(r_)] = fac
            self.r_in_override_scale = ov
            post_input_scale *= ov
            w = w * ov[post]

        # Q3bm target-specific effective fast gain. This deliberately occurs
        # AFTER balance and input-resistance construction. Only the registered
        # exact ordered pairs are replaced; no other inhibitory input is
        # rescaled as a side effect of choosing a negative hypothesis.
        requested_pairs = tuple(p.fast_edge_override_ids)
        override_gain = float(p.fast_edge_override_gain)
        if not np.isfinite(override_gain):
            raise ValueError("fast_edge_override_gain must be finite")
        self.fast_edge_override_audit = {
            "gain": override_gain,
            "requested_pair_count": len(requested_pairs),
            "selected_raw_row_count": 0,
            "selected_ordered_pair_count": 0,
            "selected_total_count": 0,
            "pairs": [],
        }
        if requested_pairs:
            if len(set(requested_pairs)) != len(requested_pairs):
                raise ValueError("duplicate fast_edge_override_ids pair")
            row_by_id = {str(bid): row for row, bid in enumerate(self.ids)}
            selected = np.zeros(len(w), dtype=bool)
            pair_records = []
            for pair in requested_pairs:
                if not isinstance(pair, tuple) or len(pair) != 2:
                    raise ValueError("fast_edge_override_ids entries must be 2-tuples")
                pre_id, post_id = pair
                if not (
                    isinstance(pre_id, str)
                    and isinstance(post_id, str)
                    and pre_id.isascii()
                    and post_id.isascii()
                    and pre_id.isdecimal()
                    and post_id.isdecimal()
                    and pre_id == str(int(pre_id))
                    and post_id == str(int(post_id))
                ):
                    raise ValueError("fast edge override IDs must be canonical decimal strings")
                if pre_id not in row_by_id or post_id not in row_by_id:
                    raise ValueError(f"fast edge override endpoint absent: {pair}")
                pre_row, post_row = row_by_id[pre_id], row_by_id[post_id]
                if float(sign[pre_row]) != 0.0:
                    raise ValueError(f"fast edge override source is not zero-fast: {pre_id}")
                pair_mask = (pre == pre_row) & (post == post_row)
                if not np.any(pair_mask):
                    raise ValueError(f"fast edge override pair has no positive raw edge: {pair}")
                selected |= pair_mask
                pair_records.append({
                    "pre_banc_888_id": pre_id,
                    "post_banc_888_id": post_id,
                    "pre_row": int(pre_row),
                    "post_row": int(post_row),
                    "raw_row_count": int(pair_mask.sum()),
                    "aggregate_count": int(cnt[pair_mask].sum()),
                })
            w = w.copy()
            w[selected] = (
                cnt[selected]
                * np.float32(p.w_syn)
                * np.float32(kernel_gain)
                * np.float32(override_gain)
                * post_input_scale[post[selected]]
            )
            self.fast_edge_override_audit = {
                "gain": override_gain,
                "requested_pair_count": len(requested_pairs),
                "selected_raw_row_count": int(selected.sum()),
                "selected_ordered_pair_count": len(pair_records),
                "selected_total_count": int(cnt[selected].sum()),
                "pairs": pair_records,
            }

        # Q3bv complete-output effective fast gain.  Resolve sources once,
        # then select every constructed outgoing row with one vectorized mask;
        # never rescan the full edge array once per target pair.  Like Q3bm,
        # this runs after balance and every input-resistance transform.
        requested_sources = tuple(p.fast_source_override_ids)
        source_override_gain = float(p.fast_source_override_gain)
        if not np.isfinite(source_override_gain):
            raise ValueError("fast_source_override_gain must be finite")
        self.fast_source_override_audit = {
            "gain": source_override_gain,
            "requested_source_count": len(requested_sources),
            "selected_raw_row_count": 0,
            "selected_ordered_pair_count": 0,
            "selected_total_count": 0,
            "sources": [],
            "pairs": [],
        }
        if requested_sources:
            if len(set(requested_sources)) != len(requested_sources):
                raise ValueError("duplicate fast_source_override_ids source")
            row_by_id = {str(bid): row for row, bid in enumerate(self.ids)}
            source_rows = []
            for source_id in requested_sources:
                if not (
                    isinstance(source_id, str)
                    and source_id.isascii()
                    and source_id.isdecimal()
                    and source_id == str(int(source_id))
                ):
                    raise ValueError(
                        "fast source override IDs must be canonical decimal strings")
                if source_id not in row_by_id:
                    raise ValueError(f"fast source override endpoint absent: {source_id}")
                source_row = int(row_by_id[source_id])
                if float(sign[source_row]) != 0.0:
                    raise ValueError(
                        f"fast source override source is not zero-fast: {source_id}")
                source_rows.append(source_row)

            source_row_array = np.asarray(source_rows, dtype=pre.dtype)
            selected = np.isin(pre, source_row_array)
            if not np.any(selected):
                raise ValueError("fast source override sources have no positive raw edge")

            # Aggregate the selected rows in one pass.  This audit is kept in
            # BANC-ID space so an executor can compare it with an independent
            # raw-edge reducer without relying on CNS row order.
            source_stats = {
                row: {"raw_row_count": 0, "aggregate_count": 0,
                      "targets": set()}
                for row in source_rows
            }
            pair_stats = {}
            for pre_row, post_row, count in zip(
                    pre[selected], post[selected], cnt[selected]):
                pre_i, post_i, count_i = int(pre_row), int(post_row), int(count)
                stat = source_stats[pre_i]
                stat["raw_row_count"] += 1
                stat["aggregate_count"] += count_i
                stat["targets"].add(post_i)
                pair = (pre_i, post_i)
                pair_stat = pair_stats.setdefault(
                    pair, {"raw_row_count": 0, "aggregate_count": 0})
                pair_stat["raw_row_count"] += 1
                pair_stat["aggregate_count"] += count_i

            for source_id, source_row in zip(requested_sources, source_rows):
                if source_stats[source_row]["raw_row_count"] == 0:
                    raise ValueError(
                        f"fast source override source has no positive raw edge: {source_id}")

            w = w.copy()
            w[selected] = (
                cnt[selected]
                * np.float32(p.w_syn)
                * np.float32(kernel_gain)
                * np.float32(source_override_gain)
                * post_input_scale[post[selected]]
            )
            source_records = []
            for source_id, source_row in sorted(
                    zip(requested_sources, source_rows), key=lambda item: int(item[0])):
                stat = source_stats[source_row]
                source_records.append({
                    "pre_banc_888_id": source_id,
                    "pre_row": source_row,
                    "raw_row_count": stat["raw_row_count"],
                    "unique_target_count": len(stat["targets"]),
                    "aggregate_count": stat["aggregate_count"],
                })
            pair_records = []
            for (pre_row, post_row), stat in sorted(pair_stats.items()):
                pair_records.append({
                    "pre_banc_888_id": str(self.ids[pre_row]),
                    "post_banc_888_id": str(self.ids[post_row]),
                    "pre_row": pre_row,
                    "post_row": post_row,
                    "raw_row_count": stat["raw_row_count"],
                    "aggregate_count": stat["aggregate_count"],
                })
            self.fast_source_override_audit = {
                "gain": source_override_gain,
                "requested_source_count": len(requested_sources),
                "selected_raw_row_count": int(selected.sum()),
                "selected_ordered_pair_count": len(pair_records),
                "selected_total_count": int(cnt[selected].sum()),
                "sources": source_records,
                "pairs": pair_records,
            }

        # C-V68 target-cell conductance normalization. This deliberately
        # replaces every incumbent transform on only the nominated edges with
        # one efficacy times raw EM count, preserving all six anatomical count
        # ratios exactly. Both backends consume the same CSC bytes: NumPy builds
        # Wg below and Metal uploads self.W.data, so no second runtime
        # implementation exists.
        requested_graded_pairs = tuple(p.graded_edge_equilibrium_ids)
        self.graded_edge_equilibrium_audit = {
            "mode": "off",
            "requested_pair_count": 0,
            "selected_raw_row_count": 0,
            "selected_total_count": 0,
            "targets": [],
            "pairs": [],
        }
        if requested_graded_pairs:
            if not (p.conductance_based and p.split_conductances):
                raise ValueError(
                    "graded edge equilibrium normalization requires split "
                    "conductance synapses")
            if int(p.graded_every) < 1:
                raise ValueError(
                    "graded edge equilibrium normalization requires a positive "
                    "graded update interval")
            if float(getattr(p, "graded_phasic", 0.0) or 0.0) != 0.0:
                raise ValueError(
                    "graded edge equilibrium normalization requires tonic "
                    "graded release")
            if len(set(requested_graded_pairs)) != len(requested_graded_pairs):
                raise ValueError("duplicate graded edge equilibrium pair")
            row_by_id = {str(bid): row for row, bid in enumerate(self.ids)}
            graded_members = set(int(row) for row in p.graded_rows)
            selected = np.zeros(len(w), dtype=bool)
            pair_records = []
            pair_rows = []
            for pair in requested_graded_pairs:
                if not isinstance(pair, tuple) or len(pair) != 2:
                    raise ValueError(
                        "graded edge equilibrium entries must be 2-tuples")
                pre_id, post_id = pair
                if not (
                    isinstance(pre_id, str)
                    and isinstance(post_id, str)
                    and pre_id.isascii()
                    and post_id.isascii()
                    and pre_id.isdecimal()
                    and post_id.isdecimal()
                    and pre_id == str(int(pre_id))
                    and post_id == str(int(post_id))
                ):
                    raise ValueError(
                        "graded edge equilibrium IDs must be canonical decimal strings")
                if pre_id not in row_by_id or post_id not in row_by_id:
                    raise ValueError(
                        f"graded edge equilibrium endpoint absent: {pair}")
                pre_row = int(row_by_id[pre_id])
                post_row = int(row_by_id[post_id])
                if pre_row not in graded_members:
                    raise ValueError(
                        f"graded edge equilibrium source is not graded: {pre_id}")
                if float(sign[pre_row]) <= 0.0:
                    raise ValueError(
                        f"graded edge equilibrium source is not excitatory: {pre_id}")
                pair_mask = (pre == pre_row) & (post == post_row)
                if not np.any(pair_mask):
                    raise ValueError(
                        f"graded edge equilibrium pair has no raw edge: {pair}")
                if np.any(w[pair_mask] <= 0.0):
                    raise ValueError(
                        f"graded edge equilibrium pair is not positive: {pair}")
                pair_counts = cnt[pair_mask].astype(np.float64)
                if (not np.all(np.isfinite(pair_counts))
                        or np.any(pair_counts <= 0.0)
                        or not np.array_equal(
                            pair_counts, np.rint(pair_counts))):
                    raise ValueError(
                        f"graded edge equilibrium pair count is not a "
                        f"positive integer: {pair}")
                selected |= pair_mask
                pair_rows.append((pre_row, post_row, pair_mask))
                pair_records.append({
                    "pre_banc_888_id": pre_id,
                    "post_banc_888_id": post_id,
                    "pre_row": pre_row,
                    "post_row": post_row,
                    "raw_row_count": int(pair_mask.sum()),
                    "aggregate_count": int(cnt[pair_mask].sum()),
                    "weight_before": float(w[pair_mask].sum()),
                })

            def _edge_vector(value, name):
                array = np.asarray(value, dtype=np.float64)
                if array.ndim == 0:
                    return np.full(self.N, float(array), dtype=np.float64)
                if array.shape != (self.N,) or not np.all(np.isfinite(array)):
                    raise ValueError(
                        f"graded edge equilibrium {name} must be finite scalar "
                        "or (N,)")
                return array

            runtime_rest = _edge_vector(p.v_rest, "v_rest")
            rest = runtime_rest.copy()
            if p.v_rest_override:
                for identifier, value in p.v_rest_override.items():
                    row = row_by_id.get(str(identifier))
                    if row is not None:
                        rest[int(row)] = float(value)
            threshold = _edge_vector(p.v_threshold, "v_threshold")
            tau_syn_vector = _edge_vector(p.tau_syn, "tau_syn")
            graded_gain = _edge_vector(p.graded_gain, "graded_gain")
            graded_v50 = _edge_vector(p.graded_v50, "graded_v50")
            graded_slope = _edge_vector(p.graded_slope, "graded_slope")
            if (np.any(graded_slope <= 0.0)
                    or np.any(tau_syn_vector <= 0.0)
                    or np.any(graded_gain < 0.0)):
                raise ValueError(
                    "graded edge equilibrium received invalid graded/tau vectors")
            # Match the shared NumPy/Metal runtime exactly: both derive the
            # conductance scale from p.v_rest before any sparse legacy
            # v_rest_override is applied. Target margins still use the actual
            # override-adjusted rest above.
            mean_rest = float(runtime_rest.mean())
            driving_force = abs(float(p.e_exc) - mean_rest)
            if not np.isfinite(driving_force) or driving_force <= 0.0:
                raise ValueError(
                    "graded edge equilibrium excitatory driving force is invalid")
            gscale_local = 1.0 / driving_force

            target_specs = []
            for post_row in sorted(set(item[1] for item in pair_rows)):
                target_pairs = [item for item in pair_rows if item[1] == post_row]
                source_rows = np.asarray(
                    [item[0] for item in target_pairs], dtype=np.int64)
                target_margin = float(threshold[post_row] - rest[post_row])
                target_exc_span = float(p.e_exc - threshold[post_row])
                if target_margin <= 0.0 or target_exc_span <= 0.0:
                    raise ValueError(
                        "graded edge equilibrium needs a positive target "
                        "membrane range")
                rest_release = 1.0 / (1.0 + np.exp(np.clip(
                    -(rest[source_rows] - graded_v50[source_rows])
                    / graded_slope[source_rows], -60.0, 60.0)))
                if np.any(rest_release <= 0.0) or np.any(rest_release >= 1.0):
                    raise ValueError(
                        "graded edge equilibrium rest release must be in (0, 1)")
                saturation_coefficient = 0.0
                rest_coefficient = 0.0
                rest_release_by_source = {
                    int(source): float(release)
                    for source, release in zip(source_rows, rest_release)
                }
                for pre_row, _, pair_mask in target_pairs:
                    count_gain = float(cnt[pair_mask].sum()) * float(
                        graded_gain[pre_row])
                    saturation_coefficient += count_gain
                    rest_coefficient += (
                        count_gain * rest_release_by_source[pre_row])
                target_conductance = target_margin / target_exc_span
                decay = float(np.exp(-float(p.dt) / tau_syn_vector[post_row]))
                saturation_conductance_per_weight_unit = (
                    saturation_coefficient / (1.0 - decay) * gscale_local)
                rest_conductance_per_weight_unit = (
                    rest_coefficient / (1.0 - decay) * gscale_local)
                if (not np.isfinite(saturation_conductance_per_weight_unit)
                        or saturation_conductance_per_weight_unit <= 0.0
                        or not np.isfinite(target_conductance)
                        or target_conductance <= 0.0):
                    raise ValueError(
                        "graded edge equilibrium population denominator is invalid")
                target_specs.append({
                    "post_banc_888_id": str(self.ids[post_row]),
                    "post_row": int(post_row),
                    "source_count": int(len(target_pairs)),
                    "rest_mv": float(rest[post_row]),
                    "threshold_mv": float(threshold[post_row]),
                    "tau_syn_ms": float(tau_syn_vector[post_row]),
                    "target_dimensionless_conductance": float(target_conductance),
                    "saturation_count_gain_coefficient": float(
                        saturation_coefficient),
                    "mean_saturation_conductance_per_weight_unit": float(
                        saturation_conductance_per_weight_unit),
                    "mean_rest_conductance_per_weight_unit": float(
                        rest_conductance_per_weight_unit),
                    "source_rest_release_min": float(rest_release.min()),
                    "source_rest_release_max": float(rest_release.max()),
                })

            population_conductance_per_weight_unit = float(sum(
                target["mean_saturation_conductance_per_weight_unit"]
                for target in target_specs))
            target_population_conductance = float(sum(
                target["target_dimensionless_conductance"]
                for target in target_specs))
            weight_per_synaptic_contact = (
                target_population_conductance
                / population_conductance_per_weight_unit)
            if (not np.isfinite(weight_per_synaptic_contact)
                    or weight_per_synaptic_contact <= 0.0):
                raise ValueError(
                    "graded edge equilibrium per-contact efficacy is invalid")
            stored_weight_per_contact = np.float32(
                weight_per_synaptic_contact)
            if (not np.isfinite(stored_weight_per_contact)
                    or stored_weight_per_contact <= 0.0):
                raise ValueError(
                    "graded edge equilibrium stored efficacy is invalid")
            w = w.copy()
            w[selected] = cnt[selected] * stored_weight_per_contact

            target_records = []
            for target in target_specs:
                record = dict(target)
                record["normalized_mean_saturation_conductance"] = float(
                    target["mean_saturation_conductance_per_weight_unit"]
                    * float(stored_weight_per_contact))
                record["normalized_mean_rest_conductance"] = float(
                    target["mean_rest_conductance_per_weight_unit"]
                    * float(stored_weight_per_contact))
                record["fraction_of_own_target_at_saturation"] = float(
                    record["normalized_mean_saturation_conductance"]
                    / target["target_dimensionless_conductance"])
                record["derived_weight_per_synaptic_contact"] = float(
                    weight_per_synaptic_contact)
                target_records.append(record)
            for record, (_, _, pair_mask) in zip(pair_records, pair_rows):
                record["weight_after"] = float(w[pair_mask].sum())
                record["derived_multiplier_over_incumbent"] = (
                    record["weight_after"] / record["weight_before"])
                record["weight_after_per_synaptic_contact"] = (
                    record["weight_after"] / record["aggregate_count"])
            self.graded_edge_equilibrium_audit = {
                "mode": "saturation_population_equilibrium",
                "formula": (
                    "sum_d(mean_ge_d_at_release_1)=sum_d((v_threshold_d-"
                    "v_rest_d)/(e_exc-v_threshold_d))"),
                "derived_weight_per_synaptic_contact": float(
                    weight_per_synaptic_contact),
                "stored_float32_weight_per_synaptic_contact": float(
                    stored_weight_per_contact),
                "runtime_conductance_scale": float(gscale_local),
                "population_mean_saturation_conductance_per_weight_unit": float(
                    population_conductance_per_weight_unit),
                "target_population_dimensionless_conductance": float(
                    target_population_conductance),
                "normalized_population_mean_saturation_conductance": float(
                    population_conductance_per_weight_unit
                    * float(stored_weight_per_contact)),
                "graded_update_every_ticks": int(p.graded_every),
                "graded_update_charge_compensated": True,
                "baseline_subtracted_from_runtime_release": False,
                "requested_pair_count": len(requested_graded_pairs),
                "selected_raw_row_count": int(selected.sum()),
                "selected_total_count": int(cnt[selected].sum()),
                "targets": target_records,
                "pairs": pair_records,
            }

        # CSC on the PREsynaptic axis: column j holds neuron j's outgoing
        # weights, so a step gathers the columns of the neurons that spiked.
        self.W = sparse.csc_matrix(
            (w, (post, pre)), shape=(self.N, self.N), dtype=np.float32)
        self.W.sort_indices()
        self._indptr = self.W.indptr
        self._indices = self.W.indices
        self._data = self.W.data

        self.n_edges = len(w)
        self.n_syn = float(cnt.sum())

        # --- WHAT THE RIG SELECTED, announced once per process ------------
        # AUDIT 2026-08-19. Five of the six bugs that have cost this project
        # a withdrawal would have been visible in this one line the moment
        # the network was built: the all-excitatory male statue (inh 0), the
        # motor selector that matched zero rows after a rename, the nerve
        # vocabulary that found zero proprioceptors, the campaniform
        # miscount, and gap='annotated' wiring nothing. The counters already
        # existed -- they were printed only under `if __name__ ==
        # "__main__"`, hard-wired to BANC, which is to say never in a real
        # run. The rule this enforces: an adapter must announce what the RIG
        # SELECTS, not what the dataset contains.
        _exc = int((self.sign > 0).sum())
        _inh = int((self.sign < 0).sum())
        _mut = int((self.sign == 0).sum())
        if not _ANNOUNCED:
            # stderr, not stdout: eval_job.py emits pure JSON on stdout
            # and this line corrupted it (caught 2026-08-19, minutes after
            # the announce was added -- an announcement that breaks a
            # machine-readable channel is its own kind of silent failure).
            print(f"[CNS] core {self.N} neurons / {self.n_edges} edges / "
                  f"{self.n_syn:.0f} synapses | exc {_exc} inh {_inh} "
                  f"muted {_mut} unknown-defaulted {self.n_unknown_nt} | "
                  f"gap_pairs {len(p.gap_pairs)} graded {len(p.graded_rows)} "
                  f"premotor_inh {len(p.premotor_inh_rows)} "
                  f"transmitter_map {self.n_transmitter_overridden} rows "
                  f"({self.n_transmitter_sign_flips} sign flips)",
                  file=sys.stderr, flush=True)
            _ANNOUNCED.append(1)
        # The one thing that must never be true again. The male ran for 14
        # hours with an all-excitatory cord and produced a 12/12 "perfect
        # posture transfer" that was rigidity; an empty string in the
        # verified-transmitter column had beaten every real prediction.
        assert _inh > 0, (
            f"NO INHIBITION in the rig core: {_exc} excitatory, 0 inhibitory "
            f"of {self.N} neurons. This is the all-excitatory statue. Check "
            f"neurotransmitter_verified for EMPTY STRINGS -- cns.py:401 "
            f"prefers verified on notna(), not on non-empty.")

        # Gap-junction graph, symmetric, as (adjacency, per-neuron degree).
        # Kept SEPARATE from W: chemical weights are signed and delayed, these
        # are unsigned and instantaneous, and mixing them would silently put
        # electrical coupling through the delay ring.
        # Graded (non-spiking) cells: their outgoing columns, pre-sliced once.
        # W is CSC on the presynaptic axis, so a column slice IS the set of a
        # neuron's outputs and the per-tick cost is one small matvec.
        self.graded = np.zeros(self.N, dtype=bool)
        self.Wg = None
        if len(p.graded_rows):
            gr = np.unique(np.asarray(p.graded_rows, dtype=np.int64))
            self.graded[gr] = True
            self.graded_idx = gr
            self.Wg = sparse.csc_matrix(
                (w, (post, pre)), shape=(self.N, self.N),
                dtype=np.float32)[:, gr].tocsr()

        self.G = None
        self.gap_deg = None
        self.gap_scale = np.float32(0.0)
        self.gap_pair_weights = np.empty(0, dtype=np.float32)
        has_pair_weights = len(p.gap_pair_weights) > 0
        if has_pair_weights and p.g_gap:
            raise ValueError(
                "gap_pair_weights and scalar g_gap are mutually exclusive")
        if has_pair_weights and not len(p.gap_pairs):
            raise ValueError("gap_pair_weights require gap_pairs")
        if (p.g_gap or p.gap_spikelet or has_pair_weights) and len(p.gap_pairs):
            raw_gp = np.asarray(p.gap_pairs)
            if raw_gp.ndim != 2 or raw_gp.shape[1] != 2:
                raise ValueError("gap_pairs must have shape (n_pairs, 2)")
            if (not np.issubdtype(raw_gp.dtype, np.integer)
                    or np.any(raw_gp < 0) or np.any(raw_gp >= self.N)):
                raise ValueError("gap_pairs contain invalid neuron row indices")
            gp = raw_gp.astype(np.int64, copy=False)
            if np.any(gp[:, 0] == gp[:, 1]):
                raise ValueError("gap_pairs cannot contain self-pairs")
            if has_pair_weights:
                pair_weights = np.asarray(
                    p.gap_pair_weights, dtype=np.float32).reshape(-1)
                if (len(pair_weights) != len(gp)
                        or not np.isfinite(pair_weights).all()
                        or np.any(pair_weights < 0.0)):
                    raise ValueError(
                        "gap_pair_weights must be one finite nonnegative "
                        "value per gap pair")
                self.gap_scale = np.float32(1.0)
            else:
                if not np.isfinite(p.g_gap) or p.g_gap < 0.0:
                    raise ValueError("g_gap must be finite and nonnegative")
                pair_weights = np.ones(len(gp), dtype=np.float32)
                self.gap_scale = np.float32(p.g_gap)
            if p.gap_rectify:
                i, j, val = gp[:, 1], gp[:, 0], pair_weights
            else:
                i = np.concatenate([gp[:, 0], gp[:, 1]])
                j = np.concatenate([gp[:, 1], gp[:, 0]])
                val = np.concatenate([pair_weights, pair_weights])
            self.G = sparse.csr_matrix((val, (i, j)), shape=(self.N, self.N),
                                       dtype=np.float32)
            self.gap_deg = np.asarray(self.G.sum(axis=1)).ravel().astype(
                np.float32)
            self.n_gap_pairs = len(gp)
            self.gap_pair_weights = pair_weights

    # ---------------------------------------------------------------- utils
    def select(self, **kw):
        """Row indices whose metadata columns match. Values may be scalars or
        iterables; a column matches if the value is in the iterable."""
        mask = np.ones(self.N, dtype=bool)
        for col, val in kw.items():
            series = self.meta[col]
            if isinstance(val, (list, tuple, set, np.ndarray, pd.Series)):
                mask &= series.isin(list(val)).to_numpy()
            else:
                mask &= (series == val).to_numpy()
        return np.flatnonzero(mask)

    # ------------------------------------------------------------------ run
    def _fold_receptor_classes(self, w, pre, post, nt):
        cell_class = self.meta["cell_class"].fillna("").to_numpy()
        mult, audit = _receptor_class_multiplier(
            self.p.receptor_classes, pre, post, nt, cell_class)
        self.receptor_class_audit = audit
        return w * mult

    def run(self, duration_ms, drive=None, seed=0, record=None,
            record_trace_every=None, record_voltage=None, drive_window=None,
            drive_fn=None, feedback_rows=None, drive_every_ms=1.0,
            force_spikes=None, first_spike=False, force_spikes_fn=None,
            feedback_graded_release_rows=None,
            graded_drive_max_rate_hz=None, graded_drive_signed=False,
            membrane_drive_rows=None, membrane_drive_mv=0.0,
            membrane_drive_fraction=None, record_input=None):
        """Simulate.

        drive : dict {neuron_index_array : rate_hz} or (indices, rate)
            Poisson spike injection. These neurons fire independently of their
            input, which is how a sensory periphery the volume does not contain
            gets represented.
        membrane_drive_rows : array of neuron indices, or None (default).
            COMMAND AS A MEMBRANE CURRENT (gap.command_membrane_drive,
            2026-09-02). Every driven row fires as a forced spike, unioned
            into `fired` after the membrane update, so no synapse onto it
            can oppose it: lane D measured the champion's command DNg100
            ungateable by its largest inhibitory input (DNge129, 439
            synapses) for exactly this reason. For the rows named here each
            drive event instead adds `membrane_drive_mv` millivolts to the
            membrane, drawn from the SAME per-step Bernoulli or renewal
            stream (the RNG sequence is unchanged), and the cell spikes only
            when its own threshold test passes on a later step, where
            inhibition through the ordinary synaptic path subtracts from it.
            Refractory rows are clamped to v_reset at the start of each step.
            A kick with more than one refractory tick remaining is therefore
            clamped away on the following step; a kick on the final refractory
            tick survives the countdown and can affect the next active step.
            A same-step natural or forced spike resets over the kick. A kick
            larger than threshold minus rest (7 mV at the Shiu defaults)
            otherwise fires the cell on the next step unless inhibited. None
            = the forced-spike path, bit-identical.
        membrane_drive_mv : float
            Depolarisation per drive event on membrane_drive_rows, mV.
        membrane_drive_fraction : float or None
            Cell-relative alternative to ``membrane_drive_mv``.  Each event
            adds this positive fraction of that row's own
            ``v_threshold - effective_v_rest`` margin.  Effective rest
            includes any per-neuron override.  Exactly one of an absolute
            kick or a fraction must be supplied when membrane rows are set.
        graded_drive_signed : bool
            Default False. With graded_drive_max_rate_hz, negative values on
            graded rows are dimensionless membrane commands after division by
            that normalization scale; they hyperpolarize and are not firing
            rates. Negative values on spiking rows are rejected.
        drive_window : (t_on_ms, t_off_ms), or None for the whole run.
            A step command rather than a constant one. This is the actual
            physiological protocol for a descending neuron — optogenetics
            switches DNs on and off, it does not hold them at a rate forever —
            and it is the only way to see onset and offset transients, which
            is where post-inhibitory rebound lives.
        record : array of neuron indices to return per-neuron spike counts for.
            None records all.
        record_trace_every : if set, also return a spike-count time series
            binned at this many ms, for the `record` set.
        record_input : dict or None
            Default-off Metal diagnostic. It records native-tick synaptic
            increments by a caller-supplied source-route partition, plus the
            exact voltage and split conductance state used by selected target
            neurons. It changes no model parameter or drive. The NumPy engine
            refuses it rather than returning a different observation.

        Returns dict with 'counts' (spikes per recorded neuron), 'rates_hz',
        'n_spikes' total, and optionally 'trace'.
        """
        p = self.p
        if (membrane_drive_fraction is not None
                and (membrane_drive_rows is None
                     or len(membrane_drive_rows) == 0)):
            raise ValueError(
                "membrane_drive_fraction requires membrane_drive_rows")
        tonic_ge = getattr(p, "tonic_exc_conductance_per_neuron", None)
        if tonic_ge is not None:
            tonic_ge = np.asarray(tonic_ge, dtype=np.float32)
            if (tonic_ge.shape != (self.N,)
                    or not np.all(np.isfinite(tonic_ge))
                    or np.any(tonic_ge < 0.0)):
                raise ValueError(
                    "tonic_exc_conductance_per_neuron must be a finite "
                    "nonnegative (N,) vector")
            if not p.conductance_based:
                raise ValueError(
                    "tonic_exc_conductance_per_neuron requires "
                    "conductance_based=True")
        if getattr(p, "backend", "numpy") == "metal":
            # tonic_exc_conductance_per_neuron runs on Metal since 2026-09-06
            # (compute seat; gate gpu/gate_v0.py --tonic): the vector rides in
            # the 4th section of the clamp-values buffer and k_step adds it to
            # ge exactly where this function does, after ge and gi are formed
            # and before either integrator.
            if (getattr(p, "serotonin_source_rows", None) is not None
                    or getattr(p, "serotonin_target_rows", None) is not None):
                raise NotImplementedError(
                    "the receptor-scoped serotonin state is not yet ported "
                    "to Metal; refusing a silent zero-effect run")
            # the coincidence seam runs on Metal since 2026-09-05 (compute
            # seat; gpu/gate_v0.py --coincidence): cns_metal.py builds the
            # same pool matrices, refuses the same empty pool, and returns
            # the same receipt under out["coincidence"].
            from cns_metal import run_metal
            return run_metal(
                self, duration_ms, drive=drive, seed=seed, record=record,
                record_trace_every=record_trace_every,
                record_voltage=record_voltage, drive_window=drive_window,
                drive_fn=drive_fn, feedback_rows=feedback_rows,
                drive_every_ms=drive_every_ms, force_spikes=force_spikes,
                first_spike=first_spike, force_spikes_fn=force_spikes_fn,
                feedback_graded_release_rows=feedback_graded_release_rows,
                graded_drive_max_rate_hz=graded_drive_max_rate_hz,
                graded_drive_signed=graded_drive_signed,
                membrane_drive_rows=membrane_drive_rows,
                membrane_drive_mv=membrane_drive_mv,
                membrane_drive_fraction=membrane_drive_fraction,
                record_input=record_input)
        elif getattr(p, "backend", "numpy") != "numpy":
            raise ValueError(f"unknown backend {p.backend!r}")
        if record_input is not None:
            raise NotImplementedError(
                "record_input is an exact Metal-engine diagnostic")
        rng = np.random.default_rng(seed)
        n_steps = int(round(duration_ms / p.dt))
        conductance_integrator = getattr(
            p, "conductance_integrator", "euler")
        if conductance_integrator not in ("euler", "analytic"):
            raise ValueError(
                "conductance_integrator must be 'euler' or 'analytic', got "
                f"{conductance_integrator!r}")
        if conductance_integrator == "analytic" and not p.conductance_based:
            raise ValueError(
                "analytic conductance integration requires "
                "conductance_based=True")
        dpn = getattr(p, "delay_per_neuron", None)
        if dpn is None:
            n_delay = max(1, int(round(p.delay / p.dt)))
            delay_slots = None
        else:
            _dpn = np.asarray(dpn, dtype=np.float64)
            if _dpn.shape != (self.N,):
                raise ValueError(
                    f"delay_per_neuron needs one value per neuron: got shape "
                    f"{_dpn.shape}, N={self.N}. If a core cut is in play, the "
                    f"vector must be built from the SAME meta the CNS was "
                    f"built from.")
            if not np.all(np.isfinite(_dpn)) or float(_dpn.min()) < 0.0:
                raise ValueError("delay_per_neuron must be finite and >= 0")
            delay_slots = np.maximum(
                1, np.rint(_dpn / p.dt)).astype(np.int64)
            n_delay = int(delay_slots.max())
        delivery = getattr(p, "delay_delivery", "bucket")
        if delivery not in ("bucket", "queue"):
            raise ValueError(
                f"delay_delivery must be 'bucket' or 'queue', got "
                f"{delivery!r}")
        # The queue needs one bucket per tick of the longest delay plus one,
        # so a spike's bucket is never popped again before it arrives: the
        # pops between push and arrival are at ticks t+1..t+d-1, and
        # (t+j) == (t+d) mod n_queue would need d-j to be a multiple of
        # n_queue, impossible for 0 < d-j < d <= n_queue-1.
        use_queue = delivery == "queue" and delay_slots is not None
        graded_delivery = getattr(p, "graded_delay_delivery", "bucket")
        if graded_delivery not in ("bucket", "queue"):
            raise ValueError(
                "graded_delay_delivery must be 'bucket' or 'queue', got "
                f"{graded_delivery!r}")
        use_graded_queue = (graded_delivery == "queue"
                            and delay_slots is not None)
        n_queue = int(delay_slots.max()) + 1 if use_queue else 0
        fire_q = [[] for _ in range(n_queue)] if use_queue else None
        n_refrac = np.broadcast_to(
            np.rint(np.asarray(p.t_refrac, dtype=np.float64) / p.dt
                    ).astype(np.int32), (self.N,)).copy()

        # tau_mem and v_rest may be scalars or per-neuron arrays.
        tau_mem = np.asarray(p.tau_mem, dtype=np.float32)
        v_rest = np.asarray(p.v_rest, dtype=np.float32)
        e_inh = np.asarray(p.e_inh, dtype=np.float32)
        if e_inh.ndim == 0:
            if not np.isfinite(e_inh):
                raise ValueError("e_inh must be finite")
        elif (e_inh.shape != (self.N,)
              or not np.all(np.isfinite(e_inh))):
            raise ValueError("e_inh must be finite scalar or (N,) array")
        if p.v_rest_override:
            v_rest = np.broadcast_to(v_rest, (self.N,)).copy()
            pos_vo = pd.Series(np.arange(self.N), index=self.ids)
            for bid, mv in p.v_rest_override.items():
                r_ = pos_vo.get(bid)
                if r_ is not None and not np.isnan(r_):
                    v_rest[int(r_)] = mv
        # v_reset is indexed by `fired`, so it must be a real array rather
        # than a scalar that happens to broadcast in some expressions and not
        # others. Broadcasting it once here costs 0.75 MB and removes a class
        # of shape bug: `v[fired] = p.v_reset` raised ValueError the first
        # time a per-neuron v_reset was passed in.
        v_reset = np.broadcast_to(
            np.asarray(p.v_reset, dtype=np.float32), (self.N,)).copy()
        decay_v = np.exp(-p.dt / tau_mem).astype(np.float32)
        dt_over_tau = np.float32(p.dt) / tau_mem
        # ⚠️ float32, deliberately. np.exp on a Python float returns a float64
        # SCALAR, and under NumPy 2's promotion rules a float64 scalar upcasts
        # the float32 array in `g *= decay_g` -- so every tick cast all 188,508
        # elements to double and back. Measured: 57.5 us -> 10.8 us on that
        # line (5.3x), ~19% of total wall per run. COST: not bit-identical to
        # runs committed before 2026-08-11 (double rounding differs by <=1 ulp,
        # and the network is chaotic) -- statistics over seeds are unaffected,
        # exact replay of old single runs requires the old code.
        tau_syn = np.asarray(p.tau_syn, dtype=np.float32)
        if p.tau_syn_exc is not None:
            decay_g = np.float32(np.exp(-p.dt / float(p.tau_syn_exc)))
            decay_g_i = np.float32(np.exp(-p.dt / float(p.tau_syn_inh)))
        elif tau_syn.ndim == 0:
            if not np.isfinite(tau_syn) or float(tau_syn) <= 0.0:
                raise ValueError("tau_syn must be finite and > 0")
            decay_g = np.float32(np.exp(-p.dt / tau_syn))
            decay_g_i = decay_g
        else:
            if (tau_syn.shape != (self.N,)
                    or not np.all(np.isfinite(tau_syn))
                    or np.any(tau_syn <= 0.0)):
                raise ValueError(
                    "tau_syn must be finite scalar or positive (N,) array")
            decay_g = np.exp(-np.float32(p.dt) / tau_syn).astype(np.float32)
            decay_g_i = decay_g
        # g is accumulated in mV (w_syn is mV/synapse). For the conductance
        # form it must be dimensionless, so rescale by the driving force a
        # spike would see at rest -- this keeps a single spike's PSP roughly
        # the same size as in the current-based model, making the two
        # comparable rather than differing by an arbitrary gain.
        gscale = 1.0 / abs(p.e_exc - float(np.mean(np.asarray(p.v_rest))))

        v = np.full(self.N, p.v_rest, dtype=np.float32) if v_rest.ndim == 0 \
            else v_rest.astype(np.float32).copy()
        g = np.zeros(self.N, dtype=np.float32)
        refrac = np.zeros(self.N, dtype=np.int32)
        counts = np.zeros(self.N, dtype=np.int64)

        # Adaptation. Entirely skipped when both coefficients are zero, so the
        # LIF path costs exactly what it did before this was added.
        adapt_a = np.asarray(p.adapt_a, dtype=np.float32)
        adapt_b = np.asarray(p.adapt_b, dtype=np.float32)
        use_adapt = bool(np.any(adapt_a != 0) or np.any(adapt_b != 0))
        w_adapt = np.zeros(self.N, dtype=np.float32) if use_adapt else None
        dt_over_tauw = np.float32(p.dt) / np.asarray(p.tau_w, dtype=np.float32)

        # Receptor-scoped neuromodulation. Configuration is all-or-nothing:
        # accepting one population without the other would produce an inert
        # mechanism that still looked enabled. Gain zero remains a real paired
        # control: the source state is measured, but no membrane arithmetic is
        # changed.
        _ser_source = getattr(p, "serotonin_source_rows", None)
        _ser_target = getattr(p, "serotonin_target_rows", None)
        serotonin_configured = _ser_source is not None or _ser_target is not None
        if serotonin_configured:
            if _ser_source is None or _ser_target is None:
                raise ValueError(
                    "serotonin source and target rows must be configured together")
            serotonin_source = np.asarray(_ser_source, dtype=np.int64)
            serotonin_target = np.asarray(_ser_target, dtype=np.int64)
            if (serotonin_source.ndim != 1 or not len(serotonin_source)
                    or len(np.unique(serotonin_source)) != len(serotonin_source)
                    or np.any(serotonin_source < 0)
                    or np.any(serotonin_source >= self.N)):
                raise ValueError("serotonin_source_rows must be unique valid rows")
            if (serotonin_target.ndim != 1 or not len(serotonin_target)
                    or len(np.unique(serotonin_target)) != len(serotonin_target)
                    or np.any(serotonin_target < 0)
                    or np.any(serotonin_target >= self.N)):
                raise ValueError("serotonin_target_rows must be unique valid rows")
            serotonin_tau_ms = float(p.serotonin_tau_ms)
            serotonin_gain = float(p.serotonin_gain_mv_per_hz)
            if not np.isfinite(serotonin_tau_ms) or serotonin_tau_ms <= 0.0:
                raise ValueError("serotonin_tau_ms must be finite and positive")
            if not np.isfinite(serotonin_gain) or serotonin_gain < 0.0:
                raise ValueError(
                    "serotonin_gain_mv_per_hz must be finite and non-negative")
            serotonin_decay = np.float32(np.exp(-p.dt / serotonin_tau_ms))
            # A source population firing steadily at r Hz receives N*r spikes
            # per second; each spike adds 1/(N*tau_seconds), so the equilibrium
            # state is r Hz independent of source-population size.
            serotonin_increment_hz = np.float32(
                1000.0 / (len(serotonin_source) * serotonin_tau_ms))
            serotonin_source_mask = np.zeros(self.N, dtype=bool)
            serotonin_source_mask[serotonin_source] = True
            serotonin_state_hz = np.float32(0.0)
            serotonin_state_max_hz = np.float32(0.0)
            serotonin_state_applied_sum_hz = 0.0
            serotonin_source_spikes = 0
            serotonin_extra = (np.zeros(self.N, dtype=np.float32)
                                if serotonin_gain > 0.0 else None)
        else:
            serotonin_source = serotonin_target = None
            serotonin_state_hz = np.float32(0.0)
            serotonin_state_max_hz = np.float32(0.0)
            serotonin_state_applied_sum_hz = 0.0
            serotonin_source_spikes = 0
            serotonin_extra = None
        use_exp = p.delta_T > 0
        G = self.G
        gap_deg = self.gap_deg
        gap_scale = self.gap_scale
        gap_spike = np.float32(p.gap_spikelet)
        # Delayed spikelet ring (gap_delay_ms > 0 only): one row per arrival
        # tick, gap_ds ticks ahead. None keeps the instantaneous path.
        gap_ring, gap_ds, gap_S = None, 0, 1
        if G is not None and gap_spike and float(p.gap_delay_ms) > 0.0:
            gap_ds = int(round(float(p.gap_delay_ms) / p.dt))
            if gap_ds < 1:
                raise ValueError("gap_delay_ms must be at least one timestep")
            gap_S = gap_ds + 1
            gap_ring = np.zeros((gap_S, self.N), dtype=np.float32)
        # Presynaptic AP waveform ring (gap_ap_mv > 0 only): per arrival tick,
        # the extra presynaptic voltage each cell shows its partners.
        ap_ring, ap_wave, ap_S = None, None, 1
        if G is not None and float(p.gap_ap_mv) > 0.0:
            n_ap = max(int(round(float(p.gap_ap_ms) / p.dt)), 2)
            n_rise = max(n_ap // 3, 1)
            up = np.linspace(0.0, 1.0, n_rise + 1)[1:]
            down = np.linspace(1.0, 0.0, n_ap - n_rise + 1)[1:]
            ap_wave = (np.concatenate([up, down]) * float(p.gap_ap_mv)).astype(np.float32)
            ap_ds = int(round(float(p.gap_delay_ms) / p.dt)) if float(p.gap_delay_ms) > 0.0 else 0
            ap_S = ap_ds + len(ap_wave) + 1
            ap_ring = np.zeros((ap_S, self.N), dtype=np.float32)
            ap_off = ap_ds
        graded = self.graded
        Wg = self.Wg
        graded_idx = getattr(self, "graded_idx", np.empty(0, np.int64))
        # Per-neuron delay x graded release.  The canonical bucket delivery
        # pre-splits Wg by source delay.  Queue delivery instead stores one
        # pending release value per graded source and arrival slot, then uses
        # one full Wg matvec at arrival.  This mirrors the spike queue and
        # avoids one sparse matvec per distinct delay when cable lengths are
        # unquantised.
        graded_delay_groups = None
        graded_release_q = None
        graded_delay_slots = None
        graded_positions = None
        if Wg is not None and delay_slots is not None:
            graded_delay_slots = delay_slots[graded_idx]
            if use_graded_queue:
                graded_positions = np.arange(len(graded_idx), dtype=np.int64)
                graded_release_q = np.zeros(
                    (n_delay, len(graded_idx)), dtype=np.float32)
            else:
                graded_delay_groups = []
                for _s in np.unique(graded_delay_slots):
                    _pos = np.flatnonzero(graded_delay_slots == _s)
                    graded_delay_groups.append(
                        (int(_s), _pos, Wg[:, _pos].tocsr()))
        _ggain = np.asarray(p.graded_gain, dtype=np.float32)
        if _ggain.ndim == 0:
            ggain = np.float32(_ggain)
            ggain_v = None
        else:
            if (_ggain.shape != (self.N,) or not np.all(np.isfinite(_ggain))
                    or np.any(_ggain < 0.0)):
                raise ValueError(
                    "graded_gain must be a finite scalar or a finite "
                    "non-negative (N,) array")
            ggain = None
            ggain_v = _ggain[graded_idx]
        # Keep the scalar path byte-for-byte shaped like the historical one,
        # while allowing a full per-neuron vector. The vector is subset once
        # to the graded columns, so the inner loop still works only on Wg.
        _gv50 = np.asarray(p.graded_v50, dtype=np.float32)
        _gslope = np.asarray(p.graded_slope, dtype=np.float32)
        if _gv50.ndim == 0:
            gv50 = np.float32(_gv50)
        else:
            if _gv50.shape != (self.N,) or not np.all(np.isfinite(_gv50)):
                raise ValueError("graded_v50 must be finite scalar or (N,) array")
            gv50 = _gv50[graded_idx]
        if _gslope.ndim == 0:
            gslope = np.float32(_gslope)
            if not np.isfinite(gslope) or gslope <= 0.0:
                raise ValueError("graded_slope must be finite and > 0")
        else:
            if (_gslope.shape != (self.N,)
                    or not np.all(np.isfinite(_gslope))
                    or np.any(_gslope <= 0.0)):
                raise ValueError(
                    "graded_slope must be positive finite scalar or (N,) array")
            gslope = _gslope[graded_idx]
        g_every = max(1, int(p.graded_every))
        gphasic = float(getattr(p, "graded_phasic", 0.0) or 0.0)
        gphasic_alpha = float(getattr(p, "graded_phasic_alpha", 0.01) or 0.01)
        graded_ema = None
        inv_dT = np.float32(1.0 / p.delta_T) if use_exp else np.float32(0.0)
        spike_at = p.v_peak if use_exp else p.v_threshold

        # ring buffer of delayed synaptic increments
        ring = np.zeros((n_delay, self.N), dtype=np.float32)
        # CB3 (2026-08-19, Robin's question about synapse sign): TRUE
        # conductance synapses need excitatory and inhibitory
        # conductances tracked SEPARATELY. Summing them into one signed
        # ring cancels them arithmetically before either can act, which
        # destroys SHUNTING -- the divisive gain control that is the
        # main point of conductance-based synapses. ring_i carries the
        # magnitude of inhibitory input; allocated only when needed.
        ring_i = (np.zeros((n_delay, self.N), dtype=np.float32)
                  if p.conductance_based and p.split_conductances else None)
        g_i = (np.zeros(self.N, dtype=np.float32)
               if ring_i is not None else None)
        _presyn_rows = getattr(p, "presynaptic_inhibition_rows", None)
        presynaptic_inhibition = _presyn_rows is not None
        if presynaptic_inhibition:
            if ring_i is None:
                raise ValueError(
                    "presynaptic inhibition requires conductance_based=True "
                    "and split_conductances=True")
            presyn_rows = np.asarray(_presyn_rows, dtype=np.int64)
            if (presyn_rows.ndim != 1 or not len(presyn_rows)
                    or len(np.unique(presyn_rows)) != len(presyn_rows)
                    or np.any(presyn_rows < 0)
                    or np.any(presyn_rows >= self.N)):
                raise ValueError(
                    "presynaptic_inhibition_rows must be unique valid rows")
            presyn_mask = np.zeros(self.N, dtype=bool)
            presyn_mask[presyn_rows] = True
            presyn_spikes = 0
            presyn_attenuated_spikes = 0
            presyn_factor_sum = 0.0
            presyn_factor_min = 1.0
        else:
            presyn_rows = np.empty(0, dtype=np.int64)
            presyn_mask = None

        _spike_reg_vector = getattr(p, "spike_reg_per_neuron", None)
        use_reg_vector = _spike_reg_vector is not None
        if use_reg_vector:
            spike_reg_vector = np.asarray(_spike_reg_vector, dtype=np.float64)
            if (spike_reg_vector.shape != (self.N,)
                    or not np.all(np.isfinite(spike_reg_vector))
                    or np.any(spike_reg_vector < 1.0)):
                raise ValueError(
                    "spike_reg_per_neuron must be a finite (N,) array >= 1")
            use_reg = bool(np.any(spike_reg_vector > 1.0))
        else:
            spike_reg_vector = None
            use_reg = float(getattr(p, "spike_reg", 1.0)) > 1.0
        if use_reg:
            _kreg = (None if use_reg_vector else
                     int(round(float(p.spike_reg))))
            # per-ENTRY accumulator and Erlang-k threshold; entries (not
            # neurons) so duplicated drive rows keep CD1's union semantics
            reg_acc = None      # lazily sized to len(drive_idx) on first use
            reg_thr = None

        force_ms = ({round(float(k), 3): np.asarray(v, dtype=np.int64)
                     for k, v in force_spikes.items()}
                    if force_spikes else None)
        # First-spike tick per neuron, -1 if never. Sub-millisecond latency is
        # the whole measurement here, so a 1 ms trace bin would quantise away
        # the quantity under test (measured TTM 0.93 ms, DLM 1.44 ms).
        t_first = np.full(self.N, -1, dtype=np.int32) if first_spike else None

        use_std = p.std_U > 0
        x_res = np.ones(self.N, dtype=np.float32) if use_std else None
        std_keep = np.float32(1.0 - p.std_U)
        std_dec = np.float32(np.exp(-p.dt / p.std_tau_rec))

        drive_idx = np.empty(0, dtype=np.int64)
        drive_rate = np.empty(0, dtype=np.float64)
        drive_prob = np.empty(0, dtype=np.float64)
        drive_reg = np.empty(0, dtype=np.float64)
        if drive:
            di, dp = [], []
            for k, rate in drive.items():
                k = np.asarray(k, dtype=np.int64)
                di.append(k)
                dp.append(np.full(len(k), rate * p.dt / 1000.0))
            drive_idx = np.concatenate(di)
            drive_prob = np.concatenate(dp)
            drive_rate = drive_prob * (1000.0 / p.dt)
            if use_reg_vector:
                drive_reg = spike_reg_vector[drive_idx]
        membrane_drive_mask = None
        membrane_drive_kick_vector = None
        membrane_drive_receipt = None
        membrane_drive_events = 0
        if membrane_drive_rows is not None and len(membrane_drive_rows):
            membrane_drive_rows = np.asarray(membrane_drive_rows,
                                             dtype=np.int64)
            if (membrane_drive_rows.ndim != 1
                    or len(np.unique(membrane_drive_rows))
                    != len(membrane_drive_rows)
                    or membrane_drive_rows.min() < 0
                    or membrane_drive_rows.max() >= self.N):
                raise IndexError(
                    "membrane_drive_rows must be unique valid rows")
            membrane_drive_mask = np.zeros(self.N, dtype=bool)
            membrane_drive_mask[membrane_drive_rows] = True
            threshold = np.broadcast_to(
                np.asarray(p.v_threshold, dtype=np.float32),
                (self.N,))
            rest = np.broadcast_to(
                np.asarray(v_rest, dtype=np.float32),
                (self.N,))
            margin = (threshold[membrane_drive_rows]
                      - rest[membrane_drive_rows]).astype(np.float32)
            if membrane_drive_fraction is not None:
                if (not np.all(np.isfinite(margin))
                        or np.any(margin <= 0.0)):
                    raise ValueError(
                        "cell-relative membrane-driven rows need finite "
                        "positive v_threshold - effective_v_rest margins")
                if (np.asarray(membrane_drive_mv).ndim != 0
                        or float(membrane_drive_mv) != 0.0):
                    raise ValueError(
                        "choose membrane_drive_mv or "
                        "membrane_drive_fraction, not both")
                fraction = float(membrane_drive_fraction)
                if not np.isfinite(fraction) or fraction <= 0.0:
                    raise ValueError(
                        "membrane_drive_fraction must be finite and > 0")
                selected_kicks = (
                    np.float32(fraction) * margin).astype(np.float32)
                membrane_drive_kick_vector = np.zeros(
                    self.N, dtype=np.float32)
                membrane_drive_kick_vector[membrane_drive_rows] = (
                    selected_kicks)
                drive_mode = "threshold_margin_fraction"
                requested = fraction
            else:
                if (np.asarray(membrane_drive_mv).ndim != 0
                        or not np.isfinite(membrane_drive_mv)
                        or membrane_drive_mv <= 0):
                    raise ValueError(
                        "membrane_drive_mv must be a finite scalar > 0")
                membrane_drive_mv = np.float32(membrane_drive_mv)
                selected_kicks = np.full(
                    len(membrane_drive_rows), membrane_drive_mv,
                    dtype=np.float32)
                drive_mode = "absolute_mv"
                requested = float(membrane_drive_mv)
            membrane_drive_receipt = {
                "mode": drive_mode,
                "rows": int(len(membrane_drive_rows)),
                "requested": float(requested),
                "threshold_margin_mv": {
                    "min": float(margin.min()),
                    "mean": float(margin.mean()),
                    "max": float(margin.max()),
                    "unique_values": int(len(np.unique(margin))),
                },
                "effective_kick_mv": {
                    "min": float(selected_kicks.min()),
                    "mean": float(selected_kicks.mean()),
                    "max": float(selected_kicks.max()),
                    "unique_values": int(len(np.unique(selected_kicks))),
                },
                "effective_v_rest_includes_overrides": True,
                "backend": "numpy",
            }
        if drive_window is None:
            step_on, step_off = 0, n_steps
        else:
            step_on = int(round(drive_window[0] / p.dt))
            step_off = int(round(drive_window[1] / p.dt))

        # CLOSED LOOP. drive_fn(t_ms, motor_counts) -> (indices, rates_hz),
        # called every drive_every_ms, replacing the fixed Poisson drive above.
        # This is what makes the sensory periphery a function of what the motor
        # neurons just did, instead of unstructured noise. motor_counts is the
        # spike count of `feedback_rows` since the previous call, so the caller
        # never has to reach into simulator state.
        fb_every = max(1, int(round(drive_every_ms / p.dt)))
        fb_rows = (None if feedback_rows is None
                   else np.asarray(feedback_rows, dtype=np.int64))
        # Optional graded feedback exposes the continuous release state already
        # delivered to downstream synapses. It replaces only the selected
        # callback entries; all other entries retain spike-count units.
        fb_graded_rows = (None if feedback_graded_release_rows is None
                          else np.asarray(
                              feedback_graded_release_rows, dtype=np.int64))
        fb_graded_callback_pos = np.empty(0, dtype=np.int64)
        fb_graded_release_pos = np.empty(0, dtype=np.int64)
        if fb_graded_rows is not None:
            if fb_rows is None:
                raise ValueError(
                    "feedback_graded_release_rows requires feedback_rows")
            if (fb_graded_rows.ndim != 1
                    or len(np.unique(fb_graded_rows)) != len(fb_graded_rows)
                    or np.any(fb_graded_rows < 0)
                    or np.any(fb_graded_rows >= self.N)):
                raise ValueError(
                    "feedback_graded_release_rows must be unique valid rows")
            row_to_callback = {int(row): pos
                               for pos, row in enumerate(fb_rows)}
            graded_to_release = {int(row): pos
                                 for pos, row in enumerate(graded_idx)}
            if any(int(row) not in row_to_callback
                   for row in fb_graded_rows):
                raise ValueError(
                    "graded feedback rows must be present in feedback_rows")
            if any(int(row) not in graded_to_release
                   for row in fb_graded_rows):
                raise ValueError(
                    "graded feedback rows must designate graded neurons")
            fb_graded_callback_pos = np.asarray(
                [row_to_callback[int(row)] for row in fb_graded_rows],
                dtype=np.int64)
            fb_graded_release_pos = np.asarray(
                [graded_to_release[int(row)] for row in fb_graded_rows],
                dtype=np.int64)
        fb_accum = (None if fb_rows is None else np.zeros(
            len(fb_rows), dtype=(np.float64 if fb_graded_rows is not None
                                 else np.int64)))
        if graded_drive_max_rate_hz is not None:
            if (type(graded_drive_max_rate_hz) not in (int, float)
                    or not np.isfinite(graded_drive_max_rate_hz)
                    or float(graded_drive_max_rate_hz) <= 0.0):
                raise ValueError(
                    "graded_drive_max_rate_hz must be finite and positive")
            graded_drive_max_rate_hz = float(graded_drive_max_rate_hz)
        if type(graded_drive_signed) is not bool:
            raise ValueError("graded_drive_signed must be Boolean")
        if graded_drive_signed and graded_drive_max_rate_hz is None:
            raise ValueError(
                "graded_drive_signed requires graded_drive_max_rate_hz")
        if drive_fn is not None and force_spikes_fn is not None:
            raise ValueError("drive_fn and force_spikes_fn are mutually exclusive")
        if force_spikes_fn is not None:
            if fb_rows is None:
                raise ValueError("force_spikes_fn requires feedback_rows")
            cadence = Fraction(str(drive_every_ms)) / Fraction(str(p.dt))
            if cadence <= 0 or cadence.denominator != 1:
                raise ValueError("force_spikes_fn cadence must be a positive integral number of CNS steps")
            fb_every = int(cadence)

        rec = np.arange(self.N) if record is None else np.asarray(record)
        # Membrane voltage for a handful of neurons, at full dt. This is the
        # raw quantity the model integrates -- spikes are just where it crosses
        # threshold -- so it is the honest view of what the simulation does.
        vrec = None if record_voltage is None else np.asarray(record_voltage)
        vtrace = (None if vrec is None
                  else np.full((n_steps, len(vrec)), np.nan, dtype=np.float32))
        trace = None
        if record_trace_every:
            n_bins = int(np.ceil(duration_ms / record_trace_every))
            trace = np.zeros((n_bins, len(rec)), dtype=np.int32)
            steps_per_bin = int(round(record_trace_every / p.dt))

        indptr, indices, data = self._indptr, self._indices, self._data

        use_release_scale = use_std or presynaptic_inhibition

        def _deliver_queued(src, release_snap, slot_):
            """One gather and one bincount for a whole arrival tick.

            Writes into ring[slot_] rather than into g so graded release,
            written into the same slot earlier, is still summed first: the
            accumulation order matches the bucket path exactly.
            """
            start = indptr[src]
            nper = indptr[src + 1] - start
            tot = int(nper.sum())
            if not tot:
                return
            off = (np.repeat(start, nper) + np.arange(tot)
                   - np.repeat(np.cumsum(nper) - nper, nper))
            rows = indices[off]
            vals = data[off]
            if release_snap is not None:
                vals = vals * np.repeat(release_snap, nper)
            if ring_i is not None:
                ring[slot_] += np.bincount(
                    rows, weights=np.maximum(vals, 0.0),
                    minlength=self.N).astype(np.float32)
                ring_i[slot_] += np.bincount(
                    rows, weights=np.maximum(-vals, 0.0),
                    minlength=self.N).astype(np.float32)
            else:
                ring[slot_] += np.bincount(
                    rows, weights=vals, minlength=self.N).astype(np.float32)
        total_spikes = 0

        # coincidence seam: two source pools per designated target, the
        # slow one delayed, multiplied into the target's synaptic drive.
        _coin = None
        if (p.coincidence_gain and p.coincidence_targets is not None
                and p.coincidence_fast_rows is not None
                and p.coincidence_slow_rows is not None):
            _ctgt = np.asarray(p.coincidence_targets, dtype=np.int64)
            if _ctgt.size:
                _tpos = np.full(self.N, -1, dtype=np.int64)
                _tpos[_ctgt] = np.arange(len(_ctgt))
                _pre_all = np.repeat(
                    np.arange(self.N, dtype=np.int64),
                    np.diff(self._indptr))
                _post_all = self._indices
                _in_t = _tpos[_post_all] >= 0
                _mats = []
                for _srcs in (p.coincidence_fast_rows,
                              p.coincidence_slow_rows):
                    _m = _in_t & np.isin(_pre_all,
                                         np.asarray(_srcs, dtype=np.int64))
                    _mats.append(sparse.csr_matrix(
                        (np.abs(self._data[_m]).astype(np.float32),
                         (_tpos[_post_all[_m]], _pre_all[_m])),
                        shape=(len(_ctgt), self.N)))
                # 2026-09-05, judge, on lane C's C-N32: the seam multiplies
                # EXISTING edge weights, so a source pool with no edges onto the
                # designated targets builds an all-zero matrix and contributes
                # exactly nothing at any gain. Lane C measured that failure as
                # bit-identical output at gain 1e15 against a positive control
                # that moved 26,313 spikes, and the assembled matrices gave no
                # warning because their COMBINED edge count was large: an
                # aggregate hiding a zero, the same shape as the calibration
                # ledger's claw rate that morning. An arm whose mechanism is
                # structurally absent is worse than one that fails, because it
                # reads as evidence. So refuse it here rather than run it.
                for _nm, _mat in (("fast", _mats[0]), ("slow", _mats[1])):
                    if _mat.nnz == 0:
                        raise ValueError(
                            f"coincidence seam: the {_nm} source pool has NO "
                            f"edges onto the {len(_ctgt)} designated targets, so "
                            f"its product term is identically zero at any gain. "
                            f"Check coincidence_{_nm}_rows against the target "
                            f"set; the other pool carries {_mats[1 - (_nm == 'fast')].nnz} edges.")
                _clag = max(1, int(round(float(p.coincidence_lag_ms)
                                         / float(p.dt))))
                _coin = {"tgt": _ctgt, "fast": _mats[0], "slow": _mats[1],
                         "ring": np.zeros((_clag, len(_ctgt)), np.float32),
                         "gain": np.float32(p.coincidence_gain),
                         "prev": np.zeros(self.N, np.float32),
                         "fs": np.zeros(len(_ctgt), np.float32),
                         "ss": np.zeros(len(_ctgt), np.float32),
                         "decay": np.float32(np.exp(
                             -float(p.dt)
                             / max(1e-6, float(p.coincidence_tau_ms)))),
                         }

        for t in range(n_steps):
            slot = t % n_delay
            if gap_ring is not None:
                # spikelets whose conduction time ends this tick
                _gs = t % gap_S
                v = v + gap_ring[_gs]
                gap_ring[_gs] = 0.0
            g *= decay_g
            if g_i is not None:
                # CB6 FIX (2026-08-19): the inhibitory pool decays on the
                # SAME kernel as the excitatory one. Without this line g_i
                # only ever accumulated -- an unbounded inhibitory
                # conductance that pinned every neuron at e_inh. Every
                # measurement made under shunt=True before this fix is
                # withdrawn. Caught by a dial moving the wrong way: more
                # inhibition produced MORE firing (0.00 -> 324 spikes/tick),
                # which is impossible.
                g_i *= decay_g_i
            if use_std:
                # exact relaxation of the vesicle pool toward 1
                x_res *= std_dec
                x_res += np.float32(1.0) - std_dec
            if graded_release_q is not None:
                _due = graded_release_q[slot]
                if np.any(_due):
                    _gr = (Wg @ _due).astype(np.float32)
                    if ring_i is not None:
                        ring[slot] += np.maximum(_gr, 0.0)
                        ring_i[slot] += np.maximum(-_gr, 0.0)
                    else:
                        ring[slot] += _gr
                    _due.fill(0.0)
            if fire_q is not None:
                _b = fire_q[t % n_queue]
                if _b:
                    fire_q[t % n_queue] = []
                    if use_release_scale:
                        _arr = np.concatenate([x[0] for x in _b])
                        _xr = np.concatenate([x[1] for x in _b])
                    else:
                        _arr, _xr = np.concatenate(_b), None
                    _deliver_queued(_arr, _xr, slot)
            g += ring[slot]
            ring[slot] = 0.0
            if _coin is not None:
                _sv = _coin["prev"]
                _coin["fs"] *= _coin["decay"]
                _coin["fs"] += _coin["fast"] @ _sv
                _coin["ss"] *= _coin["decay"]
                _coin["ss"] += _coin["slow"] @ _sv
                _k = t % _coin["ring"].shape[0]
                _slow_del = _coin["ring"][_k].copy()
                _coin["ring"][_k] = _coin["ss"]
                _capp = _coin["gain"] * _coin["fs"] * _slow_del
                g[_coin["tgt"]] += _capp
                # 2026-09-05, judge, after two wrong causes for lane C's inert
                # seam: measure the term instead of theorising about it. These
                # accumulate what the seam ACTUALLY applied, so an arm reporting
                # a bit-identical result can say whether the product was zero,
                # tiny, or large-but-ineffective. Read on the CNS object after a
                # run; four scalars per tick, no effect on the dynamics.
                _coin["applied_max"] = max(_coin.get("applied_max", 0.0),
                                           float(np.abs(_capp).max()))
                _coin["applied_sum"] = (_coin.get("applied_sum", 0.0)
                                        + float(np.abs(_capp).sum()))
                _coin["fs_max"] = max(_coin.get("fs_max", 0.0),
                                      float(_coin["fs"].max()))
                _coin["slowdel_max"] = max(_coin.get("slowdel_max", 0.0),
                                           float(_slow_del.max()))
            if ring_i is not None:
                g_i += ring_i[slot]
                ring_i[slot] = 0.0

            # Everything that enters the membrane equation as a current, in mV.
            # The adaptation variable is subtracted here so it composes with
            # both the current-based and the conductance-based synapse model
            # rather than being a third mutually exclusive branch.
            extra = np.float32(0.0)
            if serotonin_configured:
                serotonin_state_hz *= serotonin_decay
                serotonin_state_applied_sum_hz += float(serotonin_state_hz)
                if serotonin_extra is not None:
                    serotonin_extra.fill(0.0)
                    serotonin_extra[serotonin_target] = np.float32(
                        serotonin_gain) * serotonin_state_hz
                    extra = serotonin_extra
            if G is not None and gap_scale:
                # sum_j v_j - deg_i * v_i  -- the Laplacian, in mV. No delay
                # and no spike needed: this is what makes it electrical. It
                # carries SUBTHRESHOLD coupling only; the spike itself arrives
                # as a spikelet below, because a LIF spike has no waveform.
                if ap_ring is None:
                    extra = extra + gap_scale * (G @ v - gap_deg * v)
                else:
                    # the partners see v plus this tick's AP waveform value
                    _as = t % ap_S
                    extra = extra + gap_scale * (G @ (v + ap_ring[_as]) - gap_deg * v)
                    ap_ring[_as] = 0.0
            if use_adapt:
                extra = extra - w_adapt
            if use_exp:
                extra = extra + p.delta_T * np.exp(
                    np.minimum((v - p.v_threshold) * inv_dT, 20.0))

            if p.conductance_based:
                # Split the accumulated input by sign and apply each with its
                # own driving force, so inhibition saturates at e_inh instead
                # of running away.
                if g_i is not None:
                    # true shunting: BOTH conductances open at once, so
                    # total membrane conductance rises and every input
                    # produces less voltage (divisive, not subtractive)
                    ge = np.maximum(g, 0.0) * gscale
                    gi = g_i * gscale
                else:
                    ge = np.maximum(g, 0.0) * gscale
                    gi = np.maximum(-g, 0.0) * gscale
                if tonic_ge is not None:
                    ge = ge + tonic_ge
                if conductance_integrator == "analytic":
                    # Exact one-step solution with ge, gi and extra held
                    # constant over dt:
                    #   tau dV/dt = Vrest - V + ge(Ee-V) + gi(Ei-V) + I.
                    # Normalising before multiplying by reversal potentials
                    # avoids the overflow-prone ge*V product in Euler. The
                    # exponential approaches the instantaneous equilibrium
                    # monotonically for any finite non-negative conductance.
                    total_g = np.float32(1.0) + ge + gi
                    inv_total_g = np.float32(1.0) / total_g
                    v_inf = (v_rest * inv_total_g
                             + p.e_exc * (ge * inv_total_g)
                             + e_inh * (gi * inv_total_g)
                             + extra * inv_total_g)
                    conductance_decay = np.exp(
                        -dt_over_tau * total_g).astype(np.float32)
                    v = v_inf + (v - v_inf) * conductance_decay
                else:
                    v = (v_rest + (v - v_rest) * decay_v
                         + (ge * (p.e_exc - v)
                            + gi * (e_inh - v) + extra) * dt_over_tau)
            else:
                v = v_rest + (v - v_rest) * decay_v + (g + extra) * dt_over_tau

            if use_adapt:
                # Forward Euler; dt/tau_w is ~1e-3 so this is far inside the
                # stable region. w keeps evolving during refractoriness, which
                # is what makes the brake outlast the spike that set it.
                w_adapt += dt_over_tauw * (adapt_a * (v - v_rest) - w_adapt)

            active = refrac <= 0
            v = np.where(active, v, v_reset)
            if use_exp:
                np.minimum(v, p.v_peak, out=v)

            if vtrace is not None:
                vtrace[t] = v[vrec]

            # Graded cells never cross threshold -- they have no threshold.
            fired = np.flatnonzero(active & (v >= spike_at) & ~graded)

            if drive_fn is not None and t % fb_every == 0:
                idx_new, rate_new = drive_fn(t * p.dt, fb_accum)
                drive_idx = np.asarray(idx_new, dtype=np.int64)
                drive_rate = np.asarray(rate_new, dtype=np.float64)
                if drive_idx.shape != drive_rate.shape:
                    raise ValueError(
                        "drive_fn indices and rates must have matching shapes")
                drive_prob = drive_rate * p.dt / 1000.0
                if use_reg_vector:
                    drive_reg = spike_reg_vector[drive_idx]
                # Non-finite afferent guard (ledger L5, re-freeze event
                # 2026-08-31): with spike_reg set, the renewal accumulator
                # never recovers from one NaN (reg_acc += NaN poisons it
                # permanently), so a non-finite rate silently kills that
                # afferent for the rest of the run. Fail fast instead.
                # Finite-rate behavior is unchanged (probe-verified).
                if not np.all(np.isfinite(drive_prob)):
                    bad = drive_idx[~np.isfinite(drive_prob)]
                    raise FloatingPointError(
                        f"non-finite afferent rate for {len(bad)} cells "
                        f"(first ids: {bad[:5].tolist()}) at t={t * p.dt} ms")
                if fb_accum is not None:
                    fb_accum[:] = 0

            dynamic_force = None
            if force_spikes_fn is not None and t % fb_every == 0:
                callback_counts = np.frombuffer(
                    fb_accum.astype(np.int64, copy=False).tobytes(order="C"),
                    dtype=np.int64,
                )
                returned = force_spikes_fn(t * p.dt, callback_counts)
                if type(returned) is not np.ndarray or returned.ndim != 1 or \
                        returned.dtype != np.dtype(np.int64):
                    raise ValueError(
                        "force_spikes_fn must return an exact 1-D np.int64 array"
                    )
                if len(returned) and (
                    (returned < 0).any() or (returned >= self.N).any()
                ):
                    raise IndexError("force_spikes_fn returned row out of bounds")
                if len(returned) != len(np.unique(returned)):
                    raise ValueError("force_spikes_fn returned duplicate rows")
                dynamic_force = returned.copy()
                fb_accum[:] = 0

            # DETERMINISTIC injection: {t_ms: indices}. Poisson drive cannot
            # express "this named cell spikes exactly once, at t = 0", which is
            # the actual protocol of every giant-fibre latency measurement --
            # the experimenter stimulates the brain and times the muscle. A
            # rate is the wrong object for a single-shot latency.
            if force_ms is not None:
                f = force_ms.get(round(t * p.dt, 3))
                if f is not None:
                    fired = np.union1d(fired, np.asarray(f, dtype=np.int64))
            if dynamic_force is not None and len(dynamic_force):
                fired = np.union1d(fired, dynamic_force)

            # Known non-spiking cells cannot be made to spike by an external
            # Poisson boundary. When explicitly enabled, map the existing
            # sensory command continuously between each cell's own rest and
            # upper membrane anchors. Signed mode mirrors the same span below
            # rest. It adds no threshold or gain: the upper anchor is the
            # theta's existing per-row (or scalar) spike_at value.
            graded_drive_mask = None
            if graded_drive_max_rate_hz is not None and len(drive_idx):
                graded_drive_mask = graded[drive_idx]
                if (graded_drive_signed
                        and np.any(drive_rate[~graded_drive_mask] < 0.0)):
                    raise ValueError(
                        "negative signed drive requires graded neurons")
                if graded_drive_mask.any():
                    driven = drive_idx[graded_drive_mask]
                    level = np.clip(
                        drive_rate[graded_drive_mask]
                        / graded_drive_max_rate_hz,
                        -1.0 if graded_drive_signed else 0.0, 1.0)
                    upper = np.broadcast_to(
                        np.asarray(spike_at, dtype=np.float32), (self.N,))
                    rest = np.broadcast_to(
                        np.asarray(v_rest, dtype=np.float32), (self.N,))
                    v[driven] = (rest[driven]
                                 + level * (upper[driven] - rest[driven]))

            if len(drive_idx) and step_on <= t < step_off:
                # Externally clamped graded rows release continuously and can
                # never emit afferent spikes. Exclude them BEFORE drawing, not
                # after: under renewal drive, allowing their accumulators to
                # cross consumed extra gamma draws and silently changed every
                # later sensory spike when only the graded command changed.
                # This branch exists only when the graded-drive boundary is
                # explicitly enabled; the default path is unchanged.
                spiking_drive_mask = (
                    None if graded_drive_mask is None
                    else ~graded_drive_mask)
                spiking_drive_idx = (
                    drive_idx if spiking_drive_mask is None
                    else drive_idx[spiking_drive_mask])
                spiking_drive_prob = (
                    drive_prob if spiking_drive_mask is None
                    else drive_prob[spiking_drive_mask])
                if not use_reg:
                    crossed = (rng.random(len(spiking_drive_idx))
                               < spiking_drive_prob)
                    hit = spiking_drive_idx[crossed]
                elif not use_reg_vector:
                    # N1: renewal firing. The accumulator integrates the
                    # per-step firing probability (= expected spikes), and
                    # an entry fires when it crosses its Erlang-k
                    # threshold (mean 1 after normalisation by k, so the
                    # RATE matches Poisson exactly; only the TIMING
                    # regularises: CV = 1/sqrt(k)).
                    n_e = len(spiking_drive_idx)
                    if reg_acc is None or len(reg_acc) != n_e:
                        reg_acc = np.zeros(n_e)
                        reg_thr = rng.gamma(_kreg, 1.0 / _kreg, n_e)
                    reg_acc += spiking_drive_prob
                    crossed = reg_acc >= reg_thr
                    if crossed.any():
                        reg_acc[crossed] = 0.0
                        reg_thr[crossed] = rng.gamma(
                            _kreg, 1.0 / _kreg, int(crossed.sum()))
                    hit = spiking_drive_idx[crossed]
                else:
                    # Mixed boundary statistics: evidence-anchored afferents
                    # retain renewal firing while unanchored central/command
                    # rows use the existing Poisson process.
                    spiking_drive_reg = (
                        drive_reg if spiking_drive_mask is None
                        else drive_reg[spiking_drive_mask])
                    n_e = len(spiking_drive_idx)
                    renew = spiking_drive_reg > 1.0
                    if reg_acc is None or len(reg_acc) != n_e:
                        reg_acc = np.zeros(n_e)
                        reg_thr = np.full(n_e, np.nan)
                        if renew.any():
                            reg_thr[renew] = rng.gamma(
                                spiking_drive_reg[renew],
                                1.0 / spiking_drive_reg[renew])
                    crossed = np.zeros(n_e, dtype=bool)
                    poisson = ~renew
                    if poisson.any():
                        crossed[poisson] = (
                            rng.random(int(poisson.sum()))
                            < spiking_drive_prob[poisson])
                    if renew.any():
                        reg_acc[renew] += spiking_drive_prob[renew]
                        renew_crossed = renew & (reg_acc >= reg_thr)
                        crossed |= renew_crossed
                        if renew_crossed.any():
                            reg_acc[renew_crossed] = 0.0
                            reg_thr[renew_crossed] = rng.gamma(
                                spiking_drive_reg[renew_crossed],
                                1.0 / spiking_drive_reg[renew_crossed])
                    hit = spiking_drive_idx[crossed]
                if len(hit) and membrane_drive_mask is not None:
                    # Membrane delivery: the same drawn events, but as a
                    # depolarising kick the threshold test reads next step
                    # instead of a spike no synapse can veto.
                    _mem = membrane_drive_mask[hit]
                    if _mem.any():
                        membrane_drive_events += int(
                            len(np.unique(hit[_mem])))
                        if membrane_drive_kick_vector is None:
                            # Keep the legacy scalar arithmetic literal.
                            v[hit[_mem]] += membrane_drive_mv
                        else:
                            v[hit[_mem]] += (
                                membrane_drive_kick_vector[hit[_mem]])
                        hit = hit[~_mem]
                if len(hit):
                    fired = np.union1d(fired, hit[refrac[hit] <= 0])

            # 2026-09-05, lane C's C-N42, ported by the judge: this write used
            # to sit immediately after the threshold test 175 lines above,
            # BEFORE driven spikes are merged into `fired`, so `prev` never
            # marked a driven cell and the coincidence seam was blind to
            # exactly the carriers it exists to multiply. Measured by lane C:
            # Mi1 driven at 20 Hz emitted 30,589 spikes while its accumulator
            # read 0.0000 in either pool slot. It belongs after the merge,
            # where `fired` is complete for the tick.
            if _coin is not None:
                _coin["prev"][:] = 0.0
                if len(fired):
                    _coin["prev"][fired] = 1.0

            if serotonin_configured and len(fired):
                _n_serotonin = int(serotonin_source_mask[fired].sum())
                if _n_serotonin:
                    serotonin_source_spikes += _n_serotonin
                    serotonin_state_hz += (
                        serotonin_increment_hz * _n_serotonin)
                    serotonin_state_max_hz = max(
                        serotonin_state_max_hz, serotonin_state_hz)

            # GRADED RELEASE, every tick, no spike required. This is the whole
            # difference: a spiking cell delivers w at discrete times, a graded
            # cell delivers release(v)*w continuously.
            if Wg is not None and (t % g_every) == 0:
                # clip before exp: v can sit far below v50 for a cell with no
                # input, and np.exp overflows to inf. 1/(1+inf) is 0 either way,
                # but an overflow warning in the innermost loop is how a real
                # NaN later gets ignored as "just that warning again".
                rel = 1.0 / (1.0 + np.exp(np.clip(
                    -(v[graded_idx] - gv50) / gslope, -60.0, 60.0)))
                if gphasic > 0.0:
                    if graded_ema is None:
                        graded_ema = rel.copy()
                    graded_ema = graded_ema + gphasic_alpha * (rel - graded_ema)
                    rel = np.clip(
                        (rel - gphasic * graded_ema) * (1.0 + 2.0 * gphasic),
                        0.0, 1.0,
                    )
                if fb_graded_rows is not None:
                    # Mean release over the callback window. Multiplying by
                    # g_every compensates decimated updates in the same way as
                    # the delivered synaptic charge path below.
                    fb_accum[fb_graded_callback_pos] += (
                        rel[fb_graded_release_pos]
                        * (float(g_every) / float(fb_every)))
                # scaled by g_every so the DELIVERED CHARGE is unchanged when
                # the update rate changes -- otherwise decimating would quietly
                # weaken every graded synapse and look like a modelling result.
                # Per-row gain scales the release before the matvec, so a
                # vector costs one extra multiply on the graded rows; the
                # scalar branch is the historical arithmetic, untouched.
                if ggain_v is not None:
                    rel = (ggain_v * rel.astype(np.float32)).astype(np.float32)
                if graded_release_q is not None:
                    _relf = rel.astype(np.float32)
                    _queued = (g_every * _relf if ggain_v is not None else
                               ggain * g_every * _relf).astype(np.float32)
                    _arrival = (t + graded_delay_slots) % n_delay
                    graded_release_q[_arrival, graded_positions] = _queued
                elif graded_delay_groups is None:
                    _gr = ((g_every * (Wg @ rel.astype(np.float32)))
                           if ggain_v is not None else
                           ggain * g_every
                           * (Wg @ rel.astype(np.float32))).astype(np.float32)
                    if ring_i is not None:
                        ring[(t + n_delay) % n_delay] += np.maximum(_gr, 0.0)
                        ring_i[(t + n_delay) % n_delay] += np.maximum(-_gr, 0.0)
                    else:
                        ring[(t + n_delay) % n_delay] += _gr
                else:
                    _relf = rel.astype(np.float32)
                    for _s, _pos, _Wg_s in graded_delay_groups:
                        _gr = ((g_every * (_Wg_s @ _relf[_pos]))
                               if ggain_v is not None else
                               ggain * g_every
                               * (_Wg_s @ _relf[_pos])).astype(np.float32)
                        _tgt = (t + _s) % n_delay
                        if ring_i is not None:
                            ring[_tgt] += np.maximum(_gr, 0.0)
                            ring_i[_tgt] += np.maximum(-_gr, 0.0)
                        else:
                            ring[_tgt] += _gr

            refrac -= 1
            if len(fired):
                v[fired] = v_reset[fired]
                refrac[fired] = n_refrac[fired]
                counts[fired] += 1
                total_spikes += len(fired)
                if t_first is not None:
                    new = fired[t_first[fired] < 0]
                    if len(new):
                        t_first[new] = t
                if fb_accum is not None:
                    hit_fb = np.isin(fb_rows, fired, assume_unique=False)
                    if hit_fb.any():
                        fb_accum[hit_fb] += 1
                if use_adapt:
                    w_adapt[fired] += (adapt_b if adapt_b.ndim == 0
                                       else adapt_b[fired])

                # SPIKELET: instantaneous depolarisation of gap partners. No
                # ring buffer -- an electrical synapse has no transmitter
                # release and essentially no delay, which is the entire reason
                # the giant fibre is faster than any chemical path.
                if ap_ring is not None:
                    # schedule the firing cells' AP waveform for their partners,
                    # after the axon delay
                    for _k, _a in enumerate(ap_wave):
                        ap_ring[(t + ap_off + _k) % ap_S, fired] += _a
                if G is not None and gap_spike:
                    hot = np.zeros(self.N, dtype=np.float32)
                    hot[fired] = 1.0
                    if gap_ring is None:
                        v = v + gap_spike * (G @ hot)
                    else:
                        # delayed by the axon's conduction time (gap_delay_ms);
                        # delivered at the top of the arrival tick below.
                        gap_ring[(t + gap_ds) % gap_S] += gap_spike * (G @ hot)

                # Snapshot source-side release at spike time.  STD and the
                # evidence-scoped hook shunt multiply cleanly when both are
                # enabled; absent both, the historical propagation branch is
                # untouched.  gi is already in the simulator's synaptic mV
                # units, so gscale converts it to the same dimensionless
                # conductance used by the membrane equation.
                release_scale = (x_res[fired].copy() if use_std else None)
                if presynaptic_inhibition:
                    _pm = presyn_mask[fired]
                    if _pm.any():
                        _fac = (np.float32(1.0) / (
                            np.float32(1.0)
                            + g_i[fired[_pm]] * np.float32(gscale)))
                        if release_scale is None:
                            release_scale = np.ones(len(fired), dtype=np.float32)
                        release_scale[_pm] *= _fac
                        presyn_spikes += int(_pm.sum())
                        presyn_attenuated_spikes += int((_fac < 1.0).sum())
                        presyn_factor_sum += float(_fac.sum())
                        presyn_factor_min = min(
                            presyn_factor_min, float(_fac.min()))
                if use_release_scale and release_scale is None:
                    # Queue records carry a scale array uniformly when any
                    # source-side mechanism is enabled. Non-target sources
                    # therefore carry explicit unity, not a mixed tuple shape.
                    release_scale = np.ones(len(fired), dtype=np.float32)

                # ragged gather of the fired neurons' outgoing columns
                if fire_q is not None:
                    # Grouping uses a STABLE sort so a single delay class
                    # reproduces the bucket path's summation order exactly.
                    _d = delay_slots[fired]
                    _o = np.argsort(_d, kind="stable")
                    _fs, _ds = fired[_o], _d[_o]
                    _u, _st = np.unique(_ds, return_index=True)
                    _sc = (None if release_scale is None
                           else release_scale[_o])
                    _sc_parts = ([None] * len(_u) if _sc is None
                                 else np.split(_sc, _st[1:]))
                    for _dv_i, _seg, _seg_scale in zip(
                            _u, np.split(_fs, _st[1:]), _sc_parts):
                        _k = (t + int(_dv_i)) % n_queue
                        if use_release_scale:
                            # Snapshot all source-side release effects at
                            # SPIKE time, so conduction delay cannot move them
                            # to the arrival tick.
                            fire_q[_k].append((_seg, _seg_scale.copy()))
                        else:
                            fire_q[_k].append(_seg)
                    if use_std:
                        x_res[fired] *= std_keep
                    tot = 0
                else:
                    start = indptr[fired]
                    end = indptr[fired + 1]
                    nper = end - start
                    tot = int(nper.sum())
                if tot:
                    off = (np.repeat(start, nper)
                           + np.arange(tot)
                           - np.repeat(np.cumsum(nper) - nper, nper))
                    rows = indices[off]
                    vals = data[off]
                    if release_scale is not None:
                        # deliver at the CURRENT pool level, then deplete
                        vals = vals * np.repeat(release_scale, nper)
                    if use_std:
                        x_res[fired] *= std_keep
                    if delay_slots is None:
                        tgt = (t + n_delay) % n_delay
                        if ring_i is not None:
                            pos = np.maximum(vals, 0.0)
                            neg = np.maximum(-vals, 0.0)
                            ring[tgt] += np.bincount(
                                rows, weights=pos,
                                minlength=self.N).astype(np.float32)
                            ring_i[tgt] += np.bincount(
                                rows, weights=neg,
                                minlength=self.N).astype(np.float32)
                        else:
                            ring[tgt] += np.bincount(
                                rows, weights=vals,
                                minlength=self.N).astype(np.float32)
                    else:
                        # each spike arrives after ITS SOURCE's delay:
                        # partition this tick's edges by source delay slot
                        src_slot = delay_slots[fired]
                        edge_slot = np.repeat(src_slot, nper)
                        for _s in np.unique(src_slot):
                            _m = edge_slot == _s
                            tgt = (t + int(_s)) % n_delay
                            if ring_i is not None:
                                pos = np.maximum(vals[_m], 0.0)
                                neg = np.maximum(-vals[_m], 0.0)
                                ring[tgt] += np.bincount(
                                    rows[_m], weights=pos,
                                    minlength=self.N).astype(np.float32)
                                ring_i[tgt] += np.bincount(
                                    rows[_m], weights=neg,
                                    minlength=self.N).astype(np.float32)
                            else:
                                ring[tgt] += np.bincount(
                                    rows[_m], weights=vals[_m],
                                    minlength=self.N).astype(np.float32)

            if trace is not None:
                b = t // steps_per_bin
                if b < len(trace) and len(fired):
                    sel = np.isin(fired, rec)
                    if sel.any():
                        pos = np.searchsorted(rec, fired[sel])
                        np.add.at(trace[b], pos, 1)

        out = {
            "counts": counts[rec],
            **({"t_first_ms": np.where(t_first >= 0, t_first * p.dt, np.nan)}
               if t_first is not None else {}),
            "rates_hz": counts[rec] / (duration_ms / 1000.0),
            "n_spikes": total_spikes,
            "duration_ms": duration_ms,
            "recorded": rec,
        }
        if serotonin_configured:
            out["serotonin"] = {
                "source_rows": int(len(serotonin_source)),
                "target_rows": int(len(serotonin_target)),
                "source_spikes": int(serotonin_source_spikes),
                "tau_ms": float(serotonin_tau_ms),
                "gain_mv_per_hz": float(serotonin_gain),
                "state_final_hz": float(serotonin_state_hz),
                "state_max_hz": float(serotonin_state_max_hz),
                "state_mean_applied_hz": float(
                    serotonin_state_applied_sum_hz / max(1, n_steps)),
            }
        if presynaptic_inhibition:
            out["presynaptic_inhibition"] = {
                "rows": int(len(presyn_rows)),
                "source_spikes": int(presyn_spikes),
                "attenuated_spikes": int(presyn_attenuated_spikes),
                "release_factor_mean": (
                    None if not presyn_spikes
                    else float(presyn_factor_sum / presyn_spikes)),
                "release_factor_min": (
                    None if not presyn_spikes else float(presyn_factor_min)),
            }
        if membrane_drive_receipt is not None:
            membrane_drive_receipt["drawn_unique_row_tick_events"] = int(
                membrane_drive_events)
            out["membrane_drive"] = membrane_drive_receipt
        if _coin is not None:
            # The seam's own receipt, so a bit-identical arm can say WHY.
            out["coincidence"] = {
                "fast_nnz": int(_coin["fast"].nnz),
                "slow_nnz": int(_coin["slow"].nnz),
                "n_targets": int(len(_coin["tgt"])),
                "gain": float(_coin["gain"]),
                "applied_abs_max": float(_coin.get("applied_max", 0.0)),
                "applied_abs_sum": float(_coin.get("applied_sum", 0.0)),
                "fs_max": float(_coin.get("fs_max", 0.0)),
                "slow_delayed_max": float(_coin.get("slowdel_max", 0.0)),
            }
        if trace is not None:
            out["trace"] = trace
        if vtrace is not None:
            out["voltage"] = vtrace
            out["voltage_idx"] = vrec
        return out


def load(meta_path="data/banc_888_meta.feather",
         edge_path="data/banc_888_edgelist_simple_v3.feather",
         params=None):
    meta = pd.read_feather(meta_path)
    edges = pd.read_feather(edge_path)
    return CNS(meta, edges, params)


if __name__ == "__main__":
    import time
    t0 = time.time()
    net = load()
    print(f"loaded in {time.time()-t0:.1f}s")
    print(f"neurons {net.N}  edges {net.n_edges}  synapses {net.n_syn:.0f}")
    print(f"edges dropped (endpoint not in meta): {net.n_edges_dropped}")
    print(f"neurons with no transmitter at all:   {net.n_unknown_nt}")
    print(f"sign breakdown: exc {(net.sign>0).sum()}  inh {(net.sign<0).sum()}  "
          f"zero {(net.sign==0).sum()}")
    t0 = time.time()
    r = net.run(50.0, seed=0)
    print(f"50 ms in {time.time()-t0:.1f}s wall, {r['n_spikes']} spikes")
