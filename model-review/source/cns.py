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
    # POST-BALANCE EDGE GAIN (2026-09-24, design 0c25fab3). One gain on the
    # EXISTING positive final weights from one set of source rows onto one
    # set of target rows, applied to the built CSC data after every
    # constructor transform, so the balance is frozen and no other input is
    # rebalanced. (pre_rows, post_rows, gain); () is bit-identical (branch not
    # taken). Zero and negative contacts are left as they are and listed in
    # the audit. Folded into W, so the native engines read it with no kernel
    # change.
    post_balance_edge_gain: tuple = ()
    # POST-BALANCE EDGE GAIN BLOCKS (2026-10-05, design efe88957 Build 5).
    # ((pre_rows, post_rows, gain), ...) applied in order at the same point:
    # every existing final weight from a block's source rows onto its target
    # rows is multiplied by its gain, sign kept, so an inhibitory edge takes
    # it too. Exclusive with post_balance_edge_gain; a one-element tuple whose
    # block holds no negative contact gives that field's W. () is
    # bit-identical (branch not taken).
    post_balance_edge_gain_blocks: tuple = ()
    # PSI PAIR KERNEL (2026-09-24, design 7b0ad6b2, H01). Separately delayed
    # exponential conductance states for named source-target pairs. After
    # every constructor transform, post_balance_edge_gain included, each
    # pair's final weight becomes an explicit zero in W and the pair's own
    # state x = r*x + G*Q*(1-r)*n/dt carries the source's events instead, with
    # Q = a*dt/(1-r0) the area of the ordinary impulse it replaces (a the first
    # used normalized conductance, r0 the target's native decay). The tuple is
    # (pairs, delay_ms, tau_ms, gain): pairs ((src_row, tgt_row), ...);
    # delay_ms from the source event to the first used tick, None for the
    # source's inherited delivery slot; tau_ms, None for the target's inherited
    # decay; gain G > 0. () is bit-identical (branch not taken). The native
    # CUDA and Metal engines carry the states, and NumPy since 2026-10-07
    # (gpu-7, gate gpu/gate_psi_numpy.py). Design
    # 727bccb8 (H02) appends (beta, factor_from): each area becomes
    # K*Q**(1-beta) at unchanged total, its factors optionally permuted
    # across the pairs (see _psi_pair_kernel).
    psi_pair_kernel: tuple = ()
    # Design e05c0eef B1: (form, tau_rec_ms, u), the recovery of every pair of
    # psi_pair_kernel. "dead_time" (u None): an event arriving within
    # round(tau_rec_ms/dt) ticks of the pair's last delivery delivers nothing;
    # "depletion": a resource R recovers by R = 1 - (1 - R)*exp(-dt/tau_rec)
    # per tick and an event delivers inc*R and keeps R*(1 - u). () is the
    # pre-B1 channel, and so are dead time at tau_rec_ms 0 and depletion at
    # u 0. Every engine, as psi_pair_kernel (see psi_pair_recovery_values).
    psi_pair_recovery: tuple = ()
    # SHARED OCCUPANCY (2026-09-27, design 81427d47, H03). One state per
    # target row replaces the ordinary conductance of every existing entry
    # from pre_rows onto post_rows, each of which must be positive. Mode "T"
    # (occupancy): per tick o = o*exp(-dt/tau_close), then with A the tick's
    # summed arrival weight o = 1-(1-o)*exp(-alpha*A), delivering
    # float32(Gmax*o); mode "L" (linear): h = h*exp(-dt/tau_close) + beta*A,
    # delivering float32(h). The tuple is (pre_rows, post_rows, mode, k, eta,
    # tau_close_ms, interrupt), eta None for "L"; see shared_occupancy_arrays.
    # The selected entries become explicit zeros in W. () is bit-identical
    # (branch not taken). CUDA only: NumPy and Metal refuse it.
    shared_occupancy: tuple = ()
    # LEFT APL GABA (2026-09-21, design fb9d4a58, Build 1). Row
    # 720575941734526507 carries no cell type and no transmitter in BANC, so
    # it takes the unknown default (sign +1, w_unknown_scale). True sets its
    # predicted transmitter to "gaba" on a copy of meta before the
    # transmitter is resolved; balance runs as usual afterwards. False is
    # bit-identical (the branch is not taken).
    left_apl_gaba: bool = False
    # KC-TO-KC ACETYLCHOLINE (2026-09-21, design fb9d4a58, Builds 2 and 3).
    # None is bit-identical. "off" sets every edge with a Kenyon cell at both
    # ends to 0 before the balance step; "calyx_kept" multiplies each such
    # edge by the share of its synapses in MB_CA, from kc_kc_calyx, a
    # (table_sha256, ((pre_id, post_id, n, n_ca), ...)) pair whose n must
    # equal the edge list count. kc_kc_ids are the 4,552 KC banc_888_ids.
    # Entries stay explicit in W, so nnz does not change.
    kc_kc_ach: object = None
    kc_kc_ids: tuple = ()
    kc_kc_calyx: object = None
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
    # Design efe88957 Build 2: the seam's list form, one dict per triple from
    # walk_search._coincidence_triples_parse ({targets, fast, slow} rows,
    # gain, lag_ms, tau_ms, read). A graded source is read by its release
    # over graded_every, any other by its spike. Empty = off = bit-identical;
    # exclusive with coincidence_gain.
    coincidence_triples: tuple = ()
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
    # Optional conserving electrical law. Absent preserves the old kernel.
    gap_law: object = None
    gap_reverse_ratio: object = None
    gap_substep_factor: int = 1
    gap_current_tap: bool = False

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
    # One U and one tau_rec for every neuron, and one resource per
    # presynaptic neuron shared by all its outputs, is the model's
    # assumption: every outgoing synapse of neuron j shares j's spike
    # history, so the resource x_j is ONE float per neuron, not one per
    # synapse. Kim 2025 (measurement b6bf62b3) finds the DM1 PN onto LHN1
    # fits this form (U 0.052, 672 ms, within 0.0018 per spike) and onto
    # LHN2 does not (facilitation, then 1,791 ms, a resource per target).
    # Nagel 2015 (e8113fb7) gives the lab's arm values, U 0.22 and 893 ms.
    # The per-row resource is release_resource_* below. On j firing:
    # delivered weight scales by x_j, then x_j *= (1 - U). Between spikes x
    # recovers toward 1 with tau_rec. std_U = 0 is off and bit-identical.
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
    # avoid.  Needs conductance_based; runs on NumPy and, since 2026-09-06
    # (549703a6c), on Metal.
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
    # Design 302c1bfb, Build 8: the factor with a gain k >= 0,
    # 1/(1 + G*float32(k*gscale)). None is k = 1 and computes the bytes of the
    # parameter-free factor above. With source rows set, G is not the row's
    # whole inhibitory conductance but a second state g9 per listed row, fed
    # only by the inhibitory arrivals from the source rows with an
    # inhibitory sign: it decays by r9 = exp(-dt/tau) per step, tau =
    # presynaptic_inhibition_tau_ms (None is the row's own inhibitory
    # decay), and each arrival adds its g_i increment times
    # (1 - r9)/(1 - r_i), so its area equals the ordinary impulse's. The
    # arrivals still enter g_i and the membrane unchanged. Queued delivery
    # only; NumPy, Metal and, since 982bfb258 (design 9501604f, Build C1),
    # CUDA, which also refuses it without split inhibition or beside the
    # release resource or the slow inhibitory pool.
    presynaptic_inhibition_gain: object = None
    presynaptic_inhibition_source_rows: object = None
    presynaptic_inhibition_tau_ms: object = None
    # PER-ROW RELEASE RESOURCE (2026-09-21, design 0e0587be, Build 1). The
    # std_U recurrence above, restricted to named rows with one U and one
    # recovery time per row: d = float32(exp(-dt/tau)), c = 1 - d, k = float32
    # (1 - U); each tick x <- fl(fl(x*d) + c); a spike (threshold, driven or
    # forced) snapshots x, every outgoing synapse delivers fl(w*x) at its own
    # arrival tick, then x <- fl(x*k). x starts at 1.0. Rows are int64, sorted
    # and unique; U and tau_rec_ms are float64, one per row. None is
    # bit-identical: no array is allocated and no branch is taken. run()
    # returns the release log (tick, row, x) and its sha256.
    release_resource_rows: object = None
    release_resource_U: object = None
    release_resource_tau_rec_ms: object = None
    # Design 02e57559: sorted unique target rows that receive the emitted x;
    # every other output of a resource row delivers its full weight. None =
    # every target. CUDA engine only; this engine refuses it.
    release_resource_target_rows: object = None
    # ADAPTATION WARM START (design fbcd0f61, Build B6). w at t = 0 on the
    # listed rows: sorted unique int64 rows, one float32 value in mV per row;
    # every other row starts at zero. None is bit-identical. Refused when
    # adaptation is off for the network. This engine and CUDA; Metal refuses.
    adapt_w_init_rows: object = None
    adapt_w_init_mv: object = None
    # OPSIN CONDUCTANCE (design 87a3ad35, B1). One or more sets of rows, each
    # with its own reversal, maximal conductance in multiples of the row's
    # leak conductance (so it does not pass through gscale), first-order
    # opening and closing constants and light windows in simulated ms. Per
    # step, before the membrane update, each opsin row's state s moves by the
    # exact one-step solution: lit, s <- 1 + (s - 1) a_on; dark, s <- s a_off,
    # with a = float32(exp(-dt/tau)) computed once. The membrane gains
    # g_op = g_max s with reversal E_op, appended last in both integrators.
    # Step t is lit when round(t_on/dt) <= t < round(t_off/dt), the rule
    # drive_window uses. Each field is a tuple with one entry per set:
    # rows (sorted unique int64), e_rev (mV, or "e_inh", bound to the run's
    # e_inh), g_max, tau_on_ms, tau_off_ms, windows ((t_on, t_off), ...).
    # None is bit-identical. Conductance law only. opsin_arrays() validates.
    opsin_rows: object = None
    opsin_e_rev_mv: object = None
    opsin_g_max: object = None
    opsin_tau_on_ms: object = None
    opsin_tau_off_ms: object = None
    opsin_windows_ms: object = None
    # Design 2163ba7d, Build 1: a light pulse train per set, beside the
    # windows. None, or one entry per set, each None or (start_ms, stop_ms,
    # on_ms, off_ms): step t is also lit when k_start <= t < k_stop and
    # (t - k_start) mod (k_on + k_off) < k_on, every edge a whole number of
    # steps or opsin_arrays raises. A tau of exactly 0 is the instantaneous
    # switch (a = 0: s is 1 on lit steps and 0 on dark ones).
    opsin_train_ms: object = None
    # Check 2(b)'s hook (note e887d7ef, item 4): one bool per set; while true
    # s is 1 on every lit step in place of the opening update. Refused on a
    # network with any edge, so only the isolated probe cells can carry it.
    opsin_force_open: object = None
    # Arm E of design 87a3ad35, the threshold stand-in: on lit steps of the
    # windows these rows' spike test reads threshold_window_mv (mV, absolute)
    # in place of their own threshold, as silence_rows does for a whole run.
    # Drive and forced events still fire them, as they fire any row. Rides
    # the opsin channel in the native engines as slots with g_max 0.
    threshold_window_rows: object = None
    threshold_window_mv: object = None
    threshold_window_ms: object = None
    # Design 2163ba7d, Build 2: a slow inhibitory pool on the output of the
    # listed rows. Of each inhibitory increment a listed row's spike delivers,
    # the share keep = float32(1 - frac) stays in the fast pool and
    # fl(increment * sfac) goes to the target's slow conductance g_s, which
    # decays by d_s = float32(exp(-dt/tau_ms)) and joins the membrane as
    # gi = g_i*gscale + g_s*gscale, the fast pool's reversal. sfac =
    # float32(frac * A), A = (1 - d_s)/(1 - d_i) from the two float32 decays,
    # so at frac 1 the slow pool delivers the conductance-time the fast pool
    # delivers at frac 0. Rows are sorted unique int64 with every outgoing
    # weight <= 0; record rows get, per bin of record_bin_ms, the sum of
    # g_s*gscale over the bin's steps. Conductance law with split pools, the
    # scalar tau_syn and the queue delivery only; slow_inh_arrays() validates.
    # None is bit-identical, and frac 0 is bit-identical to None.
    slow_inh_rows: object = None
    slow_inh_frac: float = 0.0
    slow_inh_tau_ms: float = 200.0
    slow_inh_record_rows: object = None
    slow_inh_record_bin_ms: float = 5.0

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
    # Design 6d0042d5, Build 2: the subthreshold coupling where v < v_rest.
    # None is adapt_a on both sides, the symmetric law. A scalar or (N,) array
    # splits the term per row: adapt_a above rest and this below, so
    # a max(x, 0) is (a, 0), a min(x, 0) is (0, a) and a x is (a, a).
    adapt_a_below: object = None
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
    # Design b1746728, Build 1: the state, and its running maximum, at t = 0
    # in Hz. 0 is the historical start; a walk that must open at the dark
    # equilibrium sets the equilibrium its dark arm measured.
    serotonin_init_hz: float = 0.0
    # Design 8b6e0441, Build 2: one constant in mV added into the membrane
    # equation's extra term on every row of octopamine_target_rows, every
    # step, after the serotonin, gap and injected terms and before the
    # adaptation; negative lowers excitability. Rows None is the historical
    # path, bit-identical, and an offset of 0 adds nothing.
    octopamine_target_rows: object = None
    octopamine_offset_mv: float = 0.0
    # Design 8b6e0441, Build 1: a second modulatory state of the serotonin
    # state's shape. Every step s decays by exp(-dt/tau) and every source
    # spike adds 1000/(N_src tau), so s settles at the mean per-source rate in
    # Hz; every row of octopamine_target_rows gets gain * s mV in extra, added
    # after the serotonin term (summed on rows both reach) and before the
    # offset above. N_src is the given set's size. A source row with no
    # nonzero weight onto a target row in the built network (a descending row
    # under the neck cut) is left out of the spike count and named in the
    # receipt (critique 28ee40b6, condition 1). Source rows None is the
    # historical path, bit-identical.
    octopamine_source_rows: object = None
    octopamine_tau_ms: float = 2300.0
    octopamine_gain_mv_per_hz: float = 0.0
    octopamine_init_hz: float = 0.0
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
    # nt_transfer_rows (design e2b1a20d B5): a transmitter transferred by a
    # cross-dataset label onto rows BANC leaves untyped. Keys are banc_888_id
    # strings, values transmitter tokens. Written into
    # neurotransmitter_predicted on a COPY of meta before transmitter
    # resolution; a row with a verified transmitter or any BANC cell_type is
    # refused, so a transfer never overwrites a typed or verified call.
    # EMPTY = bit-identical.
    nt_transfer_rows: dict = field(default_factory=dict)


def release_resource_constants(U, tau_rec_ms, dt):
    """Per-row float32 (d, c, k): d = float32(exp(-dt/tau)) with the
    exponential in float64, c = float32(1) - d (exact), k = float32(1 - U)
    with the subtraction in float64."""
    tau = np.asarray(tau_rec_ms, dtype=np.float64)
    d = np.exp(-float(dt) / tau).astype(np.float32)
    c = (np.float32(1.0) - d).astype(np.float32)
    k = (1.0 - np.asarray(U, dtype=np.float64)).astype(np.float32)
    return d, c, k


def adapt_w_init_arrays(p, n_rows):
    """Design fbcd0f61, Build B6: the validated (rows int64, values float32) of
    the adaptation warm start; the caller tests adapt_w_init_rows for None."""
    rows = np.asarray(p.adapt_w_init_rows, dtype=np.int64)
    vals = np.asarray(p.adapt_w_init_mv, dtype=np.float32)
    if rows.ndim != 1 or not len(rows) or vals.shape != rows.shape:
        raise ValueError("adapt_w_init_rows and adapt_w_init_mv need one value per row")
    if np.any(np.diff(rows) <= 0) or rows[0] < 0 or rows[-1] >= n_rows:
        raise ValueError("adapt_w_init_rows must be sorted, unique and in range")
    if not np.all(np.isfinite(vals)):
        raise ValueError("adapt_w_init_mv must be finite")
    return rows, vals


def release_resource_arrays(p, n_rows, graded=None):
    """The validated per-row release resource of SimParams, or None when off.

    Refuses U outside (0, 1], a tau that is not finite and positive, an empty,
    unsorted, duplicated or out-of-range row, use with std_U > 0, and a row in
    the graded-release set (`graded`, a boolean mask over rows)."""
    rows = getattr(p, "release_resource_rows", None)
    if rows is None:
        return None
    rows = np.asarray(rows)
    if rows.ndim != 1 or not len(rows) or rows.dtype.kind not in "iu":
        raise ValueError("release_resource_rows must be a non-empty 1-D "
                         "integer array")
    rows = rows.astype(np.int64)
    if (np.any(rows < 0) or np.any(rows >= n_rows)
            or np.any(np.diff(rows) <= 0)):
        raise ValueError("release_resource_rows must be sorted, unique, "
                         "valid rows")
    U = np.asarray(getattr(p, "release_resource_U", None), dtype=np.float64)
    tau = np.asarray(getattr(p, "release_resource_tau_rec_ms", None),
                     dtype=np.float64)
    if U.shape != rows.shape or tau.shape != rows.shape:
        raise ValueError("release_resource_U and release_resource_tau_rec_ms "
                         "need one value per row")
    if not np.all(np.isfinite(U)) or np.any(U <= 0.0) or np.any(U > 1.0):
        raise ValueError("release_resource_U must lie in (0, 1]")
    if not np.all(np.isfinite(tau)) or np.any(tau <= 0.0):
        raise ValueError("release_resource_tau_rec_ms must be finite and "
                         "positive")
    if float(getattr(p, "std_U", 0.0)) > 0.0:
        raise ValueError("the per-row release resource refuses std_U > 0")
    if graded is not None and np.any(np.asarray(graded)[rows]):
        raise ValueError("a release-resource row is in the graded-release set")
    d, c, k = release_resource_constants(U, tau, p.dt)
    return {"rows": rows, "U": U, "tau_rec_ms": tau, "d": d, "c": c, "k": k}


def release_log_pack(ticks, rows, xs):
    """The release log sorted by tick then row, and its sha256 over the
    little-endian int32 ticks, int32 rows and float32 x, in that order."""
    import hashlib
    t = np.asarray(ticks, dtype=np.int32).ravel()
    r = np.asarray(rows, dtype=np.int32).ravel()
    x = np.asarray(xs, dtype=np.float32).ravel()
    o = np.lexsort((r, t))
    t, r, x = t[o], r[o], x[o]
    h = hashlib.sha256()
    for a, dt_ in ((t, "<i4"), (r, "<i4"), (x, "<f4")):
        h.update(np.ascontiguousarray(a.astype(dt_)).tobytes())
    return {"tick": t, "row": r, "x": x}, h.hexdigest()


OPSIN_MAX_WINDOWS = 4   # per set, in every engine
OPSIN_TAP_FIELDS = ("s", "g_op", "i_after", "i_start")


def opsin_arrays(p, n_rows, duration_ms, e_inh, n_edges, graded=None):
    """The validated opsin channel of SimParams (design 87a3ad35, B1), or None.

    One slot per opsin row and per threshold-window row, in set order, each
    with float32 a_on, a_off, g_max, e_rev and thr_mv, a force flag and its
    set's windows in steps. `e_inh` is the run's resolved reversal (float32
    scalar or (n_rows,) vector), which a set with e_rev "e_inh" binds to.
    Raises on: opsin fields without rows; the current-based law; a row that
    is out of range, unsorted, repeated or in two sets; a non-finite or
    negative g_max; a non-finite or non-positive tau; a reversal that is
    neither finite nor "e_inh"; a window outside the run, empty in steps, or
    overlapping another of its set; more than OPSIN_MAX_WINDOWS windows per
    set; force_open on a network with any edge; a threshold-window row that
    is graded, which has no threshold to raise; a pulse train (design
    2163ba7d) whose edges are not whole steps, that leaves the run, or whose
    on or off phase is empty. A tau of 0 gives a = 0. Threshold-window slots carry
    kind 1, g_max 0 and a = 0, so their s is 1 on lit steps and 0 otherwise
    and their conductance term is an exact zero in every engine."""
    dt = float(p.dt)
    names = ("opsin_rows", "opsin_e_rev_mv", "opsin_g_max", "opsin_tau_on_ms",
             "opsin_tau_off_ms", "opsin_windows_ms", "opsin_force_open",
             "opsin_train_ms")
    got = {k: getattr(p, k, None) for k in names}
    thr_rows = getattr(p, "threshold_window_rows", None)
    thr_mv = getattr(p, "threshold_window_mv", None)
    thr_ms = getattr(p, "threshold_window_ms", None)
    op_on = got["opsin_rows"] is not None and len(got["opsin_rows"]) > 0
    thr_on = thr_rows is not None and len(thr_rows) > 0
    if not op_on and any(got[k] is not None and len(got[k]) > 0 for k in names[1:]):
        raise ValueError("opsin parameters are set but opsin_rows is empty")
    if not thr_on and (thr_mv is not None or thr_ms is not None):
        raise ValueError("threshold_window parameters are set but its rows are empty")
    if not op_on and not thr_on:
        return None

    def _windows(spec, what):
        spec = list(spec) if spec is not None else []
        if len(spec) > OPSIN_MAX_WINDOWS:
            raise ValueError(f"{what}: more than {OPSIN_MAX_WINDOWS} windows")
        steps = []
        for w in spec:
            if len(w) != 2:
                raise ValueError(f"{what}: a window is (t_on_ms, t_off_ms)")
            a, b = float(w[0]), float(w[1])
            if not (np.isfinite(a) and np.isfinite(b)) or a < 0.0 or b > float(duration_ms) or a >= b:
                raise ValueError(f"{what}: window ({a}, {b}) lies outside the "
                                 f"{float(duration_ms)} ms run or is empty")
            k0, k1 = int(round(a / dt)), int(round(b / dt))
            if k0 >= k1:
                raise ValueError(f"{what}: window ({a}, {b}) is empty in steps")
            steps.append((k0, k1))
        steps.sort()
        for (a0, a1), (b0, _) in zip(steps, steps[1:]):
            if b0 < a1:
                raise ValueError(f"{what}: windows overlap")
        return spec, steps

    def _train(spec, what):
        if spec is None:
            return None, (0, 0, 0, 0)
        if len(spec) != 4:
            raise ValueError(f"{what}: a pulse train is (start_ms, stop_ms, on_ms, off_ms)")
        k = []
        for x in spec:
            x = float(x)
            q = x / dt
            if not np.isfinite(x) or abs(q - round(q)) > 1e-6:
                raise ValueError(f"{what}: pulse train edge {x} ms is not a whole number of "
                                 f"{dt} ms steps")
            k.append(int(round(q)))
        k0, k1, kon, koff = k
        if k0 < 0 or k1 > int(round(float(duration_ms) / dt)) or k0 >= k1 or kon <= 0 or koff <= 0:
            raise ValueError(f"{what}: pulse train {tuple(spec)} leaves the "
                             f"{float(duration_ms)} ms run or has an empty phase")
        return [float(x) for x in spec], (k0, k1, kon, kon + koff)

    def _rows(r, what):
        if isinstance(r, (int, np.integer)):
            raise ValueError(f"{what}: rows are an array")
        r = np.asarray(r)
        if r.ndim != 1 or not len(r) or r.dtype.kind not in "iu":
            raise ValueError(f"{what}: rows must be a non-empty 1-D integer array")
        r = r.astype(np.int64)
        if np.any(r < 0) or np.any(r >= n_rows) or np.any(np.diff(r) <= 0):
            raise ValueError(f"{what}: rows must be sorted, unique, valid rows")
        return r

    e_inh = np.asarray(e_inh, dtype=np.float32)
    sets, slots = [], []
    if op_on:
        if isinstance(got["opsin_rows"], np.ndarray):
            raise ValueError("opsin_rows is a tuple of row arrays, one per set")
        n_set = len(got["opsin_rows"])
        for k in names[1:6]:
            if got[k] is None or len(got[k]) != n_set:
                raise ValueError(f"{k} needs one entry per opsin set ({n_set})")
        trains = got["opsin_train_ms"]
        if trains is None:
            trains = (None,) * n_set
        if len(trains) != n_set:
            raise ValueError(f"opsin_train_ms needs one entry per opsin set ({n_set})")
        force = got["opsin_force_open"]
        if force is None:
            force = (False,) * n_set
        if len(force) != n_set or any(type(f) not in (bool, np.bool_) for f in force):
            raise ValueError("opsin_force_open needs one bool per opsin set")
        if not p.conductance_based:
            raise ValueError("the opsin conductance requires conductance_based=True: "
                             "a reversal potential has no meaning in the current-based law")
        if any(force) and n_edges:
            raise ValueError("opsin_force_open refuses a network with edges: it exists "
                             "for the isolated probe cells of checks 2(b)")
        for j in range(n_set):
            what = f"opsin set {j}"
            rows = _rows(got["opsin_rows"][j], what)
            g_max = float(got["opsin_g_max"][j])
            t_on = float(got["opsin_tau_on_ms"][j])
            t_off = float(got["opsin_tau_off_ms"][j])
            if not np.isfinite(g_max) or g_max < 0.0:
                raise ValueError(f"{what}: g_max must be finite and >= 0")
            for nm, tau in (("tau_on_ms", t_on), ("tau_off_ms", t_off)):
                if not np.isfinite(tau) or tau < 0.0:
                    raise ValueError(f"{what}: {nm} must be finite and >= 0")
            e_req = got["opsin_e_rev_mv"][j]
            if isinstance(e_req, str):
                if e_req != "e_inh":
                    raise ValueError(f"{what}: e_rev is a number in mV or 'e_inh'")
                e_rows = (np.full(len(rows), e_inh, np.float32) if e_inh.ndim == 0
                          else e_inh[rows].astype(np.float32))
            else:
                e_val = float(e_req)
                if not np.isfinite(e_val):
                    raise ValueError(f"{what}: e_rev must be finite")
                e_rows = np.full(len(rows), e_val, np.float32)
            spec, steps = _windows(got["opsin_windows_ms"][j], what)
            t_spec, t_steps = _train(trains[j], what)
            a_on = np.float32(0.0) if t_on == 0.0 else np.float32(np.exp(-dt / t_on))
            a_off = np.float32(0.0) if t_off == 0.0 else np.float32(np.exp(-dt / t_off))
            sets.append({"kind": "opsin", "rows": rows.tolist(), "n_rows": int(len(rows)),
                         "e_rev_request": e_req, "e_rev_mv": [float(x) for x in e_rows],
                         "g_max": float(np.float32(g_max)), "tau_on_ms": t_on,
                         "tau_off_ms": t_off, "a_on": float(a_on), "a_off": float(a_off),
                         "windows_ms": [[float(a), float(b)] for a, b in spec],
                         "windows_steps": [list(s) for s in steps],
                         "force_open": bool(force[j]),
                         **({"train_ms": t_spec,
                             "train_steps": {"start": t_steps[0], "stop": t_steps[1],
                                             "on": t_steps[2], "period": t_steps[3]}}
                            if t_spec is not None else {})})
            for q, r in enumerate(rows):
                slots.append((int(r), j, a_on, a_off, np.float32(g_max), e_rows[q],
                              np.float32(0.0), int(bool(force[j])), steps, 0, t_steps))
    if thr_on:
        what = "threshold window"
        rows = _rows(thr_rows, what)
        off = float(thr_mv) if thr_mv is not None else float("nan")
        if not np.isfinite(off):
            raise ValueError(f"{what}: threshold_window_mv must be finite")
        if graded is not None and np.any(np.asarray(graded)[rows]):
            raise ValueError(f"{what}: a graded row has no threshold to raise")
        spec, steps = _windows(thr_ms, what)
        if not steps:
            raise ValueError(f"{what}: threshold_window_ms needs at least one window")
        j = len(sets)
        sets.append({"kind": "threshold", "rows": rows.tolist(), "n_rows": int(len(rows)),
                     "threshold_mv": float(np.float32(off)),
                     "windows_ms": [[float(a), float(b)] for a, b in spec],
                     "windows_steps": [list(s) for s in steps]})
        for r in rows:
            slots.append((int(r), j, np.float32(0.0), np.float32(0.0), np.float32(0.0),
                          np.float32(0.0), np.float32(off), 0, steps, 1, (0, 0, 0, 0)))
    all_rows = np.array([s[0] for s in slots], dtype=np.int64)
    if len(np.unique(all_rows)) != len(all_rows):
        raise ValueError("a row is in more than one opsin or threshold-window set")
    n = len(slots)
    win = np.zeros((n, OPSIN_MAX_WINDOWS, 2), dtype=np.int64)
    n_win = np.zeros(n, dtype=np.uint32)
    for q, s in enumerate(slots):
        for w, (k0, k1) in enumerate(s[8]):
            win[q, w] = (k0, k1)
        n_win[q] = len(s[8])
    col = lambda i, dt_: np.array([s[i] for s in slots], dtype=dt_)
    train = np.array([s[10] for s in slots], dtype=np.int64).reshape(n, 4)
    return {"rows": all_rows, "set": col(1, np.int32), "a_on": col(2, np.float32),
            "a_off": col(3, np.float32), "g_max": col(4, np.float32),
            "e_rev": col(5, np.float32), "thr_mv": col(6, np.float32),
            "force": col(7, np.uint8), "kind": col(9, np.uint8),
            "win": win, "n_win": n_win,
            "train": train if np.any(train[:, 3] > 0) else None,
            "readback": {"design": "87a3ad35", "dt_ms": dt, "gscale_applied": False,
                         "lit_rule": "round(t_on/dt) <= step < round(t_off/dt)",
                         "record_time_ms": "(step + 1) * dt",
                         "sets": sets}}


def opsin_out(op, tap):
    """out["opsin"] in every engine: the readback, the slot map and, with the
    recorder on, the four (n_steps, n_slots) float32 arrays and one sha256
    over their bytes in OPSIN_TAP_FIELDS order."""
    r = {"readback": op["readback"], "slot_rows": op["rows"].tolist(),
         "slot_set": op["set"].tolist(), "slot_kind": op["kind"].tolist()}
    if tap is not None:
        import hashlib
        h = hashlib.sha256()
        for k in OPSIN_TAP_FIELDS:
            r[k] = np.ascontiguousarray(tap[k], dtype=np.float32)
            h.update(r[k].tobytes())
        r["tap_sha256"] = h.hexdigest()
    return r


def opsin_lit(win, n_win, t, train=None):
    """Per slot, whether step t lies inside one of its windows or in an on
    phase of its pulse train (design 2163ba7d; train rows are (start, stop,
    on, period) steps, period 0 for none)."""
    lit = ((win[:, :, 0] <= t) & (t < win[:, :, 1])
           & (np.arange(win.shape[1])[None, :] < n_win[:, None])).any(axis=1)
    if train is not None:
        per = train[:, 3]
        lit = lit | ((per > 0) & (train[:, 0] <= t) & (t < train[:, 1])
                     & (((t - train[:, 0]) % np.maximum(per, 1)) < train[:, 2]))
    return lit


def row_exact_arrays(p, n_rows, analytic, op):
    """Design d27f82df B8: p.row_exact_mask (B2) and p.holddown_slots (B3)
    validated as cns_engine.cu's configuration validates them, or None when p
    carries neither. Returns {"xm": the rows on the exact update, sorted
    int64, or None under the analytic integrator, where every row already
    is; "hd": the hold-down slots or None}. "hd" holds the slot arrays as the
    engines keep them under "slots", the row indices as int64, the pool rows
    as a bool mask, latch_ticks as float32 for the latch test, and each
    slot's place in "xm". The opsin is refused beside either: Metal carries
    the two in the opsin's buffers and no design runs them together."""
    mask = getattr(p, "row_exact_mask", None)
    slots = getattr(p, "holddown_slots", None)
    if mask is None and slots is None:
        return None
    if op is not None:
        raise NotImplementedError(
            "design d27f82df's exact-update mask and hold-down run beside the "
            "opsin on the native CUDA engine only")
    if not p.conductance_based:
        raise ValueError("row exact and hold-down need the conductance-based membrane")
    on = None
    if mask is not None:
        m = np.asarray(mask)
        if m.shape != (n_rows,):
            raise ValueError("row_exact_mask needs one byte per row")
        on = m != 0
        if not on.any():
            raise ValueError("row exact: no row set")
    hd = None
    if slots is not None:
        from cns_metal import HOLDDOWN_SLOT_FIELDS
        if set(slots) != set(HOLDDOWN_SLOT_FIELDS):
            raise ValueError(f"holddown_slots must carry exactly {HOLDDOWN_SLOT_FIELDS}")
        dts = {"rows": np.int32, "form": np.uint32, "latch_ticks": np.uint32}
        arr = {k: np.ascontiguousarray(slots[k], dtype=dts.get(k, np.float32))
               for k in HOLDDOWN_SLOT_FIELDS}
        n = len(arr["rows"])
        if not n or any(a.shape != (n,) for a in arr.values()):
            raise ValueError("holddown_slots arrays must share one non-zero length")
        dt = np.float32(p.dt)
        if not (np.isfinite(dt) and dt > 0):
            raise ValueError("hold-down: dt must be finite and above zero")
        if not analytic and on is None:
            raise ValueError("hold-down: no row is on the exact update")
        rows = arr["rows"].astype(np.int64)
        if (rows < 0).any() or (rows >= n_rows).any():
            raise ValueError("hold-down: row out of range")
        if len(np.unique(rows)) != n:
            raise ValueError("hold-down: row repeated")
        if not analytic and not on[rows].all():
            raise ValueError("hold-down: a row not on the exact update")
        if (arr["form"] > 1).any():
            raise ValueError("hold-down: unknown form")
        fl = ("e_rel", "decay", "dg", "gmax", "c_spk", "k", "va", "tau_c", "latch_mv")
        if not all(np.isfinite(arr[k]).all() for k in fl):
            raise ValueError("hold-down: a non-finite value")
        if not ((arr["decay"] > 0) & (arr["decay"] <= 1)).all():
            raise ValueError("hold-down: decay outside (0, 1]")
        if any((arr[k] < 0).any() for k in ("dg", "gmax", "c_spk", "k")):
            raise ValueError("hold-down: a negative dg, gmax, c_spk or k")
        if not (arr["tau_c"] > 0).all():
            raise ValueError("hold-down: tau_c must be above zero")
        if (arr["latch_ticks"] == 0).any():
            raise ValueError("hold-down: latch_ticks 0")
        hd = {**{k: arr[k] for k in fl}, "slots": arr, "rows": rows,
              "pool": arr["form"] == 1, "latch_ticks": arr["latch_ticks"].astype(np.float32),
              "dt": dt}
    xm = None if analytic else np.flatnonzero(on).astype(np.int64)
    if hd is not None and xm is not None:
        hd["xpos"] = np.searchsorted(xm, hd["rows"])
    return {"xm": xm, "hd": hd}


def holddown_out(rows, state, tap):
    """out["holddown"] in NumPy and on Metal: per slot its row, the state
    after the last step and the run's peak g_h with its step, and with the
    recorder on one (n_steps, n_slots) float32 array per channel of
    cns_metal.HOLDDOWN_TAP_CHANNELS and one sha256 over their bytes in that
    order. tap is (n_steps, n_slots, channels) float32."""
    from cns_metal import HOLDDOWN_TAP_CHANNELS
    r = {"rows": [int(x) for x in rows]}
    for k in ("c", "g_h", "L", "b"):
        r["end_" + k] = np.asarray(state[k], dtype=np.float32).copy()
    r["peak_g_h"] = np.asarray(state["peak"], dtype=np.float32).copy()
    r["peak_tick"] = np.asarray(state["peak_tick"], dtype=np.uint32).copy()
    if tap is not None:
        import hashlib
        h = hashlib.sha256()
        for j, k in enumerate(HOLDDOWN_TAP_CHANNELS):
            r[k] = np.ascontiguousarray(tap[:, :, j], dtype=np.float32)
            h.update(r[k].tobytes())
        r["tap_sha256"] = h.hexdigest()
    return r


def slow_inh_arrays(p, n_rows, indptr, indices, data, n_steps, graded=None):
    """The validated slow inhibitory pool of SimParams (design 2163ba7d,
    Build 2), or None when slow_inh_rows is None.

    Returns the listed-source mask, the target rows (every target of a
    listed row, sorted) and each row's target slot or -1, the float32 keep,
    sfac, d_s and d_i (d_i computed as run() computes decay_g_i), the area
    factor A in float64, the record rows' target slots (-1 for a record row
    no listed row reaches), the bin in steps, the number of bins, and the
    receipt. Raises on: record fields without rows; the current-based law or
    unsplit pools; a per-neuron tau_syn or the split decay (tau_syn_exc,
    tau_syn_inh); presynaptic inhibition, which reads g_i; listed rows that
    are empty, unsorted, repeated, out of range or graded; a listed row with
    an excitatory output, or no inhibitory output among them; frac outside
    [0, 1]; a non-finite or non-positive tau; record rows that are unsorted,
    repeated or out of range; a bin that is not a whole number of steps."""
    rows = getattr(p, "slow_inh_rows", None)
    rec = getattr(p, "slow_inh_record_rows", None)
    if rows is None:
        if rec is not None:
            raise ValueError("slow_inh_record_rows is set but slow_inh_rows is None")
        return None
    if not (p.conductance_based and p.split_conductances):
        raise ValueError("slow_inh needs conductance_based and split_conductances")
    if p.tau_syn_exc is not None or p.tau_syn_inh is not None:
        raise ValueError("slow_inh needs the scalar tau_syn, not tau_syn_exc/tau_syn_inh")
    tau_syn = np.asarray(p.tau_syn, dtype=np.float32)
    if tau_syn.ndim != 0 or not np.isfinite(tau_syn) or float(tau_syn) <= 0.0:
        raise ValueError("slow_inh needs a finite positive scalar tau_syn")
    if getattr(p, "presynaptic_inhibition_rows", None) is not None:
        raise ValueError("slow_inh is refused with presynaptic_inhibition_rows")

    def _sorted_rows(x, what):
        a = np.asarray(x)
        if (a.ndim != 1 or not len(a) or a.dtype.kind not in "iu"
                or np.any(np.diff(a.astype(np.int64)) <= 0)
                or int(a.min()) < 0 or int(a.max()) >= n_rows):
            raise ValueError(f"{what} must be non-empty, sorted, unique rows in range")
        return a.astype(np.int64)

    rows = _sorted_rows(rows, "slow_inh_rows")
    if graded is not None and np.any(np.asarray(graded)[rows]):
        raise ValueError("slow_inh_rows names a graded row, which releases without spikes")
    frac, tau = float(p.slow_inh_frac), float(p.slow_inh_tau_ms)
    if not (np.isfinite(frac) and 0.0 <= frac <= 1.0):
        raise ValueError("slow_inh_frac must lie in [0, 1]")
    if not (np.isfinite(tau) and tau > 0.0):
        raise ValueError("slow_inh_tau_ms must be finite and > 0")
    e = np.concatenate([np.arange(indptr[r], indptr[r + 1]) for r in rows])
    w, tg = data[e], indices[e]
    if np.any(w > 0):
        raise ValueError("a slow_inh row has an excitatory output")
    if not np.any(w < 0):
        raise ValueError("no slow_inh row has an inhibitory output")
    tgt = np.unique(tg).astype(np.int64)
    tpos = np.full(n_rows, -1, dtype=np.int64)
    tpos[tgt] = np.arange(len(tgt))
    mask = np.zeros(n_rows, dtype=bool)
    mask[rows] = True
    d_i = np.float32(np.exp(-p.dt / tau_syn))
    d_s = np.float32(np.exp(-float(p.dt) / tau))
    A = (1.0 - float(d_s)) / (1.0 - float(d_i))
    keep, sfac = np.float32(1.0 - frac), np.float32(frac * A)
    bin_ms = float(p.slow_inh_record_bin_ms)
    spb = round(bin_ms / float(p.dt))
    if spb < 1 or abs(spb * float(p.dt) - bin_ms) > 1e-9 * max(1.0, bin_ms):
        raise ValueError("slow_inh_record_bin_ms must be a whole number of steps")
    rec = (np.zeros(0, dtype=np.int64) if rec is None
           else _sorted_rows(rec, "slow_inh_record_rows"))
    n_bins = -(-int(n_steps) // spb)
    import hashlib
    receipt = {"design": "2163ba7d", "build": 2, "rows": rows.tolist(),
               "frac": frac, "tau_ms": tau, "dt_ms": float(p.dt),
               "tau_syn_ms": float(tau_syn), "keep": float(keep), "sfac": float(sfac),
               "decay_slow": float(d_s), "decay_fast": float(d_i), "area_factor": A,
               "n_targets": len(tgt),
               "targets_sha256": hashlib.sha256(tgt.astype("<i8").tobytes()).hexdigest(),
               "n_edges": len(e), "n_edges_inhibitory": int(np.count_nonzero(w < 0)),
               "record_rows": rec.tolist(),
               "record_rows_reached": int(np.count_nonzero(tpos[rec] >= 0)),
               "bin_ms": bin_ms, "steps_per_bin": spb, "n_bins": n_bins}
    return {"mask": mask, "targets": tgt, "tpos": tpos, "keep": keep, "sfac": sfac,
            "d_s": d_s, "d_i": d_i, "A": A, "rec_pos": tpos[rec], "spb": spb,
            "n_bins": n_bins, "receipt": receipt}


def slow_inh_out(sl, acc_fast, acc_slow, intg, g_end, pending, rec_acc,
                 spikes_counted, spikes_rule, spikes_emitted, gscale):
    """out["slow_inh"] in every engine (design 2163ba7d, Build 2).

    acc_fast and acc_slow are, per target slot, the float32 per-slot sums of
    the listed rows' fast and slow increments accumulated in float64;
    intg the per-step sum of fl(g_s * gscale); g_end the slow conductance
    after the last step and pending the slow increments still in the ring.
    The conductance-times are in leak-conductance ms: an increment x in the
    fast pool delivers x gscale dt / (1 - d_i), in the slow pool x gscale dt
    / (1 - d_s). slow_ct_traj is the integral the membrane read, plus the
    decay of g_end and of the pending increments after the run; it equals
    slow_ct_inc when the pool conserves what it was given. spikes_counted
    are the listed rows' spikes whose increments the accumulators hold."""
    dt, d_i, d_s = sl["receipt"]["dt_ms"], float(sl["d_i"]), float(sl["d_s"])
    gsc = float(np.float32(gscale))
    fast_ct = float(np.sum(acc_fast, dtype=np.float64)) * gsc * dt / (1.0 - d_i)
    slow_ct_inc = float(np.sum(acc_slow, dtype=np.float64)) * gsc * dt / (1.0 - d_s)
    slow_ct_traj = (float(np.sum(intg, dtype=np.float64))
                    + float(np.sum(g_end, dtype=np.float64)) * gsc * d_s / (1.0 - d_s)
                    + float(np.sum(pending, dtype=np.float64)) * gsc / (1.0 - d_s)) * dt
    n = int(spikes_counted)
    per = (lambda x: x / n if n else None)
    g_bin = np.ascontiguousarray(rec_acc, dtype=np.float32)
    import hashlib
    return {**sl["receipt"], "gscale_f32": gsc,
            "spikes_emitted": int(spikes_emitted), "spikes_counted": n,
            "spikes_rule": spikes_rule,
            "fast_ct": fast_ct, "slow_ct_inc": slow_ct_inc, "slow_ct_traj": slow_ct_traj,
            "slow_traj_over_inc": (slow_ct_traj / slow_ct_inc) if slow_ct_inc else None,
            "ct_per_spike_inc": per(fast_ct + slow_ct_inc),
            "ct_per_spike_traj": per(fast_ct + slow_ct_traj),
            "slow_g_end_sum": float(np.sum(g_end, dtype=np.float64)),
            "slow_pending_sum": float(np.sum(pending, dtype=np.float64)),
            "g_bin_rule": ("per record row and bin, the sum over the bin's steps of "
                           "fl(g_s * gscale), leak-conductance units; divide by "
                           "steps_per_bin for the mean"),
            "g_bin": g_bin,
            "g_bin_sha256": hashlib.sha256(g_bin.tobytes()).hexdigest()}


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


_RECEPTOR_CLASS_KEYS = {"pre_nt", "post_cell_class", "sign", "gain", "pre_nt_exclusive"}


def _receptor_class_multiplier(classes, pre, post, nt_series, cell_class):
    """Per-edge float32 multiplier from the receptor-class table.

    Returns (multiplier, audit). Entries are applied in order and compose
    multiplicatively where they overlap. Refuses any key outside the five
    understood ones, so a future "tau_ms" cannot be silently ignored.
    pre_nt_exclusive true matches only rows whose transmitter token set is
    exactly {pre_nt} (design 10dc4dcc); absent or false matches every row
    carrying pre_nt among its tokens, as before."""
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
        excl = entry.get("pre_nt_exclusive", False)
        if not isinstance(excl, bool):
            raise ValueError("receptor class pre_nt_exclusive must be true or false")
        if excl and pre_nt == "*":
            raise ValueError("receptor class pre_nt_exclusive needs a named pre_nt")
        post_mask = cell_class[post] == pcc
        if pre_nt == "*":
            pre_mask = np.ones(len(pre), dtype=bool)
        elif excl:
            has = np.fromiter((t == {pre_nt} for t in nt_tokens), dtype=bool,
                              count=len(nt_tokens))
            pre_mask = has[pre]
        else:
            has = np.fromiter((pre_nt in t for t in nt_tokens), dtype=bool,
                              count=len(nt_tokens))
            pre_mask = has[pre]
        m = post_mask & pre_mask
        mult[m] *= np.float32(sgn * gn)
        rec = {"pre_nt": pre_nt, "post_cell_class": pcc,
               "sign": sgn, "gain": gn, "raw_edges": int(m.sum())}
        if excl:
            rec["pre_nt_exclusive"] = True
        audit.append(rec)
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


def _left_apl_gaba_receipt(meta_absent, meta, nt, scale, sign, gain, p, row):
    """Build 1's receipt (design fb9d4a58): every per-neuron array CNS.__init__
    derives from the resolved transmitter, recomputed with the key absent and
    with it set, and the rows at which each differs. The recomputation with the
    key set must reproduce the arrays the build used, or this refuses."""
    def derive(m):
        nt_ = _resolve_nt(m, p)
        raw = _sign_of(nt_, p)
        n_unknown = int(np.isnan(raw).sum())
        if p.mn_transmitter_override:
            is_mn = (m["super_class"] == "motor").to_numpy()
            raw = np.where(is_mn, p.mn_cns_output_sign, raw)
        sc = np.where(np.isnan(raw), p.w_unknown_scale, 1.0)
        sg = np.where(np.isnan(raw), p.sign_unknown, raw)
        gn = np.where(sg < 0, p.w_inh_scale, 1.0)
        if p.premotor_inh_rows and p.premotor_inh_gain != 1.0:
            pm = np.zeros(len(sg), dtype=bool)
            pm[np.asarray(p.premotor_inh_rows, dtype=int)] = True
            gn = np.where(pm & (sg < 0), gn * p.premotor_inh_gain, gn)
        if p.premotor_inh_gain_per_neuron is not None:
            gn = np.where(sg < 0, gn * np.asarray(p.premotor_inh_gain_per_neuron,
                                                  dtype=np.float64), gn)
        if p.presyn_gain_per_neuron is not None:
            gn = gn * np.asarray(p.presyn_gain_per_neuron, dtype=np.float64)
        return {"transmitter": nt_.fillna("").astype(str).to_numpy(), "raw_sign": raw,
                "scale": sc, "sign": sg, "gain": gn}, n_unknown

    before, n_before = derive(meta_absent)
    after, n_after = derive(meta)
    used = {"transmitter": nt.fillna("").astype(str).to_numpy(), "scale": scale,
            "sign": sign, "gain": gain}
    for k, v in used.items():
        if not np.array_equal(after[k], v):
            raise ValueError(f"left_apl_gaba: the receipt's {k} does not reproduce "
                             "the build's")
    differing = {}
    for k in before:
        x, y = before[k], after[k]
        if k == "transmitter":
            d = np.flatnonzero(x != y)
        else:
            d = np.flatnonzero(~((x == y) | (np.isnan(x) & np.isnan(y))))
        if d.size:
            differing[k] = [int(r) for r in d]

    def val(a):
        return None if isinstance(a, float) and np.isnan(a) else a

    return {
        "arrays_compared": sorted(before),
        "rows_differing_from_key_absent": differing,
        "transmitter": [before["transmitter"][row] or None,
                        after["transmitter"][row] or None],
        "raw_sign": [val(float(before["raw_sign"][row])),
                     val(float(after["raw_sign"][row]))],
        "sign": [float(before["sign"][row]), float(after["sign"][row])],
        "scale": [float(before["scale"][row]), float(after["scale"][row])],
        "gain": [float(before["gain"][row]), float(after["gain"][row])],
        "n_unknown_nt": [n_before, n_after],
    }


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


def _silence_window_steps(window, n, dt, drive_every_ms):
    """Design 302c1bfb, Build 10: a silence window's rows and its first and
    past-the-last CNS steps, refused unless the rows are distinct engine rows
    and both edges fall on drive ticks, where the Metal engine switches."""
    rows = np.asarray(window["rows"], dtype=np.int64)
    if (rows.ndim != 1 or not len(rows) or (rows < 0).any() or (rows >= n).any()
            or len(np.unique(rows)) != len(rows)):
        raise ValueError("silence_window rows must be distinct engine rows")
    steps = []
    for key in ("on_ms", "off_ms"):
        ms = float(window[key])
        k = int(round(ms / dt))
        tick = int(round(float(drive_every_ms) / dt))
        if not np.isfinite(ms) or abs(k * dt - ms) > 1e-9 or k % tick:
            raise ValueError(
                f"silence_window {key} {ms} is not a drive tick of {drive_every_ms} ms")
        steps.append(k)
    if not 0 <= steps[0] < steps[1]:
        raise ValueError("silence_window needs 0 <= on_ms < off_ms")
    return rows, steps[0], steps[1]


def silence_window_receipt(mask, on, off, dt, spikes):
    return {"rows": int(mask.sum()), "on_step": int(on), "off_step": int(off),
            "on_ms": round(on * float(dt), 9), "off_ms": round(off * float(dt), 9),
            "threshold_mv": 1e9, "spikes_in_window": int(spikes)}


def coincidence_triple_seam(n, indptr, indices, data, triples, graded, dt):
    """Design efe88957, Build 2: SimParams.coincidence_triples as the engines
    run it, shared by CNS.run() and the CUDA driver so both multiply the same
    matrices. Per triple: its target rows, the two pool matrices built as the
    single key builds them (|w| of the graph's existing edges from the pool
    onto the triple's targets, CSR over the source axis) with the same
    refusal of a pool that has none, the lag in ticks and the float32 gain and
    decay as the single key rounds them, and its static receipt. A source row
    is read by its release where `graded` marks it and by its spike
    elsewhere; each pool's count by mode is checked against the triple's
    `read`, which walk_search counted from the graded set it built. Refused:
    a target repeated or shared by two triples, a count that differs."""
    pre_all = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    post_all = np.asarray(indices, dtype=np.int64)
    claimed = np.zeros(n, dtype=bool)
    out = []
    for k, tr in enumerate(triples):
        tgt = np.asarray(tr["targets"], dtype=np.int64)
        if not tgt.size or len(np.unique(tgt)) != len(tgt) or claimed[tgt].any():
            raise ValueError(
                f"coincidence triple {k}: targets must be distinct rows that no "
                "other triple names")
        claimed[tgt] = True
        tpos = np.full(n, -1, dtype=np.int64)
        tpos[tgt] = np.arange(len(tgt))
        in_t = tpos[post_all] >= 0
        mats, receipt = [], {"triple": k, "names": tr.get("names"),
                             "n_targets": int(len(tgt))}
        for nm in ("fast", "slow"):
            srcs = np.unique(np.asarray(tr[nm], dtype=np.int64))
            m = in_t & np.isin(pre_all, srcs)
            mat = sparse.csr_matrix(
                (np.abs(data[m]).astype(np.float32), (tpos[post_all[m]], pre_all[m])),
                shape=(len(tgt), n))
            if mat.nnz == 0:
                raise ValueError(
                    f"coincidence triple {k}: the {nm} source pool has NO edges "
                    f"onto its {len(tgt)} targets, so its product term is "
                    "identically zero at any gain")
            n_rel = int(graded[srcs].sum())
            want = (tr.get("read") or {}).get(nm)
            if want is not None and (want["release"], want["spike"]) != (n_rel, len(srcs) - n_rel):
                raise ValueError(
                    f"coincidence triple {k}: the engine grades {n_rel} of the "
                    f"{len(srcs)} {nm} rows, walk_search counted {want['release']} "
                    f"of {want['release'] + want['spike']}")
            mats.append(mat)
            receipt[nm + "_nnz"] = int(mat.nnz)
        lag = max(1, int(round(float(tr["lag_ms"]) / float(dt))))
        gain = np.float32(tr["gain"])
        decay = np.float32(np.exp(-float(dt) / max(1e-6, float(tr["tau_ms"]))))
        receipt.update(gain=float(gain), lag_ticks=lag, tau_ms=float(tr["tau_ms"]),
                       decay=float(decay), read=tr.get("read"))
        out.append({"tgt": tgt, "fast": mats[0], "slow": mats[1], "lag": lag,
                    "gain": gain, "decay": decay, "receipt": receipt})
    return out


def coincidence_triple_receipt(seam, stats):
    """Design efe88957, Build 2: one receipt per triple, its static part and
    the applied |product| max and sum, fast-trace max and delayed-slow max
    the engine accumulated, in that order."""
    return [{**q["receipt"], "applied_abs_max": float(s[0]),
             "applied_abs_sum": float(s[1]), "fs_max": float(s[2]),
             "slow_delayed_max": float(s[3])} for q, s in zip(seam, stats)]


def rest_drive_plan(spec, n, threshold, rest, graded, seed, dt):
    """Design 1e20a949, Build C1: CNS.run's `rest_drive` as its sorted rows,
    per-row float32 kicks, per-step event probability, own generator and
    receipt, shared by both engines so they draw and kick alike.

    Draw j of a step belongs to sorted row j. The generator is
    numpy.random.default_rng([seed, spec["stream"]]) and nothing else reads
    it. Refused: a row that is graded (it has no threshold test, and cns.py
    computes graded release after the kick where the GPU kernels read the
    pre-kick membrane), a margin that is not finite and positive, or a
    probability above 1."""
    if set(spec) != {"rows", "hz", "kick_fraction", "stream"}:
        raise ValueError("rest_drive is {rows, hz, kick_fraction, stream}")
    rows = np.sort(np.asarray(spec["rows"], dtype=np.int64))
    if (rows.ndim != 1 or not len(rows) or rows[0] < 0 or rows[-1] >= n
            or len(np.unique(rows)) != len(rows)):
        raise ValueError("rest_drive rows must be distinct engine rows")
    hz, frac = float(spec["hz"]), float(spec["kick_fraction"])
    if not np.isfinite(hz) or hz < 0.0:
        raise ValueError("rest_drive hz must be finite and >= 0")
    if not np.isfinite(frac) or frac <= 0.0:
        raise ValueError("rest_drive kick_fraction must be finite and > 0")
    if type(spec["stream"]) is not int or spec["stream"] < 0:
        raise ValueError("rest_drive stream must be a nonnegative int")
    prob = hz * float(dt) / 1000.0
    if prob > 1.0:
        raise ValueError(f"rest_drive hz {hz} exceeds one event per step")
    if graded is not None and np.asarray(graded, dtype=bool)[rows].any():
        raise ValueError("rest_drive rows must not be graded")
    margin = (np.broadcast_to(np.asarray(threshold, dtype=np.float32), (n,))[rows]
              - np.broadcast_to(np.asarray(rest, dtype=np.float32), (n,))[rows]
              ).astype(np.float32)
    if not np.all(np.isfinite(margin)) or np.any(margin <= 0.0):
        raise ValueError("rest_drive rows need finite positive "
                         "v_threshold - effective_v_rest margins")
    kick = (np.float32(frac) * margin).astype(np.float32)
    mask = np.zeros(n, dtype=bool)
    mask[rows] = True
    receipt = {"rows": int(len(rows)), "hz": hz, "kick_fraction": frac,
               "stream": int(spec["stream"]), "seed": int(seed),
               "event_probability_per_step": prob, "draws_per_step": int(len(rows)),
               "kick_mv": {"min": float(kick.min()), "median": float(np.median(kick)),
                           "max": float(kick.max())},
               "margin_mv": {"min": float(margin.min()), "median": float(np.median(margin)),
                             "max": float(margin.max())}}
    return {"rows": rows, "mask": mask, "kick": kick, "prob": prob,
            "gen": np.random.default_rng([int(seed), int(spec["stream"])]),
            "receipt": receipt, "draws": 0, "events": 0}


def rest_drive_refuse_overlap(plan, drive_idx):
    """A rest row in the drive set would take its drive events as kicks on
    the GPU engines, whose one membrane-drive mask holds the rest rows, and as
    forced spikes here, so the overlap is refused wherever the set changes."""
    if plan is not None and len(drive_idx) and plan["mask"][drive_idx].any():
        raise ValueError("rest_drive rows must not be in the drive set")


def octopamine_offset_plan(p, n):
    """Design 8b6e0441, Build 2: (rows, offset) of the octopamine offset, the
    rows int64 and the offset np.float32, or None when no row is configured or
    the offset is 0. Shared by both engines so they add the same float32 on
    the same rows."""
    rows = getattr(p, "octopamine_target_rows", None)
    off = float(getattr(p, "octopamine_offset_mv", 0.0))
    if not np.isfinite(off):
        raise ValueError("octopamine_offset_mv must be finite")
    if rows is None:
        if off != 0.0:
            raise ValueError("octopamine_offset_mv needs octopamine_target_rows")
        return None
    r = np.asarray(rows, dtype=np.int64)
    if (r.ndim != 1 or not len(r) or len(np.unique(r)) != len(r)
            or r.min() < 0 or r.max() >= n):
        raise ValueError("octopamine_target_rows must be unique valid rows")
    if off == 0.0:
        return None
    return r, np.float32(off)


def octopamine_state_plan(p, n, dt, n_steps, W):
    """Design 8b6e0441, Build 1: the octopamine state's configuration, shared
    by both engines, or None when octopamine_source_rows is None. W is the
    built network's CSC weight matrix, a column per presynaptic row; its
    stored entries are every edge of the edge list, those at weight 0
    included. A source row is left out of the spike count (`counted`
    excludes it, `excluded` names it) when none of its edges reaches a target
    row past p.row_pair_seams' gain-0 entries (critique 28ee40b6 condition 1:
    a row whose synapses onto the target set are seamed). The edge's weight
    plays no part, so an octopaminergic row the build mutes (sign 0) counts
    wherever its synapses reach the target set. The increment keeps the
    given set's size, so s keeps its unit."""
    src = getattr(p, "octopamine_source_rows", None)
    if src is None:
        return None
    tgt = getattr(p, "octopamine_target_rows", None)
    if tgt is None:
        raise ValueError("the octopamine state needs octopamine_target_rows")
    rows = {}
    for name, r in (("source", src), ("target", tgt)):
        r = np.asarray(r, dtype=np.int64)
        if (r.ndim != 1 or not len(r) or len(np.unique(r)) != len(r)
                or r.min() < 0 or r.max() >= n):
            raise ValueError(f"octopamine_{name}_rows must be unique valid rows")
        rows[name] = r
    tau = float(p.octopamine_tau_ms)
    gain = float(p.octopamine_gain_mv_per_hz)
    init = float(p.octopamine_init_hz)
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("octopamine_tau_ms must be finite and positive")
    if not np.isfinite(gain) or gain < 0.0:
        raise ValueError("octopamine_gain_mv_per_hz must be finite and non-negative")
    if not np.isfinite(init) or init < 0.0:
        raise ValueError("octopamine_init_hz must be finite and non-negative")
    tmask = np.zeros(n, dtype=bool)
    tmask[rows["target"]] = True
    cuts = []
    for pr, po, g in (getattr(p, "row_pair_seams", None) or ()):
        if float(g) == 0.0:
            ma = np.zeros(n, dtype=bool)
            ma[np.asarray(pr, dtype=np.int64)] = True
            mb = np.zeros(n, dtype=bool)
            mb[np.asarray(po, dtype=np.int64)] = True
            cuts.append((ma, mb))
    reach = np.zeros(len(rows["source"]), dtype=bool)
    for q, j in enumerate(rows["source"]):
        a, b = W.indptr[j], W.indptr[j + 1]
        t = W.indices[a:b][tmask[W.indices[a:b]]]
        live = np.ones(len(t), dtype=bool)
        for ma, mb in cuts:
            if ma[j]:
                live &= ~mb[t]
        reach[q] = bool(live.any())
    spb = max(1, int(round(5.0 / dt)))
    return {"source": rows["source"], "counted": rows["source"][reach],
            "excluded": rows["source"][~reach], "target": rows["target"],
            "tau": tau, "gain": gain, "init": init, "spb": spb,
            "n_bins": n_steps // spb,
            "decay": np.float32(np.exp(-dt / tau)),
            "inc": np.float32(1000.0 / (len(rows["source"]) * tau))}


def octopamine_state_receipt(plan, state, state_max, applied_sum, spikes, bins,
                             n_steps, dt, ids=None):
    """Design 8b6e0441, Build 1: the state's receipt, the same on both engines:
    s per bin (its value at the bin's end), its mean over 1 to 6 s and over the
    first and last 500 ms, N_src, the counted and excluded source rows, N_tgt.
    `ids` (the meta ids, optional) names the excluded rows by id."""
    bins = np.asarray(bins, dtype=np.float32)
    bin_ms = plan["spb"] * dt

    def _mean(lo_ms, hi_ms):
        a = max(0, int(round(lo_ms / bin_ms)))
        b = min(len(bins), int(round(hi_ms / bin_ms)))
        return float(np.mean(bins[a:b], dtype=np.float64)) if b > a else None

    exc = [int(r) for r in plan["excluded"]]
    return {
        "source_rows": int(len(plan["source"])),
        "counted_source_rows": int(len(plan["counted"])),
        "excluded_source_rows": exc,
        "excluded_source_ids": (None if ids is None
                                else [str(ids[r]) for r in exc]),
        "excluded_rule": ("left out of the spike count: a source row none of "
                          "whose edges reaches a target row past the "
                          "row_pair_seams entries at gain 0 (critique 28ee40b6 "
                          "condition 1), whatever the edges' weights, so a "
                          "muted octopamine row counts where its synapses "
                          "reach; N_src keeps the given set's size"),
        "target_rows": int(len(plan["target"])),
        "source_spikes": int(spikes),
        "tau_ms": float(plan["tau"]),
        "gain_mv_per_hz": float(plan["gain"]),
        "init_hz": float(plan["init"]),
        "state_final_hz": float(state),
        "state_max_hz": float(state_max),
        "state_mean_applied_hz": float(applied_sum / max(1, n_steps)),
        "state_mean_1_6s_hz": _mean(1000.0, 6000.0),
        "state_mean_first_500ms_hz": _mean(0.0, 500.0),
        "state_mean_last_500ms_hz": _mean(len(bins) * bin_ms - 500.0,
                                          len(bins) * bin_ms),
        "bin_ms": float(bin_ms),
        "state_bin_end_hz": bins.tolist(),
    }


def _inject_segments(inject, dt, n_steps, n):
    """CNS.run's `inject` as tick segments [(t0, t1, vec)], vec a float32
    n-vector of the summed amplitudes active on ticks t0 <= t < t1, or None
    where nothing is injected. Shared by both engines so they add the same
    float32 vector on the same ticks. On and off times must fall on the dt
    grid, because a rounded edge would move the step by up to half a tick.
    Copied from team-henri (373cd93d3, design 06226d13) for design afc8fd03."""
    if inject is None:
        return None
    entries = []
    for e in inject:
        if len(e) != 4:
            raise ValueError("inject entries are (row, t_on_ms, t_off_ms, amp_mv)")
        row, t_on, t_off, amp = int(e[0]), float(e[1]), float(e[2]), float(e[3])
        if not 0 <= row < n:
            raise ValueError(f"inject row {row} outside 0..{n - 1}")
        if not (np.isfinite(t_on) and np.isfinite(t_off) and np.isfinite(amp)):
            raise ValueError("inject times and amplitude must be finite")
        k0, k1 = round(t_on / dt), round(t_off / dt)
        if abs(k0 * dt - t_on) > 1e-6 or abs(k1 * dt - t_off) > 1e-6:
            raise ValueError("inject on and off times must be multiples of dt")
        if not 0 <= k0 < k1:
            raise ValueError("inject needs 0 <= t_on_ms < t_off_ms")
        entries.append((row, int(k0), int(min(k1, n_steps)), np.float32(amp)))
    edges = sorted({0, n_steps} | {k for _, a, b, _ in entries
                                    for k in (a, b) if k < n_steps})
    segs = []
    for a, b in zip(edges[:-1], edges[1:]):
        vec = None
        for row, k0, k1, amp in entries:
            if k0 <= a and b <= k1:
                if vec is None:
                    vec = np.zeros(n, dtype=np.float32)
                vec[row] = np.float32(vec[row] + amp)
        segs.append((a, b, vec))
    return segs


OCC_REF_DT_MS = 0.1   # design 81427d47: alpha, Gmax and beta are pinned at this timestep


def shared_occupancy_arrays(W, p, n_rows, graded_rows, spec, ids=None):
    """Design 81427d47 (H03), section 2: the occupancy channel's built arrays,
    read from W before its selected entries are zeroed. Returns (arrays,
    audit, selected CSC positions).

    Selected: every existing entry of W from a pre row onto a post row; a
    nonpositive or repeated entry, a graded row or an empty selection is a
    preparation fault. a_i = float32(W_i * float32(gscale)) is the normalized
    conductance the membrane first uses after one isolated source event. Per
    target j: e0_j the smallest float32 ulp exponent among its a_i, scale_j =
    2**e0_j, quanta_i = a_i/scale_j, exact integers summing below 2**53,
    S_j = sum(quanta)*scale_j (exact), and r0_j the target's native float32
    conductance decay per 0.1 ms tick. With c = 1-exp(-0.1/tau_close),
    computed as -expm1, alpha_j = float32(eta/S_j), Gmax_j =
    float32(k*S_j*c/(eta*(1-r0_j))) and beta_j = float32(k*c/(1-r0_j)), all
    pinned at 0.1 ms whatever the running dt; dec = exp(-dt/tau_close) in
    float64 at the running dt. Mode "L" takes eta None and leaves alpha and
    Gmax zero, so every L control with one (k, tau_close) is built alike.
    Connections are ordered by target row and then source row."""
    pre, post, mode, k, eta, tau, interrupt = spec
    pre = np.unique(np.asarray(pre, dtype=np.int64))
    post = np.unique(np.asarray(post, dtype=np.int64))
    if len(spec[0]) != len(pre) or len(spec[1]) != len(post) or not len(pre) or not len(post):
        raise ValueError("shared_occupancy rows must be nonempty and unique")
    if pre.min() < 0 or post.min() < 0 or max(pre.max(), post.max()) >= n_rows:
        raise ValueError("shared_occupancy rows must be valid rows")
    if mode not in ("T", "L"):
        raise ValueError("shared_occupancy mode is 'T' or 'L'")
    k, tau = float(k), float(tau)
    if not (np.isfinite(k) and k > 0.0 and np.isfinite(tau) and tau > 0.0):
        raise ValueError("shared_occupancy k and tau_close_ms must be finite and > 0")
    if mode == "T":
        eta = float(eta)
        if not (np.isfinite(eta) and eta > 0.0):
            raise ValueError("shared_occupancy eta must be finite and > 0 for mode 'T'")
    elif eta is not None:
        raise ValueError("shared_occupancy mode 'L' takes eta None")
    if interrupt not in (False, True, 0, 1):
        raise ValueError("shared_occupancy interrupt is a bool")
    if len(graded_rows) and np.isin(np.concatenate([pre, post]),
                                    np.asarray(graded_rows, dtype=np.int64)).any():
        raise NotImplementedError("shared_occupancy on a graded row")
    indptr = np.asarray(W.indptr, dtype=np.int64)
    indices = np.asarray(W.indices, dtype=np.int64)
    is_post = np.zeros(n_rows, dtype=bool)
    is_post[post] = True
    src, tgt, pos = [], [], []
    for j in pre:
        lo, hi = int(indptr[j]), int(indptr[j + 1])
        hit = np.flatnonzero(is_post[indices[lo:hi]])
        src.extend([int(j)] * len(hit))
        tgt.extend(indices[lo + hit].tolist())
        pos.extend((lo + hit).tolist())
    src, tgt, pos = (np.asarray(x, dtype=np.int64) for x in (src, tgt, pos))
    if not len(pos):
        raise ValueError("shared_occupancy selected no entry of W")
    if len(np.unique(src * n_rows + tgt)) != len(pos):
        raise ValueError("shared_occupancy selection repeats a connection")
    w = np.asarray(W.data, dtype=np.float32)[pos]
    if not np.all(w > 0):
        raise ValueError("shared_occupancy selection has a nonpositive weight")
    order = np.lexsort((src, tgt))
    src, tgt, pos, w = src[order], tgt[order], pos[order], w[order]
    tgt_rows = np.unique(tgt)
    if not np.array_equal(tgt_rows, post):
        raise ValueError("shared_occupancy: a post row receives no selected entry")
    gscale = np.float32(1.0 / abs(p.e_exc - float(np.mean(np.asarray(p.v_rest)))))
    a = (w * gscale).astype(np.float32)
    if not np.all(a >= np.finfo(np.float32).tiny) or not np.all(np.isfinite(a)):
        raise ValueError("shared_occupancy increments must be positive normal float32")
    tau_syn = np.asarray(p.tau_syn, dtype=np.float32)
    ref = OCC_REF_DT_MS
    if tau_syn.ndim == 0:
        r0_all = np.full(n_rows, np.float32(np.exp(-ref / tau_syn)), dtype=np.float32)
    else:
        r0_all = np.exp(-np.float32(ref) / tau_syn).astype(np.float32)
    c = float(-np.expm1(-ref / tau))
    dt = float(p.dt)
    dec = float(np.exp(-dt / tau))
    n_t = len(tgt_rows)
    tidx = np.searchsorted(tgt_rows, tgt)
    a64 = a.astype(np.float64)
    ulp_exp = np.frexp(a64)[1].astype(np.int64) - 24
    quanta = np.zeros(len(a), dtype=np.uint64)
    scale = np.zeros(n_t, dtype=np.float64)
    S = np.zeros(n_t, dtype=np.float64)
    r0 = r0_all[tgt_rows]
    alpha = np.zeros(n_t, dtype=np.float32)
    gmax = np.zeros(n_t, dtype=np.float32)
    beta = np.zeros(n_t, dtype=np.float32)
    per_t = []
    for t in range(n_t):
        m = tidx == t
        e0 = int(ulp_exp[m].min())
        qf = np.ldexp(a64[m], -e0)
        if not np.all(qf == np.floor(qf)):
            raise ValueError("shared_occupancy quanta are not exact integers")
        q = [int(x) for x in qf]
        if sum(q) >= 2**53:
            raise ValueError("shared_occupancy target sum reaches 2**53 quanta")
        quanta[m] = np.asarray(q, dtype=np.uint64)
        scale[t] = float(np.ldexp(1.0, e0))
        S[t] = float(sum(q)) * scale[t]
        one_r0 = 1.0 - float(r0[t])
        beta[t] = np.float32(k * c / one_r0)
        if mode == "T":
            alpha[t] = np.float32(eta / S[t])
            gmax[t] = np.float32(k * S[t] * c / (eta * one_r0))
        vals = [float(x) for x in (alpha[t], gmax[t], beta[t])]
        if not all(np.isfinite(vals)) or beta[t] <= 0 or (mode == "T" and min(vals) <= 0):
            raise ValueError("shared_occupancy produced a nonfinite or nonpositive coefficient")
        prod = np.float32(gmax[t]) * np.float32(alpha[t])
        ulps = (abs(int(np.float32(prod).view(np.uint32)) - int(beta[t].view(np.uint32)))
                if mode == "T" else None)
        per_t.append({
            "row": int(tgt_rows[t]), "banc_888_id": None if ids is None else str(ids[int(tgt_rows[t])]),
            "connections": int(m.sum()), "scale_exponent": e0, "quanta_sum": int(sum(q)),
            "S": S[t], "S_hex": S[t].hex(), "r0": float(r0[t]),
            "r0_hex": f"0x{int(r0[t].view(np.uint32)):08x}",
            "alpha": float(alpha[t]), "alpha_hex": f"0x{int(alpha[t].view(np.uint32)):08x}",
            "gmax": float(gmax[t]), "gmax_hex": f"0x{int(gmax[t].view(np.uint32)):08x}",
            "beta": float(beta[t]), "beta_hex": f"0x{int(beta[t].view(np.uint32)):08x}",
            "gmax_alpha_float32": float(prod) if mode == "T" else None,
            "gmax_alpha_vs_beta_ulps": ulps})
    arrays = {"src": src, "tgt": tgt, "tgt_rows": tgt_rows, "a": a, "quanta": quanta,
              "scale": scale, "S": S, "r0": r0, "alpha": alpha, "gmax": gmax, "beta": beta,
              "dec": np.full(n_t, dec, dtype=np.float64), "mode": 1 if mode == "T" else 2,
              "interrupt": int(bool(interrupt)), "k": k, "eta": eta, "tau_close_ms": tau,
              "positions": pos}
    audit = {
        "rule": ("T: o = o*dec, then o = 1-(1-o)*exp(-alpha*A), g = float32(Gmax*o); "
                 "L: h = h*dec + beta*A, g = float32(h); A = sum(quanta*count)*scale per tick; "
                 "alpha = float32(eta/S), Gmax = float32(k*S*c/(eta*(1-r0))), "
                 "beta = float32(k*c/(1-r0)), c = -expm1(-0.1/tau_close), "
                 "a = float32(W*float32(gscale))"),
        "mode": mode, "k": k, "k_hex": k.hex(), "eta": eta,
        "eta_hex": None if eta is None else float(eta).hex(),
        "tau_close_ms": tau, "tau_close_hex": tau.hex(), "interrupt": bool(interrupt),
        "reference_dt_ms": ref, "dt_ms": dt, "c": c, "c_hex": c.hex(),
        "dec": dec, "dec_hex": dec.hex(), "gscale_float32": float(gscale),
        "targets": per_t,
        "connections": [{"pre_row": int(src[q]), "post_row": int(tgt[q]), "position": int(pos[q]),
                         "pre_banc_888_id": None if ids is None else str(ids[int(src[q])]),
                         "post_banc_888_id": None if ids is None else str(ids[int(tgt[q])]),
                         "weight_before": float(w[q]),
                         "weight_hex": f"0x{int(w[q].view(np.uint32)):08x}",
                         "a": float(a[q]), "a_hex": f"0x{int(a[q].view(np.uint32)):08x}",
                         "quanta": int(quanta[q])} for q in range(len(pos))],
    }
    return arrays, audit, pos



PSI_PAIR_RECOVERY_FORMS = {"dead_time": 1, "depletion": 2}


def psi_pair_recovery_values(spec, dt):
    """Design e05c0eef B1: the native engines' values for a pair recovery
    (form, tau_rec_ms, u) at step dt, which every engine configuration and
    every refinement recomputes from the spec. Dead time: ticks =
    round(tau_rec_ms/dt), u None. Depletion: decay = float32(exp(-dt/tau)),
    0.0 at tau 0, and keep = float32(1 - u), u in [0, 1]. Per tick, in order:
    dead time decrements the clock, then an arrival at clock 0 delivers and
    sets it to ticks; depletion recovers R, then each arrival delivers inc*R
    and keeps R*keep."""
    if len(spec) != 3:
        raise ValueError("psi_pair_recovery is (form, tau_rec_ms, u)")
    form, tau, u = spec
    if form not in PSI_PAIR_RECOVERY_FORMS:
        raise ValueError("psi_pair_recovery form must be dead_time or depletion")
    tau = float(tau)
    if not np.isfinite(tau) or tau < 0.0:
        raise ValueError("psi_pair_recovery tau_rec_ms must be finite and >= 0")
    out = {"form": form, "form_code": PSI_PAIR_RECOVERY_FORMS[form], "tau_rec_ms": tau,
           "tau_rec_hex": tau.hex(), "dt_ms": float(dt), "u": None, "ticks": 0,
           "decay": 0.0, "keep": 1.0}
    if form == "dead_time":
        if u is not None:
            raise ValueError("psi_pair_recovery dead_time takes no u")
        out["ticks"] = int(round(tau / float(dt)))
        out["rule"] = ("per tick T = max(T - 1, 0); an arrival at T = 0 delivers inc*n and "
                       "sets T = ticks, at T > 0 delivers nothing")
    else:
        u = float(u)
        if not np.isfinite(u) or u < 0.0 or u > 1.0:
            raise ValueError("psi_pair_recovery u must be in [0, 1]")
        out["u"] = u
        out["decay"] = float(np.float32(np.exp(-float(dt) / tau)) if tau > 0.0 else np.float32(0.0))
        out["keep"] = float(np.float32(1.0 - u))
        out["rule"] = ("per tick R = 1 - (1 - R)*decay; each arrival delivers inc*R and "
                       "keeps R*keep, R = 1 at start")
    return out


def gap_law_substeps(p, N):
    """Electrical substeps per network step under gap_law, as the native
    engines take them: ceil(2 dt max(degree / tau_mem)), at least one, times
    gap_substep_factor, where a row's degree sums the weights of every pair
    it ends. At one substep the one_way law is the plain coupling over G,
    which is how Metal runs it and how CUDA and NumPy run it (gpu-7, note
    10906d21)."""
    pairs = np.asarray(p.gap_pairs, dtype=np.int64)
    tau = np.broadcast_to(np.asarray(p.tau_mem, dtype=np.float64), (N,))
    if not np.isfinite(tau).all() or np.any(tau <= 0) or not np.isfinite(p.dt) or p.dt <= 0:
        raise ValueError("gap_law requires positive finite membrane times and dt")
    degree = np.zeros(N, dtype=np.float64)
    np.add.at(degree, pairs[:, 0], np.asarray(p.gap_pair_weights, dtype=np.float64))
    np.add.at(degree, pairs[:, 1], np.asarray(p.gap_pair_weights, dtype=np.float64))
    return max(1, int(np.ceil(2*p.dt*np.max(degree/tau)))) * p.gap_substep_factor

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
                nt_tok = spec.get("neurotransmitter") if isinstance(spec, dict) else None
                if (isinstance(spec, dict) and "cell_type" in spec
                        and spec["cell_type"] is None):
                    # design 6f704792 Build 3: an untyped row keeps its null
                    # type and takes the transmitter only, so no type-keyed
                    # lookup reads a type named "None".
                    if not pd.isna(self.meta.loc[hit, "cell_type"].iloc[0]):
                        raise ValueError(
                            f"cell_type_patch[{rid}] states cell_type None; "
                            "the row is typed")
                    if not nt_tok:
                        raise ValueError(
                            f"cell_type_patch[{rid}] is untyped and patches no "
                            "transmitter")
                elif not isinstance(spec, dict) or not spec.get("cell_type"):
                    raise ValueError(f"cell_type_patch[{rid}] needs a cell_type")
                else:
                    self.meta.loc[hit, "cell_type"] = str(spec["cell_type"])
                if nt_tok:
                    if nt_tok not in p.nt_sign:
                        raise ValueError(
                            f"cell_type_patch[{rid}] neurotransmitter {nt_tok!r} "
                            f"is not a transmitter token in nt_sign")
                    self.meta.loc[hit, "neurotransmitter_predicted"] = nt_tok
                self.n_cell_type_patched += 1
        self.left_apl_gaba_audit = None
        _la_meta_absent = None
        if p.left_apl_gaba:
            _la_id = "720575941734526507"
            _la_hit = (self.meta["banc_888_id"].astype(str) == _la_id).to_numpy()
            if int(_la_hit.sum()) != 1:
                raise ValueError(f"left_apl_gaba: row {_la_id} matches "
                                 f"{int(_la_hit.sum())} meta rows, expected exactly 1")
            if p.cell_type_patch and _la_id in {str(k) for k in p.cell_type_patch}:
                raise ValueError("left_apl_gaba: an enabled cell_type_patch names "
                                 f"row {_la_id}")
            _la_row = int(np.flatnonzero(_la_hit)[0])
            for col in ("neurotransmitter_verified", "neurotransmitter_predicted"):
                if pd.notna(self.meta[col].iloc[_la_row]):
                    raise ValueError(f"left_apl_gaba: row {_la_id} already carries "
                                     f"{col} {self.meta[col].iloc[_la_row]!r}")
            _la_meta_absent = self.meta
            self.meta = self.meta.copy()
            if str(self.meta["neurotransmitter_predicted"].dtype) == "category":
                self.meta["neurotransmitter_predicted"] = \
                    self.meta["neurotransmitter_predicted"].astype(object)
            self.meta.loc[_la_hit, "neurotransmitter_predicted"] = "gaba"
            self.left_apl_gaba_audit = {"banc_888_id": _la_id, "row": _la_row,
                                        "n_left_apl_gaba": 1}
        self.nt_transfer_audit = None
        if p.nt_transfer_rows:
            _tr_ids = self.meta["banc_888_id"].astype(str)
            _tr_pos = pd.Series(np.arange(len(_tr_ids)), index=_tr_ids.to_numpy())
            if not _tr_pos.index.is_unique:
                raise ValueError("nt_transfer_rows: meta banc_888_id is not unique")
            _tr_rows = []
            for rid, tok in p.nt_transfer_rows.items():
                if str(rid) not in _tr_pos.index:
                    raise ValueError(f"nt_transfer_rows: row {rid} is not in meta")
                if tok not in p.nt_sign:
                    raise ValueError(
                        f"nt_transfer_rows[{rid}] = {tok!r} is not a transmitter "
                        "token in nt_sign")
                _tr_rows.append(int(_tr_pos[str(rid)]))
            _tr_rows = np.asarray(_tr_rows, dtype=np.int64)
            _tr_ver = self.meta["neurotransmitter_verified"].iloc[_tr_rows]
            _tr_ct = self.meta["cell_type"].iloc[_tr_rows]
            if bool(_tr_ver.notna().any()):
                raise ValueError(
                    "nt_transfer_rows: a listed row carries a verified transmitter")
            if bool((_tr_ct.notna() & (_tr_ct.astype(str) != "")).any()):
                raise ValueError(
                    "nt_transfer_rows: a listed row carries a BANC cell_type")
            before = self.meta["neurotransmitter_predicted"].iloc[_tr_rows].to_numpy()
            self.meta = self.meta.copy()
            if str(self.meta["neurotransmitter_predicted"].dtype) == "category":
                self.meta["neurotransmitter_predicted"] = \
                    self.meta["neurotransmitter_predicted"].astype(object)
            _col = self.meta.columns.get_loc("neurotransmitter_predicted")
            self.meta.iloc[_tr_rows, _col] = [
                p.nt_transfer_rows[k] for k in p.nt_transfer_rows]
            self.nt_transfer_audit = {
                "rows": _tr_rows, "predicted_before": before,
                "predicted_after": self.meta["neurotransmitter_predicted"]
                .iloc[_tr_rows].to_numpy()}
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
        if self.left_apl_gaba_audit is not None:
            self.left_apl_gaba_audit.update(_left_apl_gaba_receipt(
                _la_meta_absent, self.meta, nt, scale, sign, gain, p,
                self.left_apl_gaba_audit["row"]))
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

        # KC-to-KC acetylcholine (design fb9d4a58, Builds 2 and 3); see
        # SimParams.kc_kc_ach. Before the balance step, so each receiver's
        # scale is recomputed from the input it keeps.
        self.kc_kc_audit = None
        if p.kc_kc_ach is not None:
            if p.kc_kc_ach not in ("off", "calyx_kept"):
                raise ValueError("kc_kc_ach must be None, 'off' or 'calyx_kept'")
            _row_of = {str(b): r for r, b in enumerate(self.ids)}
            _kc_ids = [str(x) for x in p.kc_kc_ids]
            if (len(_kc_ids) != 4552 or len(set(_kc_ids)) != 4552
                    or any(b not in _row_of for b in _kc_ids)):
                raise ValueError("kc_kc_ach needs the 4,552 KC ids, all in meta")
            _kc = np.zeros(self.N, dtype=bool)
            _kc[[_row_of[b] for b in _kc_ids]] = True
            _e = np.flatnonzero(_kc[pre] & _kc[post])
            _n = cnt[_e].astype(np.int64)
            _aud = {"mode": p.kc_kc_ach, "n_kc": len(_kc_ids),
                    "kc_kc_edges": int(_e.size), "kc_kc_synapses": int(_n.sum()),
                    "receivers": int(np.unique(post[_e]).size)}
            w = w.copy()
            if p.kc_kc_ach == "off":
                w[_e] = 0.0
                _aud.update({"edges_zeroed": int(_e.size),
                             "synapses_zeroed": int(_n.sum())})
            else:
                if p.kc_kc_calyx is None:
                    raise ValueError("kc_kc_ach 'calyx_kept' needs kc_kc_calyx")
                _sha, _rows = p.kc_kc_calyx
                _at = {(int(pre[i]), int(post[i])): j for j, i in enumerate(_e)}
                _mult = np.zeros(_e.size, dtype=np.float64)
                _seen = np.zeros(_e.size, dtype=bool)
                for _a, _b, _nn, _nca in _rows:
                    _j = _at.get((_row_of.get(str(_a), -1), _row_of.get(str(_b), -1)))
                    if _j is None:
                        raise ValueError(f"kc_kc_calyx: ({_a}, {_b}) is not a "
                                         "KC-to-KC edge")
                    if _seen[_j] or int(_nn) != int(_n[_j]) or not 0 < int(_nca) <= int(_nn):
                        raise ValueError(f"kc_kc_calyx: row ({_a}, {_b}, {_nn}, {_nca}) "
                                         f"is repeated or does not fit the edge's "
                                         f"{int(_n[_j])} synapses")
                    _seen[_j] = True
                    _mult[_j] = int(_nca) / int(_nn)
                w[_e] = np.where(_mult == 0.0, 0.0, w[_e] * _mult)
                _nca_all = np.rint(_mult * _n).astype(np.int64)
                _aud.update({"table_sha256": str(_sha),
                             "calyx_entries": int(_seen.sum()),
                             "calyx_synapses": int(_nca_all.sum()),
                             "all_calyx": int((_mult == 1.0).sum()),
                             "part_calyx": int(((_mult > 0.0) & (_mult < 1.0)).sum()),
                             "edges_zeroed": int((~_seen).sum()),
                             "synapses_removed": int(_n.sum() - _nca_all.sum())})
            self.kc_kc_audit = _aud

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
            # Observational arrays for transmitter-correction build audits.
            self.balance_input_exc = E_in
            self.balance_input_inh = I_in
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
            self.balance_cap_mask = s > p.balance_max_scale
            self.balance_exempt_mask = (np.zeros(self.N, dtype=bool)
                                       if p.balance_exempt is None
                                       else np.asarray(p.balance_exempt).copy())
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
        if self.kc_kc_audit is not None:
            import hashlib as _hl
            self.kc_kc_audit["balance_scale_sha256"] = (
                None if self.balance_scale is None else _hl.sha256(np.ascontiguousarray(
                    self.balance_scale, dtype=np.float64).tobytes()).hexdigest())

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
        self.r_in_total_audit = None
        _r_total = getattr(p, "r_in_total_by_id", None)
        if p.r_in_override or _r_total:
            pos_ov = pd.Series(np.arange(self.N), index=self.ids)
            ov = np.ones(self.N, dtype=np.float32)
            for bid, fac in (p.r_in_override or {}).items():
                r_ = pos_ov.get(bid)
                if r_ is not None and not np.isnan(r_):
                    ov[int(r_)] = fac
            # Design d27f82df B1 (study eb9ff316): th["r_in_total_by_id"]
            # sets a named row's total input factor r_scale x ov to f, so its
            # entry is f / r_scale (r_scale 1 with r_in_alpha off), written
            # after r_in_override's. walk_search.r_in_total_by_id_apply checks
            # the ids against meta; these raises cover a direct build.
            if _r_total:
                self.r_in_total_audit = []
                for bid, f in _r_total.items():
                    r_ = pos_ov.get(bid)
                    if r_ is None or np.isnan(r_):
                        raise ValueError(
                            f"r_in_total_by_id id {bid} is not a row of this network")
                    if bid in (p.r_in_override or {}):
                        raise ValueError(
                            f"r_in_total_by_id id {bid} also has an r_in_override entry")
                    r_ = int(r_)
                    rs_ = 1.0 if self.r_scale is None else float(self.r_scale[r_])
                    ov[r_] = np.float32(float(f) / rs_)
                    self.r_in_total_audit.append(
                        {"banc_888_id": str(bid), "row": r_, "f": float(f),
                         "r_scale": rs_, "ov": float(ov[r_])})
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
        self.post_balance_edge_gain_audit = None
        if p.post_balance_edge_gain:
            self.post_balance_edge_gain_audit = self._post_balance_edge_gain(
                p.post_balance_edge_gain, p.graded_rows)
        self.post_balance_edge_gain_blocks_audit = None
        if p.post_balance_edge_gain_blocks:
            if p.post_balance_edge_gain:
                raise ValueError("post_balance_edge_gain and "
                                 "post_balance_edge_gain_blocks are exclusive")
            self.post_balance_edge_gain_blocks_audit = (
                self._post_balance_edge_gain_blocks(
                    p.post_balance_edge_gain_blocks, p.graded_rows))
        self.psi_pairs = None
        self.psi_pair_kernel_audit = None
        if p.psi_pair_kernel:
            self.psi_pair_kernel_audit = self._psi_pair_kernel(
                p.psi_pair_kernel, p.graded_rows)
        self.psi_pair_recovery_audit = None
        if p.psi_pair_recovery:
            if not p.psi_pair_kernel:
                raise ValueError("psi_pair_recovery needs psi_pair_kernel")
            self.psi_pair_recovery_audit = psi_pair_recovery_values(
                p.psi_pair_recovery, float(p.dt))
        self.shared_occupancy = None
        self.shared_occupancy_audit = None
        if p.shared_occupancy:
            self.shared_occupancy_audit = self._shared_occupancy(
                p.shared_occupancy, p.graded_rows)
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

        self._build_gap()

    def _post_balance_edge_gain(self, spec, graded_rows):
        """Design 0c25fab3: W'_ij = g W_ij on existing final weights with
        source j in pre_rows, target i in post_rows and W_ij > 0.

        Runs on this network's own freshly built CSC data, after the balance
        and every other constructor transform, so nothing is rebalanced and
        no other network's array is touched. The product is formed in float64
        and stored as float32, the dtype the engines read. An empty selection
        is a preparation fault, not a neutral run."""
        import hashlib
        pre_rows, post_rows, gain = spec
        pre_rows = np.asarray(pre_rows, dtype=np.int64)
        post_rows = np.asarray(post_rows, dtype=np.int64)
        gain = float(gain)
        if not np.isfinite(gain) or gain <= 0.0:
            raise ValueError("post_balance_edge_gain must be finite and > 0")
        for name, rows in (("pre", pre_rows), ("post", post_rows)):
            if (rows.ndim != 1 or len(rows) == 0
                    or len(np.unique(rows)) != len(rows)
                    or rows.min() < 0 or rows.max() >= self.N):
                raise ValueError(
                    f"post_balance_edge_gain {name} rows must be unique valid rows")
        if len(graded_rows) and np.isin(
                pre_rows, np.asarray(graded_rows, dtype=np.int64)).any():
            # The NumPy engine's graded matrix Wg is sliced from the pre-CSC
            # weights, so a graded source would carry the gain on one engine.
            raise NotImplementedError(
                "post_balance_edge_gain on a graded source row")
        W = self.W
        before_sha = hashlib.sha256(W.data.tobytes()).hexdigest()
        is_post = np.zeros(self.N, dtype=bool)
        is_post[post_rows] = True
        pos = np.concatenate([
            int(W.indptr[j]) + np.flatnonzero(
                is_post[W.indices[W.indptr[j]:W.indptr[j + 1]]])
            for j in np.sort(pre_rows)]).astype(np.int64)
        value = W.data[pos].copy()
        selected = pos[value > 0]
        if len(selected) == 0:
            raise ValueError(
                "post_balance_edge_gain selected no existing positive final weight")
        after = (np.float64(gain) * W.data[selected].astype(np.float64)
                 ).astype(np.float32)
        W.data[selected] = after
        src = np.searchsorted(W.indptr, pos, side="right") - 1
        tgt = W.indices[pos].astype(np.int64)

        def contact(k):
            return {"pre_banc_888_id": str(self.ids[src[k]]),
                    "post_banc_888_id": str(self.ids[tgt[k]]),
                    "pre_row": int(src[k]), "post_row": int(tgt[k]),
                    "position": int(pos[k])}
        chosen, excluded = [], []
        for k in range(len(pos)):
            if value[k] > 0:
                chosen.append({**contact(k), "weight_before": float(value[k]),
                               "weight_after": float(W.data[pos[k]])})
            else:
                excluded.append({**contact(k), "weight": float(value[k]),
                                 "reason": ("zero_final_weight" if value[k] == 0
                                            else "negative_final_weight")})
        return {
            "gain": gain,
            "gain_hex": gain.hex(),
            "rule": "float32(float64(g) * float64(W_ij)) where W_ij > 0",
            "requested_pre_rows": int(len(pre_rows)),
            "requested_post_rows": int(len(post_rows)),
            "contacts_between_sets": int(len(pos)),
            "selected_contacts": int(len(selected)),
            "selected": chosen,
            "excluded": excluded,
            "selected_positions_sha256": hashlib.sha256(
                np.ascontiguousarray(selected, dtype="<i8").tobytes()).hexdigest(),
            "data_sha256_before": before_sha,
            "data_sha256_after": hashlib.sha256(W.data.tobytes()).hexdigest(),
        }

    def _post_balance_edge_gain_blocks(self, blocks, graded_rows):
        """Design efe88957 Build 5: for each block (pre_rows, post_rows, g) in
        order, W'_ij = g W_ij on every existing final weight with source j in
        pre_rows and target i in post_rows, sign kept, formed in float64 and
        stored as float32 as _post_balance_edge_gain does. A later block reads
        the weights an earlier one left. The same refusals: a gain that is not
        finite and > 0, rows that are not unique valid rows, a graded source
        row (the NumPy engine's graded matrix is sliced from the pre-CSC
        weights), and a block with no existing edge between its sets."""
        import hashlib
        W = self.W
        graded = np.asarray(graded_rows, dtype=np.int64)
        out = []
        for k, (pre_rows, post_rows, gain) in enumerate(blocks):
            where = f"post_balance_edge_gain_blocks[{k}]"
            pre_rows = np.asarray(pre_rows, dtype=np.int64)
            post_rows = np.asarray(post_rows, dtype=np.int64)
            gain = float(gain)
            if not np.isfinite(gain) or gain <= 0.0:
                raise ValueError(f"{where} gain must be finite and > 0")
            for name, rows in (("pre", pre_rows), ("post", post_rows)):
                if (rows.ndim != 1 or len(rows) == 0
                        or len(np.unique(rows)) != len(rows)
                        or rows.min() < 0 or rows.max() >= self.N):
                    raise ValueError(f"{where} {name} rows must be unique valid rows")
            if len(graded) and np.isin(pre_rows, graded).any():
                raise NotImplementedError(f"{where} on a graded source row")
            before_sha = hashlib.sha256(W.data.tobytes()).hexdigest()
            is_post = np.zeros(self.N, dtype=bool)
            is_post[post_rows] = True
            pos = np.concatenate([
                int(W.indptr[j]) + np.flatnonzero(
                    is_post[W.indices[W.indptr[j]:W.indptr[j + 1]]])
                for j in np.sort(pre_rows)]).astype(np.int64)
            if len(pos) == 0:
                raise ValueError(f"{where} selected no existing edge")
            value = W.data[pos].copy()
            W.data[pos] = (np.float64(gain) * value.astype(np.float64)
                           ).astype(np.float32)
            after = W.data[pos]
            out.append({
                "gain": gain,
                "gain_hex": gain.hex(),
                "rule": "float32(float64(g) * float64(W_ij)) on every existing edge, sign kept",
                "requested_pre_rows": int(len(pre_rows)),
                "requested_post_rows": int(len(post_rows)),
                "edges": int(len(pos)),
                "edges_positive": int((value > 0).sum()),
                "edges_negative": int((value < 0).sum()),
                "edges_zero": int((value == 0).sum()),
                "sum_abs_weight_before": float(np.abs(value.astype(np.float64)).sum()),
                "sum_abs_weight_after": float(np.abs(after.astype(np.float64)).sum()),
                "sum_weight_before": float(value.astype(np.float64).sum()),
                "sum_weight_after": float(after.astype(np.float64).sum()),
                "positions_sha256": hashlib.sha256(
                    np.ascontiguousarray(pos, dtype="<i8").tobytes()).hexdigest(),
                "data_sha256_before": before_sha,
                "data_sha256_after": hashlib.sha256(W.data.tobytes()).hexdigest(),
            })
        return out

    def _shared_occupancy(self, spec, graded_rows):
        """Design 81427d47: derive the occupancy channel from the built W, then
        set its selected entries to explicit zeros, kept in the sparsity
        pattern, so nothing is delivered twice."""
        import hashlib
        W = self.W
        before = hashlib.sha256(W.data.tobytes()).hexdigest()
        occ, audit, pos = shared_occupancy_arrays(W, self.p, self.N, graded_rows, spec,
                                                  ids=self.ids)
        W.data[pos] = np.float32(0.0)
        self.shared_occupancy = occ
        audit["data_sha256_before"] = before
        audit["data_sha256_after"] = hashlib.sha256(W.data.tobytes()).hexdigest()
        return audit

    def _psi_pair_kernel(self, spec, graded_rows):
        """Design 7b0ad6b2, section 2: each named pair's ordinary impulse, as
        the native engine applies it, and the separate state that replaces it.

        a_i = float32(W_ij * float32(gscale)), the first normalized conductance
        the membrane uses after an isolated event (decay, then arrival, then
        the gscale product); r0_i the target's native float32 decay; Q_i =
        a_i*dt/(1-r0_i) in float64. The state decays by r = float32(exp(-dt/tau))
        per tick, or by r0_i itself when tau is inherited, and each arriving
        event adds inc_i = float32(G*Q_i*(1-r)/dt), so an isolated event's
        discrete area dt*sum(x) is G*Q_i. The delay counts ticks from the
        source event to the first used tick, the ordinary path's delay_slot
        when inherited. The pair's final weight is then set to an explicit
        zero, kept in the sparsity pattern, so nothing is delivered twice and
        every other array is unchanged. An absent, repeated or nonpositive
        pair is a preparation fault.

        Design 727bccb8 (H02) appends two optional entries, (..., beta,
        factor_from). With beta, each area becomes Q_i(beta) =
        K*Q_i**(1-beta), K = sum(Q)/sum(Q**(1-beta)), in float64. With
        factor_from, a permutation in pair order, pair q takes the factor
        f = Q(beta)/Q of pair factor_from[q], renormalized by C =
        sum(Q)/sum(Q*f[factor_from]) so the total is unchanged. Either total
        must equal sum(Q) to relative 1e-5 before G. beta = 0 returns Q
        exactly (K = 1, f = 1), and the 4-entry spec is unchanged."""
        import hashlib
        pairs, delay_ms, tau_ms, gain = spec[:4]
        beta = spec[4] if len(spec) > 4 else None
        factor_from = spec[5] if len(spec) > 5 else None
        if len(spec) not in (4, 6) or (factor_from is not None and beta is None):
            raise ValueError("psi_pair_kernel is (pairs, delay_ms, tau_ms, gain[, beta, factor_from])")
        pairs = np.asarray(pairs, dtype=np.int64)
        gain = float(gain)
        if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) == 0:
            raise ValueError("psi_pair_kernel pairs must be ((src_row, tgt_row), ...)")
        if pairs.min() < 0 or pairs.max() >= self.N:
            raise ValueError("psi_pair_kernel rows must be valid rows")
        if len(np.unique(pairs[:, 1])) != len(pairs):
            raise ValueError("psi_pair_kernel targets must be distinct")
        if not np.isfinite(gain) or gain <= 0.0:
            raise ValueError("psi_pair_kernel gain must be finite and > 0")
        if len(graded_rows) and np.isin(
                pairs, np.asarray(graded_rows, dtype=np.int64)).any():
            raise NotImplementedError("psi_pair_kernel on a graded row")
        p = self.p
        dt = float(p.dt)
        dpn = getattr(p, "delay_per_neuron", None)
        if dpn is None:
            slot = np.full(self.N, max(1, int(round(p.delay / dt))), dtype=np.int64)
        else:
            slot = np.maximum(1, np.rint(np.asarray(dpn, dtype=np.float64) / dt)
                              ).astype(np.int64)
        tau_syn = np.asarray(p.tau_syn, dtype=np.float32)
        if tau_syn.ndim == 0:
            decay_g = np.full(self.N, np.float32(np.exp(-dt / tau_syn)), dtype=np.float32)
        else:
            decay_g = np.exp(-np.float32(dt) / tau_syn).astype(np.float32)
        gscale = np.float32(1.0 / abs(p.e_exc - float(np.mean(np.asarray(p.v_rest)))))
        if delay_ms is None:
            delay_ticks = slot[pairs[:, 0]]
            delay_source = "inherited delivery slot of the source"
        else:
            d = float(delay_ms)
            k = d / dt
            if not np.isfinite(d) or d <= 0.0 or abs(k - round(k)) > 1e-9:
                raise ValueError("psi_pair_kernel delay_ms must be a positive whole number of ticks")
            delay_ticks = np.full(len(pairs), int(round(k)), dtype=np.int64)
            delay_source = "searched"
        if tau_ms is not None:
            t = float(tau_ms)
            if not np.isfinite(t) or t <= 0.0:
                raise ValueError("psi_pair_kernel tau_ms must be finite and > 0")
            r_new = np.float32(np.exp(-dt / t))
        W = self.W
        before_sha = hashlib.sha256(W.data.tobytes()).hexdigest()
        pos = np.empty(len(pairs), dtype=np.int64)
        for q, (j, i) in enumerate(pairs):
            lo, hi = int(W.indptr[j]), int(W.indptr[j + 1])
            k = lo + int(np.searchsorted(W.indices[lo:hi], i))
            if k >= hi or W.indices[k] != i or (k + 1 < hi and W.indices[k + 1] == i):
                raise ValueError(f"psi_pair_kernel pair {int(j)}->{int(i)} is absent or repeated in W")
            if not W.data[k] > 0:
                raise ValueError(f"psi_pair_kernel pair {int(j)}->{int(i)} has a nonpositive final weight")
            pos[q] = k
        w = W.data[pos].copy()
        a = (w.astype(np.float32) * gscale).astype(np.float32)
        r0 = decay_g[pairs[:, 1]]
        Q = a.astype(np.float64) * dt / (1.0 - r0.astype(np.float64))
        area = Q
        eq = None
        if beta is not None:
            b = float(beta)
            if not np.isfinite(b) or b < 0.0 or b > 1.0:
                raise ValueError("psi_pair_kernel beta must be in [0, 1]")
            Qp = Q ** (1.0 - b)
            K = float(Q.sum() / Qp.sum())
            area = K * Qp
            f = area / Q
            eq = {"beta": b, "beta_hex": b.hex(), "K": K, "K_hex": K.hex(),
                  "factor": [float(x) for x in f], "factor_from": None, "C": None}
            if factor_from is not None:
                ff = np.asarray(factor_from, dtype=np.int64)
                if ff.shape != (len(pairs),) or not np.array_equal(np.sort(ff), np.arange(len(pairs))):
                    raise ValueError("psi_pair_kernel factor_from must be a permutation of the pair indices")
                fp = f[ff]
                C = float(Q.sum() / (Q * fp).sum())
                area = C * Q * fp
                eq.update(factor_from=[int(x) for x in ff], C=C, C_hex=C.hex(),
                          factor_applied=[float(x) for x in fp])
            rel = abs(float(area.sum()) - float(Q.sum())) / float(Q.sum())
            if not rel <= 1e-5:
                raise ValueError(f"psi_pair_kernel redistributed area differs from sum(Q) by {rel:.3g}")
            eq.update(area_sum=float(area.sum()), Q_sum=float(Q.sum()), area_rel_error=rel)
        r = r0.copy() if tau_ms is None else np.full(len(pairs), r_new, dtype=np.float32)
        inc = (gain * area * (1.0 - r.astype(np.float64)) / dt).astype(np.float32)
        if not np.all(np.isfinite(inc)) or np.any(r >= 1.0) or np.any(r < 0.0):
            raise ValueError("psi_pair_kernel produced a nonfinite increment or a decay outside [0, 1)")
        W.data[pos] = np.float32(0.0)
        self.psi_pairs = {"src": pairs[:, 0].copy(), "tgt": pairs[:, 1].copy(),
                          "delay_ticks": delay_ticks.astype(np.int64), "r": r, "inc": inc}
        return {
            "rule": ("x[k] = r*x[k-1] + inc*n[k], inc = float32(G*Q*(1-r)/dt), "
                     "Q = a*dt/(1-r0), a = float32(W*float32(gscale))"),
            "gain": gain, "gain_hex": gain.hex(),
            "delay_ms": None if delay_ms is None else float(delay_ms),
            "tau_ms": None if tau_ms is None else float(tau_ms),
            "delay_source": delay_source,
            "tau_source": "inherited target decay" if tau_ms is None else "searched",
            "dt_ms": dt, "gscale_float32": float(gscale),
            "pairs": [{"pre_banc_888_id": str(self.ids[int(j)]),
                       "post_banc_888_id": str(self.ids[int(i)]),
                       "pre_row": int(j), "post_row": int(i), "position": int(pos[q]),
                       "weight_before": float(w[q]), "weight_after": float(W.data[pos[q]]),
                       "a": float(a[q]), "r0": float(r0[q]), "Q": float(Q[q]),
                       "Q_hex": float(Q[q]).hex(), "r": float(r[q]), "inc": float(inc[q]),
                       "inherited_slot": int(slot[int(j)]),
                       "delay_ticks": int(delay_ticks[q]),
                       **({} if eq is None else {"area": float(area[q]), "area_hex": float(area[q]).hex()})}
                      for q, (j, i) in enumerate(pairs)],
            "data_sha256_before": before_sha,
            "data_sha256_after": hashlib.sha256(W.data.tobytes()).hexdigest(),
            **({} if eq is None else {"equalize": eq}),
        }

    def _build_gap(self):
        p = self.p
        law = p.gap_law
        if law not in (None, "conserving", "one_way"):
            raise ValueError("gap_law must be conserving or one_way")
        if type(p.gap_substep_factor) is not int or p.gap_substep_factor < 1:
            raise ValueError("gap_substep_factor must be a positive integer")
        if type(p.gap_current_tap) is not bool:
            raise ValueError("gap_current_tap must be boolean")
        if law is None:
            if p.gap_reverse_ratio is not None or p.gap_substep_factor != 1 or p.gap_current_tap:
                raise ValueError("new gap controls require gap_law")
        else:
            if not len(p.gap_pairs) or not len(p.gap_pair_weights):
                raise ValueError("gap_law requires pairs and per-pair weights")
            if p.gap_rectify or p.gap_spikelet != 0:
                raise ValueError("gap_law is incompatible with gap_rectify or nonzero spikelet")
            if law == "conserving":
                rho = p.gap_reverse_ratio
                if rho is None or not np.isfinite(rho) or not 0 <= rho <= 1:
                    raise ValueError("conserving gap_law requires reverse ratio in [0, 1]")
            elif p.gap_reverse_ratio is not None:
                raise ValueError("one_way law cannot use gap_reverse_ratio")
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
            if p.gap_rectify or law == "one_way":
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

    def set_gap_pair_weights(self, weights, reverse_ratio=None):
        """Change only the existing electrical block, preserving zero entries.

        CNS.run creates a fresh native state. Close its previous allocation so
        neither its CSR weights nor its substep count can survive this change.
        """
        if self.p.gap_law is None:
            raise ValueError("set_gap_pair_weights requires gap_law")
        values = np.asarray(weights, dtype=np.float64)
        if (values.shape != (len(self.p.gap_pairs),) or not np.isfinite(values).all()
                or np.any(values < 0) or np.any(values > np.finfo(np.float32).max)):
            raise ValueError("weights must be one finite nonnegative value per pair")
        rho = self.p.gap_reverse_ratio if reverse_ratio is None else reverse_ratio
        if self.p.gap_law == "conserving":
            if rho is None or not np.isfinite(rho) or not 0 <= rho <= 1:
                raise ValueError("reverse ratio must be in [0, 1]")
        elif reverse_ratio is not None:
            raise ValueError("one_way law cannot use gap_reverse_ratio")
        engine = getattr(self, "_metal_engine", None)
        if engine is not None:
            engine.close()
        self.p.gap_pair_weights = tuple(float(x) for x in values)
        self.p.gap_reverse_ratio = rho
        self._build_gap()

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
            membrane_drive_fraction=None, record_input=None,
            light_input=None, record_voltage_every_ms=None,
            record_voltage_native=False, record_lamina=None,
            native_record=None, graded_gain_window=None,
            graded_release_record=None, record_opsin=False,
            spike_tap_rows=None, spike_tap=None, row_current=None,
            inject=None, silence_window=None, rest_drive=None,
            record_holddown=False, record_voltage_native_state=False):
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
        graded_gain_window : dict or None
            Design 336cacb4, CUDA only: {"rows", "factor", "start_ms",
            "stop_ms"} multiplies the listed graded rows' release gain by
            factor while start_ms <= t < stop_ms, window-aligned.
        graded_release_record : dict or None
            Design 336cacb4, readout only: {"windows_ms", "onset_ms"} returns
            each graded row's mean post-gain release per window, its first
            update at or after onset_ms, and its summed |outgoing weight|.
        record_opsin : bool
            Design 87a3ad35, B4. With an opsin or threshold-window set in the
            parameters, return per step and per slot the state s, g_op =
            g_max s, the opsin current g_op (E_op - v) at the potential the
            step started from ('i_start') and after the membrane update and
            refractory clamp ('i_after'), under out["opsin"]. Record k is step
            k, at time (k + 1) dt. It writes nothing to the model.
        record_holddown : bool
            Design d27f82df B8. With holddown_slots in the parameters, return
            under out["holddown"] (cns.holddown_out) each slot's state after
            the last step, its peak g_h and that step, and per step and slot
            the row current before adaptation and the hold-down's c, g_h, L
            and b after the step, in every engine. It writes nothing to the
            model.
        spike_tap_rows, spike_tap : rows and a caller-owned list, or None
            Default-off tap (design fa5adfc8). After each window the
            (CNS step, row) pairs of those rows' spikes are appended to the
            list, sorted by step then row. It reads the engine's spikes and
            changes nothing. The NumPy engine appends after every step
            (design 302c1bfb, the batch-2 identity recorder).
        row_current : dict or None
            Design 2d12a3b3 (B2), CUDA only: {"rows", "i0_mv", "i1_mv",
            "on_tick", "off_tick"}, one entry per window, ticks of dt from
            the run's start. On a listed row at tick k the membrane update
            adds I = i0 + i1 [on <= k < off] mV to its current term before
            the adaptation w is subtracted, so a held I moves the steady
            voltage by I (dt/tau)/(1 - e^(-dt/tau)); I is not scaled by the
            row's resistance factor. A row may be listed once per window
            (design 6d0042d5, Build 3) when its windows share one i0 and do
            not overlap. The Metal host and the NumPy engine refuse it;
            out["row_current"] records the row and window counts and each
            array's sha256.
        silence_window : dict {"rows", "on_ms", "off_ms"}, or None (default).
            Design 302c1bfb, Build 10: the listed rows test against a 1e9 mV
            threshold on every step with on_ms <= t < off_ms and against
            their own thresholds otherwise. It draws no random number. Both
            edges fall on drive ticks (the Metal engine switches between
            windows). out["silence_window"] counts the rows' spikes inside
            the window. NumPy and Metal; None = bit-identical.
        rest_drive : dict {"rows", "hz", "kick_fraction", "stream"}, or None.
            Design 1e20a949, Build C1 (walk_search's `sez_rest`). On every
            step each listed row draws one uniform from its own generator,
            numpy.random.default_rng([seed, stream]), and a draw below
            hz * dt / 1000 adds kick_fraction times the row's own
            v_threshold - effective_v_rest margin to its membrane after the
            threshold test, the membrane_drive_rows kick: the threshold test
            reads it next step, inhibition subtracts from it, a same-step
            spike resets over it and a refractory clamp erases it. The
            network's rng is never read, so hz 0.0 is bit-identical to None.
            Refused with membrane_drive_rows (the GPU engines hold one kick
            mask), on graded rows, and on a row in the drive set.
            out["rest_drive"] is the receipt. NumPy and Metal/CUDA.
        inject : list of (row, t_on_ms, t_off_ms, amp_mv), or None (default).
            Constant current injection, team-henri's semantics (study
            ca84ead4, design 06226d13, 2026-09-21), carried here for design
            afc8fd03 on NumPy and Metal: amp_mv is added to that row's
            membrane current `extra` on every step with t_on <= t < t_off,
            after the gap current and before adaptation, in extra's units
            (mV of steady-state depolarisation at adapt_a 0). Entries on one
            row add. Refused with drive_fn and force_spikes_fn. The team CUDA
            library does not carry it. None = bit-identical.

        Returns dict with 'counts' (spikes per recorded neuron), 'rates_hz',
        'n_spikes' total, and optionally 'trace'.
        """
        p = self.p
        if rest_drive is not None and membrane_drive_rows is not None \
                and len(membrane_drive_rows):
            raise ValueError(
                "rest_drive and membrane_drive_rows share the GPU engines' "
                "one membrane-drive mask")
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
            # the receptor-scoped serotonin state runs on the CUDA library
            # since design b1746728, Build 1, and on Metal since design
            # fcb48834, Build 0 (gate gpu/gate_v0.py --serotonin): k_ser
            # after k_step, cns.py's float32 order, the applied sum in
            # double on the host.
            # the octopamine state (design 8b6e0441, Build 1) runs on the
            # CUDA library, k_ser on its own Ser; cns_metal.py refuses it on
            # a library without cns_engine_octopamine_state_config (Metal).
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
                record_input=record_input, light_input=light_input,
                record_voltage_every_ms=record_voltage_every_ms,
                record_voltage_native=record_voltage_native,
                record_lamina=record_lamina, native_record=native_record,
                graded_gain_window=graded_gain_window,
                graded_release_record=graded_release_record,
                record_opsin=record_opsin,
                spike_tap_rows=spike_tap_rows, spike_tap=spike_tap,
                row_current=row_current, inject=inject,
                silence_window=silence_window, rest_drive=rest_drive,
                record_holddown=record_holddown,
                record_voltage_native_state=record_voltage_native_state)
        elif getattr(p, "backend", "numpy") != "numpy":
            raise ValueError(f"unknown backend {p.backend!r}")
        if row_current is not None:
            raise NotImplementedError(
                "row_current (design 2d12a3b3) runs on the native CUDA engine only")
        tap_member = None
        if spike_tap_rows is not None or spike_tap is not None:
            if spike_tap_rows is None or type(spike_tap) is not list:
                raise ValueError(
                    "spike_tap_rows needs spike_tap, a list the caller owns")
            _tap_rows = np.asarray(spike_tap_rows, dtype=np.int64)
            if (_tap_rows.ndim != 1 or not len(_tap_rows) or (_tap_rows < 0).any()
                    or (_tap_rows >= self.N).any()
                    or len(np.unique(_tap_rows)) != len(_tap_rows)):
                raise ValueError("spike_tap_rows must be distinct engine rows")
            tap_member = np.zeros(self.N, dtype=bool)
            tap_member[_tap_rows] = True
        if native_record is not None:
            raise NotImplementedError(
                "native_record (design 02e57559) requires the native CUDA engine")
        if graded_gain_window is not None or graded_release_record is not None:
            raise NotImplementedError(
                "graded_gain_window and graded_release_record (design "
                "336cacb4) run on the native engines only")
        if getattr(p, "release_resource_target_rows", None) is not None:
            raise NotImplementedError(
                "release_resource_target_rows (design 02e57559) requires the "
                "native CUDA engine")
        if getattr(self, "shared_occupancy", None) is not None:
            raise NotImplementedError(
                "shared_occupancy (design 81427d47) runs on the native CUDA engine only")
        if p.gap_law is not None and (p.gap_law != "one_way" or p.gap_current_tap
                                      or gap_law_substeps(p, self.N) != 1):
            raise NotImplementedError(
                "the conserving gap law, electrical substeps and the gap current "
                "tap require the native Metal engine")
        if record_input is not None:
            raise NotImplementedError(
                "record_input is an exact Metal-engine diagnostic")
        if (light_input is not None or record_lamina is not None
                or record_voltage_every_ms is not None or record_voltage_native
                or record_voltage_native_state):
            raise NotImplementedError("lamina input and recording require the native CUDA engine")
        rng = np.random.default_rng(seed)
        n_steps = int(round(duration_ms / p.dt))
        if inject is not None and (drive_fn is not None
                                   or force_spikes_fn is not None):
            raise ValueError("inject is refused with drive_fn and force_spikes_fn")
        inj_segs = _inject_segments(inject, p.dt, n_steps, self.N)
        inj_seg_i, inj_vec = 0, None
        _oa = octopamine_offset_plan(p, self.N)   # design 8b6e0441, Build 2
        oa_vec = None
        if _oa is not None:
            oa_vec = np.zeros(self.N, dtype=np.float32)
            oa_vec[_oa[0]] = _oa[1]
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
        adapt_a_below = getattr(p, "adapt_a_below", None)
        adapt_a_below = (None if adapt_a_below is None else np.broadcast_to(
            np.asarray(adapt_a_below, dtype=np.float32), (self.N,)).copy())
        use_adapt = bool(np.any(adapt_a != 0) or np.any(adapt_b != 0)
                         or (adapt_a_below is not None and np.any(adapt_a_below != 0)))
        w_adapt = np.zeros(self.N, dtype=np.float32) if use_adapt else None
        if getattr(p, "adapt_w_init_rows", None) is not None:
            if not use_adapt:
                raise ValueError("adapt_w_init_rows needs adaptation on the network")
            _w_rows, _w_vals = adapt_w_init_arrays(p, self.N)
            w_adapt[_w_rows] = _w_vals
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
            serotonin_init_hz = float(getattr(p, "serotonin_init_hz", 0.0))
            if not np.isfinite(serotonin_init_hz) or serotonin_init_hz < 0.0:
                raise ValueError("serotonin_init_hz must be finite and non-negative")
            serotonin_state_hz = np.float32(serotonin_init_hz)
            serotonin_state_max_hz = np.float32(serotonin_init_hz)
            serotonin_state_applied_sum_hz = 0.0
            serotonin_source_spikes = 0
            serotonin_extra = (np.zeros(self.N, dtype=np.float32)
                                if serotonin_gain > 0.0 else None)
            # Design b1746728, Build 1: the readback, s at the end of each
            # 5 ms bin, after that step's source spikes are added
            serotonin_steps_per_bin = max(1, int(round(5.0 / p.dt)))
            serotonin_state_bins = np.zeros(
                n_steps // serotonin_steps_per_bin, dtype=np.float32)
        else:
            serotonin_source = serotonin_target = None
            serotonin_state_hz = np.float32(0.0)
            serotonin_state_max_hz = np.float32(0.0)
            serotonin_state_applied_sum_hz = 0.0
            serotonin_source_spikes = 0
            serotonin_extra = None
        # Design 8b6e0441, Build 1: the octopamine state, the serotonin
        # state's arithmetic in its own variables
        _os = octopamine_state_plan(p, self.N, p.dt, n_steps, self.W)
        if _os is not None:
            os_mask = np.zeros(self.N, dtype=bool)
            os_mask[_os["counted"]] = True
            os_state_hz = np.float32(_os["init"])
            os_state_max_hz = np.float32(_os["init"])
            os_applied_sum_hz = 0.0
            os_source_spikes = 0
            os_gain = np.float32(_os["gain"])
            os_extra = (np.zeros(self.N, dtype=np.float32)
                        if _os["gain"] > 0.0 else None)
            os_bins = np.zeros(_os["n_bins"], dtype=np.float32)
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
        # Exponential IF is a spike-initiation law. A graded row has no
        # threshold and never fires, so it is never reset: the exponential
        # term would carry one held a few delta_T above v_threshold to v_peak
        # and keep it there. The term and the v_peak cap act on the rows
        # that can fire (gpu-7, standing sweep S19, 2026-10-07; no run had
        # set delta_T above 0 on a body with graded rows).
        exp_off = graded if use_exp and graded.any() else None
        sw_mask = None
        sw_spikes = 0
        if silence_window is not None:
            if use_exp:
                raise NotImplementedError(
                    "silence_window raises v_threshold; exponential IF "
                    "fires at v_peak")
            sw_rows, sw_on, sw_off = _silence_window_steps(
                silence_window, self.N, p.dt, drive_every_ms)
            sw_mask = np.zeros(self.N, dtype=bool)
            sw_mask[sw_rows] = True
            sw_thr = np.broadcast_to(
                np.asarray(spike_at), (self.N,)).copy()
            sw_thr[sw_rows] = 1e9

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
        # Design 7b0ad6b2's pair channel with design e05c0eef B1's recovery,
        # on NumPy since 2026-10-07 (gpu-7), in cns_engine.cu's order: each
        # tick takes the pair's arrivals for that tick, applies the recovery,
        # sets x = r*x + inc*n with each product and the sum rounded to
        # float32, and adds x to the target's ge after tonic and before either
        # integrator; a fired source's event arrives delay ticks later.
        # _psi_pair_kernel has zeroed every pair's ordinary weight, so nothing
        # is delivered twice. The ring holds max delay + 1 slots, the least
        # that never reuses a slot before it is read.
        pp = getattr(self, "psi_pairs", None)
        if pp is not None:
            if g_i is None:
                raise ValueError(
                    "psi_pair_kernel needs the conductance law with split "
                    "inhibition, as on the native engines")
            pp_src = np.asarray(pp["src"], dtype=np.int64)
            pp_tgt = np.asarray(pp["tgt"], dtype=np.int64)
            pp_delay = np.asarray(pp["delay_ticks"], dtype=np.int64)
            pp_r = np.asarray(pp["r"], dtype=np.float32)
            pp_inc = np.asarray(pp["inc"], dtype=np.float32)
            if int(pp_delay.min()) < 1:
                raise ValueError("psi_pair_kernel delay must be at least one tick")
            pp_nr = int(pp_delay.max()) + 1
            pp_ring = np.zeros((pp_nr, len(pp_src)), dtype=np.int64)
            pp_x = np.zeros(len(pp_src), dtype=np.float32)
            pp_rec = (psi_pair_recovery_values(p.psi_pair_recovery, float(p.dt))
                      if getattr(p, "psi_pair_recovery", ()) else None)
            pp_form = 0 if pp_rec is None else int(pp_rec["form_code"])
            pp_clk = np.zeros(len(pp_src), dtype=np.int64)
            pp_res = np.ones(len(pp_src), dtype=np.float32)
            pp_cnt = np.zeros((len(pp_src), 4), dtype=np.int64)
            pp_area = np.zeros(len(pp_src), dtype=np.float32)
            if pp_rec is not None:
                pp_ticks = int(pp_rec["ticks"])
                pp_rdecay = np.float32(pp_rec["decay"])
                pp_rkeep = np.float32(pp_rec["keep"])
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
            # design 302c1bfb, Build 8: the gain folds into the one float32
            # constant, so a gain of None or 1.0 gives the historical bytes
            _presyn_gain = getattr(p, "presynaptic_inhibition_gain", None)
            if _presyn_gain is None:
                presyn_kg = np.float32(gscale)
            else:
                if not np.isfinite(float(_presyn_gain)) or float(_presyn_gain) < 0.0:
                    raise ValueError(
                        "presynaptic_inhibition_gain must be finite and >= 0")
                presyn_kg = np.float32(float(_presyn_gain) * gscale)
            presyn_log_t, presyn_log_r = [], []
            presyn_log_f, presyn_log_G, presyn_log_gi = [], [], []
            _psi_src = getattr(p, "presynaptic_inhibition_source_rows", None)
            _psi_tau = getattr(p, "presynaptic_inhibition_tau_ms", None)
            psi9 = _psi_src is not None
            if not psi9 and _psi_tau is not None:
                raise ValueError(
                    "presynaptic_inhibition_tau_ms is the decay of the source "
                    "state and needs presynaptic_inhibition_source_rows")
            if psi9:
                if not use_queue:
                    raise ValueError(
                        "the source-fed release state needs delay_delivery "
                        "'queue', where each arrival tick is one delivery")
                _src = np.asarray(_psi_src, dtype=np.int64)
                if (_src.ndim != 1 or not len(_src)
                        or len(np.unique(_src)) != len(_src)
                        or np.any(_src < 0) or np.any(_src >= self.N)):
                    raise ValueError(
                        "presynaptic_inhibition_source_rows must be unique "
                        "valid rows")
                if np.any(graded[_src]):
                    raise ValueError(
                        "a graded source releases without a spike arrival")
                # sources with an inhibitory sign, and their edges onto the
                # listed rows that carry a nonzero final weight
                _src = _src[self.sign[_src] < 0]
                psi_src_mask = np.zeros(self.N, dtype=bool)
                psi_src_mask[_src] = True
                presyn_pos = np.full(self.N, -1, dtype=np.int64)
                presyn_pos[presyn_rows] = np.arange(len(presyn_rows))
                _nper = self._indptr[_src + 1] - self._indptr[_src]
                _off = (np.repeat(self._indptr[_src], _nper) + np.arange(int(_nper.sum()))
                        - np.repeat(np.cumsum(_nper) - _nper, _nper))
                psi_pairs = int(np.count_nonzero(
                    (presyn_pos[self._indices[_off]] >= 0) & (self._data[_off] != 0)))
                _r_i = np.broadcast_to(np.asarray(decay_g_i, dtype=np.float32),
                                       (self.N,))[presyn_rows].astype(np.float32)
                if _psi_tau is None:
                    psi_r9 = _r_i.copy()
                else:
                    if not np.isfinite(float(_psi_tau)) or float(_psi_tau) <= 0.0:
                        raise ValueError(
                            "presynaptic_inhibition_tau_ms must be finite and > 0")
                    # the expression run() gives decay_g_i for a scalar tau_syn
                    psi_r9 = np.full(len(presyn_rows), np.float32(np.exp(
                        -p.dt / np.asarray(float(_psi_tau), dtype=np.float32))),
                        dtype=np.float32)
                psi_c9 = ((1.0 - psi_r9.astype(np.float64))
                          / (1.0 - _r_i.astype(np.float64))).astype(np.float32)
                psi_g9 = np.zeros(len(presyn_rows), dtype=np.float32)
                psi_s9 = [None]
                psi_n_src = int(len(_src))
        else:
            presyn_rows = np.empty(0, dtype=np.int64)
            presyn_mask = None
            psi9 = False

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
        _rr = release_resource_arrays(p, self.N, self.graded)
        use_rr = _rr is not None
        if use_rr:
            rr_d, rr_c, rr_k = _rr["d"], _rr["c"], _rr["k"]
            rr_x = np.ones(len(_rr["rows"]), dtype=np.float32)
            rr_pos = np.full(self.N, -1, dtype=np.int64)
            rr_pos[_rr["rows"]] = np.arange(len(_rr["rows"]))
            rr_log_t, rr_log_r, rr_log_x = [], [], []

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
        rest_plan = (None if rest_drive is None else rest_drive_plan(
            rest_drive, self.N, p.v_threshold, v_rest, self.graded, seed, p.dt))
        rest_drive_refuse_overlap(rest_plan, drive_idx)
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

        use_release_scale = use_std or presynaptic_inhibition or use_rr

        # Design 2163ba7d, Build 2: the slow inhibitory pool, one slot per
        # target of a listed row. None adds no branch below.
        _sl = slow_inh_arrays(p, self.N, indptr, indices, data, n_steps, graded)
        if _sl is not None:
            if not use_queue:
                raise ValueError("slow_inh needs delay_delivery 'queue' with "
                                 "per-neuron delays, the path wm1 runs")
            if _sl["d_i"] != decay_g_i:
                raise ValueError("slow_inh: the fast decay differs from run()'s")
            sl_src, sl_tgt, sl_tpos = _sl["mask"], _sl["targets"], _sl["tpos"]
            sl_keep, sl_sfac, sl_ds = _sl["keep"], _sl["sfac"], _sl["d_s"]
            sl_nt = len(sl_tgt)
            sl_ring = np.zeros((n_delay, sl_nt), dtype=np.float32)
            sl_g = np.zeros(sl_nt, dtype=np.float32)
            sl_acc_fast = np.zeros(sl_nt, dtype=np.float64)
            sl_acc_slow = np.zeros(sl_nt, dtype=np.float64)
            sl_int = np.zeros(sl_nt, dtype=np.float64)
            sl_rec_cols = np.flatnonzero(_sl["rec_pos"] >= 0)
            sl_rec_q = _sl["rec_pos"][sl_rec_cols]
            sl_rec_acc = np.zeros((_sl["n_bins"], len(_sl["rec_pos"])),
                                  dtype=np.float64)
            sl_spb = _sl["spb"]
            sl_arrived = [0]

        def _slow_split(src, nper, rows, neg, slot_):
            """Design 2163ba7d, Build 2: on the listed rows' edges, keep the
            fast share in neg (in place) and add the slow share to the slow
            ring, each summed per target as one float32 bincount."""
            le = np.repeat(sl_src[src], nper)
            if not le.any():
                return
            sl_arrived[0] += int(np.count_nonzero(sl_src[src]))
            tp = sl_tpos[rows[le]]
            nl = neg[le]
            fast = nl * sl_keep
            slow = nl * sl_sfac
            neg[le] = fast
            ss = np.bincount(tp, weights=slow, minlength=sl_nt).astype(np.float32)
            sl_ring[slot_] += ss
            sl_acc_slow[:] += ss
            sl_acc_fast[:] += np.bincount(
                tp, weights=fast, minlength=sl_nt).astype(np.float32)

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
                neg = np.maximum(-vals, 0.0)
                if _sl is not None:
                    _slow_split(src, nper, rows, neg, slot_)
                if psi9:
                    # design 302c1bfb, Build 8: the sources' share of this
                    # tick's g_i increment on the listed rows, summed in the
                    # same order and rounded once, as ring_i's own sum
                    _m9 = np.repeat(psi_src_mask[src], nper)
                    if _m9.any():
                        _h9 = presyn_pos[rows[_m9]]
                        _k9 = _h9 >= 0
                        if _k9.any():
                            psi_s9[0] = np.bincount(
                                _h9[_k9], weights=neg[_m9][_k9],
                                minlength=len(presyn_rows)).astype(np.float32)
                ring_i[slot_] += np.bincount(
                    rows, weights=neg,
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
        # Design efe88957, Build 2: the seam's list form. Each triple runs the
        # single key's arithmetic on its own pools. `src` is every source's
        # read for the next tick: last tick's spike indicator on a spiking row
        # (`last` names the rows set), its last release over g_every on a
        # graded row, written where the release is computed.
        _coin2 = None
        if p.coincidence_triples:
            if _coin is not None:
                raise ValueError("coincidence_gain and coincidence_triples are "
                                 "two forms of one seam; give one")
            _coin2 = {"trips": coincidence_triple_seam(
                          self.N, self._indptr, self._indices, self._data,
                          p.coincidence_triples, graded, p.dt),
                      "src": np.zeros(self.N, np.float32),
                      "last": np.empty(0, np.int64)}
            for _q in _coin2["trips"]:
                _nq = len(_q["tgt"])
                _q.update(ring=np.zeros((_q["lag"], _nq), np.float32),
                          fs=np.zeros(_nq, np.float32), ss=np.zeros(_nq, np.float32),
                          applied_max=0.0, applied_sum=0.0, fs_max=0.0,
                          slowdel_max=0.0)

        # Design 87a3ad35, B1 and B4: the opsin conductance and arm E's
        # threshold window, one slot per row. None adds no branch below.
        _op = opsin_arrays(p, self.N, duration_ms, e_inh, self.n_edges, graded)
        if record_opsin and _op is None:
            raise ValueError(
                "record_opsin needs an opsin or threshold-window set")
        if _op is not None:
            op_rows = _op["rows"]
            op_s = np.zeros(len(op_rows), dtype=np.float32)
            op_force = _op["force"].astype(bool)
            op_thr = np.flatnonzero(_op["kind"] == 1)
            op_thr_rows = op_rows[op_thr]
            op_one = np.float32(1.0)
            op_tap = ({k: np.zeros((n_steps, len(op_rows)), dtype=np.float32)
                       for k in OPSIN_TAP_FIELDS} if record_opsin else None)

        # Design d27f82df B8: B2's exact-update mask and B3's hold-down, in
        # cns_engine.cu's order of operations. None adds no branch below.
        self._holddown_numpy = None
        _xh = row_exact_arrays(p, self.N, conductance_integrator == "analytic", _op)
        xm = None if _xh is None else _xh["xm"]
        _hd = None if _xh is None else _xh["hd"]
        if record_holddown and _hd is None:
            raise ValueError("record_holddown needs the hold-down (holddown_by_id)")
        if xm is not None:
            xm_vr = v_rest if v_rest.ndim == 0 else v_rest[xm]
            xm_ei = e_inh if e_inh.ndim == 0 else e_inh[xm]
            xm_dtt = dt_over_tau if np.ndim(dt_over_tau) == 0 else dt_over_tau[xm]
        if _hd is not None:
            hd_rows = _hd["rows"]
            hd_pool = _hd["pool"]
            hd_xpos = _hd.get("xpos")
            hd_vr = (np.full(len(hd_rows), v_rest, dtype=np.float32) if v_rest.ndim == 0
                     else v_rest[hd_rows].astype(np.float32))
            hd_vrel = (hd_vr + _hd["e_rel"]).astype(np.float32)
            hd_c, hd_g, hd_L, hd_b, hd_pk = (np.zeros(len(hd_rows), dtype=np.float32)
                                             for _ in range(5))
            hd_pt = np.zeros(len(hd_rows), dtype=np.uint32)
            hd_one, hd_zero = np.float32(1.0), np.float32(0.0)
            hd_tap = (np.zeros((n_steps, len(hd_rows), 5), dtype=np.float32)
                      if record_holddown else None)

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
            if psi9:
                psi_g9 *= psi_r9
            if pp is not None:
                # the pair channel: this tick's arrivals, the recovery, then
                # x = r*x + inc*n (cns_engine.cu's pair_recover and k_step)
                _pslot = t % pp_nr
                pp_na = pp_ring[_pslot].copy()
                pp_ring[_pslot] = 0
                if pp_form == 0:
                    pp_nd = pp_na.astype(np.float32)
                else:
                    pp_cnt[:, 3] += pp_na > 1
                    _arr = pp_na > 0
                    if pp_form == 1:
                        # dead time: count down, then deliver at 0 and restart
                        np.subtract(pp_clk, 1, out=pp_clk, where=pp_clk > 0)
                        _skip = _arr & (pp_clk > 0)
                        _dlv = _arr & (pp_clk == 0)
                        pp_cnt[_skip, 1] += pp_na[_skip]
                        pp_cnt[_dlv, 0] += pp_na[_dlv]
                        pp_area[_dlv] += pp_na[_dlv].astype(np.float32)
                        pp_clk[_dlv] = pp_ticks
                        pp_nd = np.where(_dlv, pp_na, 0).astype(np.float32)
                    else:
                        # depletion: recover, then each event takes R and
                        # keeps R*keep, summed in event order
                        pp_res = (np.float32(1.0)
                                  - (np.float32(1.0) - pp_res) * pp_rdecay)
                        pp_nd = np.zeros(len(pp_src), dtype=np.float32)
                        for _k in range(int(pp_na.max())):
                            _m = pp_na > _k
                            pp_cnt[_m & (pp_res < np.float32(1.0)), 2] += 1
                            pp_nd[_m] += pp_res[_m]
                            pp_res[_m] *= pp_rkeep
                        pp_cnt[_arr, 0] += pp_na[_arr]
                        pp_area[_arr] += pp_nd[_arr]
                pp_x = pp_x * pp_r + pp_inc * pp_nd
            if _sl is not None:
                sl_g *= sl_ds
            if use_std:
                # exact relaxation of the vesicle pool toward 1
                x_res *= std_dec
                x_res += np.float32(1.0) - std_dec
            if use_rr:
                # the per-row resource: two roundings, no fused multiply-add
                rr_x *= rr_d
                rr_x += rr_c
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
            if _coin2 is not None:
                _sv2 = _coin2["src"]
                for _q in _coin2["trips"]:
                    _q["fs"] *= _q["decay"]
                    _q["fs"] += _q["fast"] @ _sv2
                    _q["ss"] *= _q["decay"]
                    _q["ss"] += _q["slow"] @ _sv2
                    _k = t % _q["lag"]
                    _slow_del = _q["ring"][_k].copy()
                    _q["ring"][_k] = _q["ss"]
                    _capp = _q["gain"] * _q["fs"] * _slow_del
                    g[_q["tgt"]] += _capp
                    _q["applied_max"] = max(_q["applied_max"],
                                            float(np.abs(_capp).max()))
                    _q["applied_sum"] = (_q["applied_sum"]
                                         + float(np.abs(_capp).sum()))
                    _q["fs_max"] = max(_q["fs_max"], float(_q["fs"].max()))
                    _q["slowdel_max"] = max(_q["slowdel_max"],
                                            float(_slow_del.max()))
            if ring_i is not None:
                g_i += ring_i[slot]
                ring_i[slot] = 0.0
            if psi9 and psi_s9[0] is not None:
                psi_g9 += psi_c9 * psi_s9[0]
                psi_s9[0] = None
            if _sl is not None:
                sl_g += sl_ring[slot]
                sl_ring[slot] = 0.0

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
            if _os is not None:   # design 8b6e0441, Build 1: summed, never assigned
                os_state_hz *= _os["decay"]
                os_applied_sum_hz += float(os_state_hz)
                if os_extra is not None:
                    os_extra.fill(0.0)
                    os_extra[_os["target"]] = os_gain * os_state_hz
                    extra = extra + os_extra
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
            if inj_segs is not None:
                if t >= inj_segs[inj_seg_i][1]:
                    inj_seg_i += 1
                inj_vec = inj_segs[inj_seg_i][2]
                if inj_vec is not None:
                    extra = extra + inj_vec
            if oa_vec is not None:   # design 8b6e0441, Build 2: the offset
                extra = extra + oa_vec
            if _hd is not None and hd_tap is not None:
                # the row current before adaptation, cns_engine.cu's irow
                hd_tap[t, :, 0] = extra[hd_rows] if np.ndim(extra) else extra
            if use_adapt:
                extra = extra - w_adapt
            if use_exp:
                _ex = p.delta_T * np.exp(
                    np.minimum((v - p.v_threshold) * inv_dT, 20.0))
                if exp_off is not None:
                    _ex[exp_off] = 0
                extra = extra + _ex
            if _op is not None:
                # exact one-step gating, then the conductance this step uses
                op_lit = opsin_lit(_op["win"], _op["n_win"], t, _op["train"])
                op_s = np.where(
                    op_lit,
                    np.where(op_force, op_one,
                             op_one + (op_s - op_one) * _op["a_on"]),
                    op_s * _op["a_off"]).astype(np.float32)
                op_g = _op["g_max"] * op_s
                op_i = op_g * (_op["e_rev"] - v[op_rows])
                if op_tap is not None:
                    op_tap["s"][t] = op_s
                    op_tap["g_op"][t] = op_g
                    op_tap["i_start"][t] = op_i

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
                    if _sl is not None:
                        # design 2163ba7d, Build 2: the slow pool, the fast
                        # pool's reversal
                        _sg = sl_g * gscale
                        gi[sl_tgt] += _sg
                        sl_int += _sg
                        sl_rec_acc[t // sl_spb, sl_rec_cols] += _sg[sl_rec_q]
                else:
                    ge = np.maximum(g, 0.0) * gscale
                    gi = np.maximum(-g, 0.0) * gscale
                if tonic_ge is not None:
                    ge = ge + tonic_ge
                if pp is not None:
                    # the pair channel after tonic, before either integrator
                    ge[pp_tgt] += pp_x
                if conductance_integrator == "analytic":
                    # Exact one-step solution with ge, gi and extra held
                    # constant over dt:
                    #   tau dV/dt = Vrest - V + ge(Ee-V) + gi(Ei-V) + I.
                    # Normalising before multiplying by reversal potentials
                    # avoids the overflow-prone ge*V product in Euler. The
                    # exponential approaches the instantaneous equilibrium
                    # monotonically for any finite non-negative conductance.
                    total_g = np.float32(1.0) + ge + gi
                    if _op is not None:
                        total_g[op_rows] += op_g
                    if _hd is not None:
                        total_g[hd_rows] += hd_g   # B3, after the opsin
                    inv_total_g = np.float32(1.0) / total_g
                    v_inf = (v_rest * inv_total_g
                             + p.e_exc * (ge * inv_total_g)
                             + e_inh * (gi * inv_total_g)
                             + extra * inv_total_g)
                    if _op is not None:
                        v_inf[op_rows] += _op["e_rev"] * (
                            op_g * inv_total_g[op_rows])
                    if _hd is not None:
                        v_inf[hd_rows] += hd_vrel * (hd_g * inv_total_g[hd_rows])
                    conductance_decay = np.exp(
                        -dt_over_tau * total_g).astype(np.float32)
                    v = v_inf + (v - v_inf) * conductance_decay
                elif _op is None:
                    if xm is not None:
                        # B2: the masked rows take the analytic update above,
                        # from the potential the step started at
                        _xge, _xgi = ge[xm], gi[xm]
                        _xt = np.float32(1.0) + _xge + _xgi
                        if _hd is not None:
                            _xt[hd_xpos] += hd_g
                        _xi = np.float32(1.0) / _xt
                        _xinf = (xm_vr * _xi + p.e_exc * (_xge * _xi)
                                 + xm_ei * (_xgi * _xi)
                                 + (extra[xm] if np.ndim(extra) else extra) * _xi)
                        if _hd is not None:
                            _xinf[hd_xpos] += hd_vrel * (hd_g * _xi[hd_xpos])
                        _vx = _xinf + (v[xm] - _xinf) * np.exp(
                            -xm_dtt * _xt).astype(np.float32)
                    v = (v_rest + (v - v_rest) * decay_v
                         + (ge * (p.e_exc - v)
                            + gi * (e_inh - v) + extra) * dt_over_tau)
                    if xm is not None:
                        v[xm] = _vx
                else:
                    # the same bracket with the opsin current appended last
                    op_brk = ge * (p.e_exc - v) + gi * (e_inh - v) + extra
                    op_brk[op_rows] += op_i
                    v = v_rest + (v - v_rest) * decay_v + op_brk * dt_over_tau
            else:
                v = v_rest + (v - v_rest) * decay_v + (g + extra) * dt_over_tau

            if use_adapt:
                # Forward Euler; dt/tau_w is ~1e-3 so this is far inside the
                # stable region. w keeps evolving during refractoriness, which
                # is what makes the brake outlast the spike that set it.
                if adapt_a_below is None:
                    w_adapt += dt_over_tauw * (adapt_a * (v - v_rest) - w_adapt)
                else:
                    _dvr = v - v_rest
                    w_adapt += dt_over_tauw * (
                        np.where(_dvr < 0, adapt_a_below, adapt_a) * _dvr - w_adapt)

            if _hd is not None:
                # B3: x before the refractory clamp; form A decays, the pool
                # integrates its load
                hd_x = v[hd_rows] - hd_vr
                _hc = np.maximum(hd_c + _hd["dt"] * (
                    (_hd["k"] * np.maximum(hd_x - _hd["va"], hd_zero)) * hd_L
                    - hd_c / _hd["tau_c"]), hd_zero)
                hd_c = np.where(hd_pool, _hc, hd_c)
                hd_g = np.where(hd_pool, (_hd["gmax"] * hd_c) / (hd_one + hd_c),
                                hd_g * _hd["decay"])

            active = refrac <= 0
            if _hd is not None:
                hd_act = active[hd_rows]
            v = np.where(active, v, v_reset)
            if use_exp:
                if exp_off is None:
                    np.minimum(v, p.v_peak, out=v)
                else:
                    v = np.where(exp_off, v, np.minimum(v, p.v_peak))

            if vtrace is not None:
                vtrace[t] = v[vrec]
            if _op is not None and op_tap is not None:
                op_tap["i_after"][t] = op_g * (_op["e_rev"] - v[op_rows])

            # Graded cells never cross threshold -- they have no threshold.
            _sa = (sw_thr if sw_mask is not None and sw_on <= t < sw_off
                   else spike_at)
            if _op is None or not len(op_thr):
                fired = np.flatnonzero(active & (v >= _sa) & ~graded)
            else:
                # arm E: lit threshold-window rows test against its value
                op_above = v >= _sa
                op_on = op_thr[op_lit[op_thr]]
                op_above[op_rows[op_on]] = v[op_rows[op_on]] >= _op["thr_mv"][op_on]
                fired = np.flatnonzero(active & op_above & ~graded)

            if drive_fn is not None and t % fb_every == 0:
                idx_new, rate_new = drive_fn(t * p.dt, fb_accum)
                drive_idx = np.asarray(idx_new, dtype=np.int64)
                drive_rate = np.asarray(rate_new, dtype=np.float64)
                if drive_idx.shape != drive_rate.shape:
                    raise ValueError(
                        "drive_fn indices and rates must have matching shapes")
                rest_drive_refuse_overlap(rest_plan, drive_idx)
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

            if rest_plan is not None:
                # Design 1e20a949, Build C1: the rest drive's own draws, one
                # per row per step whatever hz is, kicked as membrane_drive
                # kicks are, outside the drive window.
                _rmark = (rest_plan["gen"].random(len(rest_plan["rows"]))
                          < rest_plan["prob"])
                rest_plan["draws"] += len(rest_plan["rows"])
                if _rmark.any():
                    rest_plan["events"] += int(_rmark.sum())
                    v[rest_plan["rows"][_rmark]] += rest_plan["kick"][_rmark]

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
            if _coin2 is not None:
                _coin2["src"][_coin2["last"]] = 0.0
                _coin2["last"] = fired[~graded[fired]]
                _coin2["src"][_coin2["last"]] = 1.0

            if serotonin_configured and len(fired):
                _n_serotonin = int(serotonin_source_mask[fired].sum())
                if _n_serotonin:
                    serotonin_source_spikes += _n_serotonin
                    serotonin_state_hz += (
                        serotonin_increment_hz * _n_serotonin)
                    serotonin_state_max_hz = max(
                        serotonin_state_max_hz, serotonin_state_hz)
            if serotonin_configured and (t + 1) % serotonin_steps_per_bin == 0:
                serotonin_state_bins[t // serotonin_steps_per_bin] = (
                    serotonin_state_hz)
            if _os is not None:   # design 8b6e0441, Build 1
                if len(fired):
                    _n_os = int(os_mask[fired].sum())
                    if _n_os:
                        os_source_spikes += _n_os
                        os_state_hz += _os["inc"] * np.float32(_n_os)
                        os_state_max_hz = max(os_state_max_hz, os_state_hz)
                if (t + 1) % _os["spb"] == 0:
                    os_bins[t // _os["spb"]] = os_state_hz

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
                if _coin2 is not None:
                    # Build 2: a graded source's seam read until the next update
                    _coin2["src"][graded_idx] = (rel.astype(np.float32)
                                                 / np.float32(g_every))
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

            if sw_mask is not None and sw_on <= t < sw_off and len(fired):
                sw_spikes += int(sw_mask[fired].sum())

            if tap_member is not None and len(fired):
                # `fired` is complete and sorted here, so the pairs are
                # sorted by step then row, as the Metal tap appends them
                _tapped = fired[tap_member[fired]]
                if len(_tapped):
                    spike_tap.extend(zip([t] * len(_tapped), _tapped.tolist()))

            refrac -= 1
            if _hd is not None:
                # B3: a spike opens the hold-down and re-arms its latch; an
                # active step without one counts toward the latch
                hd_f = np.isin(hd_rows, fired)
                if hd_f.any():
                    _fa, _fp = hd_f & ~hd_pool, hd_f & hd_pool
                    hd_g[_fa] = hd_g[_fa] + _hd["dg"][_fa]
                    hd_c[_fp] = hd_c[_fp] + _hd["c_spk"][_fp]
                    hd_g[_fp] = (_hd["gmax"][_fp] * hd_c[_fp]) / (hd_one + hd_c[_fp])
                    hd_L[hd_f] = hd_one
                    hd_b[hd_f] = hd_zero
                _lb = hd_act & ~hd_f
                hd_b = np.where(_lb, np.where(hd_x < _hd["latch_mv"], hd_b + hd_one,
                                              hd_zero), hd_b)
                hd_L = np.where(_lb & (hd_b >= _hd["latch_ticks"]), hd_zero, hd_L)
                _pk = hd_g > hd_pk
                hd_pk = np.where(_pk, hd_g, hd_pk)
                hd_pt = np.where(_pk, np.uint32(t), hd_pt).astype(np.uint32)
                if hd_tap is not None:
                    hd_tap[t, :, 1] = hd_c
                    hd_tap[t, :, 2] = hd_g
                    hd_tap[t, :, 3] = hd_L
                    hd_tap[t, :, 4] = hd_b
            if len(fired):
                v[fired] = v_reset[fired]
                refrac[fired] = n_refrac[fired]
                counts[fired] += 1
                total_spikes += len(fired)
                if pp is not None:
                    # pair events from this tick's complete fired set, driven
                    # and forced spikes included, as cns_engine.cu's `fire`
                    _pq = np.flatnonzero(np.isin(pp_src, fired))
                    if len(_pq):
                        pp_ring[(t + pp_delay[_pq]) % pp_nr, _pq] += 1
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
                        _hf = fired[_pm]
                        _G = psi_g9[presyn_pos[_hf]] if psi9 else g_i[_hf]
                        _fac = (np.float32(1.0) / (
                            np.float32(1.0) + _G * presyn_kg))
                        if release_scale is None:
                            release_scale = np.ones(len(fired), dtype=np.float32)
                        release_scale[_pm] *= _fac
                        presyn_log_t.append(np.full(len(_hf), t, dtype=np.int32))
                        presyn_log_r.append(_hf.astype(np.int32))
                        presyn_log_f.append(np.asarray(_fac, dtype=np.float32))
                        presyn_log_G.append(np.asarray(_G, dtype=np.float32))
                        presyn_log_gi.append(g_i[_hf].astype(np.float32))
                        presyn_spikes += int(_pm.sum())
                        presyn_attenuated_spikes += int((_fac < 1.0).sum())
                        presyn_factor_sum += float(_fac.sum())
                        presyn_factor_min = min(
                            presyn_factor_min, float(_fac.min()))
                if use_rr:
                    # snapshot x at spike time, log it, then deplete; the
                    # snapshot rides release_scale through the delay
                    _rp = rr_pos[fired]
                    _rs = _rp >= 0
                    if _rs.any():
                        _ri = _rp[_rs]
                        _snap = rr_x[_ri].copy()
                        if release_scale is None:
                            release_scale = np.ones(len(fired),
                                                    dtype=np.float32)
                        release_scale[_rs] *= _snap
                        rr_log_t.append(np.full(len(_ri), t, dtype=np.int32))
                        rr_log_r.append(fired[_rs].astype(np.int32))
                        rr_log_x.append(_snap)
                        rr_x[_ri] *= rr_k[_ri]
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
        if _oa is not None:
            out["octopamine_offset"] = {"offset_mv": float(_oa[1]),
                                        "target_rows": int(len(_oa[0])),
                                        "route": "cns.py extra"}
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
                "init_hz": float(serotonin_init_hz),
                "bin_ms": float(serotonin_steps_per_bin * p.dt),
                # float32 values as Python floats, so the receipt stays JSON
                "state_bin_end_hz": serotonin_state_bins.tolist(),
            }
        if _os is not None:   # design 8b6e0441, Build 1
            out["octopamine_state"] = octopamine_state_receipt(
                _os, os_state_hz, os_state_max_hz, os_applied_sum_hz,
                os_source_spikes, os_bins, n_steps, p.dt)
            out["octopamine_state"]["route"] = "cns.py extra"
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
            if _presyn_gain is not None or psi9:
                out["presynaptic_inhibition"].update({
                    "gain": None if _presyn_gain is None else float(_presyn_gain),
                    "kg": float(presyn_kg),
                    "state": "source" if psi9 else "g_i"})
            if psi9:
                out["presynaptic_inhibition"].update({
                    "source_rows": psi_n_src, "pairs": psi_pairs,
                    "tau_ms": None if _psi_tau is None else float(_psi_tau),
                    "r9": [float(x) for x in np.unique(psi_r9)],
                    "c9": [float(x) for x in np.unique(psi_c9)]})
            _lcat = (lambda parts, dt_: np.concatenate(parts) if parts
                     else np.empty(0, dtype=dt_))
            # one entry per listed-row spike: step, row, factor, the state it
            # read and the row's g_i at that step
            out["presynaptic_inhibition_log"] = {
                "step": _lcat(presyn_log_t, np.int32),
                "row": _lcat(presyn_log_r, np.int32),
                "f": _lcat(presyn_log_f, np.float32),
                "G": _lcat(presyn_log_G, np.float32),
                "g_i": _lcat(presyn_log_gi, np.float32)}
        if use_rr:
            _cat = (lambda parts, dt_: np.concatenate(parts) if parts
                    else np.empty(0, dtype=dt_))
            out["release_log"], out["release_log_sha256"] = release_log_pack(
                _cat(rr_log_t, np.int32), _cat(rr_log_r, np.int32),
                _cat(rr_log_x, np.float32))
        if membrane_drive_receipt is not None:
            membrane_drive_receipt["drawn_unique_row_tick_events"] = int(
                membrane_drive_events)
            out["membrane_drive"] = membrane_drive_receipt
        if rest_plan is not None:
            out["rest_drive"] = dict(
                rest_plan["receipt"], steps=int(n_steps),
                draws_total=int(rest_plan["draws"]),
                membrane_drive_events=int(rest_plan["events"]), backend="numpy")
        if sw_mask is not None:
            out["silence_window"] = silence_window_receipt(
                sw_mask, sw_on, sw_off, p.dt, sw_spikes)
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
        if _coin2 is not None:
            out["coincidence_triples"] = coincidence_triple_receipt(
                _coin2["trips"], [(q["applied_max"], q["applied_sum"], q["fs_max"],
                                   q["slowdel_max"]) for q in _coin2["trips"]])
        if _op is not None:
            out["opsin"] = opsin_out(_op, op_tap)
        if _hd is not None:
            _hst = {"c": hd_c, "g_h": hd_g, "L": hd_L, "b": hd_b, "peak": hd_pk,
                    "peak_tick": hd_pt}
            self._holddown_numpy = {"state": _hst, "slots": _hd["slots"]}
            if record_holddown:
                out["holddown"] = holddown_out(hd_rows, _hst, hd_tap)
        if pp is not None:
            # the pair channel's state after the run, in the fields the native
            # engines read back (pair_state, pair_recovery_readback)
            self._pair_numpy = {
                "state": pp_x.copy(), "ring": pp_ring.copy(),
                "recovery": None if pp_rec is None else {
                    **pp_rec,
                    "delivered": [int(x) for x in pp_cnt[:, 0]],
                    "skipped": [int(x) for x in pp_cnt[:, 1]],
                    "scaled": [int(x) for x in pp_cnt[:, 2]],
                    "multi_event_ticks": [int(x) for x in pp_cnt[:, 3]],
                    "area_fraction": [float(x) for x in pp_area],
                    "clock": [int(x) for x in pp_clk],
                    "resource": [float(x) for x in pp_res]}}
        if _sl is not None:
            # the queue delivers at arrival, so the accumulators hold the
            # spikes that arrived, and nothing is pending in the slow ring
            out["slow_inh"] = slow_inh_out(
                _sl, sl_acc_fast, sl_acc_slow, sl_int, sl_g, sl_ring.sum(axis=0,
                dtype=np.float64), sl_rec_acc, sl_arrived[0], "arrived",
                int(counts[_sl["receipt"]["rows"]].sum()), gscale)
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
