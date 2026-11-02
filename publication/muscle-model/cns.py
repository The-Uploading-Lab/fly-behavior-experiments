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
    tau_syn: float = 5.0         # ms
    # Scalar or (N,) array. Per-neuron refractoriness matters because this
    # one number sets every neuron's maximum firing rate identically --
    # 1000/2.2 = 455 Hz for all 188,508 of them -- and real neurons differ.
    # It was scalar-only until 2026-08-09 for no reason except that the code
    # computed n_refrac as an int.
    t_refrac: float = 2.2        # ms, scalar or (N,) array
    delay: float = 1.8           # ms, uniform axonal delay
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
    # bit-identical.
    gap_pairs: tuple = ()
    g_gap: float = 0.0           # dimensionless coupling per pair, subthreshold

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
    graded_gain: float = 0.05    # release units per tick, vs 1.0 for a spike
    graded_v50: float = -48.0    # mV at half-maximal release
    graded_slope: float = 2.5    # mV; smaller = more switch-like

    # ⚠️ "CONTINUOUS" TAKEN LITERALLY IS 884x MORE WORK THAN SPIKING. Measured:
    # 32,558 graded cells own 1,636,056 outgoing edges (12% of the graph), and
    # updating them every 0.1 ms tick is 16.4 BILLION propagation events per
    # simulated second, against 18.5 million for the spiking path -- a 4.3x
    # whole-model slowdown for +11% more spikes. But tau_mem is 20 ms, so a
    # graded cell's voltage physically CANNOT change fast: over 0.9 ms it moves
    # only 4.4% of the way to its target. Recomputing release every tick is
    # therefore mostly redundant arithmetic, not fidelity.
    graded_every: int = 1        # ticks between graded release updates

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
    e_exc: float = 0.0           # mV, cation reversal (ACh receptors)
    e_inh: float = -70.0         # mV, chloride reversal (GABA-A, GluCl)

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
    adapt_a: float = 0.0         # dimensionless, subthreshold coupling
    adapt_b: float = 0.0         # mV, spike-triggered increment
    tau_w: float = 100.0         # ms, adaptation decay
    # Exponential spike initiation (the "Ex" in AdEx). 0 keeps the hard
    # threshold, which is much cheaper -- an exp() over 188,508 neurons every
    # step is the dominant cost when it is on. It sharpens spike onset; it is
    # not the part that generates rhythm, so it defaults off.
    delta_T: float = 0.0         # mV, slope factor
    v_peak: float = -30.0        # mV, cutoff when delta_T > 0

    dt: float = 0.1              # ms

    # --- sign assignment ---
    nt_sign: dict = field(default_factory=_default_nt_sign)
    # HYPOTHESIS, not a measurement: what to do with the 34,455 neurons that
    # have neither a verified nor a predicted transmitter. Acetylcholine is
    # ~55% of the annotated population so +1 is the maximum-likelihood guess,
    # and w_unknown_scale lets a search discount them without removing them.
    sign_unknown: float = +1.0
    w_unknown_scale: float = 1.0

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


# --------------------------------------------------------------------------
# building the network
# --------------------------------------------------------------------------

def _resolve_nt(meta, p):
    """Per-neuron transmitter string, verified preferred over predicted."""
    ver = meta["neurotransmitter_verified"]
    pred = meta["neurotransmitter_predicted"]
    nt = ver.where(ver.notna(), pred) if p.prefer_verified else pred
    return nt


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
            signs[i] = 0.0     # known transmitter, no fast sign (monoamine)
    return signs


class CNS:
    """Sparse LIF over the whole BANC CNS."""

    def __init__(self, meta, edges, params=None):
        self.p = params or SimParams()
        p = self.p

        self.meta = meta.reset_index(drop=True)
        self.ids = self.meta["banc_888_id"].to_numpy()
        self.N = len(self.ids)
        idx = pd.Series(np.arange(self.N), index=self.ids)

        # Edges whose endpoints are both in the meta table. BANC's edge list
        # references a few segments the meta table does not describe; those
        # cannot be simulated because they have no transmitter, so they are
        # reported rather than silently dropped.
        pre = idx.reindex(edges["pre"].to_numpy()).to_numpy()
        post = idx.reindex(edges["post"].to_numpy()).to_numpy()
        keep = ~(np.isnan(pre) | np.isnan(post))
        self.n_edges_dropped = int((~keep).sum())
        pre, post = pre[keep].astype(np.int32), post[keep].astype(np.int32)
        cnt = edges["count"].to_numpy()[keep].astype(np.float32)

        nt = _resolve_nt(self.meta, p)
        self.nt = nt
        sign = _sign_of(nt, p)
        self.n_unknown_nt = int(np.isnan(sign).sum())

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
        kernel_gain = (p.syn_ref_tau / p.tau_syn) if p.syn_normalised else 1.0
        self.kernel_gain = kernel_gain
        w = cnt * p.w_syn * kernel_gain * sign[pre] * scale[pre] * gain[pre]

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
            with np.errstate(divide="ignore", invalid="ignore"):
                s = E_in / (I_in * p.balance_target)
            s[~np.isfinite(s)] = 1.0          # no inhibition to scale
            s[I_in <= 0] = 1.0
            self.n_balance_capped = int((s > p.balance_max_scale).sum())
            s = np.clip(s, 0.0, p.balance_max_scale)
            self.balance_scale = s
            neg = w < 0
            w = w.copy()
            w[neg] *= s[post[neg]]

        # Per-neuron input resistance, folded into the INCOMING weights at
        # build time so it costs nothing per tick.
        self.r_scale = None
        if p.r_in_alpha:
            n_syn_in = np.bincount(post, weights=cnt, minlength=self.N)
            med = np.median(n_syn_in[n_syn_in > 0])
            with np.errstate(divide="ignore", invalid="ignore"):
                rs = (med / n_syn_in) ** p.r_in_alpha
            rs[~np.isfinite(rs)] = 1.0
            rs = np.clip(rs, 1.0 / p.r_in_max, p.r_in_max)
            self.r_scale = rs.astype(np.float32)
            w = w * self.r_scale[post]

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
        if (p.g_gap or p.gap_spikelet) and len(p.gap_pairs):
            gp = np.asarray(p.gap_pairs, dtype=np.int64).reshape(-1, 2)
            i = np.concatenate([gp[:, 0], gp[:, 1]])
            j = np.concatenate([gp[:, 1], gp[:, 0]])
            val = np.full(len(i), 1.0, dtype=np.float32)
            self.G = sparse.csr_matrix((val, (i, j)), shape=(self.N, self.N),
                                       dtype=np.float32)
            self.gap_deg = np.asarray(self.G.sum(axis=1)).ravel().astype(
                np.float32)
            self.n_gap_pairs = len(gp)

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
    def run(self, duration_ms, drive=None, seed=0, record=None,
            record_trace_every=None, record_voltage=None, drive_window=None,
            drive_fn=None, feedback_rows=None, drive_every_ms=1.0,
            force_spikes=None, first_spike=False):
        """Simulate.

        drive : dict {neuron_index_array : rate_hz} or (indices, rate)
            Poisson spike injection. These neurons fire independently of their
            input, which is how a sensory periphery the volume does not contain
            gets represented.
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

        Returns dict with 'counts' (spikes per recorded neuron), 'rates_hz',
        'n_spikes' total, and optionally 'trace'.
        """
        p = self.p
        rng = np.random.default_rng(seed)
        n_steps = int(round(duration_ms / p.dt))
        n_delay = max(1, int(round(p.delay / p.dt)))
        n_refrac = np.broadcast_to(
            np.rint(np.asarray(p.t_refrac, dtype=np.float64) / p.dt
                    ).astype(np.int32), (self.N,)).copy()

        # tau_mem and v_rest may be scalars or per-neuron arrays.
        tau_mem = np.asarray(p.tau_mem, dtype=np.float32)
        v_rest = np.asarray(p.v_rest, dtype=np.float32)
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
        decay_g = np.float32(np.exp(-p.dt / p.tau_syn))
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
        use_exp = p.delta_T > 0
        G = self.G
        gap_deg = self.gap_deg
        g_gap = np.float32(p.g_gap)
        gap_spike = np.float32(p.gap_spikelet)
        graded = self.graded
        Wg = self.Wg
        graded_idx = getattr(self, "graded_idx", np.empty(0, np.int64))
        ggain = np.float32(p.graded_gain)
        gv50 = np.float32(p.graded_v50)
        gslope = np.float32(p.graded_slope)
        g_every = max(1, int(p.graded_every))
        inv_dT = np.float32(1.0 / p.delta_T) if use_exp else np.float32(0.0)
        spike_at = p.v_peak if use_exp else p.v_threshold

        # ring buffer of delayed synaptic increments
        ring = np.zeros((n_delay, self.N), dtype=np.float32)

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
        drive_prob = np.empty(0, dtype=np.float64)
        if drive:
            di, dp = [], []
            for k, rate in drive.items():
                k = np.asarray(k, dtype=np.int64)
                di.append(k)
                dp.append(np.full(len(k), rate * p.dt / 1000.0))
            drive_idx = np.concatenate(di)
            drive_prob = np.concatenate(dp)
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
        fb_accum = (None if fb_rows is None
                    else np.zeros(len(fb_rows), dtype=np.int64))

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
        total_spikes = 0

        for t in range(n_steps):
            slot = t % n_delay
            g *= decay_g
            if use_std:
                # exact relaxation of the vesicle pool toward 1
                x_res *= std_dec
                x_res += np.float32(1.0) - std_dec
            g += ring[slot]
            ring[slot] = 0.0

            # Everything that enters the membrane equation as a current, in mV.
            # The adaptation variable is subtracted here so it composes with
            # both the current-based and the conductance-based synapse model
            # rather than being a third mutually exclusive branch.
            extra = np.float32(0.0)
            if G is not None and g_gap:
                # sum_j v_j - deg_i * v_i  -- the Laplacian, in mV. No delay
                # and no spike needed: this is what makes it electrical. It
                # carries SUBTHRESHOLD coupling only; the spike itself arrives
                # as a spikelet below, because a LIF spike has no waveform.
                extra = extra + g_gap * (G @ v - gap_deg * v)
            if use_adapt:
                extra = extra - w_adapt
            if use_exp:
                extra = extra + p.delta_T * np.exp(
                    np.minimum((v - p.v_threshold) * inv_dT, 20.0))

            if p.conductance_based:
                # Split the accumulated input by sign and apply each with its
                # own driving force, so inhibition saturates at e_inh instead
                # of running away.
                ge = np.maximum(g, 0.0) * gscale
                gi = np.maximum(-g, 0.0) * gscale
                v = (v_rest + (v - v_rest) * decay_v
                     + (ge * (p.e_exc - v) + gi * (p.e_inh - v) + extra)
                     * dt_over_tau)
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
                drive_prob = (np.asarray(rate_new, dtype=np.float64)
                              * p.dt / 1000.0)
                if fb_accum is not None:
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

            if len(drive_idx) and step_on <= t < step_off:
                hit = drive_idx[rng.random(len(drive_idx)) < drive_prob]
                if len(hit):
                    fired = np.union1d(fired, hit[refrac[hit] <= 0])

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
                # scaled by g_every so the DELIVERED CHARGE is unchanged when
                # the update rate changes -- otherwise decimating would quietly
                # weaken every graded synapse and look like a modelling result.
                ring[(t + n_delay) % n_delay] += (
                    ggain * g_every
                    * (Wg @ rel.astype(np.float32))).astype(np.float32)

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
                if G is not None and gap_spike:
                    hot = np.zeros(self.N, dtype=np.float32)
                    hot[fired] = 1.0
                    v = v + gap_spike * (G @ hot)

                # ragged gather of the fired neurons' outgoing columns
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
                    if use_std:
                        # deliver at the CURRENT pool level, then deplete
                        vals = vals * np.repeat(x_res[fired], nper)
                        x_res[fired] *= std_keep
                    tgt = (t + n_delay) % n_delay
                    ring[tgt] += np.bincount(
                        rows, weights=vals, minlength=self.N).astype(np.float32)

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
