/* cns_engine.cu — CUDA implementation of cns_engine.h, ported from the tuned
 * Metal engine cns_engine.m (2026-09-08). The kernel bodies are the Metal
 * ones line for line; what changed is everything around them.
 *
 * Four structural differences from the Metal file, each forced by CUDA:
 *
 * 1. No indirect dispatch. Metal sizes k_scatter and k_fold from a triple the
 *    previous kernel wrote, so the grid matches the tick's spike count with no
 *    host round trip. CUDA has no equivalent for an ordinary launch, so both
 *    kernels run a fixed grid and stride over the work list, reading the count
 *    from the same args slot Metal's triple lived in. The mapping from work
 *    item to thread changes; the arithmetic each work item performs does not.
 * 2. No function constants. Metal specialises k_step on nine booleans at
 *    pipeline build. Here the same booleans ride in Params and the branches
 *    are uniform across the grid, so they cost a predicted branch instead of
 *    a removed load. Same arithmetic either way.
 * 3. No unified memory. Metal's shared buffers are read by the host with no
 *    copy. Here every buffer is device memory and the host sees it through
 *    explicit copies, of which the spike list is the one that would have hurt:
 *    the raw per-tick list is window_ticks x fired_cap (67 MB at the default
 *    256-tick window), so k_pack compacts it on the device and only the
 *    spikes that exist cross PCIe.
 * 4. Contraction is spelled out. Metal took `#pragma clang fp contract(off)`;
 *    nvcc's default is -fmad=true, which would fuse `a*b+c` and break bit
 *    parity with the numpy engine. Every arithmetic float operation on the
 *    parity path is written as an explicit round-to-nearest intrinsic
 *    (FA/FS/FM/FD below), so the build flag cannot silently change the
 *    answer. -fmad=false is still passed, as belt and braces.
 *
 * Everything the Metal file learned the hard way is preserved: no float
 * atomic anywhere (integer-atomic append per target bin, then a per-target
 * fold in sorted order), BIN_CAP with the exact transposed-graph path when a
 * bin overflows, k_coin summing in stored CSR order because that is what
 * numpy's csr @ vector does, and K consecutive ticks fused into one k_step
 * only when no mechanism reads mid-group state.
 */
#include "cns_engine.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef SCATTER_TG
#define SCATTER_TG 256
#endif
#ifndef SCATTER_BLOCKS
#define SCATTER_BLOCKS 256   /* grid-stride over the tick's spikes */
#endif
/* k_scatter's delivery-recorder reduction halves the block each step */
static_assert((SCATTER_TG & (SCATTER_TG - 1)) == 0, "SCATTER_TG must be a power of two");
#ifndef BIN_CAP
#define BIN_CAP 64   /* keep in step with the device copy below */
#endif
#ifndef FOLD_TG
#define FOLD_TG 256
#endif
#ifndef FOLD_BLOCKS
#define FOLD_BLOCKS 256   /* grid-stride over the tick's touched targets */
#endif
#ifndef STEP_TG
#define STEP_TG 256
#endif
#define SMALL_TG 256

/* Round-to-nearest primitives. These are what keeps the port bit-identical
 * to the numpy engine regardless of -fmad: the compiler may not fuse them. */
__device__ __forceinline__ float FA(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float FS(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float FM(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float FD(float a, float b) { return __fdiv_rn(a, b); }

struct Params {
    uint32_t n, n_ring, words, fired_cap;
    int32_t cbased, split, analytic, use_adapt;
    float e_exc, e_inh, gscale, ggain;
    uint32_t g_every, n_graded, n_vrec, n_reltap;
    /* scalar copies used when the matching uni_bits flag is set */
    float u_vrest, u_vreset, u_spike, u_decv;
    float u_dtt, u_decg, u_aa, u_ab;
    float u_dtw; int32_t u_rt; uint32_t tcnt_slot, uni_bits;
    uint32_t n_coin, coin_lag; float coin_gain, coin_decay;
    uint32_t kgroup, fbits_slot, n_gt, ftrip_slot;
    uint32_t key_shift, has_tonic, sparse_vrest, adapt_split;   /* adapt_split: design 6d0042d5 */
    uint32_t n_gap; float gap_scale;   /* gpu-7, note 10906d21: the plain electrical coupling */
};

/* Disabled by a zero-initialized descriptor. Snapshots are indexed by tick
 * within the fused group and resource index, never by the later arrival tick. */
struct Resource {
    uint32_t n, cap;
    int32_t *pos;
    float *d, *c, *keep, *x, *snapshot;
    uint32_t *count;
    int32_t *tick, *row;
    float *value;
    /* Design 02e57559: one byte per target row. NULL applies the emitted
     * snapshot to every output, the design 0e0587be behaviour; otherwise an
     * output carries it only onto a marked target and keeps its built
     * weight elsewhere. The source's own recurrence is unchanged. */
    uint8_t *tmask;
};

/* Design 02e57559, CUDA only. Delivery recorder for marked sources: every
 * output onto a marked direct target is logged with its emission tick, CSC
 * edge and the weight exactly as binned; the other outputs of the same event
 * are summed by sign in float64 in a fixed per-block order. n_src zero is
 * off and leaves k_scatter on its pre-existing path. */
struct Deliv {
    uint32_t n_src, cap_direct, cap_event;
    uint8_t *src, *tgt;
    uint32_t *count;              /* [0] direct entries, [1] events */
    int32_t *dir_tick; int64_t *dir_edge; float *dir_w;
    int32_t *ev_tick, *ev_src;
    double *ev_pos, *ev_neg;
    uint32_t *ev_nother, *ev_ndir;
};

/* Design 02e57559, CUDA only. Dense state of at most four rows: a shadow
 * conductance fed only by positive arrivals from marked direct sources, with
 * the row's own synaptic decay, and sixteen native-tick channels. It writes
 * no model state. n zero is off. */
#define PTAP_ROWS 4
#define PTAP_CH 16
struct PTap {
    uint32_t n;
    int32_t row[PTAP_ROWS];
    uint8_t *dsrc;
    float *dring;                 /* n_ring x PTAP_ROWS */
    float *gdir;                  /* PTAP_ROWS */
    float *tap;                   /* window x PTAP_ROWS x PTAP_CH */
};

__device__ __forceinline__ int ptap_index(const PTap &pt, unsigned int i) {
    int j = -1;
    for (uint32_t q = 0; q < pt.n; q++) if (pt.row[q] == (int32_t)i) j = (int)q;
    return j;
}

/* Optional external chloride conductance and raw operands. Zero pointers
 * leave the pre-existing arithmetic path unchanged. With lpos NULL the level
 * follows the scalar on/off/ramp schedule; with lpos set (design e2f5f784
 * P4-3, B2 of team-henri's 97d7a92a5), a lit row reads its level from
 * level[lpos[i]], which the host renews between windows and which is held
 * between renewals. */
struct Lamina {
    float *e_inh;
    unsigned char *mask;
    float gain;
    uint32_t on, off, ramp;
    float *state;
    int32_t *lpos;
    float *level;
};

/* Design 7b0ad6b2 (H01): separately delayed exponential conductance states
 * for named source-target pairs, and a per-row state tap. A pair q carries
 * events from src[q] to tgt[q] after delay[q] ticks and keeps its own state
 * x = r*x + inc*n, added to the target's normalized excitatory conductance
 * after tonic and before either integrator. The pair's ordinary weight must
 * be zero in the graph (checked at configuration), so nothing is delivered
 * twice. n == 0 and n_aux == 0 leave every pre-existing path unchanged. */
#define AUXW 16
struct Pairs {
    uint32_t n, nr, n_aux, _pad;
    int32_t *tgt_of;      /* per neuron: the pair whose target it is, or -1 */
    int32_t *src_first;   /* per neuron: first pair it sources, or -1 */
    int32_t *src_next;    /* per pair: next pair with the same source, or -1 */
    int32_t *src, *tgt;   /* per pair, for readback */
    uint32_t *delay;      /* per pair: transport ticks, kgroup <= delay <= nr - kgroup */
    float *r, *inc;       /* per pair: decay per tick, increment per event */
    float *x;             /* per pair: conductance state */
    uint32_t *ring;       /* nr x n event counts */
    int32_t *apos;        /* per neuron: state-tap row, or -1 */
    float *aux;           /* window x n_aux x AUXW */
    /* Design e05c0eef B1: the pair's recovery. rform 0 is the pre-B1 path.
     * 1, dead time: clk counts down by one per tick; an arrival while clk is
     * 0 delivers inc*n as before and sets clk to rticks, one while clk > 0
     * delivers nothing and leaves clk. 2, depletion: res recovers by
     * res = 1 - (1 - res)*rdecay per tick, then each arriving event delivers
     * inc*res and keeps res*rkeep. cnt per pair: events delivered, skipped,
     * delivered with res < 1, ticks with more than one arrival; area per
     * pair the delivered fraction summed in tick order. */
    uint32_t rform, rticks;
    float rdecay, rkeep;
    uint32_t *clk;
    float *res;
    uint32_t *cnt;
    float *area;
};

/* Design e05c0eef B1: one tick of a recovering pair, on its target's thread.
 * Returns the delivered multiplier, n itself where the pre-B1 path would
 * multiply inc by n, so form 1 with rticks 0 and form 2 with rkeep 1 are
 * that path's arithmetic. */
__device__ __forceinline__ float pair_recover(const Pairs &pp, uint32_t na, uint32_t &clk,
                                              float &res, uint32_t *c, float &area) {
    if (na > 1u) c[3] += 1u;
    if (pp.rform == 1u) {
        if (clk > 0u) clk -= 1u;
        if (!na) return 0.0f;
        if (clk) { c[1] += na; return 0.0f; }
        clk = pp.rticks; c[0] += na; area = FA(area, (float)na);
        return (float)na;
    }
    res = FA(1.0f, -FM(FA(1.0f, -res), pp.rdecay));
    float s = 0.0f;
    for (uint32_t k = 0; k < na; k++) {
        if (res < 1.0f) c[2] += 1u;
        s = FA(s, res);
        res = FM(res, pp.rkeep);
    }
    if (na) { c[0] += na; area = FA(area, s); }
    return s;
}

/* Design 81427d47 (H03), CUDA only. One shared state per target row for named
 * source-to-target connections whose ordinary weight is an explicit zero in
 * the graph: mode 1 (T) an open fraction o, mode 2 (L) a linear conductance h.
 * Per native tick, on the target's thread and in this order: the state decays
 * by dec; the tick's arrivals on the target's connections are summed exactly,
 * as integer counts times integer quanta, A = sum * scale with scale a power
 * of two, so no ordering of sources can change A; then T sets
 * o = o + (1 - o) * -expm1(-alpha * A) and L sets h = h + beta * A, each
 * operation IEEE double with round-to-nearest. The conductance
 * float32(gmax * o), or float32(h), is added to the normalized excitatory
 * conductance after tonic, the phase at which the ordinary arrival it replaces
 * is first used. A source event adds one count to each of its connections'
 * ring slots at tick + delay, the delay being the source's own delivery slot,
 * which a later launch reads. interrupt suppresses the counts, so the state
 * receives nothing. n_edge == 0 leaves every pre-existing path unchanged. */
#define OCC_TGT 4
#define OCC_CH 20
struct Occ {
    uint32_t n_edge, n_tgt, nr, mode;
    uint32_t interrupt, log_on, tap_on, _pad;
    int32_t *tgt_of;       /* per neuron: target index, or -1 */
    int32_t *src_first;    /* per neuron: first connection it sources, or -1 */
    int32_t *src_next;     /* per connection: next connection of the same source, or -1 */
    int32_t *src, *tgt;    /* per connection: rows, for readback */
    uint32_t *delay;       /* per connection: ticks, kgroup <= delay <= nr - kgroup */
    uint64_t *quanta;      /* per connection: a / scale[target], an exact integer */
    uint32_t *t_begin;     /* n_tgt + 1: target j's connections are [t_begin[j], t_begin[j+1]) */
    double *alpha, *gmax, *beta, *dec, *scale;   /* per target */
    double *state;         /* per target: o (T) or h (L) */
    uint32_t *ring;        /* nr x n_edge arrival counts */
    /* recorder: each target owns one region of each log, written by its own thread */
    uint32_t *a_off, *s_off;          /* n_tgt + 1 region offsets */
    uint32_t *a_count, *s_count;      /* per target: entries appended, may exceed the region */
    int32_t *a_tick; uint32_t *a_edge, *a_n;
    int32_t *s_tick; uint32_t *s_n; double *s_val;   /* s_val: A, before, decayed, after */
    float *s_g;
    double *tap;           /* window x n_tgt x OCC_CH */
};

/* Design 87a3ad35 (B2), CUDA only, no header entry: cns.py's opsin_arrays,
 * one slot per row. Per tick, before the membrane, a slot's gate s takes the
 * exact one-step update, lit s = 1 + (s - 1) a_on (1 when force is set), dark
 * s = s a_off; g = g_max s joins the conductance-based membrane last in
 * either integrator, with reversal e_rev, and a kind-1 slot (arm E's
 * threshold window, g_max 0) tests its spike against thr on lit ticks. A
 * tick is lit when win[2w] <= tick < win[2w + 1] for one of the slot's n_win
 * windows. The tap holds, per window tick and slot, s, g, the current
 * g (e_rev - v) after the refractory clamp and at the tick's starting
 * membrane: cns.OPSIN_TAP_FIELDS. n == 0 leaves every path unchanged. */
#define OPSIN_WIN 4
#define OPSIN_CH 4
struct Opsin {
    uint32_t n, tap_on, _pad[2];
    int32_t *pos;          /* per neuron: slot, or -1 */
    float *a_on, *a_off, *g_max, *e_rev, *thr;   /* per slot */
    uint8_t *force, *kind; /* per slot */
    uint32_t *win;         /* per slot: OPSIN_WIN x (on, off) ticks */
    uint32_t *n_win;       /* per slot */
    float *s;              /* per slot: the gate */
    float *tap;            /* window x n x OPSIN_CH, or NULL */
    uint32_t *train;       /* design 2163ba7d: per slot (start, stop, on, period)
                            * ticks, period 0 for none; NULL when no slot has one */
};

/* Design 2d12a3b3 (B2), CUDA only, no header entry: a held current on named
 * rows, in mV, the unit of k_step's `extra`. On slot q's row at network step
 * k, I = i0 + i1 when on <= k < off and I = i0 otherwise. k_step sets
 * extra = I before it subtracts the adaptation, so extra = I - w in every
 * membrane branch (analytic, Euler, current-based), and I is not multiplied
 * by the row's resistance factor. n == 0 leaves every path unchanged.
 * Design 6d0042d5, Build 3: a row may hold several slots whose windows do not
 * overlap and whose i0 are equal. pos names the row's earliest window and
 * next links the rest in order of on, so a row with one slot reads exactly
 * the expression above. */
struct RowCur {
    uint32_t n, _pad[3];
    int32_t *pos;          /* per neuron: its earliest slot, or -1 */
    float *i0, *i1;        /* per slot, mV */
    uint32_t *on, *off;    /* per slot, network steps */
    int32_t *next;         /* per slot: the row's next slot by on, or -1 */
};

/* Design d27f82df (study eb9ff316), CUDA only, no header entry.
 * B2: amask[i] != 0 puts row i on the exact update whatever the integrator,
 * NULL leaves every row where p.analytic puts it.
 * B3: a hold-down conductance g_h, in leak units, on n named rows (each on the
 * exact update), reversing at the row's rest + e_rel. The update uses the
 * previous tick's g_h. After the adaptation, with x = v - rest taken before
 * the refractory clamp, form 0 ("A") decays g_h by `decay` and form 1
 * ("pool") loads c <- max(c + dt (k max(x - va, 0) L - c / tau_c), 0) and
 * sets g_h = gmax c / (1 + c). A spike adds dg to g_h (A) or c_spk to c
 * (pool), and opens the latch: L = 1, b = 0. On a tick that was active and
 * did not fire, b counts the ticks with x < latch_mv and resets otherwise;
 * L = 0 once b reaches latch_ticks. c, g_h, L and b are zeroed at
 * configuration, and an engine is built per run. peak and peak_tick hold the
 * run's largest end-of-tick g_h and its tick; tap, when set, holds per tick
 * and slot i_row (the held current before the adaptation), c, g_h, L and b.
 * n == 0 and amask NULL leave every path unchanged. */
#define HD_CH 5
struct HoldDown {
    uint32_t n, _pad[3];
    uint8_t *amask;        /* per neuron (B2), or NULL */
    int32_t *pos;          /* per neuron: its slot, or -1 */
    uint32_t *form;        /* per slot: 0 A, 1 pool */
    float *e_rel, *decay, *dg;                      /* per slot */
    float *gmax, *c_spk, *k, *va, *tau_c;           /* per slot, the pool */
    float *latch_mv; uint32_t *latch_ticks;         /* per slot */
    float *c, *g_h, *L, *b, *peak;                  /* per slot, state */
    uint32_t *peak_tick;
    float *tap;            /* (window ticks, n, HD_CH), or NULL */
    float dt, _fpad[3];
};

/* Design 2163ba7d (Build 2), CUDA only, no header entry: cns.py's
 * slow_inh_arrays, one slot per target of a listed row. For an edge from a
 * listed source with w < 0, k_fold sends fl(-w keep) to the fast inhibitory
 * ring, where an unlisted edge sends -w, and fl(-w sfac) to the target's slow
 * ring; each is summed compensated per arrival slot and rounded once, as the
 * fast pool is. k_step decays the slot's conductance by `decay`, adds its
 * arrivals, and adds fl(g gscale) to the inhibitory conductance before the
 * lamina term. acc_fast and acc_slow hold, per slot in double, the rounded
 * per-slot sums of the listed rows' fast and slow increments; intg the
 * per-tick fl(g gscale); rec_acc, per bin of steps_per_bin ticks and record
 * column, the same sum. n == 0 leaves every path unchanged. */
struct Slow {
    uint32_t n, n_rec, steps_per_bin, n_bins;
    float keep, sfac, decay, _pad;
    uint8_t *src;          /* per neuron: 1 for a listed source */
    int32_t *pos;          /* per neuron: target slot, or -1 */
    int32_t *rec;          /* per target slot: record column, or -1 */
    float *ring;           /* n_ring x n */
    float *g;              /* per target slot */
    double *acc_fast, *acc_slow, *intg;   /* per target slot */
    double *rec_acc;       /* n_bins x n_rec */
};

/* Design 302c1bfb, Build 8, on CUDA (design 9501604f, Build C1), no header
 * entry: cns_engine.m's release factor. n listed rows (distinct, ungraded)
 * release each spike's synapses at f = 1/(1 + G kg). With has9 each listed
 * row carries a second state g9: every tick it decays by r9 and takes c9
 * times the tick's inhibitory arrivals from the src9 rows, which k_fold sums
 * into ring9 with the inhibitory ring's own compensated fold and overflow
 * path; G is g9, or without has9 the row's g_i after k_step. fac is 1.0 on
 * every row but a listed one, whose entry k_psi sets on each tick the row
 * fires. n == 0 leaves every path unchanged. */
struct Psi {
    uint32_t n, has9;
    float kg, _pad;
    int32_t *prow;         /* per listed row: its neuron */
    int32_t *ppos;         /* per neuron: listed position, or -1 (has9) */
    uint8_t *src9;         /* per neuron: 1 on a source row (has9) */
    float *g9, *r9, *c9;   /* per listed row (has9) */
    float *ring9;          /* n_ring x n (has9) */
    float *fac;            /* per neuron */
    float *plog;           /* window x n x (f, G, g_i, fired flag) */
};

/* Design b1746728 (study fd7a2867), Build 1, CUDA only, no header entry:
 * cns.py's receptor-scoped serotonin state, one population value s in Hz.
 * Each tick k_step gives every target row extra = fl(gain fl(s decay)),
 * where cns.py's serotonin_extra enters (before the held current and the
 * adaptation), and counts the tick's spikes on source rows; k_ser then
 * forms fl(s decay), adds it to the double applied sum, adds fl(inc n) when
 * n > 0 (and raises the max), and stores s at the end of each bin of
 * steps_per_bin ticks. k_step reads s as the last tick's k_ser left it, so
 * the state needs kgroup 1. tgt is NULL at gain 0, where cns.py leaves the
 * membrane alone. n_src == 0 leaves every path unchanged. */
struct Ser {
    uint32_t n_src, n_tgt, steps_per_bin, n_bins;
    float gain, decay, inc, _pad;
    uint8_t *src;          /* per neuron: 1 on a source row */
    uint8_t *tgt;          /* per neuron: 1 on a target row, or NULL at gain 0 */
    float *s;              /* the state, then its running maximum */
    uint32_t *n_tick;      /* this tick's source spikes */
    double *applied;       /* the sum over ticks of fl(s decay) */
    unsigned long long *spikes;   /* source spikes over the run */
    float *bins;           /* n_bins: s at the end of each bin */
};

/* Design 8b6e0441 (study a3f66049), Build 2, CUDA only, no header entry:
 * cns.py's octopamine offset, one float32 in mV that k_step adds to extra on
 * every target row each tick, after the serotonin term and the held current
 * and before the adaptation (cns.py adds it after the injected current, which
 * this library does not carry). tgt is NULL when off, and then no arithmetic
 * changes. */
struct OaOff {
    uint8_t *tgt;          /* per neuron: 1 on a target row, or NULL when off */
    float off;             /* mV */
    uint32_t n_tgt;
};

/* Design 8b6e0441 (study a3f66049), Build 1, CUDA only, no header entry:
 * cns.py's octopamine state, a second Ser beside the serotonin state with the
 * same arithmetic and the same k_ser. Its target term is ADDED to extra after
 * the serotonin term (cns.py sums the two slots on rows both reach), before
 * the held current (refused with it) and the octopamine offset. src lists the
 * rows whose spikes are counted, which may be none (every source severed by a
 * seam); inc carries the given set's size, from the host. Configured when s
 * is non-NULL. */

/* Metal's function constants, now runtime flags. Bit k of uni_bits:
 * 0 v_rest 1 v_reset 2 spike_at 3 decay_v 4 dt_over_tau 5 decay_g
 * 6 refrac 7 adaptation. sparse_vrest is its own field. */
#define UNI(p, k) ((((p).uni_bits >> (k)) & 1u) != 0u)

/* ---------------------------------------------------------------- kernels */

/* Per tick, in cns.py's order: conductance decay, ring consume, membrane
 * integrate (current-based, or conductance-based with optional true shunting
 * and the analytic one-step integrator), adaptation, refractory clamp,
 * threshold, drive candidates and forced spikes, refractory countdown, reset. */
struct Compass {
    uint32_t n;
    int32_t *pos;
    uint8_t *group;
    int32_t cut_tick;
    float *ring;             /* compact targets x 14: 12 classes, noEE, EE count */
    float *g;                /* compact targets x 12, observation only */
    float *tap;              /* window x compact targets x 29 */
    uint32_t *first_tick;
    uint8_t *snapshot;
    uint64_t snapshot_bytes;
};

__global__ void k_compass_trigger(const Compass c, uint32_t tick, uint32_t slots) {
    const uint32_t i = threadIdx.x;
    if (tick < 17000 || *c.first_tick != UINT32_MAX) return;
    for (uint32_t j = i; j < c.n; j += blockDim.x)
        if (c.ring[((size_t)(tick % slots) * c.n + j) * 14] > 0.0f)
            atomicMin(c.first_tick, tick);
}

__global__ void k_compass_snapshot(const Compass c, uint32_t tick, uint32_t n,
        uint32_t slots, const float *v, const float *g, const float *gi,
        const float *w, const short *refrac, const uint32_t *counts,
        const float *ring, const float *ring_i, bool adapt) {
    if (*c.first_tick != tick) return;
    const uint8_t *sources[] = {(const uint8_t *)v, (const uint8_t *)g,
        (const uint8_t *)gi, (const uint8_t *)w, (const uint8_t *)refrac,
        (const uint8_t *)counts, (const uint8_t *)ring, (const uint8_t *)ring_i,
        (const uint8_t *)c.ring, (const uint8_t *)c.g};
    const uint64_t sizes[] = {(uint64_t)n*4,(uint64_t)n*4,(uint64_t)n*4,
        adapt ? (uint64_t)n*4 : 0,(uint64_t)n*2,(uint64_t)n*4,
        (uint64_t)n*slots*4,(uint64_t)n*slots*4,
        (uint64_t)c.n*slots*14*4,(uint64_t)c.n*12*4};
    uint64_t offset = 0;
    for (uint32_t a = 0; a < 10; a++) {
        for (uint64_t b = threadIdx.x; b < sizes[a]; b += blockDim.x)
            c.snapshot[offset + b] = sources[a][b];
        offset += sizes[a];
    }
}

__device__ __forceinline__ int compass_class(const Compass c, uint32_t src,
        uint32_t dst, float weight) {
    const int a = c.group[src], b = c.group[dst];
    if (!a) return 5;
    if (a == 1 && b == 1 && weight > 0.0f) return 0;
    if (a == 2 && b == 2 && weight < 0.0f) return 1;
    if (a == 1 && b == 2) return 2;
    if (a == 2 && b == 1) return 3;
    return 4;
}

/* gpu-7, note 10906d21: Metal's k_gap_current, the plain electrical coupling
 * gap_scale * (G @ v - gap_deg * v) over the compact junction rows, which the
 * build body's one_way block at one substep takes on Metal. All voltages are
 * read before k_step integrates. The dot product keeps G's stored CSR order
 * and fuses each product into the running sum, as Metal's fma and SciPy's
 * ARM CSR matvec do. k_step adds clamp_val's 5th section. */
__global__ void k_gap_current(const Params p, const int32_t *target,
                              const int64_t *indptr, const int32_t *source,
                              const float *weight, const float *degree,
                              const float *v, float *current) {
    const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= p.n_gap) return;
    const unsigned int i = (unsigned int)target[j];
    float sum = 0.0f;
    for (int64_t q = indptr[j]; q < indptr[j + 1]; q++)
        sum = __fmaf_rn(weight[q], v[target[source[q]]], sum);
    current[4u * p.n + i] = FM(p.gap_scale, FS(sum, FM(degree[j], v[i])));
}

__global__ void k_step(const Params p,
                       const float *v_rest, const float *v_reset,
                       const float *spike_at, const float *decay_v,
                       const float *dt_over_tau, const float *decay_g,
                       const short *ref_ticks, const unsigned char *graded,
                       const float *adapt_a, const float *adapt_b,
                       const float *dt_over_tauw,
                       float *v, float *g, float *g_i, short *refrac,
                       float *w_adapt, unsigned int *counts,
                       float *ring, float *ring_i,
                       const unsigned int *cand_bits,
                       const unsigned int *force_bits,
                       unsigned int *fired, unsigned int *args,
                       const unsigned int t0, const unsigned int K,
                       const unsigned int kw0, const unsigned int have_bits,
                       float *v_pre, const unsigned int *clamp_bits,
                       const float *clamp_val, const Resource rr, const Compass cc,
                       const Lamina lm, const PTap pt, const Pairs pp, const Occ oc,
                       const Opsin op, const RowCur rc, const Slow sl,
                       const HoldDown hd, const Ser se, const OaOff oa,
                       const Ser oas)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n) return;
    {   /* the other parity's fire bits belong to the previous group, whose folds
         * are done; clear them for the next group (strided: kgroup*words may exceed n) */
        const unsigned int fo = p.fbits_slot
            + (((have_bits >> 8) & 1u) ^ 1u) * p.kgroup * p.words;
        for (unsigned int w = i; w < p.kgroup * p.words; w += p.n) args[fo + w] = 0u;
    }
    const unsigned int word = i >> 5, bit = 1u << (i & 31u);
    const float vr   = UNI(p, 0) ? p.u_vrest
                     : (p.sparse_vrest ? ((clamp_bits[p.words + word] & bit) ? v_rest[i] : p.u_vrest)
                                       : v_rest[i]);   /* override mask lives in the 2nd half of clamp_bits */
    const float vrst = UNI(p, 1) ? p.u_vreset : v_reset[i];
    const float sa   = UNI(p, 2) ? p.u_spike  : spike_at[i];
    const float dv   = UNI(p, 3) ? p.u_decv   : decay_v[i];
    const float dtt  = UNI(p, 4) ? p.u_dtt    : dt_over_tau[i];
    const float dg   = UNI(p, 5) ? p.u_decg   : decay_g[i];
    const short rt   = UNI(p, 6) ? (short)p.u_rt : ref_ticks[i];
    const bool isg = p.n_graded ? (graded[i] != 0) : false;
    const bool split = p.split != 0, cb = p.cbased != 0, ua = p.use_adapt != 0;
    const float aa  = !ua ? 0.0f : (UNI(p, 7) ? p.u_aa  : adapt_a[i]);
    const float ab  = !ua ? 0.0f : (UNI(p, 7) ? p.u_ab  : adapt_b[i]);
    const float dtw = !ua ? 0.0f : (UNI(p, 7) ? p.u_dtw : dt_over_tauw[i]);
    float vv = v[i], gg = g[i], ggi = split ? g_i[i] : 0.0f;
    float ww = ua ? w_adapt[i] : 0.0f;
    short r = refrac[i];
    unsigned int cnt = 0;
    const bool cbits = (have_bits & 1u) != 0, fbits = (have_bits & 2u) != 0;
    const bool membits = (have_bits & 8u) != 0;
    /* coincidence seam: target mask in the 4th quarter of clamp_bits, the
     * per-tick product in the 3rd third of clamp_val (k_coin wrote it) */
    const bool coin = (have_bits & 16u) != 0 && (clamp_bits[3u * p.words + word] & bit) != 0;
    const bool clamped = (have_bits & 4u) != 0 && (clamp_bits[word] & bit) != 0;
    const float cval = clamped ? clamp_val[i] : 0.0f;
    /* design 02e57559: a replaced row emits only its forced events; its
     * membrane and inputs evolve, its threshold and drive candidates do not
     * fire. Mask in the 5th section of clamp_bits. */
    const bool repl = (have_bits & 32u) != 0 && (clamp_bits[4u * p.words + word] & bit) != 0;
    const int tj = pt.n ? ptap_index(pt, i) : -1;
    float gdir = tj >= 0 ? pt.gdir[tj] : 0.0f;
    const int32_t rp = rr.n ? rr.pos[i] : -1;
    float rx = rp >= 0 ? rr.x[rp] : 1.0f;
    const int32_t pq = pp.n ? pp.tgt_of[i] : -1;       /* pair channel target */
    const int32_t ap = pp.n_aux ? pp.apos[i] : -1;     /* state-tap row */
    float px = pq >= 0 ? pp.x[pq] : 0.0f;
    const bool prc = pq >= 0 && pp.rform != 0u;          /* design e05c0eef B1 */
    uint32_t pclk = prc ? pp.clk[pq] : 0u;
    float pres = prc ? pp.res[pq] : 1.0f, parea = prc ? pp.area[pq] : 0.0f;
    uint32_t pcn[4] = {0u, 0u, 0u, 0u};
    const int32_t oj = oc.n_edge ? oc.tgt_of[i] : -1;  /* occupancy target index */
    double os = oj >= 0 ? oc.state[oj] : 0.0;
    const int32_t oq = op.n ? op.pos[i] : -1;          /* opsin slot */
    float osv = oq >= 0 ? op.s[oq] : 0.0f;
    const int32_t cq = rc.n ? rc.pos[i] : -1;          /* held-current slot, the row's earliest */
    const int32_t sq = sl.n ? sl.pos[i] : -1;          /* slow-pool target slot */
    const int32_t sc = sq >= 0 ? sl.rec[sq] : -1;      /* its record column */
    float gs = sq >= 0 ? sl.g[sq] : 0.0f;
    double gsi = sq >= 0 ? sl.intg[sq] : 0.0;
    const bool exu = p.analytic || (hd.amask && hd.amask[i]);   /* design d27f82df B2 */
    const int32_t hq = hd.n ? hd.pos[i] : -1;          /* B3: hold-down slot */
    float hc = hq >= 0 ? hd.c[hq] : 0.0f, hgh = hq >= 0 ? hd.g_h[hq] : 0.0f;
    float hL = hq >= 0 ? hd.L[hq] : 0.0f, hb = hq >= 0 ? hd.b[hq] : 0.0f;
    float hpk = hq >= 0 ? hd.peak[hq] : 0.0f;
    uint32_t hpt = hq >= 0 ? hd.peak_tick[hq] : 0u;
    const bool se_t = se.tgt && se.tgt[i];             /* design b1746728 B1 */
    const bool oa_t = oa.tgt && oa.tgt[i];             /* design 8b6e0441 B2 */
    const bool se_s = se.n_src && se.src[i];
    const bool os_t = oas.tgt && oas.tgt[i];           /* design 8b6e0441 B1 (oas: os is the occupancy state) */
    const bool os_s = oas.src && oas.src[i];
    for (unsigned int k = 0; k < K; k++) {
        const unsigned int tick = t0 + k, kw = kw0 + k;
        if (rp >= 0) rx = __fadd_rn(__fmul_rn(rx, rr.d[rp]), rr.c[rp]);
        const unsigned int slot = tick % p.n_ring;
        const size_t ri = (size_t)slot * p.n + i;
        float *tp = tj >= 0 ? pt.tap + ((size_t)kw * PTAP_ROWS + tj) * PTAP_CH : NULL;
        if (tp) {
            tp[0] = vv;
            tp[2] = ww;
            tp[14] = ring[ri];
            tp[15] = split ? ring_i[ri] : 0.0f;
            gdir = FM(gdir, dg);
            const size_t dq = (size_t)slot * PTAP_ROWS + tj;
            tp[12] = pt.dring[dq];
            gdir = FA(gdir, pt.dring[dq]);
            pt.dring[dq] = 0.0f;
            tp[11] = gdir;
        }
        gg = FM(gg, dg);
        if (split) ggi = FM(ggi, dg);
        float arrival = ring[ri];
        const int32_t cp = cc.n ? cc.pos[i] : -1;
        float *ct = cp >= 0 ? cc.tap + ((size_t)kw * cc.n + cp) * 29 : NULL;
        if (cp >= 0) {
            float *cr = cc.ring + ((size_t)slot * cc.n + cp) * 14;
            const bool cut = cc.cut_tick >= 0 && tick >= (uint32_t)cc.cut_tick;
            /* Keep ordinary pending events intact until their arrival. The
             * parallel positive ring excludes E-E during the same native
             * folds, avoiding cancellation when the class is interrupted. */
            if (cut) arrival = cr[12];
            ct[24] = ww;
            ct[26] = cr[0]; ct[27] = cut ? 1.0f : 0.0f;
            ct[28] = cut ? 0.0f : cr[13];
            for (uint32_t j = 0; j < 12; j++) {
                const float x = cut && j == 0 ? 0.0f : cr[j];
                ct[j] = x;
                const size_t cg = (size_t)cp * 12 + j;
                cc.g[cg] = FA(FM(cc.g[cg], dg), x);
                ct[12 + j] = cc.g[cg];
                cr[j] = 0.0f;
            }
            cr[12] = cr[13] = 0.0f;
        }
        /* state tap: fields written where they are formed, so none stays live */
        float *ta = ap >= 0 ? pp.aux + ((size_t)kw * pp.n_aux + (uint32_t)ap) * AUXW : NULL;
        const float arr_i = split ? ring_i[ri] : 0.0f;
        if (ta) { ta[0] = vv; ta[2] = arrival; ta[3] = arr_i; ta[13] = ww; ta[14] = (float)r; }
        gg = FA(gg, arrival); ring[ri] = 0.0f;
        if (split) { ggi = FA(ggi, arr_i); ring_i[ri] = 0.0f; }
        if (sq >= 0) {   /* design 2163ba7d, Build 2: decay, then this tick's arrivals */
            const size_t sr = (size_t)slot * sl.n + (uint32_t)sq;
            gs = FA(FM(gs, sl.decay), sl.ring[sr]);
            sl.ring[sr] = 0.0f;
        }
        if (coin) gg = FA(gg, clamp_val[2u * p.n + i]);
        if (tp) { tp[5] = gg; tp[6] = ggi; }
        uint32_t na = 0u;
        if (pq >= 0) {   /* this pair's events arriving on this tick, then x = r*x + inc*n */
            const size_t pr = (size_t)(tick % pp.nr) * pp.n + (uint32_t)pq;
            na = pp.ring[pr]; pp.ring[pr] = 0u;
            const float nd = prc ? pair_recover(pp, na, pclk, pres, pcn, parea) : (float)na;
            px = FA(FM(px, pp.r[pq]), FM(pp.inc[pq], nd));
        }
        if (ta) { ta[4] = gg; ta[5] = ggi; ta[6] = (float)na; ta[7] = px; }
        /* design 81427d47: decay, this tick's summed arrivals, the update, and
         * the conductance the membrane update below uses */
        float og = 0.0f;
        double *ot = (oj >= 0 && oc.tap_on)
            ? oc.tap + ((size_t)kw * oc.n_tgt + (uint32_t)oj) * OCC_CH : NULL;
        if (oj >= 0) {
            const double ob = os;
            os = __dmul_rn(os, oc.dec[oj]);
            const double od = os;
            const size_t orow = (size_t)(tick % oc.nr) * oc.n_edge;
            uint64_t qs = 0u;
            uint32_t nsum = 0u;
            for (uint32_t q = oc.t_begin[oj]; q < oc.t_begin[oj + 1]; q++) {
                const uint32_t c = oc.ring[orow + q];
                if (!c) continue;
                oc.ring[orow + q] = 0u;
                qs += (uint64_t)c * oc.quanta[q];
                nsum += c;
                if (oc.log_on) {   /* this target's own region, in tick then edge order */
                    const uint32_t at = oc.a_count[oj]++;
                    if (at < oc.a_off[oj + 1] - oc.a_off[oj]) {
                        const uint32_t w = oc.a_off[oj] + at;
                        oc.a_tick[w] = (int32_t)tick; oc.a_edge[w] = q; oc.a_n[w] = c;
                    }
                }
            }
            double A = 0.0;
            if (nsum) {
                A = __dmul_rn((double)qs, oc.scale[oj]);   /* exact: qs < 2^53, scale a power of two */
                os = oc.mode == 1u
                    ? __dadd_rn(os, __dmul_rn(__dsub_rn(1.0, os),
                                              -expm1(-__dmul_rn(oc.alpha[oj], A))))
                    : __dadd_rn(os, __dmul_rn(oc.beta[oj], A));
            }
            og = oc.mode == 1u ? __double2float_rn(__dmul_rn(oc.gmax[oj], os))
                               : __double2float_rn(os);
            if (nsum && oc.log_on) {
                const uint32_t at = oc.s_count[oj]++;
                if (at < oc.s_off[oj + 1] - oc.s_off[oj]) {
                    const uint32_t w = oc.s_off[oj] + at;
                    oc.s_tick[w] = (int32_t)tick; oc.s_n[w] = nsum;
                    oc.s_val[4u * w] = A; oc.s_val[4u * w + 1u] = ob;
                    oc.s_val[4u * w + 2u] = od; oc.s_val[4u * w + 3u] = os;
                    oc.s_g[w] = og;
                }
            }
            if (ot) {
                ot[0] = vv; ot[2] = ww; ot[3] = (double)r; ot[4] = (double)nsum;
                ot[5] = A; ot[6] = ob; ot[7] = od; ot[8] = os; ot[9] = (double)og;
                ot[10] = oc.mode == 1u ? oc.gmax[oj] : oc.beta[oj];
                ot[17] = arrival; ot[18] = arr_i;
            }
        }
        /* design 87a3ad35: the gate's exact one-step update, then the
         * conductance and the current at the tick's starting membrane */
        bool olit = false;
        float og_op = 0.0f;
        float *otp = (oq >= 0 && op.tap_on)
            ? op.tap + ((size_t)kw * op.n + (uint32_t)oq) * OPSIN_CH : NULL;
        if (oq >= 0) {
            for (uint32_t w = 0; w < op.n_win[oq]; w++) {
                const uint32_t *wp = op.win + ((size_t)oq * OPSIN_WIN + w) * 2u;
                if (tick >= wp[0] && tick < wp[1]) olit = true;
            }
            if (op.train) {   /* design 2163ba7d: an on phase of the pulse train */
                const uint32_t *tp = op.train + (size_t)oq * 4u;
                if (tp[3] && tick >= tp[0] && tick < tp[1] && (tick - tp[0]) % tp[3] < tp[2])
                    olit = true;
            }
            osv = olit ? (op.force[oq] ? 1.0f : FA(1.0f, FM(FS(osv, 1.0f), op.a_on[oq])))
                       : FM(osv, op.a_off[oq]);
            og_op = FM(op.g_max[oq], osv);
            if (otp) { otp[0] = osv; otp[1] = og_op; otp[3] = FM(og_op, FS(op.e_rev[oq], vv)); }
        }
        float extra = 0.0f;
        /* design b1746728 B1: cns.py's gain * (s * decay); the held current
         * is refused with the state, so the two never meet */
        if (se_t) extra = FM(se.gain, FM(se.s[0], se.decay));
        /* design 8b6e0441 B1: cns.py's extra + gain * (s * decay), summed */
        if (os_t) extra = FA(extra, FM(oas.gain, FM(oas.s[0], oas.decay)));
        /* gpu-7, note 10906d21: the electrical current, where cns.py adds it
         * after the octopamine state; the held current, which assigns extra,
         * is refused with it */
        if (p.n_gap && p.gap_scale != 0.0f) extra = FA(extra, clamp_val[4u * p.n + i]);
        if (cq >= 0) {   /* design 2d12a3b3: the held current, before the adaptation */
            extra = rc.i0[cq];
            for (int32_t q = cq; q >= 0 && tick >= rc.on[q]; q = rc.next[q])   /* Build 3 */
                if (tick < rc.off[q]) { extra = FA(rc.i0[q], rc.i1[q]); break; }
        }
        if (oa_t) extra = FA(extra, oa.off);   /* design 8b6e0441 B2: the offset */
        const float irow = extra;   /* design d27f82df B5: the hold-down tap's i_row */
        if (ua) extra = FS(extra, ww);
        float vi;
        if (cb) {
            float ge = FM(fmaxf(gg, 0.0f), p.gscale);
            float gi = split ? FM(ggi, p.gscale) : FM(fmaxf(-gg, 0.0f), p.gscale);
            if (sq >= 0) {   /* design 2163ba7d, Build 2: the slow pool, the fast pool's reversal */
                const float sg = FM(gs, p.gscale);
                gi = FA(gi, sg);
                gsi = __dadd_rn(gsi, (double)sg);
                const uint32_t b = tick / sl.steps_per_bin;
                if (sc >= 0 && b < sl.n_bins) {
                    double *ra = sl.rec_acc + (size_t)b * sl.n_rec + (uint32_t)sc;
                    *ra = __dadd_rn(*ra, (double)sg);
                }
            }
            const float ei = lm.e_inh ? lm.e_inh[i] : p.e_inh;
            float gl = 0.0f;
            if (lm.mask && lm.mask[i]) {
                float level = 0.0f;
                if (lm.lpos)
                    level = lm.level[lm.lpos[i]];
                else if (tick >= lm.on && tick < lm.off)
                    level = lm.ramp ? fminf(1.0f, FD((float)(tick-lm.on), (float)lm.ramp)) : 1.0f;
                else if (tick >= lm.off && lm.ramp)
                    level = FS(1.0f, fminf(1.0f, FD((float)(tick-lm.off), (float)lm.ramp)));
                gl = FM(lm.gain, level);
                gi = FA(gi, gl);
            }
            /* tonic excitatory conductance, cns.py's `ge = ge + tonic_ge`,
             * here because it sits after ge and gi are formed and before
             * either integrator. Rides in the 4th section of clamp_val. */
            if (p.has_tonic) ge = FA(ge, clamp_val[3u * p.n + i]);
            if (ta) ta[8] = ge;   /* ordinary excitatory conductance, tonic included */
            if (pq >= 0) ge = FA(ge, px);
            if (oj >= 0) {
                if (ot) ot[11] = ge;   /* excitatory conductance before the occupancy term */
                ge = FA(ge, og);
                if (ot) {
                    ot[12] = ge; ot[13] = gi;
                    ot[14] = FM(og, FS(p.e_exc, vv));
                    ot[15] = FM(ge, FS(p.e_exc, vv));
                    ot[16] = FM(gi, FS(ei, vv));
                }
            }
            if (ta) {   /* normalized drives at the tick's starting membrane */
                ta[9] = gi;
                ta[10] = FM(px, FS(p.e_exc, vv));
                ta[11] = FM(ge, FS(p.e_exc, vv));
                ta[12] = FM(gi, FS(ei, vv));
            }
            if (lm.state) {
                lm.state[i] = ge;
                lm.state[p.n+i] = gi;
                lm.state[2u*p.n+i] = ww; /* before this update advances adaptation */
                lm.state[3u*p.n+i] = gl;
            }
            if (tp) {   /* the two current terms the update below multiplies by dt/tau */
                tp[7] = ge; tp[8] = gi;
                tp[9] = FM(gi, FS(ei, vv));
                tp[10] = FM(ge, FS(p.e_exc, vv));
            }
            if (exu) {
                float total = FA(FA(1.0f, ge), gi);
                if (oq >= 0) total = FA(total, og_op);   /* the opsin, last */
                if (hq >= 0) total = FA(total, hgh);     /* design d27f82df B3, after it */
                const float inv = FD(1.0f, total);
                float vinf = FA(FA(FA(FM(vr, inv), FM(p.e_exc, FM(ge, inv))),
                                   FM(ei, FM(gi, inv))), FM(extra, inv));
                if (oq >= 0) vinf = FA(vinf, FM(op.e_rev[oq], FM(og_op, inv)));
                if (hq >= 0) vinf = FA(vinf, FM(FA(vr, hd.e_rel[hq]), FM(hgh, inv)));
                const float cdec = expf(FM(-dtt, total));
                vi = FA(vinf, FM(FS(vv, vinf), cdec));
            } else {
                float brk = FA(FA(FM(ge, FS(p.e_exc, vv)), FM(gi, FS(ei, vv))), extra);
                if (oq >= 0) brk = FA(brk, FM(og_op, FS(op.e_rev[oq], vv)));   /* the opsin, last */
                vi = FA(FA(vr, FM(FS(vv, vr), dv)), FM(brk, dtt));
            }
        } else {
            if (ta) { ta[8] = ta[9] = ta[10] = ta[11] = ta[12] = 0.0f; }
            vi = FA(FA(vr, FM(FS(vv, vr), dv)), FM(FA(gg, extra), dtt));
        }
        if (ua) {
            /* design 6d0042d5 Build 2: with adapt_split the second half of
             * adapt_a holds each row's coefficient below rest */
            const float xr = FS(vi, vr);
            const float ae = (p.adapt_split && xr < 0.0f) ? adapt_a[p.n + i] : aa;
            ww = FA(ww, FM(dtw, FS(FM(ae, xr), ww)));
        }
        float hx = 0.0f;
        if (hq >= 0) {   /* design d27f82df B3: x before the refractory clamp */
            hx = FS(vi, vr);
            if (hd.form[hq] == 0u) hgh = FM(hgh, hd.decay[hq]);
            else {
                const float load = FM(FM(hd.k[hq], fmaxf(FS(hx, hd.va[hq]), 0.0f)), hL);
                hc = fmaxf(FA(hc, FM(hd.dt, FS(load, FD(hc, hd.tau_c[hq])))), 0.0f);
                hgh = FD(FM(hd.gmax[hq], hc), FA(1.0f, hc));
            }
        }
        const bool active = r <= 0;
        if (!active) vi = vrst;
        if (otp) otp[2] = FM(og_op, FS(op.e_rev[oq], vi));
        const bool cand_event = cbits
            && (cand_bits[(size_t)kw * p.words + word] & bit) != 0u;
        const bool mem_event = cand_event && membits
            && (clamp_bits[2u * p.words + word] & bit) != 0u;
        /* arm E: a lit threshold-window slot tests against its own value */
        const float sa_t = (oq >= 0 && olit && op.kind[oq] == 1u) ? op.thr[oq] : sa;
        const bool thr = active && vi >= sa_t && !isg;
        bool fire = thr && !repl;
        if (cand_event && active && !mem_event && !repl) fire = true;
        const bool forced = fbits && (force_bits[(size_t)kw * p.words + word] & bit) != 0u;
        if (forced) fire = true;
        if (clamped) vi = cval;   /* graded drive: set after detection, before release */
        if (tp) tp[1] = vi;
        if (ot) {   /* flags as the state tap's: 0 active, 1 threshold, 4 forced, 5 fired */
            ot[1] = vi;
            ot[19] = (double)((active ? 1u : 0u) | (thr ? 2u : 0u) | (forced ? 16u : 0u)
                              | (fire ? 32u : 0u));
        }
        if (ta) {   /* bit 0 active, 1 threshold, 2 candidate, 3 membrane kick,
                     * 4 forced, 5 fired, 6 clamped, 7 graded */
            ta[1] = vi;
            ta[15] = (float)((active ? 1u : 0u) | (thr ? 2u : 0u) | (cand_event ? 4u : 0u)
                             | (mem_event ? 8u : 0u) | (forced ? 16u : 0u) | (fire ? 32u : 0u)
                             | (clamped ? 64u : 0u) | (isg ? 128u : 0u));
        }
        /* The pre-reset membrane, kept for the voltage tap and for graded
         * release: cns.py computes release from v[graded_idx] before it
         * resets cells that fired, so a graded cell in the drive set
         * releases at the voltage that crossed, not at v_reset. */
        if (p.n_vrec) {
            v_pre[i] = vi;
            v_pre[p.n + i] = vv;
            v_pre[2u * p.n + i] = gg;
            v_pre[3u * p.n + i] = ggi;
        } else if (isg) {
            v_pre[i] = vi;
        }
        if (mem_event) vi = FA(vi, clamp_val[p.n + i]);   /* threshold reads it next tick */
        r = (r > 0) ? (short)(r - 1) : (short)0;
        if (fire) {
            if (rp >= 0) {
                rr.snapshot[(size_t)k * rr.n + rp] = rx;
                const uint32_t at = atomicAdd(rr.count, 1u);
                if (at < rr.cap) {
                    rr.tick[at] = (int32_t)tick;
                    rr.row[at] = (int32_t)i;
                    rr.value[at] = rx;
                }
                rx = __fmul_rn(rx, rr.keep[rp]);
            }
            unsigned int s = atomicAdd(&args[kw * 3], 1u);
            if (s < p.fired_cap) fired[(size_t)kw * p.fired_cap + s] = i;
            if (se_s) atomicAdd(se.n_tick, 1u);   /* design b1746728 B1 */
            if (os_s) atomicAdd(oas.n_tick, 1u);   /* design 8b6e0441 B1 */
            /* fire bit for k_fold's overflow path; parity in have_bits bit 8 */
            atomicOr(&args[p.fbits_slot + ((have_bits >> 8) & 1u) * p.kgroup * p.words
                           + k * p.words + word], bit);
            /* pair events: arrival slot tick + delay, which a later launch
             * reads (configuration refuses delay < kgroup or delay + kgroup > nr) */
            if (pp.n)
                for (int32_t q = pp.src_first[i]; q >= 0; q = pp.src_next[q])
                    atomicAdd(&pp.ring[(size_t)((tick + pp.delay[q]) % pp.nr) * pp.n
                                       + (uint32_t)q], 1u);
            /* occupancy arrivals, the same slot rule; interrupt withholds them */
            if (oc.n_edge && !oc.interrupt)
                for (int32_t q = oc.src_first[i]; q >= 0; q = oc.src_next[q])
                    atomicAdd(&oc.ring[(size_t)((tick + oc.delay[q]) % oc.nr) * oc.n_edge
                                       + (uint32_t)q], 1u);
            vi = vrst; r = rt; cnt++;
            if (ua) ww = FA(ww, ab);
            if (hq >= 0) {   /* design d27f82df B3 */
                if (hd.form[hq] == 0u) hgh = FA(hgh, hd.dg[hq]);
                else { hc = FA(hc, hd.c_spk[hq]); hgh = FD(FM(hd.gmax[hq], hc), FA(1.0f, hc)); }
                hL = 1.0f; hb = 0.0f;
            }
        }
        if (hq >= 0) {   /* B3: the latch count on an active tick that did not fire */
            if (active && !fire) {
                hb = hx < hd.latch_mv[hq] ? FA(hb, 1.0f) : 0.0f;
                if (hb >= (float)hd.latch_ticks[hq]) hL = 0.0f;
            }
            if (hgh > hpk) { hpk = hgh; hpt = tick; }
            if (hd.tap) {
                float *ht = hd.tap + ((size_t)kw * hd.n + (uint32_t)hq) * HD_CH;
                ht[0] = irow; ht[1] = hc; ht[2] = hgh; ht[3] = hL; ht[4] = hb;
            }
        }
        vv = vi;
        if (ct) ct[25] = ww;
        if (tp) { tp[3] = ww; tp[4] = (float)r; tp[13] = fire ? 1.0f : 0.0f; }
    }
    v[i] = vv; g[i] = gg; refrac[i] = r;
    if (split) g_i[i] = ggi;
    if (ua) w_adapt[i] = ww;
    if (cnt) counts[i] += cnt;
    if (rp >= 0) rr.x[rp] = rx;
    if (tj >= 0) pt.gdir[tj] = gdir;
    if (pq >= 0) pp.x[pq] = px;
    if (prc) {
        pp.clk[pq] = pclk; pp.res[pq] = pres; pp.area[pq] = parea;
        for (int k = 0; k < 4; k++) pp.cnt[4 * pq + k] += pcn[k];
    }
    if (oj >= 0) oc.state[oj] = os;
    if (oq >= 0) op.s[oq] = osv;
    if (sq >= 0) { sl.g[sq] = gs; sl.intg[sq] = gsi; }
    if (hq >= 0) {
        hd.c[hq] = hc; hd.g_h[hq] = hgh; hd.L[hq] = hL; hd.b[hq] = hb;
        hd.peak[hq] = hpk; hd.peak_tick[hq] = hpt;
    }
}

/* Design b1746728, Build 1: the serotonin state's tick, once after k_step
 * (kgroup 1) in one thread, in cns.py's order: the decay that k_step's
 * targets already used, the applied sum, then this tick's source spikes. */
__global__ void k_ser(const Ser se, const uint32_t tick) {
    const float sd = FM(se.s[0], se.decay);
    *se.applied = __dadd_rn(*se.applied, (double)sd);
    const uint32_t n = *se.n_tick;
    float s = sd;
    if (n) {
        s = FA(sd, FM(se.inc, (float)n));
        *se.spikes += n;
        se.s[1] = fmaxf(se.s[1], s);
    }
    se.s[0] = s;
    *se.n_tick = 0u;
    if ((tick + 1u) % se.steps_per_bin == 0u) {
        const uint32_t b = tick / se.steps_per_bin;
        if (b < se.n_bins) se.bins[b] = s;
    }
}

/* Design 302c1bfb, Build 8, on CUDA (design 9501604f, Build C1): the release
 * factor of the listed rows, once per tick between k_step and k_scatter
 * (kgroup 1), cns_engine.m's k_psi in its operation order. With has9 each
 * listed row's source-fed state first decays and takes this tick's arrivals,
 * cns.py's `psi_g9 *= psi_r9` and `psi_g9 += psi_c9 * s9`; the state G is
 * that, or the row's g_i after k_step. A listed row that fired this tick gets
 * f = 1/(1 + G kg) for k_scatter and one log entry (f, G, g_i, 1) at its
 * window tick; the others log a 0 flag. */
__global__ void k_psi(const Params p, const Psi ps, const float *g_i,
                      const unsigned int *args, const unsigned int tick,
                      const unsigned int par, const unsigned int wt)
{
    const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ps.n) return;
    const unsigned int i = (unsigned int)ps.prow[j];
    float G;
    if (ps.has9) {
        const size_t ri = (size_t)(tick % p.n_ring) * ps.n + j;
        float s = FM(ps.g9[j], ps.r9[j]);
        s = FA(s, FM(ps.c9[j], ps.ring9[ri]));
        ps.ring9[ri] = 0.0f;
        ps.g9[j] = s;
        G = s;
    } else {
        G = g_i[i];
    }
    const unsigned int bits = args[p.fbits_slot + par * p.kgroup * p.words + (i >> 5)];
    float *lg = ps.plog + ((size_t)wt * ps.n + j) * 4u;
    if (bits & (1u << (i & 31u))) {
        const float f = FD(1.0f, FA(1.0f, FM(G, ps.kg)));
        ps.fac[i] = f;
        lg[0] = f; lg[1] = G; lg[2] = g_i[i]; lg[3] = 1.0f;
    } else {
        lg[3] = 0.0f;
    }
}

/* Spike delivery, first half: one block per spike, threads stride the
 * source's CSC column, and every edge is appended to its TARGET's bin with an
 * integer atomic. Integer addition commutes, so the bin's contents are the
 * same set in any schedule; k_fold puts them in order. The key sorts by
 * arrival slot, then source, which per ring slot is the C reference's order
 * (bins hold one tick, so tick order is the dispatch order).
 *
 * Metal launched exactly one threadgroup per spike through an indirect
 * dispatch. Here the grid is fixed and blocks stride the spike list. */
__global__ void k_scatter(const Params p,
                          const unsigned int *fired,
                          const int64_t *indptr, const int32_t *indices,
                          const float *weight, const int32_t *delay_slot,
                          unsigned int *bcount, uint2 *bin,
                          unsigned int *touched, const unsigned int tick,
                          unsigned int *spk_tick,
                          unsigned int *ovr, unsigned int *args,
                          const unsigned int wt, const unsigned int k,
                          const Resource rr, const Deliv dl, const Psi ps)
{
    __shared__ double s_pos[SCATTER_TG], s_neg[SCATTER_TG];
    __shared__ unsigned int s_no[SCATTER_TG], s_nd[SCATTER_TG];
    unsigned int nfired = args[wt * 3];
    if (nfired > p.fired_cap) nfired = p.fired_cap;   /* entries past the cap were never written */
    for (unsigned int sp = blockIdx.x; sp < nfired; sp += gridDim.x) {
        const unsigned int src = fired[sp];
        const int32_t rp = rr.n ? rr.pos[src] : -1;
        const float release = rp >= 0 ? rr.snapshot[(size_t)k * rr.n + rp] : 1.0f;
        /* design 302c1bfb Build 8 (C1): cns.py's data * release_snap, 1.0 on
         * a row k_psi does not list */
        const float fs = ps.n ? ps.fac[src] : 1.0f;
        /* block-uniform: one source per block iteration */
        const bool rec = dl.n_src && dl.src[src];
        double pos_sum = 0.0, neg_sum = 0.0;
        unsigned int n_other = 0, n_dir = 0;
        /* the seam's "prev": which sources fired on this tick, read next tick */
        if (p.n_coin && threadIdx.x == 0) spk_tick[src] = tick;
        const unsigned int arr = (tick + (unsigned int)delay_slot[src]) % p.n_ring;
        const unsigned int key = (arr << p.key_shift) | src;   /* source low, arrival slot above; both checked at create */
        const int64_t e0 = indptr[src], e1 = indptr[src + 1];
        for (int64_t e = e0 + threadIdx.x; e < e1; e += blockDim.x) {
            const unsigned int tgt = (unsigned int)indices[e];
            const unsigned int pos = atomicAdd(&bcount[tgt], 1u);
            const bool sel = rp >= 0 && (!rr.tmask || rr.tmask[tgt]);
            const float w = sel ? __fmul_rn(weight[e], release)
                          : ps.n ? __fmul_rn(weight[e], fs) : weight[e];
            if (rec) {
                if (dl.tgt[tgt]) {
                    const uint32_t at = atomicAdd(dl.count, 1u);
                    if (at < dl.cap_direct) {
                        dl.dir_tick[at] = (int32_t)tick;
                        dl.dir_edge[at] = e;
                        dl.dir_w[at] = w;
                    }
                    n_dir++;
                } else {
                    if (w > 0.0f) pos_sum = __dadd_rn(pos_sum, (double)w);
                    else if (w < 0.0f) neg_sum = __dadd_rn(neg_sum, (double)w);
                    n_other++;
                }
            }
            if (pos < BIN_CAP) {
                bin[(size_t)tgt * BIN_CAP + pos] = make_uint2(key, __float_as_uint(w));
            } else if (pos == BIN_CAP) {
                atomicAdd(ovr, 1u);
            }
            if ((pos & 7u) == 7u) atomicMax(ovr + 1, pos + 1u);   /* largest bin, to 8 */
            if (pos == 3u || pos == 7u || pos == 15u || pos == 31u)   /* bin-size histogram: reached 4, 8, 16, 32 */
                atomicAdd(ovr + 2 + (pos == 3u ? 0 : pos == 7u ? 1 : pos == 15u ? 2 : 3), 1u);
            if (pos == 0) {   /* first arrival at this target this tick: enlist it for k_fold */
                const unsigned int t = atomicAdd(&args[p.tcnt_slot + wt], 1u);
                touched[t] = tgt;
            }
        }
        if (rec) {   /* fixed tree over threads: the same sum in every schedule */
            const unsigned int t = threadIdx.x;
            s_pos[t] = pos_sum; s_neg[t] = neg_sum; s_no[t] = n_other; s_nd[t] = n_dir;
            __syncthreads();
            for (unsigned int h = blockDim.x >> 1; h > 0; h >>= 1) {
                if (t < h) {
                    s_pos[t] = __dadd_rn(s_pos[t], s_pos[t + h]);
                    s_neg[t] = __dadd_rn(s_neg[t], s_neg[t + h]);
                    s_no[t] += s_no[t + h]; s_nd[t] += s_nd[t + h];
                }
                __syncthreads();
            }
            if (t == 0) {
                const uint32_t at = atomicAdd(dl.count + 1, 1u);
                if (at < dl.cap_event) {
                    dl.ev_tick[at] = (int32_t)tick; dl.ev_src[at] = (int32_t)src;
                    dl.ev_pos[at] = s_pos[0]; dl.ev_neg[at] = s_neg[0];
                    dl.ev_nother[at] = s_no[0]; dl.ev_ndir[at] = s_nd[0];
                }
            }
            __syncthreads();
        }
    }
}

/* Knuth's two-sum: with round-to-nearest and no contraction, s = a + b and
 * e together represent a + b exactly, s = fl(a+b). The fold uses it to add a
 * tick's arrivals the way cns.py does: np.bincount sums a target's weights
 * for one arrival slot in float64 and .astype(np.float32) rounds that sum
 * ONCE, then `ring[slot] +=` rounds once more. Summing in float32 one edge at
 * a time rounds c times instead. */
__device__ __forceinline__ float2 twosum(float a, float b) {
    const float s = FA(a, b);
    const float bb = FS(s, a);
    return make_float2(s, FA(FS(a, FS(s, bb)), FS(b, bb)));
}
__device__ __forceinline__ void dfadd(float &h, float &l, float w) {
    const float2 t = twosum(h, w);
    h = t.x; l = FA(l, t.y);
}

/* Observe the same sorted event bin used by the native fold. No model state
 * is read back or advanced by a second simulator. */
__device__ __noinline__ void compass_bin(const Compass c, const Params p,
        uint32_t dst, uint32_t slot, const unsigned int *keys,
        const float *weights, uint32_t begin, uint32_t end) {
    const int32_t pos = c.pos[dst];
    if (pos < 0) return;
    float hi[13] = {}, lo[13] = {};
    uint32_t count = 0;
    for (uint32_t j = begin; j < end; j++) {
        const uint32_t src = keys[j] & ((1u << p.key_shift) - 1u);
        const float w = weights[j];
        if (w == 0.0f) continue;
        const int cls = compass_class(c, src, dst, w);
        const int ch = cls * 2 + (w < 0.0f);
        dfadd(hi[ch], lo[ch], fabsf(w));
        if (w > 0.0f && cls != 0) dfadd(hi[12], lo[12], w);
        if (cls == 0) count++;
    }
    float *row = c.ring + ((size_t)slot * c.n + pos) * 14;
    for (int j = 0; j < 13; j++) {
        const float sum = twosum(hi[j], lo[j]).x;
        if (sum != 0.0f) row[j] = FA(row[j], sum);
    }
    row[13] = FA(row[13], (float)count);
}

__device__ __forceinline__ void compass_edge(const Compass c, uint32_t src,
        uint32_t dst, uint32_t slot, float w) {
    const int32_t pos = c.pos[dst];
    if (pos < 0 || w == 0.0f) return;
    const int cls = compass_class(c, src, dst, w);
    float *row = c.ring + ((size_t)slot * c.n + pos) * 14;
    const int ch = cls * 2 + (w < 0.0f);
    row[ch] = FA(row[ch], fabsf(w));
    if (w > 0.0f && cls != 0) row[12] = FA(row[12], w);
    if (cls == 0) row[13] = FA(row[13], 1.0f);
}

/* Spike delivery, second half, once per TICK (a fused group runs scatter,
 * fold, scatter, fold: bins hold one tick, so overflow stays rare; folding a
 * whole 18-tick group overflowed most hubs and cost 50x on Metal, 2026-09-05).
 * One thread per TOUCHED target: it sorts its bin on the key and adds the
 * weights into the ring in that order, so every ring slot is summed in one
 * fixed order, tick then source ascending, the C reference's and scipy's.
 * Only this thread writes ring[.][i] in the dispatch, so the ring takes plain
 * loads and stores and no float atomic is needed. A target with more than
 * BIN_CAP arrivals in the tick walks its incoming edges (the transposed
 * graph, sources ascending) against the tick's fire bits instead: the same
 * source order, exact, at the price of the whole in-degree. */
__global__ void k_fold(const Params p,
                       unsigned int *bcount, const uint2 *bin,
                       const unsigned int *touched,
                       float *ring, float *ring_i, const unsigned int *args,
                       const int64_t *tindptr, const int32_t *tsrc,
                       const float *tw, const int32_t *delay_slot,
                       const unsigned int t0, const unsigned int k,
                       const unsigned int par, const unsigned int wt,
                       const Resource rr, const Compass cc, const PTap pt,
                       const Slow sl, const Psi ps)
{
    const unsigned int fb = p.fbits_slot + par * p.kgroup * p.words;
    const unsigned int ntouched = args[p.tcnt_slot + wt];
    const unsigned int stride = gridDim.x * blockDim.x;
    const bool split = p.split != 0;
    const unsigned int smask = (1u << p.key_shift) - 1u;
    for (unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
         t < ntouched; t += stride) {
        const unsigned int i = touched[t];
        const unsigned int c = bcount[i];   /* at least 1: every enlisted target had an arrival */
        bcount[i] = 0;
        const int tj = pt.n ? ptap_index(pt, i) : -1;
        const int hp = ps.has9 ? ps.ppos[i] : -1;   /* design 302c1bfb Build 8 (C1): listed position */
        if (c <= BIN_CAP) {
            unsigned int kk[BIN_CAP]; float ww[BIN_CAP];
            for (unsigned int j = 0; j < c; j++) {
                const uint2 b = bin[(size_t)i * BIN_CAP + j];
                kk[j] = b.x; ww[j] = __uint_as_float(b.y);
            }
            for (unsigned int j = 1; j < c; j++) {   /* insertion sort, at most BIN_CAP */
                const unsigned int kj = kk[j]; const float wj = ww[j]; int m = (int)j - 1;
                while (m >= 0 && kk[m] > kj) { kk[m + 1] = kk[m]; ww[m + 1] = ww[m]; m--; }
                kk[m + 1] = kj; ww[m + 1] = wj;
            }
            /* The bin is sorted by (arrival slot, source), so a run of equal
             * slot is contiguous: sum it compensated, round the sum once, add
             * once, which is np.bincount(...).astype(np.float32) then `+=`. */
            for (unsigned int j = 0; j < c; ) {
                const unsigned int slot = kk[j] >> p.key_shift;
                float ph = 0.0f, pl = 0.0f, nh = 0.0f, nl = 0.0f;
                float dh = 0.0f, dlo = 0.0f;
                float fh = 0.0f, flo = 0.0f, sh = 0.0f, slo = 0.0f;   /* design 2163ba7d */
                float h9 = 0.0f, l9 = 0.0f;   /* design 302c1bfb Build 8 (C1) */
                unsigned int e = j;
                for (; e < c && (kk[e] >> p.key_shift) == slot; e++) {
                    const float w = ww[e];
                    if (split) {
                        if (w > 0.0f) dfadd(ph, pl, w);
                        else if (w < 0.0f) {
                            if (sl.n && sl.src[kk[e] & smask]) {   /* the listed rows' split */
                                const float fw = FM(-w, sl.keep);
                                dfadd(nh, nl, fw); dfadd(fh, flo, fw);
                                dfadd(sh, slo, FM(-w, sl.sfac));
                            } else dfadd(nh, nl, -w);
                            if (hp >= 0 && ps.src9[kk[e] & smask]) dfadd(h9, l9, -w);
                        }
                    }
                    else dfadd(ph, pl, w);
                    if (tj >= 0 && w > 0.0f && pt.dsrc[kk[e] & smask]) dfadd(dh, dlo, w);
                }
                const size_t ri = (size_t)slot * p.n + i;
                const float sp = twosum(ph, pl).x;   /* = fl(exact sum), the reference's astype */
                if (sp != 0.0f) ring[ri] = FA(ring[ri], sp);
                if (split) {
                    const float sn = twosum(nh, nl).x;
                    if (sn != 0.0f) ring_i[ri] = FA(ring_i[ri], sn);
                }
                if (sl.n) {
                    const float ss = twosum(sh, slo).x, sf = twosum(fh, flo).x;
                    if (ss != 0.0f || sf != 0.0f) {   /* only a listed row's target gets here */
                        const uint32_t q = (uint32_t)sl.pos[i];
                        if (ss != 0.0f) {
                            float *sr = sl.ring + (size_t)slot * sl.n + q;
                            *sr = FA(*sr, ss);
                            sl.acc_slow[q] = __dadd_rn(sl.acc_slow[q], (double)ss);
                        }
                        if (sf != 0.0f) sl.acc_fast[q] = __dadd_rn(sl.acc_fast[q], (double)sf);
                    }
                }
                if (hp >= 0) {   /* design 302c1bfb Build 8 (C1): rounded once, as ring_i's sum */
                    const float s9 = twosum(h9, l9).x;
                    if (s9 != 0.0f) {
                        float *q9 = ps.ring9 + (size_t)slot * ps.n + (uint32_t)hp;
                        *q9 = FA(*q9, s9);
                    }
                }
                if (tj >= 0) {
                    const float sd = twosum(dh, dlo).x;
                    const size_t dq = (size_t)slot * PTAP_ROWS + tj;
                    if (sd != 0.0f) pt.dring[dq] = FA(pt.dring[dq], sd);
                }
                if (cc.n) compass_bin(cc, p, i, slot, kk, ww, j, e);
                j = e;
            }
        } else {
            /* NOT compensated: a target's in-edges arrive here in source
             * order, not slot order, so the run of one arrival slot is not
             * contiguous and cannot be summed in one pass. This path rounds
             * per edge. It runs only when a bin passes BIN_CAP, which
             * cns_engine_fold_overflows() counts. */
            const int64_t e0 = tindptr[i], e1 = tindptr[i + 1];
            for (int64_t e = e0; e < e1; e++) {
                const unsigned int src = (unsigned int)tsrc[e];
                const unsigned int bits = args[fb + k * p.words + (src >> 5)];
                if (!(bits & (1u << (src & 31u)))) continue;
                const unsigned int arr = (t0 + k + (unsigned int)delay_slot[src]) % p.n_ring;
                const size_t ri = (size_t)arr * p.n + i;
                const int32_t rp = rr.n ? rr.pos[src] : -1;
                const bool sel = rp >= 0 && (!rr.tmask || rr.tmask[i]);
                const float w = sel ? __fmul_rn(tw[e], rr.snapshot[(size_t)k * rr.n + rp])
                              : ps.n ? __fmul_rn(tw[e], ps.fac[src]) : tw[e];
                if (split) {
                    if (w > 0.0f) ring[ri] = FA(ring[ri], w);
                    else if (w < 0.0f) {
                        if (sl.n && sl.src[src]) {   /* design 2163ba7d: the listed rows' split */
                            const float fw = FM(-w, sl.keep), sw = FM(-w, sl.sfac);
                            ring_i[ri] = FA(ring_i[ri], fw);
                            const uint32_t q = (uint32_t)sl.pos[i];
                            if (sw != 0.0f) {
                                float *sr = sl.ring + (size_t)arr * sl.n + q;
                                *sr = FA(*sr, sw);
                                sl.acc_slow[q] = __dadd_rn(sl.acc_slow[q], (double)sw);
                            }
                            if (fw != 0.0f) sl.acc_fast[q] = __dadd_rn(sl.acc_fast[q], (double)fw);
                        } else ring_i[ri] = FA(ring_i[ri], -w);
                        if (hp >= 0 && ps.src9[src]) {   /* design 302c1bfb Build 8 (C1) */
                            float *q9 = ps.ring9 + (size_t)arr * ps.n + (uint32_t)hp;
                            *q9 = FA(*q9, -w);
                        }
                    }
                }
                else ring[ri] = FA(ring[ri], w);
                if (tj >= 0 && w > 0.0f && pt.dsrc[src]) {
                    const size_t dq = (size_t)arr * PTAP_ROWS + tj;
                    pt.dring[dq] = FA(pt.dring[dq], w);
                }
                if (cc.n) compass_edge(cc, src, i, arr, w);
            }
        }
    }
}

/* Graded release, first half: one thread per graded cell computes its
 * release and the per-edge scale ggain * rel. */
__global__ void k_graded(const Params p, const int32_t *graded_idx,
                         const float *v,   /* the PRE-reset membrane; see k_step */
                         const float *gv50, const float *gslope,
                         float *rel_cur, const float *ggain, float *gscale)
{
    const unsigned int tg = blockIdx.x * blockDim.x + threadIdx.x;
    if (tg >= p.n_graded) return;
    const unsigned int src = (unsigned int)graded_idx[tg];
    float x = FD(-FS(v[src], gv50[tg]), gslope[tg]);
    x = fminf(fmaxf(x, -60.0f), 60.0f);
    const float rel = FD(1.0f, FA(1.0f, expf(x)));
    rel_cur[tg] = rel;
    gscale[tg] = FM(ggain[tg], rel);
}

/* Graded release, second half: one thread per target with graded input. Its
 * row is stored slot-major, source-ascending; the thread sums weight * scale
 * over each slot's run from zero, splits the sum by sign and adds it to the
 * live ring (numpy: csr @ rel per delay group, np.maximum after). */
__global__ void k_gsum(const Params p, const int32_t *gt_rows,
                       const int64_t *gt_indptr, const int32_t *gt_gidx,
                       const float *gt_w, const int32_t *gt_slot,
                       const float *gscale, float *ring, float *ring_i,
                       const unsigned int tick, const Compass cc)
{
    const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= p.n_gt) return;
    const unsigned int tgt = (unsigned int)gt_rows[j];
    const int64_t e0 = gt_indptr[j], e1 = gt_indptr[j + 1];
    int32_t cur = gt_slot[e0];
    float x = 0.0f;
    for (int64_t e = e0; ; e++) {
        if (e == e1 || gt_slot[e] != cur) {
            if (x != 0.0f) {
                const size_t ri = (size_t)((tick + (unsigned int)cur) % p.n_ring) * p.n + tgt;
                if (p.split) { if (x > 0.0f) ring[ri] = FA(ring[ri], x); else ring_i[ri] = FA(ring_i[ri], -x); }
                else ring[ri] = FA(ring[ri], x);
                /* wm1's graded rows are outside the compass populations.
                 * The Python configuration verifies this before enabling
                 * these native source-class taps. */
                const int32_t cp = cc.n ? cc.pos[tgt] : -1;
                if (cp >= 0) {
                    const uint32_t slot = (tick + (unsigned int)cur) % p.n_ring;
                    float *row = cc.ring + ((size_t)slot * cc.n + cp) * 14;
                    const uint32_t ch = x > 0.0f ? 10 : 11;
                    row[ch] = FA(row[ch], fabsf(x));
                    if (x > 0.0f) row[12] = FA(row[12], x);
                }
            }
            if (e == e1) break;
            cur = gt_slot[e]; x = 0.0f;
        }
        const float t = FM(gt_w[e], gscale[gt_gidx[e]]);
        x = FA(x, t);
    }
}

/* The coincidence seam, one thread per designated target, once per tick
 * BEFORE k_step: two low-passed source-pool traces from last tick's spikes,
 * the slow one read back coin_lag ticks later, their product scaled by the
 * gain written where k_step adds it to the target's g. Every sum runs in
 * stored CSR order from zero, which is scipy's csr @ vector order.
 * spk_tick[src] == tick - 1 is cns.run()'s prev vector (all zero on tick 0;
 * the buffer starts at 0xFFFFFFFF so an unfired source never matches). */
__global__ void k_coin(const Params p, const int32_t *tgt,
                       const int64_t *fip, const int32_t *fsrc, const float *fw,
                       const int64_t *sip, const int32_t *ssrc, const float *sw,
                       const unsigned int *spk_tick, float *fs, float *ss,
                       float *cring, float *coin_add, float *stats,
                       const unsigned int tick)
{
    const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= p.n_coin) return;
    const bool any = tick > 0u;
    const unsigned int prev = tick - 1u;
    float df = 0.0f;
    for (int64_t e = fip[j]; e < fip[j + 1]; e++)
        if (any && spk_tick[fsrc[e]] == prev) df = FA(df, fw[e]);
    float f = FM(fs[j], p.coin_decay);
    f = FA(f, df);
    fs[j] = f;
    float ds = 0.0f;
    for (int64_t e = sip[j]; e < sip[j + 1]; e++)
        if (any && spk_tick[ssrc[e]] == prev) ds = FA(ds, sw[e]);
    float s = FM(ss[j], p.coin_decay);
    s = FA(s, ds);
    ss[j] = s;
    const size_t ri = (size_t)(tick % p.coin_lag) * p.n_coin + j;
    const float sd = cring[ri];
    cring[ri] = s;
    const float capp = FM(FM(p.coin_gain, f), sd);
    coin_add[2u * p.n + tgt[j]] = capp;   /* third section of clamp_val, where k_step reads it */
    const unsigned int nc = p.n_coin;
    stats[j] = fmaxf(stats[j], fabsf(capp));
    stats[nc + j] = FA(stats[nc + j], fabsf(capp));
    stats[2u * nc + j] = fmaxf(stats[2u * nc + j], f);
    stats[3u * nc + j] = fmaxf(stats[3u * nc + j], sd);
}

/* Design efe88957, Build 2: the seam's list form. k_coin's arithmetic with
 * each target's own triple gain, decay and lag, and each pool edge read by
 * its source's mode: a graded source by its last release over g_every
 * (k_graded's rel_cur, held between updates, which is cns.run()'s rho), any
 * other source by last tick's spike stamp. Both reads are zero on tick 0. */
__global__ void k_coin2(const Params p, const int32_t *tgt,
                        const int64_t *fip, const int32_t *fsrc, const float *fw,
                        const int32_t *fgp,
                        const int64_t *sip, const int32_t *ssrc, const float *sw,
                        const int32_t *sgp,
                        const float *gain, const float *decay, const uint32_t *lag,
                        const unsigned int *spk_tick, const float *rel_cur,
                        float *fs, float *ss, float *cring, float *coin_add,
                        float *stats, const unsigned int tick)
{
    const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= p.n_coin) return;
    const bool any = tick > 0u;
    const unsigned int prev = tick - 1u;
    const float gev = (float)p.g_every;
    float df = 0.0f;
    for (int64_t e = fip[j]; any && e < fip[j + 1]; e++) {
        if (fgp[e] >= 0) df = FA(df, FM(fw[e], FD(rel_cur[fgp[e]], gev)));
        else if (spk_tick[fsrc[e]] == prev) df = FA(df, fw[e]);
    }
    float f = FM(fs[j], decay[j]);
    f = FA(f, df);
    fs[j] = f;
    float ds = 0.0f;
    for (int64_t e = sip[j]; any && e < sip[j + 1]; e++) {
        if (sgp[e] >= 0) ds = FA(ds, FM(sw[e], FD(rel_cur[sgp[e]], gev)));
        else if (spk_tick[ssrc[e]] == prev) ds = FA(ds, sw[e]);
    }
    float s = FM(ss[j], decay[j]);
    s = FA(s, ds);
    ss[j] = s;
    const size_t ri = (size_t)(tick % lag[j]) * p.n_coin + j;
    const float sd = cring[ri];
    cring[ri] = s;
    const float capp = FM(FM(gain[j], f), sd);
    coin_add[2u * p.n + tgt[j]] = capp;
    const unsigned int nc = p.n_coin;
    stats[j] = fmaxf(stats[j], fabsf(capp));
    stats[nc + j] = FA(stats[nc + j], fabsf(capp));
    stats[2u * nc + j] = fmaxf(stats[2u * nc + j], f);
    stats[3u * nc + j] = fmaxf(stats[3u * nc + j], sd);
}

__global__ void k_reltap(const int32_t *pos, const float *rel_cur, float *reltap,
                         const unsigned int n_tap, const unsigned int kw)
{
    const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_tap) return;
    reltap[(size_t)kw * n_tap + j] = rel_cur[pos[j]];
}

__global__ void k_vtap(const int32_t *vrec, const float *v_pre, float *vtap,
                       const unsigned int n_vrec, const unsigned int kw,
                       const unsigned int n)
{
    const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_vrec) return;
    const unsigned int row = (unsigned int)vrec[j];
    const size_t out = ((size_t)kw * n_vrec + j) * 4u;
    vtap[out] = v_pre[row];
    vtap[out + 1u] = v_pre[n + row];
    vtap[out + 2u] = v_pre[2u * n + row];
    vtap[out + 3u] = v_pre[3u * n + row];
}

/* Metal reset its per-tick dispatch triples through the shared args buffer
 * with a host memset; here that would be a PCIe write per window, so a
 * one-block kernel does it instead. */
__global__ void k_reset_args(unsigned int *args, const unsigned int n_ticks,
                             const unsigned int ftrip_slot,
                             const unsigned int tcnt_slot)
{
    const unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_ticks) return;
    args[k * 3] = 0u; args[k * 3 + 1] = 1u; args[k * 3 + 2] = 1u;
    args[ftrip_slot + k * 3] = 0u; args[ftrip_slot + k * 3 + 1] = 1u;
    args[ftrip_slot + k * 3 + 2] = 1u;
    args[tcnt_slot + k] = 0u;
}

/* Compact the window's per-tick spike lists into two dense arrays, so only
 * the spikes that exist cross PCIe. Metal read the sparse list in place out
 * of unified memory; here the raw list is window_ticks x fired_cap, which is
 * 67 MB at the default 256-tick window and would dominate the run. */
__global__ void k_pack(const unsigned int *args, const unsigned int *fired,
                       const unsigned int fcap, const unsigned int n_ticks,
                       unsigned int *out_tick, int32_t *out_id)
{
    const unsigned int k = blockIdx.x;
    if (k >= n_ticks) return;
    __shared__ unsigned int base;
    if (threadIdx.x == 0) {
        unsigned int b = 0;
        for (unsigned int j = 0; j < k; j++) {
            const unsigned int c = args[j * 3];
            b += (c > fcap) ? fcap : c;
        }
        base = b;
    }
    __syncthreads();
    unsigned int c = args[k * 3];
    if (c > fcap) c = fcap;
    const unsigned int *lst = fired + (size_t)k * fcap;
    for (unsigned int s = threadIdx.x; s < c; s += blockDim.x) {
        out_tick[base + s] = k;
        out_id[base + s] = (int32_t)lst[s];
    }
}

/* ------------------------------------------------------------------- host */

/* Design 02e57559, CUDA only: one saved complete state on the device. */
#define STATE_SLOTS 2
struct StateSlot { uint8_t *d; uint64_t bytes; int full; uint64_t scalars[4]; };

struct cns_engine {
    int dev_id;
    cudaStream_t stream;
    Params P;
    uint32_t window_ticks, kgroup, n_gslots, min_delay, n_gt, grp;
    uint64_t nnz, n_gt_entries;   /* lengths cns_engine_read_buffer copies */
    float *d_vrest, *d_vreset, *d_spike, *d_decv, *d_dtt, *d_decg;
    short *d_rt;
    unsigned char *d_graded;
    float *d_aa, *d_ab, *d_dtw;
    float *d_v, *d_g, *d_gi;
    short *d_ref;
    float *d_w;
    unsigned int *d_counts;
    float *d_ring, *d_ring_i;
    unsigned int *d_cand, *d_force, *d_fired, *d_args;
    int64_t *d_indptr; int32_t *d_indices; float *d_weight; int32_t *d_delay;
    int32_t *d_gidx; float *d_gv50, *d_gslope, *d_ggain;
    int32_t *d_vrec; float *d_vpre, *d_vtap;
    float *d_relcur; int32_t *d_relpos; float *d_reltap;
    unsigned int *d_clampbits; float *d_clampval;
    unsigned int *d_ovr;
    unsigned int *d_bcount; uint2 *d_bin; unsigned int *d_touched;
    int64_t *d_tindptr; int32_t *d_tsrc; float *d_tw;
    int32_t *d_gtrows; int64_t *d_gtindptr; int32_t *d_gtgidx;
    float *d_gtw; int32_t *d_gtslot; float *d_gscale;
    int32_t *d_ctgt; int64_t *d_cfip; int32_t *d_cfsrc; float *d_cfw;
    int64_t *d_csip; int32_t *d_cssrc; float *d_csw;
    unsigned int *d_spk; float *d_cfs, *d_css, *d_cring, *d_cstats;
    /* Design efe88957, Build 2: the seam's list form, k_coin2. Per target its
     * triple's gain, decay and lag; per pool edge its source's graded
     * position, or -1 where the source is read by its spike. */
    float *d_c2gain, *d_c2decay; uint32_t *d_c2lag; int32_t *d_c2fgp, *d_c2sgp;
    uint64_t coin_fnnz, coin_snnz; int coin2;
    unsigned int *d_outtick; int32_t *d_outid;
    /* host mirrors of everything the ABI hands back as a pointer or a scalar */
    unsigned int *h_trip;    /* window_ticks * 3 */
    unsigned int *h_tcnt;    /* window_ticks */
    float *h_vtap, *h_reltap, *h_cstats;
    unsigned int h_ovr[6];
    int clamp_on, membrane_drive_on;
    uint64_t touched_total, ticks_total;
    bool uniform[8];
    bool sparse_vrest;
    double gpu_s;
    int profile;
    cudaEvent_t ev_a, ev_b;
    cudaEvent_t *ev_pool;       /* profile mode: one pair per launch, resolved after the sync */
    unsigned char *ev_cat;
    uint32_t ev_cap, ev_used;
    double kns[5];
    uint64_t kcount[5];
    char err[512];
    char devname[128];
    Resource rr;
    Compass cc;
    float *h_compass;
    Lamina lm;
    uint32_t n_lit;   /* design e2f5f784 P4-3: the length of lm.level */
    float *d_lamina_tap, *h_lamina_tap;
    /* design 02e57559, CUDA only */
    Deliv dl;
    PTap pt;
    float *h_ptap;
    int repl_on;
    StateSlot slots[STATE_SLOTS];
    Pairs pp;
    float *h_aux;
    /* design 81427d47, CUDA only */
    Occ oc;
    double *h_otap;
    uint64_t occ_a_total, occ_s_total;
    uint32_t h_aoff[OCC_TGT + 1], h_soff[OCC_TGT + 1];
    /* design 87a3ad35, CUDA only */
    Opsin op;
    float *h_optap;
    /* design 2d12a3b3, CUDA only */
    RowCur rc;
    /* design 2163ba7d (Build 2), CUDA only */
    Slow sl;
    /* design d27f82df B2 and B3, CUDA only */
    HoldDown hd;
    /* design 302c1bfb Build 8 on CUDA (design 9501604f, Build C1) */
    Psi ps;
    float *h_plog, *h_g9;
    float *h_hdtap;   /* design d27f82df B5 */
    /* design b1746728 Build 1, CUDA only */
    Ser se;
    /* design 8b6e0441 Build 2, CUDA only */
    OaOff oa;
    /* design 8b6e0441 Build 1, CUDA only */
    Ser os;
    /* gpu-7, note 10906d21: the compact junction block, k_gap_current's CSR */
    int32_t *d_gap_tgt, *d_gap_src; int64_t *d_gap_ip; float *d_gap_w, *d_gap_deg;
};

static int all_same_f(const float *a, uint32_t n) {
    for (uint32_t i = 1; i < n; i++) if (a[i] != a[0]) return 0;
    return 1;
}
static int all_same_i16(const int16_t *a, uint32_t n) {
    for (uint32_t i = 1; i < n; i++) if (a[i] != a[0]) return 0;
    return 1;
}

static void *dmalloc(size_t bytes, const void *src) {
    if (bytes == 0) bytes = 4;
    void *p = NULL;
    if (cudaMalloc(&p, bytes) != cudaSuccess) return NULL;
    if (src) {
        if (cudaMemcpy(p, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(p); return NULL;
        }
    } else if (cudaMemset(p, 0, bytes) != cudaSuccess) {
        cudaFree(p); return NULL;
    }
    return p;
}

static void seterr(char *err, size_t n, const char *msg) {
    if (err && n) { strncpy(err, msg, n - 1); err[n - 1] = 0; }
}

/* Per-kernel attribution (CNS_ENGINE_PROFILE=1): an event pair around each
 * launch, taken from a pool and resolved only after the window's single
 * stream synchronise. Nothing blocks mid-window, so unlike Metal's
 * one-encoder-per-dispatch mode this measures the wall the fast path
 * achieves, not a slower one. The first attempt did synchronise on every
 * pair and inflated its own numbers: 302 ms of measured kernel time against
 * 209 ms of fast-path GPU time on the default net, with the surplus landing
 * on whichever kernel was launched most. */
#define LAUNCH(e, cat, call) do {                                            \
    const unsigned int _i = (e)->ev_used;                                    \
    const int _rec = (e)->profile && (_i + 1u) < (e)->ev_cap;                \
    if (_rec) cudaEventRecord((e)->ev_pool[_i], (e)->stream);                \
    call;                                                                    \
    if (_rec) {                                                              \
        cudaEventRecord((e)->ev_pool[_i + 1u], (e)->stream);                 \
        (e)->ev_cat[_i >> 1] = (unsigned char)(cat);                         \
        (e)->ev_used = _i + 2u;                                              \
    }                                                                        \
} while (0)

cns_engine *cns_engine_create(const cns_engine_desc *d, char *err, size_t errlen) {
    if (d->n_gap && (d->n_gap > d->n || !d->gap_tgt || !d->gap_indptr
                     || !d->gap_src || !d->gap_weight || !d->gap_deg)) {
        seterr(err, errlen, "invalid gap-junction descriptor");
        return NULL;
    }
    if (d->n_gap && d->analytic) {
        seterr(err, errlen, "analytic integration with gap junctions lacks exact voltage parity");
        return NULL;
    }
    if (d->n_gap && (d->gap_spikelet != 0.0f || d->gap_delay || d->gap_n_ap)) {
        seterr(err, errlen, "CUDA backend: gap spikelets, delays and action-potential "
               "waveforms run on Metal only; CUDA carries the plain coupling");
        return NULL;
    }
    cns_engine *e = (cns_engine *)calloc(1, sizeof *e);
    if (!e) { seterr(err, errlen, "out of host memory"); return NULL; }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
        seterr(err, errlen, "no CUDA device visible to this process (no GPU on the "
               "node, or the job asked for none). Use the numpy backend for BOTH arms "
               "of the comparison; the engine will not switch by itself.");
        free(e); return NULL;
    }
    cudaGetDevice(&e->dev_id);
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, e->dev_id) == cudaSuccess)
        snprintf(e->devname, sizeof e->devname, "%s", prop.name);
    else
        snprintf(e->devname, sizeof e->devname, "CUDA device %d", e->dev_id);
    if (cudaStreamCreate(&e->stream) != cudaSuccess) {
        seterr(err, errlen, "cudaStreamCreate failed"); free(e); return NULL;
    }
    cudaEventCreate(&e->ev_a); cudaEventCreate(&e->ev_b);
    const char *pf = getenv("CNS_ENGINE_PROFILE");
    e->profile = (pf && pf[0] == '1') ? 1 : 0;

    const uint32_t n0 = d->n;
    bool uni[8];
    uni[0] = all_same_f(d->v_rest, n0) != 0;
    uni[1] = all_same_f(d->v_reset, n0) != 0;
    uni[2] = all_same_f(d->spike_at, n0) != 0;
    uni[3] = all_same_f(d->decay_v, n0) != 0;
    uni[4] = all_same_f(d->dt_over_tau, n0) != 0;
    uni[5] = all_same_f(d->decay_g, n0) != 0;
    uni[6] = all_same_i16(d->refrac_ticks, n0) != 0;
    uni[7] = d->use_adapt ? (all_same_f(d->adapt_a, n0) && all_same_f(d->adapt_b, n0)
                             && all_same_f(d->dt_over_tauw, n0)) : true;
    /* v_rest: uniform, or the mode value plus a bitmask of the few neurons
     * that differ (378 size-principle overrides on the champion), or dense */
    bool sparse_vr = false; float vr_mode = d->v_rest[0]; uint32_t *ovr = NULL;
    size_t words0 = ((size_t)n0 + 31) / 32;
    if (!uni[0]) {
        uint32_t ndiff = 0; vr_mode = d->v_rest[0];
        for (uint32_t i = 0; i < n0; i++) if (d->v_rest[i] != vr_mode) ndiff++;
        if (ndiff > n0 / 2) {   /* first element was an outlier: pick the majority */
            float alt = d->v_rest[0]; uint32_t best = ndiff;
            for (uint32_t i = 1; i < n0 && i < 64; i++) {
                uint32_t c = 0; for (uint32_t j = 0; j < n0; j++) if (d->v_rest[j] != d->v_rest[i]) c++;
                if (c < best) { best = c; alt = d->v_rest[i]; }
            }
            vr_mode = alt; ndiff = best;
        }
        if (ndiff <= n0 / 8) {
            sparse_vr = true;
            ovr = (uint32_t *)calloc(words0, 4);
            for (uint32_t i = 0; i < n0; i++)
                if (d->v_rest[i] != vr_mode) ovr[i >> 5] |= 1u << (i & 31u);
        }
    }
    memcpy(e->uniform, uni, sizeof uni);
    e->sparse_vrest = sparse_vr;

    const uint32_t n = d->n;
    const uint32_t words = (n + 31) / 32;
    uint32_t cap = d->fired_cap ? d->fired_cap : 65536u;
    if (cap > n) cap = n;
    e->window_ticks = d->window_ticks ? d->window_ticks : 64;

    memset(&e->P, 0, sizeof e->P);
    e->P.n = n; e->P.n_ring = d->n_ring; e->P.words = words; e->P.fired_cap = cap;
    e->P.cbased = d->cbased; e->P.split = d->split;
    e->P.analytic = d->analytic; e->P.use_adapt = d->use_adapt;
    e->P.e_exc = d->e_exc; e->P.e_inh = d->e_inh;
    e->P.gscale = d->gscale; e->P.ggain = d->ggain_scaled;
    e->P.g_every = d->g_every ? d->g_every : 1;
    e->P.n_graded = d->n_graded; e->P.n_vrec = d->n_vrec; e->P.n_reltap = d->n_reltap;
    e->P.u_vrest = vr_mode; e->P.u_vreset = d->v_reset[0];
    e->P.u_spike = d->spike_at[0]; e->P.u_decv = d->decay_v[0];
    e->P.u_dtt = d->dt_over_tau[0]; e->P.u_decg = d->decay_g[0];
    e->P.u_aa = d->use_adapt ? d->adapt_a[0] : 0.0f;
    e->P.u_ab = d->use_adapt ? d->adapt_b[0] : 0.0f;
    e->P.u_dtw = d->use_adapt ? d->dt_over_tauw[0] : 0.0f;
    e->P.u_rt = d->refrac_ticks[0];
    e->P.sparse_vrest = sparse_vr ? 1u : 0u;
    e->P.n_gap = d->n_gap; e->P.gap_scale = d->n_gap ? d->gap_scale : 0.0f;
    for (int k = 0; k < 8; k++) if (uni[k]) e->P.uni_bits |= 1u << k;

    int32_t mind = d->delay_slot[0];
    for (uint32_t i = 1; i < n; i++) if (d->delay_slot[i] < mind) mind = d->delay_slot[i];
    if (mind < 1) { seterr(err, errlen, "delay_slot must be >= 1"); cns_engine_destroy(e); return NULL; }
    {   /* k_scatter packs (arrival slot, source) into one 32-bit key: the
         * source takes as many low bits as n needs and the slot gets the rest,
         * so a long-delay body is limited by its own graph size rather than by
         * a constant. */
        uint32_t sh = 0; while ((1u << sh) < n) sh++;
        if (sh > 31 || d->n_ring > (1u << (32 - sh))) {
            char m[256];
            snprintf(m, sizeof m, "bin key packing: %u neurons need %u bits, "
                     "leaving %u for the arrival slot (%u max), but n_ring is %u",
                     n, sh, 32 - sh, 1u << (32 - sh), d->n_ring);
            seterr(err, errlen, m); cns_engine_destroy(e); return NULL;
        }
        e->P.key_shift = sh;
    }
    e->min_delay = (uint32_t)mind;
    uint32_t kg = d->kmax ? d->kmax : (uint32_t)mind;
    if (kg > (uint32_t)mind) kg = (uint32_t)mind;
    /* the seam reads last tick's spikes; the gap current reads this tick's v */
    if (d->n_graded || d->n_vrec || d->n_coin || d->n_gap) kg = 1;
    e->kgroup = kg < 1 ? 1 : kg;

    /* distinct delay slots among graded sources: k_gsum's rows are stored
     * slot-major in this order, source-ascending within a slot */
    uint32_t *gsl = NULL; e->n_gslots = 0;
    if (d->n_graded) {
        uint8_t *seen = (uint8_t *)calloc(d->n_ring, 1);
        gsl = (uint32_t *)malloc((size_t)d->n_ring * sizeof(uint32_t));
        for (uint32_t k = 0; k < d->n_graded; k++) {
            uint32_t s = (uint32_t)d->delay_slot[d->graded_idx[k]];
            if (!seen[s]) { seen[s] = 1; gsl[e->n_gslots++] = s; }
        }
        free(seen);
    }

    const size_t nf = (size_t)n * 4;
    e->P.n_coin = d->n_coin;
    e->P.coin_lag = d->coin_lag ? d->coin_lag : 1;
    e->P.coin_gain = d->coin_gain;
    e->P.coin_decay = d->coin_decay;

    e->d_vrest  = (float *)dmalloc(nf, d->v_rest);
    e->d_vreset = (float *)dmalloc(nf, d->v_reset);
    e->d_spike  = (float *)dmalloc(nf, d->spike_at);
    e->d_decv   = (float *)dmalloc(nf, d->decay_v);
    e->d_dtt    = (float *)dmalloc(nf, d->dt_over_tau);
    e->d_decg   = (float *)dmalloc(nf, d->decay_g);
    e->d_rt     = (short *)dmalloc((size_t)n * 2, d->refrac_ticks);
    e->d_graded = (unsigned char *)dmalloc((size_t)n, d->graded_mask);
    e->d_aa  = (float *)dmalloc(d->use_adapt ? nf : 0, d->use_adapt ? d->adapt_a : NULL);
    e->d_ab  = (float *)dmalloc(d->use_adapt ? nf : 0, d->use_adapt ? d->adapt_b : NULL);
    e->d_dtw = (float *)dmalloc(d->use_adapt ? nf : 0, d->use_adapt ? d->dt_over_tauw : NULL);
    e->d_v   = (float *)dmalloc(nf, d->v_rest);
    e->d_g   = (float *)dmalloc(nf, NULL);
    e->d_gi  = (float *)dmalloc(d->split ? nf : 0, NULL);
    e->d_ref = (short *)dmalloc((size_t)n * 2, NULL);
    e->d_w   = (float *)dmalloc(d->use_adapt ? nf : 0, NULL);
    e->d_counts = (unsigned int *)dmalloc(nf, NULL);
    const size_t ringb = (size_t)d->n_ring * n * 4;
    e->d_ring   = (float *)dmalloc(ringb, NULL);
    e->d_ring_i = (float *)dmalloc(d->split ? ringb : 0, NULL);

    {   /* the graph transposed: CSR by target, sources ascending (the source
         * loop below runs ascending), for k_fold's overflow path */
        int64_t *tip = (int64_t *)calloc((size_t)n + 1, sizeof(int64_t));
        int32_t *tsrc = (int32_t *)malloc((size_t)d->nnz * 4 + 4);
        float *tw = (float *)malloc((size_t)d->nnz * 4 + 4);
        int64_t *fill = (int64_t *)malloc((size_t)n * sizeof(int64_t));
        for (uint64_t q = 0; q < d->nnz; q++) tip[d->indices[q] + 1]++;
        for (uint32_t i = 0; i < n; i++) tip[i + 1] += tip[i];
        memcpy(fill, tip, (size_t)n * sizeof(int64_t));
        for (uint32_t src = 0; src < n; src++)
            for (int64_t q = d->indptr[src]; q < d->indptr[src + 1]; q++) {
                const int64_t at = fill[d->indices[q]]++;
                tsrc[at] = (int32_t)src; tw[at] = d->weight[q];
            }
        e->d_tindptr = (int64_t *)dmalloc((size_t)(n + 1) * 8, tip);
        e->d_tsrc = (int32_t *)dmalloc((size_t)d->nnz * 4, tsrc);
        e->d_tw = (float *)dmalloc((size_t)d->nnz * 4, tw);
        free(tip); free(tsrc); free(tw); free(fill);
    }
    {   /* graded gather rows: one per target with graded input; entries
         * (graded position, weight, arrival slot) slot-major in gsl order,
         * source-ascending within a slot */
        const uint32_t ng_ = d->n_graded;
        uint32_t *order = (uint32_t *)malloc((size_t)ng_ * 4 + 4);
        for (uint32_t k = 0; k < ng_; k++) order[k] = k;
        for (uint32_t a = 1; a < ng_; a++) {   /* insertion sort; normally sorted already */
            uint32_t x = order[a]; int64_t b = (int64_t)a - 1;
            while (b >= 0 && d->graded_idx[order[b]] > d->graded_idx[x]) { order[b + 1] = order[b]; b--; }
            order[b + 1] = x;
        }
        int32_t *rowid = (int32_t *)malloc((size_t)n * 4);
        for (uint32_t i = 0; i < n; i++) rowid[i] = -1;
        int64_t *cnt = (int64_t *)calloc((size_t)n, sizeof(int64_t));
        uint64_t tot = 0;
        for (uint32_t k = 0; k < ng_; k++) {
            int32_t src = d->graded_idx[k];
            for (int64_t q = d->indptr[src]; q < d->indptr[src + 1]; q++) { cnt[d->indices[q]]++; tot++; }
        }
        uint32_t ngt = 0;
        for (uint32_t i = 0; i < n; i++) if (cnt[i]) rowid[i] = (int32_t)ngt++;
        int32_t *rows = (int32_t *)malloc((size_t)ngt * 4 + 4);
        int64_t *gip = (int64_t *)calloc((size_t)ngt + 1, sizeof(int64_t));
        for (uint32_t i = 0; i < n; i++) if (cnt[i]) { rows[rowid[i]] = (int32_t)i; gip[rowid[i] + 1] = cnt[i]; }
        for (uint32_t r = 0; r < ngt; r++) gip[r + 1] += gip[r];
        int64_t *gfill = (int64_t *)malloc((size_t)ngt * sizeof(int64_t) + 8);
        memcpy(gfill, gip, (size_t)ngt * sizeof(int64_t));
        int32_t *ggidx = (int32_t *)malloc(tot * 4 + 4);
        float *gw = (float *)malloc(tot * 4 + 4);
        int32_t *gslot = (int32_t *)malloc(tot * 4 + 4);
        for (uint32_t si = 0; si < e->n_gslots; si++) {
            const int32_t slot = (int32_t)gsl[si];
            for (uint32_t a = 0; a < ng_; a++) {
                const uint32_t k = order[a];
                const int32_t src = d->graded_idx[k];
                if (d->delay_slot[src] != slot) continue;
                for (int64_t q = d->indptr[src]; q < d->indptr[src + 1]; q++) {
                    const int64_t at = gfill[rowid[d->indices[q]]]++;
                    ggidx[at] = (int32_t)k; gw[at] = d->weight[q]; gslot[at] = slot;
                }
            }
        }
        e->n_gt = ngt;
        e->n_gt_entries = tot;
        e->d_gtrows = (int32_t *)dmalloc((size_t)ngt * 4, rows);
        e->d_gtindptr = (int64_t *)dmalloc((size_t)(ngt + 1) * 8, gip);
        e->d_gtgidx = (int32_t *)dmalloc(tot * 4, ggidx);
        e->d_gtw = (float *)dmalloc(tot * 4, gw);
        e->d_gtslot = (int32_t *)dmalloc(tot * 4, gslot);
        e->d_gscale = (float *)dmalloc((size_t)ng_ * 4, NULL);
        free(order); free(rowid); free(cnt); free(rows); free(gip);
        free(gfill); free(ggidx); free(gw); free(gslot);
    }
    free(gsl);
    e->d_bcount = (unsigned int *)dmalloc(nf, NULL);
    e->d_bin = (uint2 *)dmalloc((size_t)n * BIN_CAP * 8, NULL);   /* (key, weight bits) per arrival */
    e->d_touched = (unsigned int *)dmalloc(nf, NULL);             /* targets hit this tick */
    const size_t bitsb = (size_t)e->window_ticks * words * 4;
    e->d_cand = (unsigned int *)dmalloc(bitsb, NULL);
    e->d_force = (unsigned int *)dmalloc(bitsb, NULL);
    e->d_fired = (unsigned int *)dmalloc((size_t)e->window_ticks * cap * 4, NULL);
    /* args: per-tick dispatch triples, then two parities of fire bits
     * (kgroup ticks x words) for k_fold's overflow path. The triples the
     * CUDA port does not dispatch from are kept so the layout, and every
     * offset derived from it, reads the same as the Metal file's. */
    e->P.fbits_slot = e->window_ticks * 3 + 4;
    e->P.kgroup = e->kgroup;
    e->P.ftrip_slot = e->P.fbits_slot + 2u * e->kgroup * words;
    e->P.tcnt_slot = e->P.ftrip_slot + e->window_ticks * 3;
    e->d_args = (unsigned int *)dmalloc(((size_t)e->P.tcnt_slot + e->window_ticks) * 4, NULL);
    e->d_indptr = (int64_t *)dmalloc((size_t)(n + 1) * 8, d->indptr);
    e->d_indices = (int32_t *)dmalloc((size_t)d->nnz * 4, d->indices);
    e->d_weight = (float *)dmalloc((size_t)d->nnz * 4, d->weight);
    e->nnz = d->nnz;
    e->d_delay = (int32_t *)dmalloc(nf, d->delay_slot);
    e->d_gidx = (int32_t *)dmalloc((size_t)d->n_graded * 4, d->graded_idx);
    e->d_gv50 = (float *)dmalloc((size_t)d->n_graded * 4, d->gv50);
    e->d_gslope = (float *)dmalloc((size_t)d->n_graded * 4, d->gslope);
    {   /* per-row gain: the scalar broadcast to every row when absent, so the
         * kernel's ggain[tg] * rel is the same two floats the scalar form
         * multiplied. */
        float *gg = d->n_graded ? (float *)malloc((size_t)d->n_graded * 4) : NULL;
        for (uint32_t k = 0; k < d->n_graded; k++)
            gg[k] = d->ggain_rows ? d->ggain_rows[k] : d->ggain_scaled;
        e->d_ggain = (float *)dmalloc((size_t)d->n_graded * 4, gg);
        free(gg);
    }
    e->d_vrec = (int32_t *)dmalloc((size_t)d->n_vrec * 4, d->vrec);
    e->d_vpre = (float *)dmalloc(d->n_vrec ? nf * 4 : d->n_graded ? nf : 0, NULL);
    e->d_vtap = (float *)dmalloc((size_t)d->n_vrec * e->window_ticks * 4 * 4, NULL);
    e->d_relcur = (float *)dmalloc((size_t)d->n_graded * 4, NULL);
    e->d_relpos = (int32_t *)dmalloc((size_t)d->n_reltap * 4, d->reltap_pos);
    e->d_reltap = (float *)dmalloc((size_t)d->n_reltap * e->window_ticks * 4, NULL);
    /* Packed the way the Metal file packed them, so the two files' offsets
     * read alike: bits = [graded clamp | v_rest override | membrane command |
     * coincidence target | event replacement (design 02e57559, CUDA only)],
     * values = [clamp voltage | membrane kick mV | coincidence product |
     * tonic ge | gap current (gpu-7, note 10906d21, when n_gap)]. */
    e->d_clampbits = (unsigned int *)dmalloc((size_t)words * 20, NULL);
    e->d_clampval = (float *)dmalloc(nf * (d->n_gap ? 5 : 4), NULL);
    if (d->n_gap) {
        const uint64_t gnz = (uint64_t)d->gap_indptr[d->n_gap];
        e->d_gap_tgt = (int32_t *)dmalloc((size_t)d->n_gap * 4, d->gap_tgt);
        e->d_gap_ip = (int64_t *)dmalloc(((size_t)d->n_gap + 1) * 8, d->gap_indptr);
        e->d_gap_src = (int32_t *)dmalloc((size_t)gnz * 4, gnz ? d->gap_src : NULL);
        e->d_gap_w = (float *)dmalloc((size_t)gnz * 4, gnz ? d->gap_weight : NULL);
        e->d_gap_deg = (float *)dmalloc((size_t)d->n_gap * 4, d->gap_deg);
        if (!e->d_gap_tgt || !e->d_gap_ip || !e->d_gap_src || !e->d_gap_w || !e->d_gap_deg) {
            seterr(err, errlen, "gap junctions: device allocation failed");
            free(ovr); cns_engine_destroy(e); return NULL;
        }
    }
    if (d->tonic_ge && e->d_clampval) {
        cudaMemcpy(e->d_clampval + (size_t)n * 3, d->tonic_ge, nf, cudaMemcpyHostToDevice);
        e->P.has_tonic = 1;
    }
    if (ovr && e->d_clampbits)
        cudaMemcpy(e->d_clampbits + words0, ovr, words0 * 4, cudaMemcpyHostToDevice);
    {   /* coincidence seam buffers; 4-byte placeholders when off */
        const uint32_t nc = d->n_coin;
        const size_t fnnz = nc ? (size_t)d->coin_fast_indptr[nc] : 0;
        const size_t snnz = nc ? (size_t)d->coin_slow_indptr[nc] : 0;
        e->d_ctgt = (int32_t *)dmalloc((size_t)nc * 4, nc ? d->coin_tgt : NULL);
        e->d_cfip = (int64_t *)dmalloc((size_t)(nc + 1) * 8, nc ? d->coin_fast_indptr : NULL);
        e->d_cfsrc = (int32_t *)dmalloc(fnnz * 4, fnnz ? d->coin_fast_src : NULL);
        e->d_cfw = (float *)dmalloc(fnnz * 4, fnnz ? d->coin_fast_w : NULL);
        e->d_csip = (int64_t *)dmalloc((size_t)(nc + 1) * 8, nc ? d->coin_slow_indptr : NULL);
        e->d_cssrc = (int32_t *)dmalloc(snnz * 4, snnz ? d->coin_slow_src : NULL);
        e->d_csw = (float *)dmalloc(snnz * 4, snnz ? d->coin_slow_w : NULL);
        e->coin_fnnz = fnnz; e->coin_snnz = snnz;
        e->d_spk = (unsigned int *)dmalloc(nc ? nf : 0, NULL);
        if (nc && e->d_spk) cudaMemset(e->d_spk, 0xFF, nf);   /* never fired */
        e->d_cfs = (float *)dmalloc((size_t)nc * 4, NULL);
        e->d_css = (float *)dmalloc((size_t)nc * 4, NULL);
        e->d_cring = (float *)dmalloc((size_t)e->P.coin_lag * nc * 4, NULL);
        e->d_cstats = (float *)dmalloc((size_t)nc * 16, NULL);
        if (nc && e->d_clampbits) {
            uint32_t *tm = (uint32_t *)calloc(words, 4);
            for (uint32_t j = 0; j < nc; j++) {
                uint32_t r = (uint32_t)d->coin_tgt[j];
                tm[r >> 5] |= 1u << (r & 31u);
            }
            cudaMemcpy(e->d_clampbits + (size_t)words * 3, tm, (size_t)words * 4,
                       cudaMemcpyHostToDevice);
            free(tm);
        }
        e->h_cstats = (float *)calloc(nc ? (size_t)nc * 4 : 4, sizeof(float));
    }
    /* [0] k_fold overflows, [1] largest bin seen rounded to 8,
     * [2..5] target-ticks whose bin reached 4, 8, 16, 32 */
    e->d_ovr = (unsigned int *)dmalloc(24, NULL);
    /* the window's compacted spike list; the raw per-tick list never crosses PCIe */
    e->d_outtick = (unsigned int *)dmalloc((size_t)e->window_ticks * cap * 4, NULL);
    e->d_outid = (int32_t *)dmalloc((size_t)e->window_ticks * cap * 4, NULL);
    e->P.n_gt = e->n_gt;
    free(ovr);
    e->clamp_on = 0;
    e->membrane_drive_on = 0;
    e->h_trip = (unsigned int *)calloc((size_t)e->window_ticks * 3 + 4, 4);
    e->h_tcnt = (unsigned int *)calloc((size_t)e->window_ticks + 1, 4);
    e->h_vtap = (float *)calloc((size_t)d->n_vrec * e->window_ticks * 4 + 4, 4);
    e->h_reltap = (float *)calloc((size_t)d->n_reltap * e->window_ticks + 4, 4);
    if (e->profile) {   /* worst case per window: coin, step, vtap, K x (scatter, fold), graded, reltap, gsum */
        e->ev_cap = 2u * (8u * e->window_ticks + 8u);
        e->ev_pool = (cudaEvent_t *)calloc(e->ev_cap, sizeof(cudaEvent_t));
        e->ev_cat = (unsigned char *)calloc(e->ev_cap / 2u + 1u, 1);
        for (uint32_t k = 0; e->ev_pool && k < e->ev_cap; k++)
            if (cudaEventCreate(&e->ev_pool[k]) != cudaSuccess) { e->ev_cap = k; break; }
        if (!e->ev_pool || !e->ev_cat) { e->profile = 0; e->ev_cap = 0; }
    }

    void *all[] = { e->d_vrest, e->d_vreset, e->d_spike, e->d_decv, e->d_dtt,
        e->d_decg, e->d_rt, e->d_graded, e->d_aa, e->d_ab, e->d_dtw, e->d_v,
        e->d_g, e->d_gi, e->d_ref, e->d_w, e->d_counts, e->d_ring, e->d_ring_i,
        e->d_cand, e->d_force, e->d_fired, e->d_args, e->d_indptr, e->d_indices,
        e->d_weight, e->d_delay, e->d_gidx, e->d_gv50, e->d_gslope, e->d_ggain,
        e->d_vrec, e->d_vpre, e->d_vtap, e->d_relcur, e->d_relpos, e->d_reltap,
        e->d_clampbits, e->d_clampval, e->d_ovr, e->d_ctgt, e->d_cfip,
        e->d_cfsrc, e->d_cfw, e->d_csip, e->d_cssrc, e->d_csw, e->d_spk,
        e->d_cfs, e->d_css, e->d_cring, e->d_cstats, e->d_bcount, e->d_bin,
        e->d_touched, e->d_tindptr, e->d_tsrc, e->d_tw, e->d_gtrows,
        e->d_gtindptr, e->d_gtgidx, e->d_gtw, e->d_gtslot, e->d_gscale,
        e->d_outtick, e->d_outid, e->h_trip, e->h_tcnt, e->h_vtap,
        e->h_reltap, e->h_cstats };
    for (size_t k = 0; k < sizeof all / sizeof all[0]; k++)
        if (!all[k]) {
            /* This fires when the card is already carrying other workers, not
             * when "the GPU is busy". The engine never falls back to numpy on
             * its own: a silent engine switch would put two arms of a
             * comparison on two engines, which is the one thing the fleet rule
             * forbids. */
            char m[320];
            snprintf(m, sizeof m,
                     "device allocation failed: this engine needs about %llu MB on %s "
                     "(%u ring slots x %u neurons). Run fewer workers on the card, or "
                     "use the numpy backend for BOTH arms of the comparison; the "
                     "engine will not switch by itself.",
                     (unsigned long long)(((uint64_t)d->n_ring * d->n * 4ull *
                         (d->split ? 2ull : 1ull) + (uint64_t)d->n * (BIN_CAP * 8ull + 8ull)
                         + d->nnz * 16ull) / 1000000ull),
                     e->devname, d->n_ring, d->n);
            seterr(err, errlen, m);
            cns_engine_destroy(e); return NULL;
        }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        seterr(err, errlen, cudaGetErrorString(cudaGetLastError()));
        cns_engine_destroy(e); return NULL;
    }
    return e;
}

void cns_engine_destroy(cns_engine *e) {
    if (!e) return;
    void *dev[] = { e->d_vrest, e->d_vreset, e->d_spike, e->d_decv, e->d_dtt,
        e->d_decg, e->d_rt, e->d_graded, e->d_aa, e->d_ab, e->d_dtw, e->d_v,
        e->d_g, e->d_gi, e->d_ref, e->d_w, e->d_counts, e->d_ring, e->d_ring_i,
        e->d_cand, e->d_force, e->d_fired, e->d_args, e->d_indptr, e->d_indices,
        e->d_weight, e->d_delay, e->d_gidx, e->d_gv50, e->d_gslope, e->d_ggain,
        e->d_vrec, e->d_vpre, e->d_vtap, e->d_relcur, e->d_relpos, e->d_reltap,
        e->d_clampbits, e->d_clampval, e->d_ovr, e->d_ctgt, e->d_cfip,
        e->d_cfsrc, e->d_cfw, e->d_csip, e->d_cssrc, e->d_csw, e->d_spk,
        e->d_cfs, e->d_css, e->d_cring, e->d_cstats, e->d_bcount, e->d_bin,
        e->d_touched, e->d_tindptr, e->d_tsrc, e->d_tw, e->d_gtrows,
        e->d_gtindptr, e->d_gtgidx, e->d_gtw, e->d_gtslot, e->d_gscale,
        e->d_outtick, e->d_outid, e->rr.pos, e->rr.d, e->rr.c, e->rr.keep,
        e->rr.x, e->rr.snapshot, e->rr.count, e->rr.tick, e->rr.row, e->rr.value,
        e->cc.pos, e->cc.group, e->cc.ring, e->cc.g, e->cc.tap,
        e->cc.first_tick, e->cc.snapshot, e->lm.e_inh, e->lm.mask,
        e->lm.state, e->lm.lpos, e->lm.level, e->d_lamina_tap, e->rr.tmask, e->dl.src, e->dl.tgt,
        e->dl.count, e->dl.dir_tick, e->dl.dir_edge, e->dl.dir_w,
        e->dl.ev_tick, e->dl.ev_src, e->dl.ev_pos, e->dl.ev_neg,
        e->dl.ev_nother, e->dl.ev_ndir, e->pt.dsrc, e->pt.dring, e->pt.gdir,
        e->pt.tap,
        e->pp.tgt_of, e->pp.src_first,
        e->pp.src_next, e->pp.src, e->pp.tgt, e->pp.delay, e->pp.r, e->pp.inc,
        e->pp.x, e->pp.ring, e->pp.apos, e->pp.aux,
        e->pp.clk, e->pp.res, e->pp.cnt, e->pp.area,
        e->oc.tgt_of, e->oc.src_first, e->oc.src_next, e->oc.src, e->oc.tgt,
        e->oc.delay, e->oc.quanta, e->oc.t_begin, e->oc.alpha, e->oc.gmax,
        e->oc.beta, e->oc.dec, e->oc.scale, e->oc.state, e->oc.ring,
        e->oc.a_off, e->oc.s_off, e->oc.a_count, e->oc.s_count, e->oc.a_tick,
        e->oc.a_edge, e->oc.a_n, e->oc.s_tick, e->oc.s_n, e->oc.s_val, e->oc.s_g,
        e->oc.tap,
        e->op.pos, e->op.a_on, e->op.a_off, e->op.g_max, e->op.e_rev, e->op.thr,
        e->op.force, e->op.kind, e->op.win, e->op.n_win, e->op.s, e->op.tap,
        e->op.train,
        e->rc.pos, e->rc.i0, e->rc.i1, e->rc.on, e->rc.off, e->rc.next,
        e->sl.src, e->sl.pos, e->sl.rec, e->sl.ring, e->sl.g, e->sl.acc_fast,
        e->sl.acc_slow, e->sl.intg, e->sl.rec_acc,
        e->ps.prow, e->ps.ppos, e->ps.src9, e->ps.g9, e->ps.r9, e->ps.c9,
        e->ps.ring9, e->ps.fac, e->ps.plog,
        e->hd.amask, e->hd.pos, e->hd.form, e->hd.e_rel, e->hd.decay, e->hd.dg,
        e->hd.gmax, e->hd.c_spk, e->hd.k, e->hd.va, e->hd.tau_c, e->hd.latch_mv,
        e->hd.latch_ticks, e->hd.c, e->hd.g_h, e->hd.L, e->hd.b, e->hd.peak,
        e->hd.peak_tick, e->hd.tap,
        e->se.src, e->se.tgt, e->se.s, e->se.n_tick, e->se.applied, e->se.spikes,
        e->se.bins, e->oa.tgt, e->os.src, e->os.tgt, e->os.s, e->os.n_tick,
        e->os.applied, e->os.spikes, e->os.bins, e->d_c2gain, e->d_c2decay, e->d_c2lag, e->d_c2fgp,
        e->d_c2sgp, e->d_gap_tgt, e->d_gap_ip, e->d_gap_src, e->d_gap_w, e->d_gap_deg };
    for (size_t k = 0; k < sizeof dev / sizeof dev[0]; k++) if (dev[k]) cudaFree(dev[k]);
    free(e->h_trip); free(e->h_tcnt); free(e->h_vtap); free(e->h_reltap); free(e->h_cstats);
    free(e->h_compass);
    free(e->h_lamina_tap);
    free(e->h_ptap); free(e->h_hdtap);
    for (int k = 0; k < STATE_SLOTS; k++) if (e->slots[k].d) cudaFree(e->slots[k].d);
    free(e->h_aux);
    free(e->h_otap);
    free(e->h_optap);
    free(e->h_plog); free(e->h_g9);
    if (e->ev_a) cudaEventDestroy(e->ev_a);
    if (e->ev_b) cudaEventDestroy(e->ev_b);
    for (uint32_t k = 0; e->ev_pool && k < e->ev_cap; k++)
        if (e->ev_pool[k]) cudaEventDestroy(e->ev_pool[k]);
    free(e->ev_pool); free(e->ev_cat);
    if (e->stream) cudaStreamDestroy(e->stream);
    free(e);
}

/* Design fbcd0f61: the release resource beside the release factor. Both
 * scale a spiking source's outgoing weights (k_scatter, k_fold), and their
 * order on one source is not defined (design 9501604f, Build C1), so a
 * resource row may be neither a listed row nor, with has9, a source of the
 * listed rows' state. On disjoint rows each source carries one factor: a
 * resource row's fac stays 1.0, a listed row's rr.pos is -1, and every path
 * is the one its mechanism takes alone. rpos is the host per-neuron resource
 * slot (-1 off), prow the listed rows, src9 the has9 source mask or NULL. */
static const char *rr_psi_overlap(const int32_t *rpos, uint32_t n, const int32_t *prow,
                                  uint32_t np, const uint8_t *src9) {
    for (uint32_t j = 0; j < np; j++)
        if (rpos[prow[j]] >= 0) return "a release-resource row is a listed release-factor row";
    for (uint32_t i = 0; src9 && i < n; i++)
        if (src9[i] && rpos[i] >= 0)
            return "a release-resource row is a source of the release factor's state";
    return NULL;
}

int cns_engine_resource_config(cns_engine *e, uint32_t nr,
        const int32_t *pos, const float *d, const float *c,
        const float *keep, uint32_t cap) {
    if (!nr || nr > e->P.n || !cap || e->rr.n || e->ticks_total) {
        seterr(e->err, sizeof e->err, "invalid or repeated release-resource configuration");
        return -1;
    }
    if (e->ps.n) {   /* design fbcd0f61: beside the release factor on disjoint rows only */
        const uint32_t np = e->ps.n;
        int32_t *prow = (int32_t *)malloc((size_t)np * 4);
        uint8_t *src9 = e->ps.has9 ? (uint8_t *)malloc((size_t)e->P.n) : NULL;
        if (!prow || (e->ps.has9 && !src9)) {
            free(prow); free(src9);
            seterr(e->err, sizeof e->err, "release resource: out of host memory");
            return -2;
        }
        if (cudaMemcpy(prow, e->ps.prow, (size_t)np * 4, cudaMemcpyDeviceToHost) != cudaSuccess
            || (src9 && cudaMemcpy(src9, e->ps.src9, (size_t)e->P.n,
                                   cudaMemcpyDeviceToHost) != cudaSuccess)) {
            free(prow); free(src9);
            seterr(e->err, sizeof e->err, "release resource: release-factor readback failed");
            return -2;
        }
        const char *bad = rr_psi_overlap(pos, e->P.n, prow, np, src9);
        free(prow); free(src9);
        if (bad) {
            snprintf(e->err, sizeof e->err, "release resource: %s", bad);
            return -1;
        }
    }
    Resource &r = e->rr;
    r.n = nr; r.cap = cap;
    float *ones = (float *)malloc((size_t)nr * sizeof(float));
    if (!ones) return -2;
    for (uint32_t i = 0; i < nr; i++) ones[i] = 1.0f;
    r.pos = (int32_t *)dmalloc((size_t)e->P.n * 4, pos);
    r.d = (float *)dmalloc((size_t)nr * 4, d);
    r.c = (float *)dmalloc((size_t)nr * 4, c);
    r.keep = (float *)dmalloc((size_t)nr * 4, keep);
    r.x = (float *)dmalloc((size_t)nr * 4, ones);
    free(ones);
    r.snapshot = (float *)dmalloc((size_t)nr * e->kgroup * 4, NULL);
    r.count = (uint32_t *)dmalloc(4, NULL);
    r.tick = (int32_t *)dmalloc((size_t)cap * 4, NULL);
    r.row = (int32_t *)dmalloc((size_t)cap * 4, NULL);
    r.value = (float *)dmalloc((size_t)cap * 4, NULL);
    void *all[] = {r.pos, r.d, r.c, r.keep, r.x, r.snapshot, r.count, r.tick, r.row, r.value};
    for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) if (!all[i]) {
        seterr(e->err, sizeof e->err, "release-resource device allocation failed");
        return -2;
    }
    return 0;
}

int64_t cns_engine_resource_log(cns_engine *e, int32_t *tick, int32_t *row,
        float *value, uint32_t cap) {
    if (!e->rr.n) return 0;
    uint32_t n = 0;
    if (cudaMemcpy(&n, e->rr.count, 4, cudaMemcpyDeviceToHost) != cudaSuccess) return -2;
    if (n > e->rr.cap || n > cap) {
        seterr(e->err, sizeof e->err, "release log exceeded its declared refractory bound");
        return -1;
    }
    if (cudaMemcpy(tick, e->rr.tick, (size_t)n * 4, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(row, e->rr.row, (size_t)n * 4, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(value, e->rr.value, (size_t)n * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return -2;
    return (int64_t)n;
}

int cns_engine_compass_config(cns_engine *e, uint32_t targets,
        const int32_t *pos, const uint8_t *group, int32_t cut_tick) {
    if (!targets || targets > e->P.n || e->ticks_total || e->cc.n ||
            !e->P.split || e->kgroup != 1 || e->rr.n || e->P.n_coin) {
        seterr(e->err, sizeof e->err, "invalid compass configuration"); return -2;
    }
    for (uint32_t i = 0; i < e->P.n; i++) {
        if (pos[i] < -1 || pos[i] >= (int32_t)targets || group[i] > 2 ||
                (pos[i] >= 0 && group[i] == 0)) {
            seterr(e->err, sizeof e->err, "invalid compass row mapping"); return -2;
        }
    }
    Compass &c = e->cc;
    c.n = targets; c.cut_tick = cut_tick;
    c.pos = (int32_t *)dmalloc((size_t)e->P.n * 4, pos);
    c.group = (uint8_t *)dmalloc(e->P.n, group);
    c.ring = (float *)dmalloc((size_t)e->P.n_ring * targets * 14 * 4, NULL);
    c.g = (float *)dmalloc((size_t)targets * 12 * 4, NULL);
    c.tap = (float *)dmalloc((size_t)e->window_ticks * targets * 29 * 4, NULL);
    const uint32_t unset = UINT32_MAX;
    c.first_tick = (uint32_t *)dmalloc(4, &unset);
    c.snapshot_bytes = (uint64_t)e->P.n * (e->P.use_adapt ? 22u : 18u)
        + (uint64_t)e->P.n * e->P.n_ring * 8
        + (uint64_t)targets * e->P.n_ring * 14 * 4 + (uint64_t)targets * 12 * 4;
    c.snapshot = (uint8_t *)dmalloc(c.snapshot_bytes, NULL);
    e->h_compass = (float *)calloc((size_t)e->window_ticks * targets * 29, 4);
    if (!c.pos || !c.group || !c.ring || !c.g || !c.tap || !c.first_tick || !c.snapshot || !e->h_compass) {
        seterr(e->err, sizeof e->err, "compass allocation failed"); return -2;
    }
    return 0;
}

const float *cns_engine_compass_tap(cns_engine *e) { return e->h_compass; }

uint64_t cns_engine_compass_snapshot(cns_engine *e, uint8_t *buffer,
        uint64_t capacity, uint32_t *tick) {
    if (!e->cc.n) return 0;
    uint32_t first = UINT32_MAX;
    if (cudaMemcpy(&first,e->cc.first_tick,4,cudaMemcpyDeviceToHost) != cudaSuccess) return 0;
    if (tick) *tick = first;
    if (first == UINT32_MAX) return 0;
    const uint64_t size = e->cc.snapshot_bytes;
    if (buffer && capacity >= size &&
            cudaMemcpy(buffer,e->cc.snapshot,size,cudaMemcpyDeviceToHost) != cudaSuccess) return 0;
    return size;
}

int64_t cns_engine_run(cns_engine *e, uint32_t t0, uint32_t n_ticks,
                       const uint32_t *cand_bits, const uint32_t *force_bits,
                       uint32_t *out_tick, int32_t *out_id, int64_t cap) {
    if (n_ticks == 0) return 0;
    if (n_ticks > e->window_ticks) { seterr(e->err, sizeof e->err, "n_ticks > window_ticks"); return -2; }
    const uint32_t n = e->P.n, words = e->P.words, fcap = e->P.fired_cap;
    const size_t bitsb = (size_t)n_ticks * words * 4;
    uint32_t have = 0;
    if (cand_bits) {
        cudaMemcpyAsync(e->d_cand, cand_bits, bitsb, cudaMemcpyHostToDevice, e->stream);
        have |= 1u;
    }
    if (force_bits) {
        cudaMemcpyAsync(e->d_force, force_bits, bitsb, cudaMemcpyHostToDevice, e->stream);
        have |= 2u;
    }
    if (e->clamp_on) have |= 4u;
    if (e->membrane_drive_on) have |= 8u;
    if (e->P.n_coin) have |= 16u;
    if (e->repl_on) have |= 32u;

    e->ev_used = 0;
    cudaEventRecord(e->ev_a, e->stream);
    k_reset_args<<<(n_ticks + SMALL_TG - 1) / SMALL_TG, SMALL_TG, 0, e->stream>>>(
        e->d_args, n_ticks, e->P.ftrip_slot, e->P.tcnt_slot);

    const dim3 gridN((n + STEP_TG - 1) / STEP_TG), tgN(STEP_TG);
    const uint32_t ng = e->P.n_graded, gev = e->P.g_every;
    for (uint32_t kw = 0; kw < n_ticks; kw += e->kgroup) {
        const uint32_t K = kw + e->kgroup <= n_ticks ? e->kgroup : n_ticks - kw;
        const uint32_t tick0 = t0 + kw;
        const uint32_t par = e->grp & 1u; e->grp++;
        const uint32_t have_g = have | (par << 8);
        if (e->cc.n && tick0 >= 17000) {
            k_compass_trigger<<<1,256,0,e->stream>>>(e->cc,tick0,e->P.n_ring);
            k_compass_snapshot<<<1,256,0,e->stream>>>(e->cc,tick0,n,e->P.n_ring,
                e->d_v,e->d_g,e->d_gi,e->d_w,e->d_ref,e->d_counts,
                e->d_ring,e->d_ring_i,e->P.use_adapt != 0);
        }
        if (e->P.n_gap && e->P.gap_scale != 0.0f) {   /* K == 1 whenever n_gap > 0 */
            const uint32_t ngap = e->P.n_gap;
            LAUNCH(e, 0, (k_gap_current<<<(ngap + SMALL_TG - 1) / SMALL_TG, SMALL_TG, 0, e->stream>>>(
                e->P, e->d_gap_tgt, e->d_gap_ip, e->d_gap_src, e->d_gap_w, e->d_gap_deg,
                e->d_v, e->d_clampval)));
        }
        if (e->P.n_coin) {   /* K == 1 whenever n_coin > 0: reads last tick's stamps */
            const uint32_t nc = e->P.n_coin;
            if (e->coin2)
                LAUNCH(e, 4, (k_coin2<<<(nc + SMALL_TG - 1) / SMALL_TG, SMALL_TG, 0, e->stream>>>(
                    e->P, e->d_ctgt, e->d_cfip, e->d_cfsrc, e->d_cfw, e->d_c2fgp,
                    e->d_csip, e->d_cssrc, e->d_csw, e->d_c2sgp, e->d_c2gain, e->d_c2decay,
                    e->d_c2lag, e->d_spk, e->d_relcur, e->d_cfs, e->d_css,
                    e->d_cring, e->d_clampval, e->d_cstats, tick0)));
            else
                LAUNCH(e, 4, (k_coin<<<(nc + SMALL_TG - 1) / SMALL_TG, SMALL_TG, 0, e->stream>>>(
                    e->P, e->d_ctgt, e->d_cfip, e->d_cfsrc, e->d_cfw,
                    e->d_csip, e->d_cssrc, e->d_csw, e->d_spk, e->d_cfs, e->d_css,
                    e->d_cring, e->d_clampval, e->d_cstats, tick0)));
        }
        LAUNCH(e, 0, (k_step<<<gridN, tgN, 0, e->stream>>>(
            e->P, e->d_vrest, e->d_vreset, e->d_spike, e->d_decv, e->d_dtt,
            e->d_decg, e->d_rt, e->d_graded, e->d_aa, e->d_ab, e->d_dtw,
            e->d_v, e->d_g, e->d_gi, e->d_ref, e->d_w, e->d_counts,
            e->d_ring, e->d_ring_i, e->d_cand, e->d_force, e->d_fired,
            e->d_args, tick0, K, kw, have_g, e->d_vpre,
            e->d_clampbits, e->d_clampval, e->rr, e->cc, e->lm, e->pt, e->pp, e->oc,
            e->op, e->rc, e->sl, e->hd, e->se, e->oa, e->os)));
        if (e->se.n_src)   /* K == 1 whenever se.n_src > 0 */
            LAUNCH(e, 0, (k_ser<<<1, 1, 0, e->stream>>>(e->se, tick0)));
        if (e->os.s)   /* design 8b6e0441 B1: K == 1 whenever configured */
            LAUNCH(e, 0, (k_ser<<<1, 1, 0, e->stream>>>(e->os, tick0)));
        if (e->P.n_vrec) {   /* K == 1 whenever n_vrec > 0 */
            const uint32_t nv = e->P.n_vrec;
            LAUNCH(e, 0, (k_vtap<<<(nv + SMALL_TG - 1) / SMALL_TG, SMALL_TG, 0, e->stream>>>(
                e->d_vrec, e->d_vpre, e->d_vtap, nv, kw, e->P.n)));
            if (e->lm.state)
                LAUNCH(e, 0, (k_vtap<<<(nv + SMALL_TG - 1) / SMALL_TG, SMALL_TG, 0, e->stream>>>(
                    e->d_vrec, e->lm.state, e->d_lamina_tap, nv, kw, e->P.n)));
        }
        if (e->ps.n) {   /* K == 1 whenever ps.n: this tick's factors, before the scatter */
            const uint32_t nps = e->ps.n;
            LAUNCH(e, 0, (k_psi<<<(nps + SMALL_TG - 1) / SMALL_TG, SMALL_TG, 0, e->stream>>>(
                e->P, e->ps, e->d_gi, e->d_args, tick0, par, kw)));
        }
        for (uint32_t k = 0; k < K; k++) {
            const uint32_t tick = tick0 + k;
            const uint32_t wt = kw + k;   /* window tick: the args slot both kernels use */
            LAUNCH(e, 1, (k_scatter<<<SCATTER_BLOCKS, SCATTER_TG, 0, e->stream>>>(
                e->P, e->d_fired + (size_t)wt * fcap, e->d_indptr, e->d_indices,
                e->d_weight, e->d_delay, e->d_bcount, e->d_bin, e->d_touched,
                tick, e->d_spk, e->d_ovr, e->d_args, wt, k, e->rr, e->dl, e->ps)));
            LAUNCH(e, 3, (k_fold<<<FOLD_BLOCKS, FOLD_TG, 0, e->stream>>>(
                e->P, e->d_bcount, e->d_bin, e->d_touched, e->d_ring, e->d_ring_i,
                e->d_args, e->d_tindptr, e->d_tsrc, e->d_tw, e->d_delay,
                tick0, k, par, wt, e->rr, e->cc, e->pt, e->sl, e->ps)));
        }
        if (ng && (tick0 % gev) == 0) {   /* K == 1 whenever ng > 0 */
            LAUNCH(e, 2, (k_graded<<<(ng + SMALL_TG - 1) / SMALL_TG, SMALL_TG, 0, e->stream>>>(
                e->P, e->d_gidx, e->d_vpre, e->d_gv50, e->d_gslope,
                e->d_relcur, e->d_ggain, e->d_gscale)));
            if (e->P.n_reltap) {
                const uint32_t nt_ = e->P.n_reltap;
                LAUNCH(e, 2, (k_reltap<<<(nt_ + SMALL_TG - 1) / SMALL_TG, SMALL_TG, 0, e->stream>>>(
                    e->d_relpos, e->d_relcur, e->d_reltap, nt_, kw)));
            }
            if (e->n_gt) {
                const uint32_t ngt = e->n_gt;
                LAUNCH(e, 2, (k_gsum<<<(ngt + SMALL_TG - 1) / SMALL_TG, SMALL_TG, 0, e->stream>>>(
                    e->P, e->d_gtrows, e->d_gtindptr, e->d_gtgidx, e->d_gtw,
                    e->d_gtslot, e->d_gscale, e->d_ring, e->d_ring_i, tick0, e->cc)));
            }
        }
    }
    k_pack<<<n_ticks, SMALL_TG, 0, e->stream>>>(e->d_args, e->d_fired, fcap,
                                                n_ticks, e->d_outtick, e->d_outid);
    cudaEventRecord(e->ev_b, e->stream);

    cudaMemcpyAsync(e->h_trip, e->d_args, (size_t)n_ticks * 3 * 4,
                    cudaMemcpyDeviceToHost, e->stream);
    cudaMemcpyAsync(e->h_tcnt, e->d_args + e->P.tcnt_slot, (size_t)n_ticks * 4,
                    cudaMemcpyDeviceToHost, e->stream);
    cudaMemcpyAsync(e->h_ovr, e->d_ovr, 24, cudaMemcpyDeviceToHost, e->stream);
    if (e->P.n_coin)
        cudaMemcpyAsync(e->h_cstats, e->d_cstats, (size_t)e->P.n_coin * 16,
                        cudaMemcpyDeviceToHost, e->stream);
    if (e->P.n_vrec)
        cudaMemcpyAsync(e->h_vtap, e->d_vtap,
                        (size_t)e->P.n_vrec * e->window_ticks * 4 * 4,
                        cudaMemcpyDeviceToHost, e->stream);
    if (e->lm.state)
        cudaMemcpyAsync(e->h_lamina_tap, e->d_lamina_tap,
                        (size_t)e->P.n_vrec * n_ticks * 4 * 4,
                        cudaMemcpyDeviceToHost, e->stream);
    if (e->pp.n_aux)
        cudaMemcpyAsync(e->h_aux, e->pp.aux, (size_t)e->pp.n_aux * n_ticks * AUXW * 4,
                        cudaMemcpyDeviceToHost, e->stream);
    if (e->oc.tap_on)
        cudaMemcpyAsync(e->h_otap, e->oc.tap, (size_t)e->oc.n_tgt * n_ticks * OCC_CH * 8,
                        cudaMemcpyDeviceToHost, e->stream);
    if (e->op.tap_on)
        cudaMemcpyAsync(e->h_optap, e->op.tap, (size_t)e->op.n * n_ticks * OPSIN_CH * 4,
                        cudaMemcpyDeviceToHost, e->stream);
    if (e->cc.n)
        cudaMemcpyAsync(e->h_compass, e->cc.tap,
                        (size_t)e->cc.n * n_ticks * 29 * 4,
                        cudaMemcpyDeviceToHost, e->stream);
    if (e->P.n_reltap)
        cudaMemcpyAsync(e->h_reltap, e->d_reltap,
                        (size_t)e->P.n_reltap * e->window_ticks * 4,
                        cudaMemcpyDeviceToHost, e->stream);
    if (e->pt.n)
        cudaMemcpyAsync(e->h_ptap, e->pt.tap,
                        (size_t)n_ticks * PTAP_ROWS * PTAP_CH * 4,
                        cudaMemcpyDeviceToHost, e->stream);
    if (e->hd.tap)
        cudaMemcpyAsync(e->h_hdtap, e->hd.tap, (size_t)n_ticks * e->hd.n * HD_CH * 4,
                        cudaMemcpyDeviceToHost, e->stream);
    if (e->ps.n) {   /* design 9501604f, Build C1: the window's factor log, and g9 */
        cudaMemcpyAsync(e->h_plog, e->ps.plog, (size_t)n_ticks * e->ps.n * 16,
                        cudaMemcpyDeviceToHost, e->stream);
        if (e->ps.has9)
            cudaMemcpyAsync(e->h_g9, e->ps.g9, (size_t)e->ps.n * 4,
                            cudaMemcpyDeviceToHost, e->stream);
    }
    if (cudaStreamSynchronize(e->stream) != cudaSuccess) {
        seterr(e->err, sizeof e->err, cudaGetErrorString(cudaGetLastError()));
        return -2;
    }
    if (e->rr.n) {
        uint32_t nlog = 0;
        if (cudaMemcpy(&nlog, e->rr.count, 4, cudaMemcpyDeviceToHost) != cudaSuccess) return -2;
        if (nlog > e->rr.cap) {
            seterr(e->err, sizeof e->err, "release log exceeded its declared refractory bound");
            return -4;
        }
    }
    if (e->oc.log_on) {
        uint32_t ca[OCC_TGT], cs[OCC_TGT];
        const size_t nb = (size_t)e->oc.n_tgt * 4;
        if (cudaMemcpy(ca, e->oc.a_count, nb, cudaMemcpyDeviceToHost) != cudaSuccess
                || cudaMemcpy(cs, e->oc.s_count, nb, cudaMemcpyDeviceToHost) != cudaSuccess)
            return -2;
        for (uint32_t j = 0; j < e->oc.n_tgt; j++)
            if (ca[j] > e->h_aoff[j + 1] - e->h_aoff[j] || cs[j] > e->h_soff[j + 1] - e->h_soff[j]) {
                seterr(e->err, sizeof e->err, "occupancy log exceeded its declared bound");
                return -4;
            }
    }
    {
        cudaError_t ce = cudaGetLastError();
        if (ce != cudaSuccess) { seterr(e->err, sizeof e->err, cudaGetErrorString(ce)); return -2; }
    }
    {
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, e->ev_a, e->ev_b) == cudaSuccess)
            e->gpu_s += (double)ms / 1e3;
    }
    for (uint32_t i = 0; e->profile && i + 1u < e->ev_used; i += 2u) {
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, e->ev_pool[i], e->ev_pool[i + 1u]) != cudaSuccess) continue;
        const unsigned char cat = e->ev_cat[i >> 1];
        e->kns[cat] += (double)ms * 1e6; e->kcount[cat]++;
    }

    for (uint32_t k = 0; k < n_ticks; k++) e->touched_total += e->h_tcnt[k];
    e->ticks_total += n_ticks;
    int64_t total = 0;
    for (uint32_t k = 0; k < n_ticks; k++) {
        const uint32_t c = e->h_trip[k * 3];
        if (c > fcap) { seterr(e->err, sizeof e->err, "spike list overflow"); return -3; }
        total += c;
    }
    if (total > cap) return -1;
    if (total) {
        cudaMemcpy(out_tick, e->d_outtick, (size_t)total * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(out_id, e->d_outid, (size_t)total * 4, cudaMemcpyDeviceToHost);
    }
    return total;
}

const float *cns_engine_vtap(cns_engine *e) { return e->h_vtap; }
const float *cns_engine_reltap(cns_engine *e) { return e->h_reltap; }

void cns_engine_set_clamp(cns_engine *e, const uint32_t *clamp_bits, const float *clamp_val) {
    if (!clamp_bits || !clamp_val) { e->clamp_on = 0; return; }
    cudaMemcpy(e->d_clampbits, clamp_bits, (size_t)e->P.words * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(e->d_clampval, clamp_val, (size_t)e->P.n * 4, cudaMemcpyHostToDevice);
    e->clamp_on = 1;
}

void cns_engine_set_membrane_drive(cns_engine *e, const uint32_t *row_bits,
                                   const float *kick_mv) {
    if (!row_bits || !kick_mv) { e->membrane_drive_on = 0; return; }
    cudaMemcpy(e->d_clampbits + (size_t)e->P.words * 2, row_bits,
               (size_t)e->P.words * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(e->d_clampval + (size_t)e->P.n, kick_mv,
               (size_t)e->P.n * 4, cudaMemcpyHostToDevice);
    e->membrane_drive_on = 1;
}

void cns_engine_read_state(cns_engine *e, float *v, float *g, float *g_i,
                           float *w_adapt, uint32_t *counts) {
    const size_t nf = (size_t)e->P.n * 4;
    if (v) cudaMemcpy(v, e->d_v, nf, cudaMemcpyDeviceToHost);
    if (g) cudaMemcpy(g, e->d_g, nf, cudaMemcpyDeviceToHost);
    if (g_i && e->P.split) cudaMemcpy(g_i, e->d_gi, nf, cudaMemcpyDeviceToHost);
    if (w_adapt && e->P.use_adapt) cudaMemcpy(w_adapt, e->d_w, nf, cudaMemcpyDeviceToHost);
    if (counts) cudaMemcpy(counts, e->d_counts, nf, cudaMemcpyDeviceToHost);
}

double cns_engine_gpu_seconds(cns_engine *e) { return e->gpu_s; }

/* CUDA only, no header entry (design 0c25fab3): copy one device-resident
 * graph buffer, as the kernels read it, into out. 0 weight (nnz float32, CSC
 * by source), 1 indptr (n+1 int64), 2 indices (nnz int32), 3 delay_slot (n
 * int32), 4 transposed weight (nnz float32, CSR by target), 5 transposed
 * source (nnz int32), 6 transposed indptr (n+1 int64), 7 graded gather weight
 * (float32), 8 graded gather target rows (int32), 9 graded gather indptr
 * (int64), 10 graded gather source position (int32), 11 graded gather arrival
 * slot (int32). Design 7b0ad6b2 adds the per-neuron arrays and state the
 * kernels read: 12 decay_g, 13 refractory counter (int16), 14-20 the pair
 * channel's source, target, delay (uint32), decay, increment, state (float32)
 * and event ring (nr x pairs, uint32), 21 v_rest, 22 v_reset, 23 spike_at,
 * 24 decay_v, 25 dt_over_tau, 26 refractory ticks (int16), 27 adapt_a,
 * 28 adapt_b, 29 dt_over_tauw, 30 inhibitory state, 31 excitatory ring and
 * 32 inhibitory ring (n_ring x n float32), 33 membrane, 34 excitatory state,
 * 35 adaptation, 36 spike counts (uint32); 37 to 47 are design
 * 02e57559's and 48 to 71 design 81427d47's (see the cases). Returns the byte length, and with
 * out NULL only the length; -1 unknown buffer, -2 capacity too small, -3
 * device error. */
extern "C" int64_t cns_engine_read_buffer(cns_engine *e, uint32_t which, void *out,
                                          uint64_t capacity) {
    const uint64_t n = e->P.n, nnz = e->nnz, ngt = e->n_gt, tot = e->n_gt_entries;
    const uint64_t np = e->pp.n, ua = e->P.use_adapt ? 1 : 0, sp = e->P.split ? 1 : 0;
    const uint64_t ringn = (uint64_t)e->P.n_ring * n;
    const uint64_t ne = e->oc.n_edge, nt = e->oc.n_edge ? e->oc.n_tgt : 0;
    const void *src = NULL; uint64_t bytes = 0;
    switch (which) {
    case 0: src = e->d_weight; bytes = nnz * 4; break;
    case 1: src = e->d_indptr; bytes = (n + 1) * 8; break;
    case 2: src = e->d_indices; bytes = nnz * 4; break;
    case 3: src = e->d_delay; bytes = n * 4; break;
    case 4: src = e->d_tw; bytes = nnz * 4; break;
    case 5: src = e->d_tsrc; bytes = nnz * 4; break;
    case 6: src = e->d_tindptr; bytes = (n + 1) * 8; break;
    case 7: src = e->d_gtw; bytes = tot * 4; break;
    case 8: src = e->d_gtrows; bytes = ngt * 4; break;
    case 9: src = e->d_gtindptr; bytes = (ngt + 1) * 8; break;
    case 10: src = e->d_gtgidx; bytes = tot * 4; break;
    case 11: src = e->d_gtslot; bytes = tot * 4; break;
    case 12: src = e->d_decg; bytes = n * 4; break;
    case 13: src = e->d_ref; bytes = n * 2; break;
    case 14: src = e->pp.src; bytes = np * 4; break;
    case 15: src = e->pp.tgt; bytes = np * 4; break;
    case 16: src = e->pp.delay; bytes = np * 4; break;
    case 17: src = e->pp.r; bytes = np * 4; break;
    case 18: src = e->pp.inc; bytes = np * 4; break;
    case 19: src = e->pp.x; bytes = np * 4; break;
    case 20: src = e->pp.ring; bytes = (uint64_t)e->pp.nr * np * 4; break;
    case 21: src = e->d_vrest; bytes = n * 4; break;
    case 22: src = e->d_vreset; bytes = n * 4; break;
    case 23: src = e->d_spike; bytes = n * 4; break;
    case 24: src = e->d_decv; bytes = n * 4; break;
    case 25: src = e->d_dtt; bytes = n * 4; break;
    case 26: src = e->d_rt; bytes = n * 2; break;
    case 27: src = e->d_aa; bytes = ua * n * 4; break;
    case 28: src = e->d_ab; bytes = ua * n * 4; break;
    case 29: src = e->d_dtw; bytes = ua * n * 4; break;
    case 30: src = e->d_gi; bytes = sp * n * 4; break;
    case 31: src = e->d_ring; bytes = ringn * 4; break;
    case 32: src = e->d_ring_i; bytes = sp * ringn * 4; break;
    case 33: src = e->d_v; bytes = n * 4; break;
    case 34: src = e->d_g; bytes = n * 4; break;
    case 35: src = e->d_w; bytes = ua * n * 4; break;
    case 36: src = e->d_counts; bytes = n * 4; break;
    /* design 02e57559: the resource, replacement, recorder and tap
     * selectors as the kernels read them; zero bytes when unconfigured */
    case 37: src = e->rr.pos; bytes = e->rr.n ? n * 4 : 0; break;
    case 38: src = e->rr.tmask; bytes = e->rr.tmask ? n : 0; break;
    case 39: src = e->rr.d; bytes = (uint64_t)e->rr.n * 4; break;
    case 40: src = e->rr.c; bytes = (uint64_t)e->rr.n * 4; break;
    case 41: src = e->rr.keep; bytes = (uint64_t)e->rr.n * 4; break;
    case 42: src = e->d_clampbits + (size_t)e->P.words * 4; bytes = e->repl_on ? (uint64_t)e->P.words * 4 : 0; break;
    case 43: src = e->dl.src; bytes = e->dl.n_src ? n : 0; break;
    case 44: src = e->dl.tgt; bytes = e->dl.n_src ? n : 0; break;
    case 45: src = e->pt.dsrc; bytes = e->pt.n ? n : 0; break;
    case 46: src = e->rr.x; bytes = (uint64_t)e->rr.n * 4; break;
    case 47: src = e->pt.gdir; bytes = e->pt.n ? PTAP_ROWS * 4 : 0; break;
    /* design 81427d47: the occupancy channel and its recorder as the kernels
     * read them; zero bytes when unconfigured */
    case 48: src = e->oc.src; bytes = ne * 4; break;
    case 49: src = e->oc.tgt; bytes = ne * 4; break;
    case 50: src = e->oc.delay; bytes = ne * 4; break;
    case 51: src = e->oc.quanta; bytes = ne * 8; break;
    case 52: src = e->oc.t_begin; bytes = nt ? (nt + 1) * 4 : 0; break;
    case 53: src = e->oc.alpha; bytes = nt * 8; break;
    case 54: src = e->oc.gmax; bytes = nt * 8; break;
    case 55: src = e->oc.beta; bytes = nt * 8; break;
    case 56: src = e->oc.dec; bytes = nt * 8; break;
    case 57: src = e->oc.scale; bytes = nt * 8; break;
    case 58: src = e->oc.state; bytes = nt * 8; break;
    case 59: src = e->oc.ring; bytes = (uint64_t)e->oc.nr * ne * 4; break;
    case 60: src = e->oc.a_count; bytes = e->oc.log_on ? nt * 4 : 0; break;
    case 61: src = e->oc.s_count; bytes = e->oc.log_on ? nt * 4 : 0; break;
    case 62: src = e->oc.a_tick; bytes = e->occ_a_total * 4; break;
    case 63: src = e->oc.a_edge; bytes = e->occ_a_total * 4; break;
    case 64: src = e->oc.a_n; bytes = e->occ_a_total * 4; break;
    case 65: src = e->oc.s_tick; bytes = e->occ_s_total * 4; break;
    case 66: src = e->oc.s_n; bytes = e->occ_s_total * 4; break;
    case 67: src = e->oc.s_val; bytes = e->occ_s_total * 32; break;
    case 68: src = e->oc.s_g; bytes = e->occ_s_total * 4; break;
    case 69: src = e->oc.tgt_of; bytes = ne ? n * 4 : 0; break;
    case 70: src = e->oc.src_first; bytes = ne ? n * 4 : 0; break;
    case 71: src = e->oc.src_next; bytes = ne * 4; break;
    /* design e05c0eef B1: the pair recovery's state and counters */
    case 72: src = e->pp.clk; bytes = np * 4; break;
    case 73: src = e->pp.res; bytes = np * 4; break;
    case 74: src = e->pp.cnt; bytes = np * 16; break;
    case 75: src = e->pp.area; bytes = np * 4; break;
    default: return -1;
    }
    if (!out) return (int64_t)bytes;
    if (capacity < bytes) return -2;
    if (!bytes) return 0;
    if (cudaStreamSynchronize(e->stream) != cudaSuccess
            || cudaMemcpy(out, src, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        snprintf(e->err, sizeof e->err, "cns_engine_read_buffer %u: %s", which,
                 cudaGetErrorString(cudaGetLastError()));
        return -3;
    }
    return (int64_t)bytes;
}

/* per-kernel seconds (step, scatter, graded+gsum+reltap, fold, coin) and
 * launch counts; zeros unless CNS_ENGINE_PROFILE=1 */
void cns_engine_profile(cns_engine *e, double *secs, uint64_t *counts) {
    for (int k = 0; k < 5; k++) {
        if (secs) secs[k] = e->kns[k] / 1e9;
        if (counts) counts[k] = e->kcount[k];
    }
}

void cns_engine_coin_stats(cns_engine *e, double *out4) {
    const uint32_t nc = e->P.n_coin;
    const float *s = e->h_cstats;
    double mx = 0.0, sm = 0.0, fm = 0.0, sd = 0.0;
    for (uint32_t j = 0; nc && j < nc; j++) {
        if (s[j] > mx) mx = s[j];
        sm += s[nc + j];
        if (s[2u * nc + j] > fm) fm = s[2u * nc + j];
        if (s[3u * nc + j] > sd) sd = s[3u * nc + j];
    }
    out4[0] = mx; out4[1] = sm; out4[2] = fm; out4[3] = sd;
}

uint32_t cns_engine_kgroup(cns_engine *e) { return e->kgroup; }

/* gpu-7, note 10906d21: the junction law on CUDA is the case Metal runs on
 * k_gap_current, the one-way law (2) at one substep with no tap, whose
 * coupling is the G the descriptor already carries. The pair checks are
 * Metal's; the conserving law, substeps and the tap run on Metal only. */
int cns_engine_gap_law_config(cns_engine *e, uint32_t law, uint32_t pairs,
        const int32_t *ends, const float *weights, float rho,
        uint32_t steps, uint32_t tap) {
    if (!e || !e->P.n_gap || e->ticks_total || (law != 1 && law != 2)
            || !pairs || !ends || !weights || !steps || tap > 1
            || !isfinite(rho) || rho < 0 || rho > 1) return -1;
    for (uint32_t k = 0; k < pairs; k++)
        if (ends[2*k] < 0 || (uint32_t)ends[2*k] >= e->P.n_gap
                || ends[2*k+1] < 0 || (uint32_t)ends[2*k+1] >= e->P.n_gap
                || ends[2*k] == ends[2*k+1] || !isfinite(weights[k]) || weights[k] < 0) return -1;
    if (law != 2 || steps != 1 || tap) {
        seterr(e->err, sizeof e->err, "CUDA backend: the conserving gap law, electrical "
               "substeps and the gap current tap run on Metal only");
        return -3;
    }
    return 0;
}
const float *cns_engine_gap_tap(cns_engine *e) { (void)e; return NULL; }
uint32_t cns_engine_fold_overflows(cns_engine *e) { return e->h_ovr[0]; }
uint32_t cns_engine_bin_cap(void) { return BIN_CAP; }
uint32_t cns_engine_bin_max(cns_engine *e) { return e->h_ovr[1]; }
void cns_engine_bin_hist(cns_engine *e, uint32_t *out4) {
    for (int q = 0; q < 4; q++) out4[q] = e->h_ovr[2 + q];
}
double cns_engine_touched_per_tick(cns_engine *e) {
    return e->ticks_total ? (double)e->touched_total / (double)e->ticks_total : 0.0;
}
uint32_t cns_engine_uniform_mask(cns_engine *e) {
    uint32_t m = 0;
    for (int k = 0; k < 8; k++) if (e->uniform[k]) m |= 1u << k;
    if (e->sparse_vrest) m |= 1u << 8;
    return m;
}
const char *cns_engine_device(cns_engine *e) { return e->devname; }
#ifndef CNS_ENGINE_SRC_SHA256
#define CNS_ENGINE_SRC_SHA256 "unset: built without the Makefile"
#endif
const char *cns_engine_source_sha256(void) { return CNS_ENGINE_SRC_SHA256; }
const char *cns_engine_last_error(cns_engine *e) { return e->err; }

/* Separate optional ABI: old callers and the Metal parameter struct are unchanged.
 * lpos (design e2f5f784 P4-3): per row, the row's position in the level vector
 * or -1; NULL keeps the scalar schedule. n_lit is the level vector's length. */
extern "C" int cns_engine_lamina_config(cns_engine *e, const float *e_inh,
        const unsigned char *mask, float gain, uint32_t on, uint32_t off,
        uint32_t ramp, int record, const int32_t *lpos, uint32_t n_lit) {
    if (e->lm.e_inh || e->lm.mask || e->lm.state || e->lm.lpos) return -1;
    if (!e->P.cbased || (record && !e->P.n_vrec)) return -2;
    if (lpos && (!mask || n_lit == 0)) return -4;
    if (e_inh) e->lm.e_inh = (float *)dmalloc((size_t)e->P.n*4, e_inh);
    if (mask) e->lm.mask = (unsigned char *)dmalloc(e->P.n, mask);
    if (lpos) {
        e->lm.lpos = (int32_t *)dmalloc((size_t)e->P.n*4, lpos);
        e->lm.level = (float *)dmalloc((size_t)n_lit*4, NULL);
        e->n_lit = n_lit;
    }
    e->lm.gain=gain; e->lm.on=on; e->lm.off=off; e->lm.ramp=ramp;
    if (record) {
        size_t bytes=(size_t)e->P.n_vrec*e->window_ticks*16;
        e->lm.state=(float *)dmalloc((size_t)e->P.n*16, NULL);
        e->d_lamina_tap=(float *)dmalloc(bytes, NULL);
        e->h_lamina_tap=(float *)calloc(1,bytes);
    }
    if ((e_inh && !e->lm.e_inh) || (mask && !e->lm.mask) ||
        (lpos && (!e->lm.lpos || !e->lm.level)) ||
        (record && (!e->lm.state || !e->d_lamina_tap || !e->h_lamina_tap))) return -3;
    return 0;
}
/* Design e2f5f784 P4-3: the level of every lit position, held until the next
 * call. Ordered on the engine's stream, so it takes effect from the next window. */
extern "C" int cns_engine_lamina_level(cns_engine *e, const float *level) {
    if (!e->lm.level) return -1;
    return cudaMemcpyAsync(e->lm.level, level, (size_t)e->n_lit*4,
                           cudaMemcpyHostToDevice, e->stream) == cudaSuccess ? 0 : -2;
}
extern "C" const float *cns_engine_lamina_tap(cns_engine *e) { return e->h_lamina_tap; }
extern "C" int cns_engine_inh_readback(cns_engine *e, float *out) {
    if (e->lm.e_inh)
        return cudaMemcpy(out,e->lm.e_inh,(size_t)e->P.n*4,cudaMemcpyDeviceToHost)==cudaSuccess ? 0 : -1;
    for (uint32_t i=0;i<e->P.n;i++) out[i]=e->P.e_inh;
    return 0;
}
extern "C" int cns_engine_dt_readback(cns_engine *e, float *out) {
    return cudaMemcpy(out,e->d_dtt,(size_t)e->P.n*4,cudaMemcpyDeviceToHost)==cudaSuccess ? 0 : -1;
}

/* ---------------------------------------------------------------------------
 * Design 02e57559, CUDA only, no header entry: the Mac library and its source
 * stamp are unchanged, and the Metal driver refuses every option below.
 * Every function is optional configuration made before the first run, or a
 * read; an engine that calls none of them runs the pre-existing path.
 */

/* Restrict the emitted release snapshot to marked target rows (one byte per
 * row). Call after cns_engine_resource_config and before the first run. */
extern "C" int cns_engine_resource_targets(cns_engine *e, const uint8_t *mask) {
    if (!e->rr.n || e->rr.tmask || e->ticks_total || !mask) {
        seterr(e->err, sizeof e->err, "resource targets need one configured resource, once, before any run");
        return -1;
    }
    uint32_t marked = 0;
    for (uint32_t i = 0; i < e->P.n; i++) marked += mask[i] ? 1u : 0u;
    if (!marked) { seterr(e->err, sizeof e->err, "resource target mask marks no row"); return -1; }
    e->rr.tmask = (uint8_t *)dmalloc(e->P.n, mask);
    if (!e->rr.tmask) { seterr(e->err, sizeof e->err, "resource target allocation failed"); return -2; }
    return 0;
}

/* Complete event replacement for marked rows (bit words, the clamp layout):
 * a marked row fires only on its forced events. NULL clears. */
extern "C" int cns_engine_set_replacement(cns_engine *e, const uint32_t *bits) {
    if (!bits) { e->repl_on = 0; return 0; }
    if (cudaMemcpy(e->d_clampbits + (size_t)e->P.words * 4, bits, (size_t)e->P.words * 4,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        seterr(e->err, sizeof e->err, "replacement upload failed"); return -2;
    }
    e->repl_on = 1;
    return 0;
}

extern "C" int cns_engine_deliv_config(cns_engine *e, const uint8_t *src, const uint8_t *tgt,
                                       uint32_t cap_direct, uint32_t cap_event) {
    if (e->dl.n_src || e->ticks_total || !src || !tgt || !cap_direct || !cap_event) {
        seterr(e->err, sizeof e->err, "delivery recorder: invalid or repeated configuration");
        return -1;
    }
    uint32_t ns = 0, nt = 0;
    for (uint32_t i = 0; i < e->P.n; i++) { ns += src[i] ? 1u : 0u; nt += tgt[i] ? 1u : 0u; }
    if (!ns || !nt) { seterr(e->err, sizeof e->err, "delivery recorder marks no source or no target"); return -1; }
    Deliv &d = e->dl;
    d.cap_direct = cap_direct; d.cap_event = cap_event;
    d.src = (uint8_t *)dmalloc(e->P.n, src);
    d.tgt = (uint8_t *)dmalloc(e->P.n, tgt);
    d.count = (uint32_t *)dmalloc(8, NULL);
    d.dir_tick = (int32_t *)dmalloc((size_t)cap_direct * 4, NULL);
    d.dir_edge = (int64_t *)dmalloc((size_t)cap_direct * 8, NULL);
    d.dir_w = (float *)dmalloc((size_t)cap_direct * 4, NULL);
    d.ev_tick = (int32_t *)dmalloc((size_t)cap_event * 4, NULL);
    d.ev_src = (int32_t *)dmalloc((size_t)cap_event * 4, NULL);
    d.ev_pos = (double *)dmalloc((size_t)cap_event * 8, NULL);
    d.ev_neg = (double *)dmalloc((size_t)cap_event * 8, NULL);
    d.ev_nother = (uint32_t *)dmalloc((size_t)cap_event * 4, NULL);
    d.ev_ndir = (uint32_t *)dmalloc((size_t)cap_event * 4, NULL);
    void *all[] = {d.src, d.tgt, d.count, d.dir_tick, d.dir_edge, d.dir_w, d.ev_tick, d.ev_src,
                   d.ev_pos, d.ev_neg, d.ev_nother, d.ev_ndir};
    for (size_t k = 0; k < sizeof all / sizeof all[0]; k++) if (!all[k]) {
        seterr(e->err, sizeof e->err, "delivery recorder allocation failed"); return -2;
    }
    d.n_src = ns;
    return 0;
}

/* counts[0] direct entries, counts[1] events, as appended (either may exceed
 * its capacity, which is an error for the caller to report). */
extern "C" int cns_engine_deliv_counts(cns_engine *e, uint32_t *counts) {
    counts[0] = counts[1] = 0;
    if (!e->dl.n_src) return 0;
    if (cudaMemcpy(counts, e->dl.count, 8, cudaMemcpyDeviceToHost) != cudaSuccess) return -3;
    return 0;
}

/* The logs in append order (the order within a tick is the atomics'; the
 * caller sorts). -1 a log exceeded its capacity, -2 caller capacity too
 * small, -3 device error. */
extern "C" int cns_engine_deliv_read(cns_engine *e,
        int32_t *dir_tick, int64_t *dir_edge, float *dir_w, uint32_t cap_d,
        int32_t *ev_tick, int32_t *ev_src, double *ev_pos, double *ev_neg,
        uint32_t *ev_nother, uint32_t *ev_ndir, uint32_t cap_e) {
    uint32_t c[2];
    if (cns_engine_deliv_counts(e, c)) return -3;
    const Deliv &d = e->dl;
    if (c[0] > d.cap_direct || c[1] > d.cap_event) {
        seterr(e->err, sizeof e->err, "delivery log exceeded its capacity"); return -1;
    }
    if (c[0] > cap_d || c[1] > cap_e) return -2;
    bool ok = true;
    if (c[0]) ok = ok && cudaMemcpy(dir_tick, d.dir_tick, (size_t)c[0] * 4, cudaMemcpyDeviceToHost) == cudaSuccess
                   && cudaMemcpy(dir_edge, d.dir_edge, (size_t)c[0] * 8, cudaMemcpyDeviceToHost) == cudaSuccess
                   && cudaMemcpy(dir_w, d.dir_w, (size_t)c[0] * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
    if (c[1]) ok = ok && cudaMemcpy(ev_tick, d.ev_tick, (size_t)c[1] * 4, cudaMemcpyDeviceToHost) == cudaSuccess
                   && cudaMemcpy(ev_src, d.ev_src, (size_t)c[1] * 4, cudaMemcpyDeviceToHost) == cudaSuccess
                   && cudaMemcpy(ev_pos, d.ev_pos, (size_t)c[1] * 8, cudaMemcpyDeviceToHost) == cudaSuccess
                   && cudaMemcpy(ev_neg, d.ev_neg, (size_t)c[1] * 8, cudaMemcpyDeviceToHost) == cudaSuccess
                   && cudaMemcpy(ev_nother, d.ev_nother, (size_t)c[1] * 4, cudaMemcpyDeviceToHost) == cudaSuccess
                   && cudaMemcpy(ev_ndir, d.ev_ndir, (size_t)c[1] * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
    return ok ? 0 : -3;
}

/* Dense state of up to PTAP_ROWS rows, sixteen channels per native tick (see
 * k_step); dsrc marks the sources whose positive arrivals feed the direct
 * shadow conductance. Needs the split conductance-based membrane. */
extern "C" int cns_engine_ptap_config(cns_engine *e, uint32_t n, const int32_t *rows,
                                      const uint8_t *dsrc) {
    if (e->pt.n || e->ticks_total || !n || n > PTAP_ROWS || !rows || !dsrc
            || !e->P.cbased || !e->P.split) {
        seterr(e->err, sizeof e->err, "state tap: invalid or repeated configuration");
        return -1;
    }
    for (uint32_t q = 0; q < n; q++) {
        if (rows[q] < 0 || (uint32_t)rows[q] >= e->P.n) {
            seterr(e->err, sizeof e->err, "state tap row out of range"); return -1;
        }
        for (uint32_t r = 0; r < q; r++) if (rows[r] == rows[q]) {
            seterr(e->err, sizeof e->err, "state tap row repeated"); return -1;
        }
    }
    PTap &t = e->pt;
    for (uint32_t q = 0; q < PTAP_ROWS; q++) t.row[q] = q < n ? rows[q] : -1;
    t.dsrc = (uint8_t *)dmalloc(e->P.n, dsrc);
    t.dring = (float *)dmalloc((size_t)e->P.n_ring * PTAP_ROWS * 4, NULL);
    t.gdir = (float *)dmalloc(PTAP_ROWS * 4, NULL);
    t.tap = (float *)dmalloc((size_t)e->window_ticks * PTAP_ROWS * PTAP_CH * 4, NULL);
    e->h_ptap = (float *)calloc((size_t)e->window_ticks * PTAP_ROWS * PTAP_CH, 4);
    if (!t.dsrc || !t.dring || !t.gdir || !t.tap || !e->h_ptap) {
        seterr(e->err, sizeof e->err, "state tap allocation failed"); return -2;
    }
    t.n = n;
    return 0;
}

/* The last window's tap, window_ticks x PTAP_ROWS x PTAP_CH float32. */
extern "C" const float *cns_engine_ptap(cns_engine *e) { return e->h_ptap; }

/* Complete dynamic state, for branching executions from a shared prefix.
 * Sections in a fixed order; the host form is a 64-byte header of eight
 * uint64 (magic, version, section count, grp, ticks_total, touched_total,
 * flags, total bytes) followed by the sections back to back. Per-window
 * scratch that every tick rewrites before it reads (spike bins, touched
 * lists, fired lists, candidate and forced bits, output lists, taps) is not
 * state and is not saved. The compass, lamina recorder and coincidence seam
 * carry state of their own and are refused, as is design 7b0ad6b2's pair
 * channel. */
#define STATE_MAGIC 0x3232353565373230ull   /* the bytes of "02e57522" */
#define STATE_MAX_SECTIONS 40
static uint32_t state_sections(cns_engine *e, void **ptr, uint64_t *bytes, uint32_t cap) {
    const uint64_t n = e->P.n, nf = n * 4, words = e->P.words;
    const uint64_t ringb = (uint64_t)e->P.n_ring * n * 4;
    const Resource &r = e->rr; const Deliv &d = e->dl; const PTap &t = e->pt;
    uint32_t k = 0;
#define SEC(p_, b_) do { if (k < cap) { ptr[k] = (void *)(p_); bytes[k] = (uint64_t)(b_); } k++; } while (0)
    SEC(e->d_v, nf); SEC(e->d_g, nf); SEC(e->d_gi, e->P.split ? nf : 0);
    SEC(e->d_ref, n * 2); SEC(e->d_w, e->P.use_adapt ? nf : 0); SEC(e->d_counts, nf);
    SEC(e->d_ring, ringb); SEC(e->d_ring_i, e->P.split ? ringb : 0);
    SEC(e->d_vpre, e->P.n_vrec ? nf * 4 : e->P.n_graded ? nf : 0);
    SEC(e->d_relcur, (uint64_t)e->P.n_graded * 4); SEC(e->d_gscale, (uint64_t)e->P.n_graded * 4);
    SEC(e->d_ovr, 24); SEC(e->d_bcount, nf);
    SEC(e->d_args, ((uint64_t)e->P.tcnt_slot + e->window_ticks) * 4);
    SEC(e->d_clampbits, words * 20); SEC(e->d_clampval, nf * 4);
    SEC(r.x, (uint64_t)r.n * 4); SEC(r.snapshot, (uint64_t)r.n * e->kgroup * 4);
    SEC(r.count, r.n ? 4 : 0); SEC(r.tick, (uint64_t)r.cap * 4);
    SEC(r.row, (uint64_t)r.cap * 4); SEC(r.value, (uint64_t)r.cap * 4);
    SEC(d.count, d.n_src ? 8 : 0); SEC(d.dir_tick, (uint64_t)d.cap_direct * 4);
    SEC(d.dir_edge, (uint64_t)d.cap_direct * 8); SEC(d.dir_w, (uint64_t)d.cap_direct * 4);
    SEC(d.ev_tick, (uint64_t)d.cap_event * 4); SEC(d.ev_src, (uint64_t)d.cap_event * 4);
    SEC(d.ev_pos, (uint64_t)d.cap_event * 8); SEC(d.ev_neg, (uint64_t)d.cap_event * 8);
    SEC(d.ev_nother, (uint64_t)d.cap_event * 4); SEC(d.ev_ndir, (uint64_t)d.cap_event * 4);
    SEC(t.dring, t.n ? (uint64_t)e->P.n_ring * PTAP_ROWS * 4 : 0);
    SEC(t.gdir, t.n ? PTAP_ROWS * 4 : 0);
#undef SEC
    return k;
}

static int state_refused(cns_engine *e) {
    if (e->cc.n || e->P.n_coin || e->lm.state || e->pp.n || e->op.n || e->rc.n || e->sl.n
            || e->ps.n || e->hd.n || e->se.n_src || e->os.s) {
        seterr(e->err, sizeof e->err,
               "state save does not cover the compass, the lamina recorder, the "
               "coincidence seam, the pair channel, the opsin channel, the held current, "
               "the slow inhibitory pool, the release factor, the hold-down, the "
               "serotonin state or the octopamine state");
        return 1;
    }
    return 0;
}

/* Design 81427d47: save and load do not carry the occupancy state, ring or
 * recorder, so both are refused while the channel is configured. The live
 * form stays readable, with flag bit 3 set; the channel's own state is read
 * through cns_engine_read_buffer. */
static int occ_refused(cns_engine *e) {
    if (e->oc.n_edge) {
        seterr(e->err, sizeof e->err, "state save and load do not cover the occupancy channel");
        return 1;
    }
    return 0;
}

static uint64_t state_payload(cns_engine *e, void **ptr, uint64_t *bytes, uint32_t *ns) {
    *ns = state_sections(e, ptr, bytes, STATE_MAX_SECTIONS);
    uint64_t tot = 0;
    for (uint32_t k = 0; k < *ns && k < STATE_MAX_SECTIONS; k++) tot += bytes[k];
    return tot;
}

static void state_scalars(cns_engine *e, uint64_t *out4) {
    out4[0] = e->grp; out4[1] = e->ticks_total; out4[2] = e->touched_total;
    out4[3] = (uint64_t)(e->clamp_on ? 1 : 0) | ((uint64_t)(e->membrane_drive_on ? 1 : 0) << 1)
            | ((uint64_t)(e->repl_on ? 1 : 0) << 2) | ((uint64_t)(e->oc.n_edge ? 1 : 0) << 3);
}

/* Section count and byte lengths in their fixed order; returns the count. */
extern "C" uint32_t cns_engine_state_sections(cns_engine *e, uint64_t *bytes, uint32_t cap) {
    void *ptr[STATE_MAX_SECTIONS]; uint64_t b[STATE_MAX_SECTIONS];
    const uint32_t ns = state_sections(e, ptr, b, STATE_MAX_SECTIONS);
    for (uint32_t k = 0; k < ns && k < cap; k++) bytes[k] = b[k];
    return ns;
}

/* Bytes of the host form: the 64-byte header and every section. */
extern "C" uint64_t cns_engine_state_bytes(cns_engine *e) {
    void *ptr[STATE_MAX_SECTIONS]; uint64_t b[STATE_MAX_SECTIONS]; uint32_t ns;
    return 64 + state_payload(e, ptr, b, &ns);
}

/* Copy the live state into device slot `slot` (0 or 1), with the host
 * scalars the next window reads: the group parity, the tick and touched
 * totals and the clamp, membrane and replacement flags. */
extern "C" int cns_engine_state_save(cns_engine *e, uint32_t slot) {
    if (slot >= STATE_SLOTS || state_refused(e) || occ_refused(e)) return -1;
    void *ptr[STATE_MAX_SECTIONS]; uint64_t b[STATE_MAX_SECTIONS]; uint32_t ns;
    const uint64_t tot = state_payload(e, ptr, b, &ns);
    if (ns > STATE_MAX_SECTIONS) { seterr(e->err, sizeof e->err, "state section table overflow"); return -1; }
    StateSlot &s = e->slots[slot];
    if (s.d && s.bytes != tot) { cudaFree(s.d); s.d = NULL; }
    if (!s.d) {
        if (cudaMalloc((void **)&s.d, tot ? tot : 4) != cudaSuccess) {
            s.d = NULL; seterr(e->err, sizeof e->err, "state slot allocation failed"); return -2;
        }
        s.bytes = tot;
    }
    uint64_t at = 0;
    for (uint32_t k = 0; k < ns; k++) {
        if (b[k] && cudaMemcpyAsync(s.d + at, ptr[k], b[k], cudaMemcpyDeviceToDevice, e->stream) != cudaSuccess) {
            seterr(e->err, sizeof e->err, "state save copy failed"); return -3;
        }
        at += b[k];
    }
    if (cudaStreamSynchronize(e->stream) != cudaSuccess) {
        seterr(e->err, sizeof e->err, cudaGetErrorString(cudaGetLastError())); return -3;
    }
    state_scalars(e, s.scalars);
    s.full = 1;
    return 0;
}

/* Replace the live state with device slot `slot`. The slot must have been
 * saved by this engine, whose configuration fixes every section length. */
extern "C" int cns_engine_state_load(cns_engine *e, uint32_t slot) {
    if (slot >= STATE_SLOTS || state_refused(e) || occ_refused(e)) return -1;
    StateSlot &s = e->slots[slot];
    void *ptr[STATE_MAX_SECTIONS]; uint64_t b[STATE_MAX_SECTIONS]; uint32_t ns;
    const uint64_t tot = state_payload(e, ptr, b, &ns);
    if (!s.full || s.bytes != tot) {
        seterr(e->err, sizeof e->err, "state load: slot empty or saved under another configuration");
        return -1;
    }
    uint64_t at = 0;
    for (uint32_t k = 0; k < ns; k++) {
        if (b[k] && cudaMemcpyAsync(ptr[k], s.d + at, b[k], cudaMemcpyDeviceToDevice, e->stream) != cudaSuccess) {
            seterr(e->err, sizeof e->err, "state load copy failed"); return -3;
        }
        at += b[k];
    }
    if (cudaStreamSynchronize(e->stream) != cudaSuccess) {
        seterr(e->err, sizeof e->err, cudaGetErrorString(cudaGetLastError())); return -3;
    }
    e->grp = (uint32_t)s.scalars[0]; e->ticks_total = s.scalars[1]; e->touched_total = s.scalars[2];
    e->clamp_on = (int)(s.scalars[3] & 1u); e->membrane_drive_on = (int)((s.scalars[3] >> 1) & 1u);
    e->repl_on = (int)((s.scalars[3] >> 2) & 1u);
    return 0;
}

/* The host form of the live state (slot < 0) or of a saved slot, for
 * hashing and byte comparison. Returns the bytes written, -2 when cap is
 * too small, -1 refused or empty, -3 device error. */
extern "C" int64_t cns_engine_state_read(cns_engine *e, int32_t slot, uint8_t *out, uint64_t cap) {
    if (state_refused(e)) return -1;
    void *ptr[STATE_MAX_SECTIONS]; uint64_t b[STATE_MAX_SECTIONS]; uint32_t ns;
    const uint64_t tot = state_payload(e, ptr, b, &ns);
    if (cap < 64 + tot) return -2;
    uint64_t hdr[8];
    hdr[0] = STATE_MAGIC; hdr[1] = 1; hdr[2] = ns;
    if (slot < 0) {
        state_scalars(e, hdr + 3);
        if (cudaStreamSynchronize(e->stream) != cudaSuccess) return -3;
        uint64_t at = 64;
        for (uint32_t k = 0; k < ns; k++) {
            if (b[k] && cudaMemcpy(out + at, ptr[k], b[k], cudaMemcpyDeviceToHost) != cudaSuccess) return -3;
            at += b[k];
        }
    } else {
        if (slot >= STATE_SLOTS || !e->slots[slot].full || e->slots[slot].bytes != tot) {
            seterr(e->err, sizeof e->err, "state read: slot empty or saved under another configuration");
            return -1;
        }
        memcpy(hdr + 3, e->slots[slot].scalars, 4 * 8);
        if (tot && cudaMemcpy(out + 64, e->slots[slot].d, tot, cudaMemcpyDeviceToHost) != cudaSuccess) return -3;
    }
    hdr[7] = 64 + tot;
    memcpy(out, hdr, 64);
    return (int64_t)(64 + tot);
}
/* Design 7b0ad6b2 (H01), CUDA only, no header entry. np pairs (src[q],
 * tgt[q]), each with its own transport delay[q] in ticks, decay r[q] per tick
 * and increment inc[q] per arriving event. Refused: a repeated or late
 * configuration; the current-based law or unsplit inhibition; a row out of
 * range; a repeated target; a graded source or target; a delay outside
 * [kgroup, nr - kgroup], so that every arrival is read by a later launch than
 * the one that wrote it and its slot is not one the writing launch still
 * reads; r outside [0, 1); inc not finite or negative; and a pair
 * whose ordinary weight in the uploaded graph is not exactly zero, because the
 * channel replaces that contribution and must never duplicate it. */
extern "C" int cns_engine_pair_config(cns_engine *e, uint32_t np, uint32_t nr,
        const int32_t *src, const int32_t *tgt, const uint32_t *delay,
        const float *r, const float *inc) {
    const uint32_t n = e->P.n;
    const char *bad = NULL;
    if (e->pp.n || e->ticks_total) bad = "pair channel: repeated or late configuration";
    else if (e->oc.n_edge) bad = "pair channel: the occupancy channel is configured on this engine";
    else if (!np || nr < 2) bad = "pair channel: no pairs, or a ring under two slots";
    else if (!e->P.cbased || !e->P.split)
        bad = "pair channel: needs the conductance law with split inhibition";
    if (bad) { seterr(e->err, sizeof e->err, bad); return -1; }
    unsigned char *graded = (unsigned char *)malloc(n);
    int32_t *tgt_of = (int32_t *)malloc((size_t)n * 4);
    int32_t *src_first = (int32_t *)malloc((size_t)n * 4);
    int32_t *src_next = (int32_t *)malloc((size_t)np * 4);
    if (!graded || !tgt_of || !src_first || !src_next) {
        free(graded); free(tgt_of); free(src_first); free(src_next);
        seterr(e->err, sizeof e->err, "pair channel: out of host memory"); return -2;
    }
    if (cudaMemcpy(graded, e->d_graded, n, cudaMemcpyDeviceToHost) != cudaSuccess)
        bad = "pair channel: graded mask readback failed";
    for (uint32_t i = 0; i < n; i++) tgt_of[i] = src_first[i] = -1;
    for (uint32_t q = 0; q < np && !bad; q++) {
        if (src[q] < 0 || (uint32_t)src[q] >= n || tgt[q] < 0 || (uint32_t)tgt[q] >= n)
            bad = "pair channel: row out of range";
        else if (tgt_of[tgt[q]] >= 0) bad = "pair channel: repeated target";
        else if (graded[src[q]] || graded[tgt[q]]) bad = "pair channel: graded source or target";
        else if (delay[q] < e->kgroup || delay[q] + e->kgroup > nr)
            bad = "pair channel: delay outside [kgroup, nr - kgroup]";
        else if (!(r[q] >= 0.0f && r[q] < 1.0f)) bad = "pair channel: decay outside [0, 1)";
        else if (!isfinite(inc[q]) || !(inc[q] >= 0.0f))
            bad = "pair channel: increment not finite and nonnegative";
        else tgt_of[tgt[q]] = (int32_t)q;
    }
    for (uint32_t q = 0; q < np && !bad; q++) {   /* the ordinary weight must already be zero */
        int64_t ip[2];
        if (cudaMemcpy(ip, e->d_indptr + src[q], 16, cudaMemcpyDeviceToHost) != cudaSuccess) {
            bad = "pair channel: graph readback failed"; break;
        }
        const size_t m = (size_t)(ip[1] - ip[0]);
        int32_t *idx = (int32_t *)malloc(m * 4 + 4);
        float *w = (float *)malloc(m * 4 + 4);
        if (!idx || !w
                || cudaMemcpy(idx, e->d_indices + ip[0], m * 4, cudaMemcpyDeviceToHost) != cudaSuccess
                || cudaMemcpy(w, e->d_weight + ip[0], m * 4, cudaMemcpyDeviceToHost) != cudaSuccess)
            bad = "pair channel: graph readback failed";
        for (size_t j = 0; !bad && j < m; j++)
            if (idx[j] == tgt[q] && w[j] != 0.0f)
                bad = "pair channel: the pair's ordinary weight is not zero (it would be delivered twice)";
        free(idx); free(w);
    }
    if (!bad)   /* per-source lists, ascending in q */
        for (uint32_t q = np; q-- > 0;) { src_next[q] = src_first[src[q]]; src_first[src[q]] = (int32_t)q; }
    if (bad) {
        free(graded); free(tgt_of); free(src_first); free(src_next);
        seterr(e->err, sizeof e->err, bad); return -1;
    }
    Pairs &pp = e->pp;
    pp.tgt_of = (int32_t *)dmalloc((size_t)n * 4, tgt_of);
    pp.src_first = (int32_t *)dmalloc((size_t)n * 4, src_first);
    pp.src_next = (int32_t *)dmalloc((size_t)np * 4, src_next);
    pp.src = (int32_t *)dmalloc((size_t)np * 4, src);
    pp.tgt = (int32_t *)dmalloc((size_t)np * 4, tgt);
    pp.delay = (uint32_t *)dmalloc((size_t)np * 4, delay);
    pp.r = (float *)dmalloc((size_t)np * 4, r);
    pp.inc = (float *)dmalloc((size_t)np * 4, inc);
    pp.x = (float *)dmalloc((size_t)np * 4, NULL);
    pp.ring = (uint32_t *)dmalloc((size_t)nr * np * 4, NULL);
    free(graded); free(tgt_of); free(src_first); free(src_next);
    float *ones = (float *)malloc((size_t)np * 4);
    if (ones) for (uint32_t q = 0; q < np; q++) ones[q] = 1.0f;
    pp.clk = (uint32_t *)dmalloc((size_t)np * 4, NULL);
    pp.res = ones ? (float *)dmalloc((size_t)np * 4, ones) : NULL;
    pp.cnt = (uint32_t *)dmalloc((size_t)np * 16, NULL);
    pp.area = (float *)dmalloc((size_t)np * 4, NULL);
    free(ones);
    if (!pp.tgt_of || !pp.src_first || !pp.src_next || !pp.src || !pp.tgt || !pp.delay
            || !pp.r || !pp.inc || !pp.x || !pp.ring || !pp.clk || !pp.res || !pp.cnt || !pp.area) {
        seterr(e->err, sizeof e->err, "pair channel: device allocation failed"); return -2;
    }
    pp.n = np; pp.nr = nr;
    pp.rform = 0u; pp.rticks = 0u; pp.rdecay = 0.0f; pp.rkeep = 1.0f;
    return 0;
}

/* Design e05c0eef B1, CUDA and Metal: the configured pair channel's recovery,
 * form 1 (dead time, rticks) or 2 (depletion, rdecay = exp(-dt/tau_rec) and
 * rkeep = 1 - u). Refused: no pair channel, a repeated or late configuration,
 * another form, rdecay or rkeep outside [0, 1]. Unconfigured, every pair
 * runs the pre-B1 arithmetic. */
extern "C" int cns_engine_pair_recovery_config(cns_engine *e, uint32_t form, uint32_t rticks,
        float rdecay, float rkeep) {
    const char *bad = NULL;
    if (!e->pp.n) bad = "pair recovery: no pair channel is configured";
    else if (e->pp.rform || e->ticks_total) bad = "pair recovery: repeated or late configuration";
    else if (form != 1u && form != 2u) bad = "pair recovery: form must be 1 (dead time) or 2 (depletion)";
    else if (!(rdecay >= 0.0f && rdecay <= 1.0f) || !(rkeep >= 0.0f && rkeep <= 1.0f))
        bad = "pair recovery: decay or keep outside [0, 1]";
    if (bad) { seterr(e->err, sizeof e->err, bad); return -1; }
    e->pp.rform = form; e->pp.rticks = rticks; e->pp.rdecay = rdecay; e->pp.rkeep = rkeep;
    return 0;
}

/* Per-tick state of n_aux named rows, AUXW floats each, window-major:
 * 0 membrane at tick start, 1 membrane after integration and the refractory
 * clamp (the value tested at threshold, before reset), 2 ordinary excitatory
 * arrival, 3 ordinary inhibitory arrival (both in weight units), 4 ordinary
 * excitatory state after arrival, 5 inhibitory state, 6 pair events arriving,
 * 7 pair conductance used, 8 ordinary normalized excitatory conductance used
 * (tonic included), 9 normalized inhibitory conductance used, 10-12 pair,
 * total excitatory and inhibitory drive g*(E - V) at the starting membrane,
 * 13 adaptation and 14 refractory counter at tick start, 15 flags (see k_step). */
extern "C" int cns_engine_aux_config(cns_engine *e, uint32_t n_aux, const int32_t *rows) {
    const uint32_t n = e->P.n;
    if (e->pp.n_aux || e->ticks_total || !n_aux || n_aux > n) {
        seterr(e->err, sizeof e->err, "state tap: repeated, late or empty configuration"); return -1;
    }
    int32_t *apos = (int32_t *)malloc((size_t)n * 4);
    if (!apos) return -2;
    for (uint32_t i = 0; i < n; i++) apos[i] = -1;
    for (uint32_t j = 0; j < n_aux; j++) {
        if (rows[j] < 0 || (uint32_t)rows[j] >= n || apos[rows[j]] >= 0) {
            free(apos);
            seterr(e->err, sizeof e->err, "state tap: row out of range or repeated"); return -1;
        }
        apos[rows[j]] = (int32_t)j;
    }
    const size_t bytes = (size_t)e->window_ticks * n_aux * AUXW * 4;
    e->pp.apos = (int32_t *)dmalloc((size_t)n * 4, apos);
    e->pp.aux = (float *)dmalloc(bytes, NULL);
    e->h_aux = (float *)calloc(1, bytes);
    free(apos);
    if (!e->pp.apos || !e->pp.aux || !e->h_aux) {
        seterr(e->err, sizeof e->err, "state tap: allocation failed"); return -2;
    }
    e->pp.n_aux = n_aux;
    return 0;
}
extern "C" const float *cns_engine_aux_tap(cns_engine *e) { return e->h_aux; }
extern "C" uint32_t cns_engine_aux_width(void) { return AUXW; }

/* Design 81427d47 (H03), CUDA only, no header entry. n_edge connections
 * (src[q], tgt[q]), sorted by target index and then by source row, the
 * targets being tgt_rows[0..n_tgt) in that order. A connection's arrival
 * weight is quanta[q] * scale[j] for its target j. mode 1 (T) reads alpha and
 * gmax, mode 2 (L) beta; dec is either state's decay per tick; interrupt 1
 * withholds every arrival. Refused: a repeated or late configuration; the
 * pair channel on the same engine; the current-based law or unsplit
 * inhibition; no connection, more than OCC_TGT targets, or a target with no
 * connection; a row out of range, a repeated target, a connection whose
 * target is unlisted or out of order, or a repeated connection; a graded
 * source or target; a delay other than the source's own delivery slot, or
 * one outside [kgroup, nr - kgroup], so that every arrival is read by a later
 * launch than the one that wrote it; a scale that is not a positive power of
 * two, or a target whose largest possible tick sum reaches 2^53 quanta, since
 * the sum is then no longer exact in double; alpha, gmax or beta not finite
 * and positive where the mode reads them; dec outside [0, 1); and a
 * connection that is not exactly one explicit zero in the uploaded graph,
 * because the state replaces that contribution and must never duplicate it. */
extern "C" int cns_engine_occ_config(cns_engine *e, uint32_t n_edge, uint32_t nr, uint32_t mode,
        uint32_t interrupt, const int32_t *src, const int32_t *tgt, const uint32_t *delay,
        const uint64_t *quanta, uint32_t n_tgt, const int32_t *tgt_rows,
        const double *alpha, const double *gmax, const double *beta,
        const double *dec, const double *scale) {
    const uint32_t n = e->P.n;
    const char *bad = NULL;
    if (e->oc.n_edge || e->ticks_total) bad = "occupancy: repeated or late configuration";
    else if (e->pp.n) bad = "occupancy: the pair channel is configured on this engine";
    else if (!n_edge || !n_tgt || n_tgt > OCC_TGT || nr < 2)
        bad = "occupancy: no connection, too many targets, or a ring under two slots";
    else if (mode != 1u && mode != 2u) bad = "occupancy: mode is 1 (occupancy) or 2 (linear)";
    else if (interrupt > 1u) bad = "occupancy: interrupt is 0 or 1";
    else if (!e->P.cbased || !e->P.split)
        bad = "occupancy: needs the conductance law with split inhibition";
    if (bad) { seterr(e->err, sizeof e->err, bad); return -1; }
    unsigned char *graded = (unsigned char *)malloc(n);
    int32_t *dslot = (int32_t *)malloc((size_t)n * 4);
    int32_t *tgt_of = (int32_t *)malloc((size_t)n * 4);
    int32_t *src_first = (int32_t *)malloc((size_t)n * 4);
    int32_t *src_next = (int32_t *)malloc((size_t)n_edge * 4);
    uint32_t t_begin[OCC_TGT + 1] = {0};
    uint64_t qsum[OCC_TGT] = {0};
    if (!graded || !dslot || !tgt_of || !src_first || !src_next) {
        free(graded); free(dslot); free(tgt_of); free(src_first); free(src_next);
        seterr(e->err, sizeof e->err, "occupancy: out of host memory"); return -2;
    }
    if (cudaMemcpy(graded, e->d_graded, n, cudaMemcpyDeviceToHost) != cudaSuccess
            || cudaMemcpy(dslot, e->d_delay, (size_t)n * 4, cudaMemcpyDeviceToHost) != cudaSuccess)
        bad = "occupancy: graded mask or delay readback failed";
    for (uint32_t i = 0; i < n; i++) tgt_of[i] = src_first[i] = -1;
    for (uint32_t j = 0; j < n_tgt && !bad; j++) {
        int ex = 0;
        if (tgt_rows[j] < 0 || (uint32_t)tgt_rows[j] >= n) bad = "occupancy: target row out of range";
        else if (tgt_of[tgt_rows[j]] >= 0) bad = "occupancy: repeated target";
        else if (graded[tgt_rows[j]]) bad = "occupancy: graded target";
        else if (!(isfinite(scale[j]) && scale[j] > 0.0 && frexp(scale[j], &ex) == 0.5))
            bad = "occupancy: scale is not a positive power of two";
        else if (!(isfinite(dec[j]) && dec[j] >= 0.0 && dec[j] < 1.0))
            bad = "occupancy: decay outside [0, 1)";
        else if (mode == 1u && !(isfinite(alpha[j]) && alpha[j] > 0.0 && isfinite(gmax[j]) && gmax[j] > 0.0))
            bad = "occupancy: alpha or gmax not finite and positive";
        else if (mode == 2u && !(isfinite(beta[j]) && beta[j] > 0.0))
            bad = "occupancy: beta not finite and positive";
        else tgt_of[tgt_rows[j]] = (int32_t)j;
    }
    int32_t prev_j = -1, prev_src = -1;
    for (uint32_t q = 0; q < n_edge && !bad; q++) {
        if (src[q] < 0 || (uint32_t)src[q] >= n || tgt[q] < 0 || (uint32_t)tgt[q] >= n) {
            bad = "occupancy: connection row out of range"; break;
        }
        const int32_t j = tgt_of[tgt[q]];
        if (j < 0) bad = "occupancy: connection onto an unlisted target";
        else if (j < prev_j || (j == prev_j && src[q] <= prev_src))
            bad = "occupancy: connections not sorted by target then source, or repeated";
        else if (graded[src[q]]) bad = "occupancy: graded source";
        else if (delay[q] != (uint32_t)dslot[src[q]])
            bad = "occupancy: delay differs from the source's delivery slot";
        else if (delay[q] < e->kgroup || delay[q] + e->kgroup > nr)
            bad = "occupancy: delay outside [kgroup, nr - kgroup]";
        else if (quanta[q] == 0u || quanta[q] >= (1ull << 53) - qsum[j])
            bad = "occupancy: zero quanta, or a tick sum that could reach 2^53";
        else {
            qsum[j] += quanta[q];
            t_begin[j + 1]++;
            prev_j = j; prev_src = src[q];
        }
    }
    for (uint32_t j = 0; j < n_tgt && !bad; j++) {
        if (!t_begin[j + 1]) bad = "occupancy: a target with no connection";
        t_begin[j + 1] += t_begin[j];
    }
    for (uint32_t q = 0; q < n_edge && !bad; q++) {   /* exactly one explicit zero in the graph */
        int64_t ip[2];
        if (cudaMemcpy(ip, e->d_indptr + src[q], 16, cudaMemcpyDeviceToHost) != cudaSuccess) {
            bad = "occupancy: graph readback failed"; break;
        }
        const size_t m = (size_t)(ip[1] - ip[0]);
        int32_t *idx = (int32_t *)malloc(m * 4 + 4);
        float *w = (float *)malloc(m * 4 + 4);
        uint32_t found = 0;
        if (!idx || !w
                || cudaMemcpy(idx, e->d_indices + ip[0], m * 4, cudaMemcpyDeviceToHost) != cudaSuccess
                || cudaMemcpy(w, e->d_weight + ip[0], m * 4, cudaMemcpyDeviceToHost) != cudaSuccess)
            bad = "occupancy: graph readback failed";
        for (size_t k = 0; !bad && k < m; k++)
            if (idx[k] == tgt[q]) {
                found++;
                if (w[k] != 0.0f)
                    bad = "occupancy: the connection's ordinary weight is not zero (it would be delivered twice)";
            }
        if (!bad && found != 1u) bad = "occupancy: the connection is not exactly one entry of the graph";
        free(idx); free(w);
    }
    if (!bad)   /* per-source lists, ascending in q */
        for (uint32_t q = n_edge; q-- > 0;) { src_next[q] = src_first[src[q]]; src_first[src[q]] = (int32_t)q; }
    if (bad) {
        free(graded); free(dslot); free(tgt_of); free(src_first); free(src_next);
        seterr(e->err, sizeof e->err, bad); return -1;
    }
    Occ &oc = e->oc;
    oc.tgt_of = (int32_t *)dmalloc((size_t)n * 4, tgt_of);
    oc.src_first = (int32_t *)dmalloc((size_t)n * 4, src_first);
    oc.src_next = (int32_t *)dmalloc((size_t)n_edge * 4, src_next);
    oc.src = (int32_t *)dmalloc((size_t)n_edge * 4, src);
    oc.tgt = (int32_t *)dmalloc((size_t)n_edge * 4, tgt);
    oc.delay = (uint32_t *)dmalloc((size_t)n_edge * 4, delay);
    oc.quanta = (uint64_t *)dmalloc((size_t)n_edge * 8, quanta);
    oc.t_begin = (uint32_t *)dmalloc((size_t)(n_tgt + 1) * 4, t_begin);
    oc.alpha = (double *)dmalloc((size_t)n_tgt * 8, alpha);
    oc.gmax = (double *)dmalloc((size_t)n_tgt * 8, gmax);
    oc.beta = (double *)dmalloc((size_t)n_tgt * 8, beta);
    oc.dec = (double *)dmalloc((size_t)n_tgt * 8, dec);
    oc.scale = (double *)dmalloc((size_t)n_tgt * 8, scale);
    oc.state = (double *)dmalloc((size_t)n_tgt * 8, NULL);
    oc.ring = (uint32_t *)dmalloc((size_t)nr * n_edge * 4, NULL);
    free(graded); free(dslot); free(tgt_of); free(src_first); free(src_next);
    if (!oc.tgt_of || !oc.src_first || !oc.src_next || !oc.src || !oc.tgt || !oc.delay
            || !oc.quanta || !oc.t_begin || !oc.alpha || !oc.gmax || !oc.beta || !oc.dec
            || !oc.scale || !oc.state || !oc.ring) {
        seterr(e->err, sizeof e->err, "occupancy: device allocation failed"); return -2;
    }
    oc.n_edge = n_edge; oc.n_tgt = n_tgt; oc.nr = nr; oc.mode = mode; oc.interrupt = interrupt;
    return 0;
}

/* The occupancy recorder, after cns_engine_occ_config and before the first
 * run: with log_on, a_cap[j] arrival entries (tick, connection, count) and
 * s_cap[j] updates (tick, arrivals, A, state before decay, after decay, after
 * the arrivals, delivered conductance) per target, each target appending to
 * its own region in tick order; with tap_on, OCC_CH doubles per target per
 * native tick of every window (see k_step), read after each run. */
extern "C" int cns_engine_occ_record_config(cns_engine *e, uint32_t log_on, const uint32_t *a_cap,
                                            const uint32_t *s_cap, uint32_t tap_on) {
    Occ &oc = e->oc;
    if (!oc.n_edge || e->ticks_total || oc.log_on || oc.tap_on || log_on > 1u || tap_on > 1u) {
        seterr(e->err, sizeof e->err, "occupancy recorder: unconfigured channel, late or repeated");
        return -1;
    }
    if (log_on) {
        uint64_t at = 0, st = 0;
        e->h_aoff[0] = e->h_soff[0] = 0;
        for (uint32_t j = 0; j < oc.n_tgt; j++) {
            at += a_cap[j]; st += s_cap[j];
            if (at >= (1ull << 31) || st >= (1ull << 31)) {
                seterr(e->err, sizeof e->err, "occupancy recorder: capacity exceeds int32");
                return -1;
            }
            e->h_aoff[j + 1] = (uint32_t)at; e->h_soff[j + 1] = (uint32_t)st;
        }
        oc.a_off = (uint32_t *)dmalloc((size_t)(oc.n_tgt + 1) * 4, e->h_aoff);
        oc.s_off = (uint32_t *)dmalloc((size_t)(oc.n_tgt + 1) * 4, e->h_soff);
        oc.a_count = (uint32_t *)dmalloc((size_t)oc.n_tgt * 4, NULL);
        oc.s_count = (uint32_t *)dmalloc((size_t)oc.n_tgt * 4, NULL);
        oc.a_tick = (int32_t *)dmalloc(at * 4, NULL);
        oc.a_edge = (uint32_t *)dmalloc(at * 4, NULL);
        oc.a_n = (uint32_t *)dmalloc(at * 4, NULL);
        oc.s_tick = (int32_t *)dmalloc(st * 4, NULL);
        oc.s_n = (uint32_t *)dmalloc(st * 4, NULL);
        oc.s_val = (double *)dmalloc(st * 32, NULL);
        oc.s_g = (float *)dmalloc(st * 4, NULL);
        if (!oc.a_off || !oc.s_off || !oc.a_count || !oc.s_count || !oc.a_tick || !oc.a_edge
                || !oc.a_n || !oc.s_tick || !oc.s_n || !oc.s_val || !oc.s_g) {
            seterr(e->err, sizeof e->err, "occupancy recorder: allocation failed"); return -2;
        }
        e->occ_a_total = at; e->occ_s_total = st;
        oc.log_on = 1u;
    }
    if (tap_on) {
        const size_t bytes = (size_t)e->window_ticks * oc.n_tgt * OCC_CH * 8;
        oc.tap = (double *)dmalloc(bytes, NULL);
        e->h_otap = (double *)calloc(1, bytes);
        if (!oc.tap || !e->h_otap) {
            seterr(e->err, sizeof e->err, "occupancy recorder: tap allocation failed"); return -2;
        }
        oc.tap_on = 1u;
    }
    return 0;
}
extern "C" const double *cns_engine_occ_tap(cns_engine *e) { return e->h_otap; }
extern "C" uint32_t cns_engine_occ_width(void) { return OCC_CH; }

/* Design 87a3ad35 (B2), CUDA only, no header entry. n_slot slots in the slot
 * order of cns.py's opsin_arrays: rows[q] is the slot's network row; a_on,
 * a_off, g_max, e_rev and thr are its float32 gate factors, peak conductance,
 * reversal and arm-E threshold; force and kind are 0 or 1 (kind 1 is a
 * threshold-window slot); win[q * OPSIN_WIN + w] holds the (on, off) network
 * steps of window w < n_win[q]. With tap_on, OPSIN_CH floats per slot per
 * native tick of every window, in cns.OPSIN_TAP_FIELDS order: s, g_op, the
 * current after the refractory clamp, the current at the starting membrane.
 * Refused: a repeated or late configuration; no slot; a row out of range or
 * repeated; a non-finite value; g_max below zero; force or kind above 1; a
 * kind-1 slot whose g_max, a_on or a_off is not zero, since its conductance
 * term must be an exact zero; an opsin slot on the current-based law, whose
 * update has no reversal to drive towards; n_win above OPSIN_WIN; a window
 * with on >= off, or out of order or overlapping within its slot. */
extern "C" int cns_engine_opsin_config(cns_engine *e, uint32_t n_slot, const int32_t *rows,
        const float *a_on, const float *a_off, const float *g_max, const float *e_rev,
        const float *thr, const uint8_t *force, const uint8_t *kind, const uint32_t *win,
        const uint32_t *n_win, uint32_t tap_on) {
    const uint32_t n = e->P.n;
    const char *bad = NULL;
    if (e->op.n || e->ticks_total || !n_slot || n_slot > n || tap_on > 1u)
        bad = "opsin: repeated, late or empty configuration";
    int32_t *pos = bad ? NULL : (int32_t *)malloc((size_t)n * 4);
    if (!bad && !pos) return -2;
    if (pos) for (uint32_t i = 0; i < n; i++) pos[i] = -1;
    for (uint32_t q = 0; !bad && q < n_slot; q++) {
        if (rows[q] < 0 || (uint32_t)rows[q] >= n || pos[rows[q]] >= 0)
            bad = "opsin: row out of range or repeated";
        else if (!isfinite(a_on[q]) || !isfinite(a_off[q]) || !isfinite(g_max[q])
                 || !isfinite(e_rev[q]) || !isfinite(thr[q]) || g_max[q] < 0.0f)
            bad = "opsin: a non-finite value or a negative g_max";
        else if (force[q] > 1u || kind[q] > 1u)
            bad = "opsin: force and kind are 0 or 1";
        else if (kind[q] == 1u && (g_max[q] != 0.0f || a_on[q] != 0.0f || a_off[q] != 0.0f))
            bad = "opsin: a threshold-window slot carries g_max 0 and a 0";
        else if (kind[q] == 0u && !e->P.cbased)
            bad = "opsin: the conductance needs the conductance-based law";
        else if (n_win[q] > OPSIN_WIN)
            bad = "opsin: more than OPSIN_WIN windows";
        else
            for (uint32_t w = 0; !bad && w < n_win[q]; w++) {
                const uint32_t *wp = win + ((size_t)q * OPSIN_WIN + w) * 2u;
                if (wp[0] >= wp[1] || (w && wp[0] < wp[-1]))
                    bad = "opsin: a window is empty, out of order or overlapping";
            }
        if (!bad) pos[rows[q]] = (int32_t)q;
    }
    if (bad) { free(pos); seterr(e->err, sizeof e->err, bad); return -1; }
    Opsin &op = e->op;
    op.pos = (int32_t *)dmalloc((size_t)n * 4, pos);
    op.a_on = (float *)dmalloc((size_t)n_slot * 4, a_on);
    op.a_off = (float *)dmalloc((size_t)n_slot * 4, a_off);
    op.g_max = (float *)dmalloc((size_t)n_slot * 4, g_max);
    op.e_rev = (float *)dmalloc((size_t)n_slot * 4, e_rev);
    op.thr = (float *)dmalloc((size_t)n_slot * 4, thr);
    op.force = (uint8_t *)dmalloc(n_slot, force);
    op.kind = (uint8_t *)dmalloc(n_slot, kind);
    op.win = (uint32_t *)dmalloc((size_t)n_slot * OPSIN_WIN * 2 * 4, win);
    op.n_win = (uint32_t *)dmalloc((size_t)n_slot * 4, n_win);
    op.s = (float *)dmalloc((size_t)n_slot * 4, NULL);
    free(pos);
    if (!op.pos || !op.a_on || !op.a_off || !op.g_max || !op.e_rev || !op.thr || !op.force
            || !op.kind || !op.win || !op.n_win || !op.s) {
        seterr(e->err, sizeof e->err, "opsin: allocation failed"); return -2;
    }
    if (tap_on) {
        const size_t bytes = (size_t)e->window_ticks * n_slot * OPSIN_CH * 4;
        op.tap = (float *)dmalloc(bytes, NULL);
        e->h_optap = (float *)calloc(1, bytes);
        if (!op.tap || !e->h_optap) {
            seterr(e->err, sizeof e->err, "opsin: tap allocation failed"); return -2;
        }
        op.tap_on = 1u;
    }
    op.n = n_slot;
    return 0;
}
extern "C" const float *cns_engine_opsin_tap(cns_engine *e) { return e->h_optap; }
extern "C" uint32_t cns_engine_opsin_width(void) { return OPSIN_CH; }

/* Design 2d12a3b3 (B2), CUDA only, no header entry. n_slot slots: rows[q] is
 * the slot's network row, i0[q] and i1[q] its held and stepped current in mV,
 * on[q] and off[q] the network steps of the step, [on, off). n_ticks is the
 * run's length in network steps. Refused: a repeated or late configuration;
 * no slot; a row out of range; a non-finite value; off before on; a step past
 * the run's end. Design 6d0042d5, Build 3: a row may take several slots when
 * their i0 are equal and their windows do not overlap (a zero-length window
 * overlaps nothing); a row whose slots differ in i0 or overlap is refused.
 * The Python driver also refuses a Darwin host and a row that set_clamp
 * holds. */
static int rc_order_rows(const int32_t *rows, const uint32_t *on, uint32_t a, uint32_t b) {
    return rows[a] != rows[b] ? (rows[a] < rows[b] ? -1 : 1)
         : on[a] != on[b] ? (on[a] < on[b] ? -1 : 1) : (a < b ? -1 : a > b);
}
extern "C" int cns_engine_row_current_config(cns_engine *e, uint32_t n_slot,
        const int32_t *rows, const float *i0, const float *i1,
        const uint32_t *on, const uint32_t *off, uint32_t n_ticks) {
    const uint32_t n = e->P.n;
    const char *bad = NULL;
    if (e->rc.n || e->ticks_total || !n_slot)
        bad = "row current: repeated, late or empty configuration";
    else if (e->se.n_src)   /* design b1746728 B1: both would set extra */
        bad = "row current: not defined with the serotonin state";
    else if (e->os.s)   /* design 8b6e0441 B1: the held current assigns extra */
        bad = "row current: not defined with the octopamine state";
    else if (e->P.n_gap)   /* gpu-7, note 10906d21: it would overwrite the gap current */
        bad = "row current: not defined with gap junctions";
    int32_t *pos = bad ? NULL : (int32_t *)malloc((size_t)n * 4);
    int32_t *next = bad ? NULL : (int32_t *)malloc((size_t)n_slot * 4);
    uint32_t *ord = bad ? NULL : (uint32_t *)malloc((size_t)n_slot * 4);
    if (!bad && (!pos || !next || !ord)) { free(pos); free(next); free(ord); return -2; }
    if (pos) for (uint32_t i = 0; i < n; i++) pos[i] = -1;
    for (uint32_t q = 0; !bad && q < n_slot; q++) {
        if (rows[q] < 0 || (uint32_t)rows[q] >= n)
            bad = "row current: row out of range";
        else if (!isfinite(i0[q]) || !isfinite(i1[q]))
            bad = "row current: a non-finite value";
        else if (off[q] < on[q])
            bad = "row current: off before on";
        else if (on[q] > n_ticks || off[q] > n_ticks)
            bad = "row current: a step outside the run";
        if (!bad) { ord[q] = q; next[q] = -1; }
    }
    /* order the slots by (row, on, slot) with an insertion sort, stable and
     * short for the few slots a run names, then link each row's slots */
    for (uint32_t k = 1; !bad && k < n_slot; k++) {
        const uint32_t x = ord[k];
        uint32_t j = k;
        while (j > 0 && rc_order_rows(rows, on, ord[j - 1], x) > 0) { ord[j] = ord[j - 1]; j--; }
        ord[j] = x;
    }
    uint32_t reach = 0;   /* the latest off of the row's non-empty windows so far */
    for (uint32_t k = 0; !bad && k < n_slot; k++) {
        const uint32_t q = ord[k];
        if (k == 0 || rows[ord[k - 1]] != rows[q]) { pos[rows[q]] = (int32_t)q; reach = 0; }
        else if (!(i0[ord[k - 1]] == i0[q]))
            bad = "row current: one row's slots differ in their held current";
        else if (on[q] < off[q] && on[q] < reach)
            bad = "row current: one row's windows overlap";
        else next[ord[k - 1]] = (int32_t)q;
        if (!bad && on[q] < off[q] && off[q] > reach) reach = off[q];
    }
    free(ord);
    if (bad) { free(pos); free(next); seterr(e->err, sizeof e->err, bad); return -1; }
    RowCur &rc = e->rc;
    rc.pos = (int32_t *)dmalloc((size_t)n * 4, pos);
    rc.i0 = (float *)dmalloc((size_t)n_slot * 4, i0);
    rc.i1 = (float *)dmalloc((size_t)n_slot * 4, i1);
    rc.on = (uint32_t *)dmalloc((size_t)n_slot * 4, on);
    rc.off = (uint32_t *)dmalloc((size_t)n_slot * 4, off);
    rc.next = (int32_t *)dmalloc((size_t)n_slot * 4, next);
    free(pos); free(next);
    if (!rc.pos || !rc.i0 || !rc.i1 || !rc.on || !rc.off || !rc.next) {
        seterr(e->err, sizeof e->err, "row current: allocation failed"); return -2;
    }
    rc.n = n_slot;
    return 0;
}
/* Design d27f82df B2: the exact-update mask, one byte per row, after
 * creation and before the first tick. Refused: a repeated or late call, NULL,
 * the current-based membrane, or a mask with no row set. */
extern "C" int cns_engine_row_exact_config(cns_engine *e, const uint8_t *mask) {
    const uint32_t n = e->P.n;
    const char *bad = NULL;
    if (!mask || e->hd.amask || e->ticks_total)
        bad = "row exact: NULL, repeated or late configuration";
    else if (!e->P.cbased)
        bad = "row exact: needs the conductance-based membrane";
    uint32_t on = 0;
    for (uint32_t i = 0; !bad && i < n; i++) on += mask[i] != 0;
    if (!bad && !on) bad = "row exact: no row set";
    if (bad) { seterr(e->err, sizeof e->err, bad); return -1; }
    e->hd.amask = (uint8_t *)dmalloc(n, mask);
    if (!e->hd.amask) { seterr(e->err, sizeof e->err, "row exact: allocation failed"); return -2; }
    return 0;
}
/* Design d27f82df B3: n_slot hold-down rows, after cns_engine_row_exact_config
 * and before the first tick. Per slot: form 0 (A: decay, dg) or 1 (pool:
 * gmax, c_spk, k, va, tau_c), e_rel in mV from the row's rest, latch_mv and
 * latch_ticks; dt in ms, the network step the pool integrates over; tap
 * nonzero keeps the per-tick hold-down tap. The state starts at zero.
 * Refused: a repeated or late call, a row out of range, repeated or not on
 * the exact update, an unknown form, a non-finite value, a decay outside
 * (0, 1], a negative dg, gmax, c_spk or k, a tau_c or dt not above zero, or
 * latch_ticks 0. */
extern "C" int cns_engine_holddown_config(cns_engine *e, uint32_t n_slot,
        const int32_t *rows, const uint32_t *form, const float *e_rel,
        const float *decay, const float *dg, const float *gmax, const float *c_spk,
        const float *k, const float *va, const float *tau_c, const float *latch_mv,
        const uint32_t *latch_ticks, float dt, uint32_t tap) {
    const uint32_t n = e->P.n;
    const char *bad = NULL;
    if (e->hd.n || e->ticks_total || !n_slot)
        bad = "hold-down: repeated, late or empty configuration";
    else if (!e->P.cbased)
        bad = "hold-down: needs the conductance-based membrane";
    else if (!(dt > 0.0f) || !isfinite(dt))
        bad = "hold-down: dt must be finite and above zero";
    uint8_t *hm = NULL;
    if (!bad && !e->P.analytic && !e->hd.amask)
        bad = "hold-down: no row is on the exact update";
    if (!bad && !e->P.analytic) {
        hm = (uint8_t *)malloc(n);
        if (!hm || cudaMemcpy(hm, e->hd.amask, n, cudaMemcpyDeviceToHost) != cudaSuccess) {
            free(hm); return -2;
        }
    }
    int32_t *pos = bad ? NULL : (int32_t *)malloc((size_t)n * 4);
    if (!bad && !pos) { free(hm); return -2; }
    if (pos) for (uint32_t i = 0; i < n; i++) pos[i] = -1;
    for (uint32_t q = 0; !bad && q < n_slot; q++) {
        if (rows[q] < 0 || (uint32_t)rows[q] >= n) bad = "hold-down: row out of range";
        else if (pos[rows[q]] >= 0) bad = "hold-down: row repeated";
        else if (hm && !hm[rows[q]]) bad = "hold-down: a row not on the exact update";
        else if (form[q] > 1u) bad = "hold-down: unknown form";
        else if (!isfinite(e_rel[q]) || !isfinite(decay[q]) || !isfinite(dg[q])
                 || !isfinite(gmax[q]) || !isfinite(c_spk[q]) || !isfinite(k[q])
                 || !isfinite(va[q]) || !isfinite(tau_c[q]) || !isfinite(latch_mv[q]))
            bad = "hold-down: a non-finite value";
        else if (!(decay[q] > 0.0f && decay[q] <= 1.0f)) bad = "hold-down: decay outside (0, 1]";
        else if (dg[q] < 0.0f || gmax[q] < 0.0f || c_spk[q] < 0.0f || k[q] < 0.0f)
            bad = "hold-down: a negative dg, gmax, c_spk or k";
        else if (!(tau_c[q] > 0.0f)) bad = "hold-down: tau_c must be above zero";
        else if (!latch_ticks[q]) bad = "hold-down: latch_ticks 0";
        if (!bad) pos[rows[q]] = (int32_t)q;
    }
    free(hm);
    if (bad) { free(pos); seterr(e->err, sizeof e->err, bad); return -1; }
    HoldDown &h = e->hd;
    const size_t fs = (size_t)n_slot * 4;
    h.pos = (int32_t *)dmalloc((size_t)n * 4, pos);
    free(pos);
    h.form = (uint32_t *)dmalloc(fs, form);
    h.e_rel = (float *)dmalloc(fs, e_rel); h.decay = (float *)dmalloc(fs, decay);
    h.dg = (float *)dmalloc(fs, dg); h.gmax = (float *)dmalloc(fs, gmax);
    h.c_spk = (float *)dmalloc(fs, c_spk); h.k = (float *)dmalloc(fs, k);
    h.va = (float *)dmalloc(fs, va); h.tau_c = (float *)dmalloc(fs, tau_c);
    h.latch_mv = (float *)dmalloc(fs, latch_mv);
    h.latch_ticks = (uint32_t *)dmalloc(fs, latch_ticks);
    h.c = (float *)dmalloc(fs, NULL); h.g_h = (float *)dmalloc(fs, NULL);
    h.L = (float *)dmalloc(fs, NULL); h.b = (float *)dmalloc(fs, NULL);
    h.peak = (float *)dmalloc(fs, NULL); h.peak_tick = (uint32_t *)dmalloc(fs, NULL);
    if (tap) {
        h.tap = (float *)dmalloc((size_t)e->window_ticks * fs * HD_CH, NULL);
        e->h_hdtap = (float *)calloc((size_t)e->window_ticks * n_slot * HD_CH, 4);
    }
    if (!h.pos || !h.form || !h.e_rel || !h.decay || !h.dg || !h.gmax || !h.c_spk || !h.k
            || !h.va || !h.tau_c || !h.latch_mv || !h.latch_ticks || !h.c || !h.g_h || !h.L
            || !h.b || !h.peak || !h.peak_tick || (tap && (!h.tap || !e->h_hdtap))) {
        seterr(e->err, sizeof e->err, "hold-down: allocation failed"); return -2;
    }
    h.dt = dt;
    h.n = n_slot;
    return 0;
}
/* Design d27f82df B3: the state after the last window, per slot, float32 c,
 * g_h, L, b and peak, and uint32 peak_tick. */
extern "C" int cns_engine_holddown_read(cns_engine *e, float *c, float *g_h, float *L,
                                        float *b, float *peak, uint32_t *peak_tick) {
    const HoldDown &h = e->hd;
    if (!h.n) { seterr(e->err, sizeof e->err, "hold-down: not configured"); return -1; }
    const size_t fs = (size_t)h.n * 4;
    cudaStreamSynchronize(e->stream);
    const bool ok = cudaMemcpy(c, h.c, fs, cudaMemcpyDeviceToHost) == cudaSuccess
        && cudaMemcpy(g_h, h.g_h, fs, cudaMemcpyDeviceToHost) == cudaSuccess
        && cudaMemcpy(L, h.L, fs, cudaMemcpyDeviceToHost) == cudaSuccess
        && cudaMemcpy(b, h.b, fs, cudaMemcpyDeviceToHost) == cudaSuccess
        && cudaMemcpy(peak, h.peak, fs, cudaMemcpyDeviceToHost) == cudaSuccess
        && cudaMemcpy(peak_tick, h.peak_tick, fs, cudaMemcpyDeviceToHost) == cudaSuccess;
    return ok ? 0 : -3;
}
/* The last window's hold-down tap, window_ticks x n x HD_CH float32. */
extern "C" const float *cns_engine_holddown_tap(cns_engine *e) { return e->h_hdtap; }
/* Design 6d0042d5, Build 2: the subthreshold coefficient below rest, one
 * float per row, under cns_engine.m's name and signature. After creation and
 * before the first tick. Row i's update uses below[i] in place of adapt_a[i]
 * while v < v_rest, so (a, 0) is a max(v - v_rest, 0), (0, a) a
 * min(v - v_rest, 0) and (a, a) the symmetric law. d_aa grows to 2n floats
 * with these in its second half, and the all-same path is turned off so each
 * row reads its own coefficients, which hold the values the scalar path
 * used. Refused: a repeated or late call, NULL, an engine without
 * adaptation, or a non-finite value. */
extern "C" int cns_engine_adapt_below_config(cns_engine *e, const float *below) {
    const uint32_t n = e->P.n;
    const char *bad = NULL;
    if (!below || e->P.adapt_split || e->ticks_total)
        bad = "adapt below: NULL, repeated or late configuration";
    else if (!e->P.use_adapt)
        bad = "adapt below: the engine carries no adaptation";
    for (uint32_t i = 0; !bad && i < n; i++)
        if (!isfinite(below[i])) bad = "adapt below: a non-finite value";
    if (bad) { seterr(e->err, sizeof e->err, bad); return -1; }
    const size_t nf = (size_t)n * 4;
    float *aa = (float *)dmalloc(2 * nf, NULL);
    if (!aa || cudaMemcpy(aa, e->d_aa, nf, cudaMemcpyDeviceToDevice) != cudaSuccess
            || cudaMemcpy(aa + n, below, nf, cudaMemcpyHostToDevice) != cudaSuccess) {
        if (aa) cudaFree(aa);
        seterr(e->err, sizeof e->err, "adapt below: allocation failed");
        return -2;
    }
    cudaFree(e->d_aa);
    e->d_aa = aa;
    e->P.uni_bits &= ~(1u << 7);
    e->P.adapt_split = 1;
    return 0;
}
/* Design 2163ba7d (Build 1), CUDA only, no header entry: one pulse train per
 * opsin slot, after cns_engine_opsin_config and before the first tick.
 * train[4q .. 4q + 3] = (start, stop, on, period) network steps; a slot with
 * period 0 carries none and must be all zero. A tick is then also lit when
 * start <= tick < stop and (tick - start) mod period < on, the rule of
 * cns.opsin_lit. Refused: no configuration, a repeated or late call, or a
 * train with start >= stop or on outside (0, period). */
extern "C" int cns_engine_opsin_train(cns_engine *e, const uint32_t *train) {
    Opsin &op = e->op;
    const char *bad = NULL;
    if (!op.n || op.train || e->ticks_total)
        bad = "opsin train: no configuration, or a repeated or late call";
    for (uint32_t q = 0; !bad && q < op.n; q++) {
        const uint32_t *t = train + (size_t)q * 4u;
        if (t[3] == 0u ? (t[0] || t[1] || t[2]) : (t[0] >= t[1] || !t[2] || t[2] >= t[3]))
            bad = "opsin train: start >= stop, or on outside (0, period)";
    }
    if (bad) { seterr(e->err, sizeof e->err, bad); return -1; }
    op.train = (uint32_t *)dmalloc((size_t)op.n * 4 * 4, train);
    if (!op.train) { seterr(e->err, sizeof e->err, "opsin train: allocation failed"); return -2; }
    return 0;
}
/* Design 2163ba7d (Build 2), CUDA only, no header entry: cns.py's
 * slow_inh_arrays. n_tgt target slots; src[i] = 1 marks row i as a listed
 * source; pos[i] is row i's target slot or -1, each slot named by one row;
 * rec[q] is slot q's record column below n_rec, or -1, each column named at
 * most once; keep and sfac scale the fast and slow shares and decay is the
 * slow pool's float32 factor; the record has n_bins bins of steps_per_bin
 * ticks. Refused: a repeated or late configuration; no target; the
 * current-based law or unsplit pools; a non-finite factor, keep outside
 * [0, 1], sfac below 0 or decay outside (0, 1); a zero bin; src above 1; a
 * slot out of range, named twice or by no row; a record column out of range
 * or named twice. */
extern "C" int cns_engine_slow_config(cns_engine *e, uint32_t n_tgt, const uint8_t *src,
        const int32_t *pos, const int32_t *rec, uint32_t n_rec, float keep, float sfac,
        float decay, uint32_t steps_per_bin, uint32_t n_bins) {
    const uint32_t n = e->P.n;
    const char *bad = NULL;
    if (e->sl.n || e->ticks_total || !n_tgt || n_tgt > n)
        bad = "slow pool: repeated, late or empty configuration";
    else if (!e->P.cbased || !e->P.split)
        bad = "slow pool: needs the conductance-based law with split pools";
    else if (!isfinite(keep) || !isfinite(sfac) || !isfinite(decay) || keep < 0.0f
             || keep > 1.0f || sfac < 0.0f || !(decay > 0.0f) || !(decay < 1.0f))
        bad = "slow pool: a factor out of range";
    else if (!steps_per_bin)
        bad = "slow pool: a zero record bin";
    else if (e->ps.n)   /* design 9501604f, Build C1: g9 takes the whole g_i increment */
        bad = "slow pool: not defined with the release factor";
    uint8_t *seen = bad ? NULL : (uint8_t *)calloc((size_t)n_tgt + n_rec + 1, 1);
    if (!bad && !seen) return -2;
    for (uint32_t i = 0; !bad && i < n; i++) {
        if (src[i] > 1u) bad = "slow pool: src is 0 or 1";
        else if (pos[i] < -1 || pos[i] >= (int32_t)n_tgt) bad = "slow pool: a slot out of range";
        else if (pos[i] >= 0 && seen[pos[i]]++) bad = "slow pool: a slot named twice";
    }
    for (uint32_t q = 0; !bad && q < n_tgt; q++) {
        if (!seen[q]) bad = "slow pool: a slot no row names";
        else if (rec[q] < -1 || rec[q] >= (int32_t)n_rec) bad = "slow pool: a record column out of range";
        else if (rec[q] >= 0 && seen[n_tgt + rec[q]]++) bad = "slow pool: a record column named twice";
    }
    free(seen);
    if (bad) { seterr(e->err, sizeof e->err, bad); return -1; }
    Slow &s = e->sl;
    s.src = (uint8_t *)dmalloc(n, src);
    s.pos = (int32_t *)dmalloc((size_t)n * 4, pos);
    s.rec = (int32_t *)dmalloc((size_t)n_tgt * 4, rec);
    s.ring = (float *)dmalloc((size_t)e->P.n_ring * n_tgt * 4, NULL);
    s.g = (float *)dmalloc((size_t)n_tgt * 4, NULL);
    s.acc_fast = (double *)dmalloc((size_t)n_tgt * 8, NULL);
    s.acc_slow = (double *)dmalloc((size_t)n_tgt * 8, NULL);
    s.intg = (double *)dmalloc((size_t)n_tgt * 8, NULL);
    s.rec_acc = (double *)dmalloc((size_t)n_bins * n_rec * 8, NULL);
    if (!s.src || !s.pos || !s.rec || !s.ring || !s.g || !s.acc_fast || !s.acc_slow
            || !s.intg || !s.rec_acc) {
        seterr(e->err, sizeof e->err, "slow pool: allocation failed"); return -2;
    }
    s.keep = keep; s.sfac = sfac; s.decay = decay;
    s.n_rec = n_rec; s.steps_per_bin = steps_per_bin; s.n_bins = n_bins;
    s.n = n_tgt;
    return 0;
}
/* Design 2163ba7d (Build 2): the slow pool after the last window. Per target
 * slot acc_fast, acc_slow and intg (double) and g (float); the ring, n_ring
 * x n_tgt floats slot-major; the record, n_bins x n_rec doubles. n_ring and
 * n_tgt must be the engine's. Refused without a configuration. */
extern "C" int cns_engine_slow_read(cns_engine *e, uint32_t n_ring, uint32_t n_tgt,
        double *acc_fast, double *acc_slow, double *intg, float *g, float *ring,
        double *rec_acc) {
    const Slow &s = e->sl;
    if (!s.n || n_tgt != s.n || n_ring != e->P.n_ring) {
        seterr(e->err, sizeof e->err, "slow pool: not configured, or a size differs");
        return -1;
    }
    const size_t nt = s.n;
    if (cudaStreamSynchronize(e->stream) != cudaSuccess
            || cudaMemcpy(acc_fast, s.acc_fast, nt * 8, cudaMemcpyDeviceToHost) != cudaSuccess
            || cudaMemcpy(acc_slow, s.acc_slow, nt * 8, cudaMemcpyDeviceToHost) != cudaSuccess
            || cudaMemcpy(intg, s.intg, nt * 8, cudaMemcpyDeviceToHost) != cudaSuccess
            || cudaMemcpy(g, s.g, nt * 4, cudaMemcpyDeviceToHost) != cudaSuccess
            || cudaMemcpy(ring, s.ring, (size_t)n_ring * nt * 4, cudaMemcpyDeviceToHost) != cudaSuccess
            || (s.n_rec && cudaMemcpy(rec_acc, s.rec_acc, (size_t)s.n_bins * s.n_rec * 8,
                                      cudaMemcpyDeviceToHost) != cudaSuccess)) {
        seterr(e->err, sizeof e->err, "slow pool: read failed");
        return -2;
    }
    return 0;
}
/* ---------------------------------------------------------------------------
 * Design 336cacb4 (study efc222c7), CUDA only, no header entry: the per-row
 * graded gain k_graded reads as ggain[tg], replaced between windows so a
 * commanded row's release carries a stimulus-side multiplier during its
 * command window. The array is in graded_idx order with graded_every already
 * folded in, exactly as the build uploads it. The copy runs on the engine's
 * stream and is complete when the call returns, so the next window's k_graded
 * reads the new values; an engine that never calls it runs the built gains.
 */
extern "C" int cns_engine_set_graded_gain(cns_engine *e, const float *gain, uint32_t n) {
    if (!gain || !e->d_ggain || !n || n != e->P.n_graded) {
        seterr(e->err, sizeof e->err, "graded gain: one value per graded row is required");
        return -1;
    }
    if (cudaMemcpyAsync(e->d_ggain, gain, (size_t)n * 4, cudaMemcpyHostToDevice,
                        e->stream) != cudaSuccess
            || cudaStreamSynchronize(e->stream) != cudaSuccess) {
        seterr(e->err, sizeof e->err, "graded gain upload failed");
        return -2;
    }
    return 0;
}
/* The gains k_graded will read next, copied back for the run's record. */
extern "C" int cns_engine_get_graded_gain(cns_engine *e, float *out, uint32_t n) {
    if (!out || !e->d_ggain || !n || n != e->P.n_graded) {
        seterr(e->err, sizeof e->err, "graded gain read: one value per graded row is required");
        return -1;
    }
    if (cudaMemcpyAsync(out, e->d_ggain, (size_t)n * 4, cudaMemcpyDeviceToHost,
                        e->stream) != cudaSuccess
            || cudaStreamSynchronize(e->stream) != cudaSuccess) {
        seterr(e->err, sizeof e->err, "graded gain read failed");
        return -2;
    }
    return 0;
}

/* Design 302c1bfb, Build 8, on CUDA (design 9501604f, Build C1), no header
 * entry: cns_engine.m's cns_engine_psi_config with its C signature. n_psi
 * listed rows (distinct, ungraded) release each spike's synapses at
 * f = 1/(1 + G*kg), kg = float32(k * gscale). With src9 (one byte per row, 1
 * on a source) G is a second state per listed row: decay r9, and each
 * inhibitory arrival from a source row adds c9 times its g_i increment;
 * without it G is the row's g_i. Refused as on Metal: a late or repeated
 * configuration, kgroup > 1 (the factor is set between single-tick groups),
 * the source state without split inhibition, a row out of range, repeated
 * or graded. Refused here as well: no split inhibition at all, since k_psi
 * reads g_i; the slow inhibitory pool, which changes the same increments and
 * has no defined order with it; and, with the release resource configured, a
 * listed row or a has9 source that is a resource row (rr_psi_overlap, design
 * fbcd0f61), where the two factors would scale one source's weights. */
extern "C" int cns_engine_psi_config(cns_engine *e, uint32_t n_psi, const int32_t *rows,
        const uint8_t *src9, const float *r9, const float *c9, float kg) {
    const uint32_t n = e->P.n;
    const char *bad = NULL;
    if (e->ps.n || e->ticks_total) bad = "release factor: repeated or late configuration";
    else if (!n_psi || n_psi > n || !rows) bad = "release factor: no rows";
    else if (e->kgroup != 1)
        bad = "release factor: the engine sets it between single-tick groups (kgroup 1)";
    else if (!e->P.split) bad = "release factor: needs split inhibition";
    else if (src9 && (!r9 || !c9))
        bad = "release factor: the source-fed state needs split inhibition, r9 and c9";
    else if (!isfinite(kg) || !(kg >= 0.0f)) bad = "release factor: kg not finite and nonnegative";
    else if (e->sl.n)
        bad = "release factor: not defined with the slow inhibitory pool";
    if (bad) { seterr(e->err, sizeof e->err, bad); return -1; }
    uint8_t *graded = (uint8_t *)malloc((size_t)n);
    int32_t *pos = (int32_t *)malloc((size_t)n * 4);
    float *ones = (float *)malloc((size_t)n * 4);
    if (!graded || !pos || !ones) {
        free(graded); free(pos); free(ones);
        seterr(e->err, sizeof e->err, "release factor: out of host memory"); return -2;
    }
    if (cudaMemcpy(graded, e->d_graded, (size_t)n, cudaMemcpyDeviceToHost) != cudaSuccess) {
        free(graded); free(pos); free(ones);
        seterr(e->err, sizeof e->err, "release factor: graded mask readback failed"); return -2;
    }
    for (uint32_t i = 0; i < n; i++) { pos[i] = -1; ones[i] = 1.0f; }
    for (uint32_t j = 0; j < n_psi && !bad; j++) {
        if (rows[j] < 0 || (uint32_t)rows[j] >= n) bad = "release factor: row out of range";
        else if (pos[rows[j]] >= 0) bad = "release factor: repeated row";
        else if (graded[rows[j]]) bad = "release factor: graded row";
        else pos[rows[j]] = (int32_t)j;
    }
    free(graded);
    if (!bad && e->rr.n) {   /* design fbcd0f61: beside the release resource on disjoint rows only */
        int32_t *rpos = (int32_t *)malloc((size_t)n * 4);
        if (!rpos || cudaMemcpy(rpos, e->rr.pos, (size_t)n * 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
            free(rpos); free(pos); free(ones);
            seterr(e->err, sizeof e->err, "release factor: release-resource readback failed"); return -2;
        }
        const char *ov = rr_psi_overlap(rpos, n, rows, n_psi, src9);
        free(rpos);
        if (ov) {
            free(pos); free(ones);
            snprintf(e->err, sizeof e->err, "release factor: %s", ov); return -1;
        }
    }
    if (bad) { free(pos); free(ones); seterr(e->err, sizeof e->err, bad); return -1; }
    Psi &s = e->ps;
    s.prow = (int32_t *)dmalloc((size_t)n_psi * 4, rows);
    s.fac = (float *)dmalloc((size_t)n * 4, ones);   /* 1.0 until k_psi writes a listed row */
    s.plog = (float *)dmalloc((size_t)e->window_ticks * n_psi * 16, NULL);
    e->h_plog = (float *)calloc((size_t)e->window_ticks * n_psi * 4, 4);
    int ok = s.prow && s.fac && s.plog && e->h_plog;
    if (ok && src9) {
        s.g9 = (float *)dmalloc((size_t)n_psi * 4, NULL);
        s.ring9 = (float *)dmalloc((size_t)e->P.n_ring * n_psi * 4, NULL);
        s.r9 = (float *)dmalloc((size_t)n_psi * 4, r9);
        s.c9 = (float *)dmalloc((size_t)n_psi * 4, c9);
        s.ppos = (int32_t *)dmalloc((size_t)n * 4, pos);
        s.src9 = (uint8_t *)dmalloc((size_t)n, src9);
        e->h_g9 = (float *)calloc(n_psi, 4);
        ok = s.g9 && s.ring9 && s.r9 && s.c9 && s.ppos && s.src9 && e->h_g9;
    }
    free(pos); free(ones);
    if (!ok) { seterr(e->err, sizeof e->err, "release factor: allocation failed"); return -2; }
    s.kg = kg;
    s.has9 = src9 ? 1u : 0u;
    s.n = n_psi;   /* last: a refused or failed configuration leaves every path off */
    return 0;
}

/* The last window's release-factor log, window_ticks x n_psi x (f, G, g_i,
 * fired flag); a fired flag of 0 marks a tick the row did not spike. */
extern "C" const float *cns_engine_psi_log(cns_engine *e) { return e->h_plog; }

/* The listed rows' source-fed state after the last window (n_psi floats). */
extern "C" const float *cns_engine_psi_state(cns_engine *e) { return e->h_g9; }

/* Design 302c1bfb, Build 10, on CUDA (design 9501604f, Build C2), no header
 * entry: cns_engine.m's cns_engine_set_spike_at, spike_at on n_rows rows
 * between windows. d_spike holds every row's threshold, and while every row's
 * was equal at creation k_step reads the scalar u_spike instead (uni_bits bit
 * 2), so the first call clears that bit and k_step reads the per-row array,
 * whose other entries hold the same value. */
extern "C" int cns_engine_set_spike_at(cns_engine *e, uint32_t n_rows, const int32_t *rows,
                                       const float *vals) {
    const uint32_t n = e->P.n;
    for (uint32_t q = 0; q < n_rows; q++)
        if (rows[q] < 0 || (uint32_t)rows[q] >= n) {
            seterr(e->err, sizeof e->err, "set_spike_at: row out of range"); return -1;
        }
    float *sa = (float *)malloc((size_t)n * 4);
    if (!sa) { seterr(e->err, sizeof e->err, "set_spike_at: out of host memory"); return -2; }
    if (cudaMemcpyAsync(sa, e->d_spike, (size_t)n * 4, cudaMemcpyDeviceToHost,
                        e->stream) != cudaSuccess
            || cudaStreamSynchronize(e->stream) != cudaSuccess) {
        free(sa); seterr(e->err, sizeof e->err, "set_spike_at: threshold readback failed"); return -2;
    }
    for (uint32_t q = 0; q < n_rows; q++) sa[rows[q]] = vals[q];
    if (cudaMemcpyAsync(e->d_spike, sa, (size_t)n * 4, cudaMemcpyHostToDevice,
                        e->stream) != cudaSuccess
            || cudaStreamSynchronize(e->stream) != cudaSuccess) {
        free(sa); seterr(e->err, sizeof e->err, "set_spike_at: threshold upload failed"); return -2;
    }
    free(sa);
    if (e->uniform[2]) { e->P.uni_bits &= ~(1u << 2); e->uniform[2] = false; }
    return 0;
}
/* Design fbcd0f61, Build B6, CUDA only, no header entry: the adaptation state
 * w on n_rows rows at t = 0, after creation and before the first tick; every
 * other row keeps the zero it was allocated with. Refused: an engine without
 * adaptation, a late call, a row out of range or named twice, a value that is
 * not finite. */
extern "C" int cns_engine_set_adapt_w(cns_engine *e, uint32_t n_rows, const int32_t *rows,
                                      const float *vals) {
    const uint32_t n = e->P.n;
    const char *bad = NULL;
    if (!e->P.use_adapt || !e->d_w) bad = "set_adapt_w: adaptation is off on this engine";
    else if (e->ticks_total) bad = "set_adapt_w: late call, the engine has run";
    else if (!n_rows || !rows || !vals) bad = "set_adapt_w: no rows";
    if (bad) { seterr(e->err, sizeof e->err, bad); return -1; }
    uint8_t *seen = (uint8_t *)calloc(n, 1);
    float *w = (float *)malloc((size_t)n * 4);
    if (!seen || !w) {
        free(seen); free(w); seterr(e->err, sizeof e->err, "set_adapt_w: out of host memory"); return -2;
    }
    for (uint32_t q = 0; !bad && q < n_rows; q++) {
        if (rows[q] < 0 || (uint32_t)rows[q] >= n) bad = "set_adapt_w: a row out of range";
        else if (seen[rows[q]]++) bad = "set_adapt_w: a row named twice";
        else if (!isfinite(vals[q])) bad = "set_adapt_w: a value not finite";
    }
    free(seen);
    if (bad) { free(w); seterr(e->err, sizeof e->err, bad); return -1; }
    if (cudaMemcpyAsync(w, e->d_w, (size_t)n * 4, cudaMemcpyDeviceToHost,
                        e->stream) != cudaSuccess
            || cudaStreamSynchronize(e->stream) != cudaSuccess) {
        free(w); seterr(e->err, sizeof e->err, "set_adapt_w: readback failed"); return -2;
    }
    for (uint32_t q = 0; q < n_rows; q++) w[rows[q]] = vals[q];
    if (cudaMemcpyAsync(e->d_w, w, (size_t)n * 4, cudaMemcpyHostToDevice,
                        e->stream) != cudaSuccess
            || cudaStreamSynchronize(e->stream) != cudaSuccess) {
        free(w); seterr(e->err, sizeof e->err, "set_adapt_w: upload failed"); return -2;
    }
    free(w);
    return 0;
}
/* Design b1746728 (study fd7a2867), Build 1, CUDA only, no header entry:
 * cns.py's receptor-scoped serotonin state (struct Ser), after creation and
 * before the first tick. src and tgt list n_src and n_tgt distinct rows;
 * gain (mV/Hz), decay and inc are cns.py's float32 values; init is s and its
 * maximum at t = 0; the readback has n_bins bins of steps_per_bin ticks.
 * Refused: a repeated or late call, an engine with kgroup above 1, the held
 * current, an empty, repeated or out-of-range row list, a gain or init that
 * is negative or not finite, a decay outside (0, 1], an inc that is not
 * finite and positive, and a zero bin. */
extern "C" int cns_engine_serotonin_config(cns_engine *e, uint32_t n_src, const int32_t *src,
        uint32_t n_tgt, const int32_t *tgt, float gain, float decay, float inc, float init,
        uint32_t steps_per_bin, uint32_t n_bins) {
    const uint32_t n = e->P.n;
    const char *bad = NULL;
    if (e->se.n_src || e->ticks_total) bad = "serotonin: repeated or late configuration";
    else if (e->kgroup != 1)
        bad = "serotonin: k_step reads the state k_ser leaves each tick (kgroup 1)";
    else if (e->rc.n) bad = "serotonin: not defined with the held current";
    else if (!n_src || !n_tgt || n_src > n || n_tgt > n || !src || !tgt)
        bad = "serotonin: no source or no target rows";
    else if (!isfinite(gain) || !(gain >= 0.0f) || !isfinite(init) || !(init >= 0.0f))
        bad = "serotonin: gain or init negative or not finite";
    else if (!isfinite(decay) || !(decay > 0.0f) || !(decay <= 1.0f))
        bad = "serotonin: decay outside (0, 1]";
    else if (!isfinite(inc) || !(inc > 0.0f)) bad = "serotonin: inc not finite and positive";
    else if (!steps_per_bin) bad = "serotonin: a zero readback bin";
    if (bad) { seterr(e->err, sizeof e->err, bad); return -1; }
    uint8_t *ms = (uint8_t *)calloc(n, 1), *mt = (uint8_t *)calloc(n, 1);
    if (!ms || !mt) {
        free(ms); free(mt); seterr(e->err, sizeof e->err, "serotonin: out of host memory"); return -2;
    }
    for (uint32_t q = 0; !bad && q < n_src; q++) {
        if (src[q] < 0 || (uint32_t)src[q] >= n) bad = "serotonin: a source row out of range";
        else if (ms[src[q]]++) bad = "serotonin: a source row named twice";
    }
    for (uint32_t q = 0; !bad && q < n_tgt; q++) {
        if (tgt[q] < 0 || (uint32_t)tgt[q] >= n) bad = "serotonin: a target row out of range";
        else if (mt[tgt[q]]++) bad = "serotonin: a target row named twice";
    }
    if (bad) { free(ms); free(mt); seterr(e->err, sizeof e->err, bad); return -1; }
    Ser &s = e->se;
    const float s2[2] = {init, init};
    s.src = (uint8_t *)dmalloc(n, ms);
    s.tgt = gain > 0.0f ? (uint8_t *)dmalloc(n, mt) : NULL;
    s.s = (float *)dmalloc(8, s2);
    s.n_tick = (uint32_t *)dmalloc(4, NULL);
    s.applied = (double *)dmalloc(8, NULL);
    s.spikes = (unsigned long long *)dmalloc(8, NULL);
    s.bins = (float *)dmalloc((size_t)n_bins * 4, NULL);
    free(ms); free(mt);
    if (!s.src || (gain > 0.0f && !s.tgt) || !s.s || !s.n_tick || !s.applied || !s.spikes
            || !s.bins) {
        seterr(e->err, sizeof e->err, "serotonin: allocation failed"); return -2;
    }
    s.gain = gain; s.decay = decay; s.inc = inc;
    s.n_tgt = n_tgt; s.steps_per_bin = steps_per_bin; s.n_bins = n_bins;
    s.n_src = n_src;
    return 0;
}
/* Design b1746728, Build 1: the state after the last window: s and its
 * maximum, the applied sum, the source spikes and the n_bins readback, which
 * must be the configured count. Refused without a configuration. */
extern "C" int cns_engine_serotonin_read(cns_engine *e, float *state2, double *applied,
        uint64_t *spikes, float *bins, uint32_t n_bins) {
    const Ser &s = e->se;
    if (!s.n_src || n_bins != s.n_bins) {
        seterr(e->err, sizeof e->err, "serotonin: not configured, or the bin count differs");
        return -1;
    }
    unsigned long long k = 0;
    if (cudaStreamSynchronize(e->stream) != cudaSuccess
            || cudaMemcpy(state2, s.s, 8, cudaMemcpyDeviceToHost) != cudaSuccess
            || cudaMemcpy(applied, s.applied, 8, cudaMemcpyDeviceToHost) != cudaSuccess
            || cudaMemcpy(&k, s.spikes, 8, cudaMemcpyDeviceToHost) != cudaSuccess
            || (n_bins && cudaMemcpy(bins, s.bins, (size_t)n_bins * 4,
                                     cudaMemcpyDeviceToHost) != cudaSuccess)) {
        seterr(e->err, sizeof e->err, "serotonin: read failed");
        return -2;
    }
    *spikes = (uint64_t)k;
    return 0;
}
/* Design 8b6e0441 (study a3f66049), Build 2, CUDA only, no header entry:
 * cns.py's octopamine offset (struct OaOff), after creation and before the
 * first tick. tgt lists n_tgt distinct rows; off is cns.py's float32 value in
 * mV. Refused: a repeated or late call, an empty, repeated or out-of-range row
 * list, and an offset that is zero or not finite (cns.py configures nothing
 * at 0). */
extern "C" int cns_engine_octopamine_offset_config(cns_engine *e, uint32_t n_tgt,
        const int32_t *tgt, float off) {
    const uint32_t n = e->P.n;
    const char *bad = NULL;
    if (e->oa.tgt || e->ticks_total) bad = "octopamine offset: repeated or late configuration";
    else if (!n_tgt || n_tgt > n || !tgt) bad = "octopamine offset: no target rows";
    else if (!isfinite(off) || off == 0.0f) bad = "octopamine offset: zero or not finite";
    if (bad) { seterr(e->err, sizeof e->err, bad); return -1; }
    uint8_t *mt = (uint8_t *)calloc(n, 1);
    if (!mt) { seterr(e->err, sizeof e->err, "octopamine offset: out of host memory"); return -2; }
    for (uint32_t q = 0; !bad && q < n_tgt; q++) {
        if (tgt[q] < 0 || (uint32_t)tgt[q] >= n) bad = "octopamine offset: a target row out of range";
        else if (mt[tgt[q]]++) bad = "octopamine offset: a target row named twice";
    }
    if (bad) { free(mt); seterr(e->err, sizeof e->err, bad); return -1; }
    e->oa.tgt = (uint8_t *)dmalloc(n, mt);
    free(mt);
    if (!e->oa.tgt) { seterr(e->err, sizeof e->err, "octopamine offset: allocation failed"); return -2; }
    e->oa.off = off;
    e->oa.n_tgt = n_tgt;
    return 0;
}
/* Design 8b6e0441 (study a3f66049), Build 1, CUDA only, no header entry:
 * cns.py's octopamine state (a second struct Ser), after creation and before
 * the first tick, cns_engine_serotonin_config's arguments and checks with two
 * differences: n_src may be 0 (every source row severed: the state only
 * decays), and the state is configured whatever n_src is. Also refused beside
 * the held current, which assigns extra. */
extern "C" int cns_engine_octopamine_state_config(cns_engine *e, uint32_t n_src,
        const int32_t *src, uint32_t n_tgt, const int32_t *tgt, float gain, float decay,
        float inc, float init, uint32_t steps_per_bin, uint32_t n_bins) {
    const uint32_t n = e->P.n;
    const char *bad = NULL;
    if (e->os.s || e->ticks_total) bad = "octopamine state: repeated or late configuration";
    else if (e->kgroup != 1)
        bad = "octopamine state: k_step reads the state k_ser leaves each tick (kgroup 1)";
    else if (e->rc.n) bad = "octopamine state: not defined with the held current";
    else if (n_src > n || !n_tgt || n_tgt > n || (n_src && !src) || !tgt)
        bad = "octopamine state: no target rows, or a row list too long";
    else if (!isfinite(gain) || !(gain >= 0.0f) || !isfinite(init) || !(init >= 0.0f))
        bad = "octopamine state: gain or init negative or not finite";
    else if (!isfinite(decay) || !(decay > 0.0f) || !(decay <= 1.0f))
        bad = "octopamine state: decay outside (0, 1]";
    else if (!isfinite(inc) || !(inc > 0.0f)) bad = "octopamine state: inc not finite and positive";
    else if (!steps_per_bin) bad = "octopamine state: a zero readback bin";
    if (bad) { seterr(e->err, sizeof e->err, bad); return -1; }
    uint8_t *ms = (uint8_t *)calloc(n, 1), *mt = (uint8_t *)calloc(n, 1);
    if (!ms || !mt) {
        free(ms); free(mt);
        seterr(e->err, sizeof e->err, "octopamine state: out of host memory"); return -2;
    }
    for (uint32_t q = 0; !bad && q < n_src; q++) {
        if (src[q] < 0 || (uint32_t)src[q] >= n) bad = "octopamine state: a source row out of range";
        else if (ms[src[q]]++) bad = "octopamine state: a source row named twice";
    }
    for (uint32_t q = 0; !bad && q < n_tgt; q++) {
        if (tgt[q] < 0 || (uint32_t)tgt[q] >= n) bad = "octopamine state: a target row out of range";
        else if (mt[tgt[q]]++) bad = "octopamine state: a target row named twice";
    }
    if (bad) { free(ms); free(mt); seterr(e->err, sizeof e->err, bad); return -1; }
    Ser &s = e->os;
    const float s2[2] = {init, init};
    s.src = (uint8_t *)dmalloc(n, ms);
    s.tgt = gain > 0.0f ? (uint8_t *)dmalloc(n, mt) : NULL;
    s.s = (float *)dmalloc(8, s2);
    s.n_tick = (uint32_t *)dmalloc(4, NULL);
    s.applied = (double *)dmalloc(8, NULL);
    s.spikes = (unsigned long long *)dmalloc(8, NULL);
    s.bins = (float *)dmalloc((size_t)n_bins * 4, NULL);
    free(ms); free(mt);
    if (!s.src || (gain > 0.0f && !s.tgt) || !s.s || !s.n_tick || !s.applied || !s.spikes
            || !s.bins) {
        seterr(e->err, sizeof e->err, "octopamine state: allocation failed"); return -2;
    }
    s.gain = gain; s.decay = decay; s.inc = inc;
    s.n_tgt = n_tgt; s.steps_per_bin = steps_per_bin; s.n_bins = n_bins;
    s.n_src = n_src;
    return 0;
}
/* Design 8b6e0441, Build 1: cns_engine_serotonin_read for the octopamine
 * state. Refused without a configuration. */
extern "C" int cns_engine_octopamine_state_read(cns_engine *e, float *state2, double *applied,
        uint64_t *spikes, float *bins, uint32_t n_bins) {
    const Ser &s = e->os;
    if (!s.s || n_bins != s.n_bins) {
        seterr(e->err, sizeof e->err, "octopamine state: not configured, or the bin count differs");
        return -1;
    }
    unsigned long long k = 0;
    if (cudaStreamSynchronize(e->stream) != cudaSuccess
            || cudaMemcpy(state2, s.s, 8, cudaMemcpyDeviceToHost) != cudaSuccess
            || cudaMemcpy(applied, s.applied, 8, cudaMemcpyDeviceToHost) != cudaSuccess
            || cudaMemcpy(&k, s.spikes, 8, cudaMemcpyDeviceToHost) != cudaSuccess
            || (n_bins && cudaMemcpy(bins, s.bins, (size_t)n_bins * 4,
                                     cudaMemcpyDeviceToHost) != cudaSuccess)) {
        seterr(e->err, sizeof e->err, "octopamine state: read failed");
        return -2;
    }
    *spikes = (uint64_t)k;
    return 0;
}

/* Design efe88957, Build 2: switch the coincidence seam to its list form
 * before the first window. Per target, in the descriptor's coin_tgt order,
 * its triple's float32 gain and decay and its lag in ticks, at most the
 * descriptor's coin_lag (the ring's depth); per pool edge, in the
 * descriptor's CSR order, the source's graded position or -1. Refused: no
 * seam, a repeated or late configuration, a target or edge count other than
 * the descriptor's, a lag outside [1, coin_lag], a non-finite gain, a decay
 * outside (0, 1), a graded position out of range. */
extern "C" int cns_engine_coin2_config(cns_engine *e, uint32_t n_coin, const float *gain,
        const float *decay, const uint32_t *lag, uint64_t fnnz, const int32_t *fast_gpos,
        uint64_t snnz, const int32_t *slow_gpos) {
    const char *bad = NULL;
    const int32_t ng = (int32_t)e->P.n_graded;
    if (!e->P.n_coin || e->coin2 || e->ticks_total)
        bad = "coincidence list: no seam, or a repeated or late configuration";
    else if (n_coin != e->P.n_coin || fnnz != e->coin_fnnz || snnz != e->coin_snnz)
        bad = "coincidence list: a target or edge count other than the descriptor's";
    for (uint32_t j = 0; !bad && j < n_coin; j++) {
        if (!lag[j] || lag[j] > e->P.coin_lag) bad = "coincidence list: a lag out of range";
        else if (!isfinite(gain[j]) || !(decay[j] > 0.0f) || !(decay[j] < 1.0f))
            bad = "coincidence list: a gain or decay out of range";
    }
    for (uint64_t k = 0; !bad && k < fnnz; k++)
        if (fast_gpos[k] < -1 || fast_gpos[k] >= ng) bad = "coincidence list: a graded position out of range";
    for (uint64_t k = 0; !bad && k < snnz; k++)
        if (slow_gpos[k] < -1 || slow_gpos[k] >= ng) bad = "coincidence list: a graded position out of range";
    if (bad) { seterr(e->err, sizeof e->err, bad); return -1; }
    e->d_c2gain = (float *)dmalloc((size_t)n_coin * 4, gain);
    e->d_c2decay = (float *)dmalloc((size_t)n_coin * 4, decay);
    e->d_c2lag = (uint32_t *)dmalloc((size_t)n_coin * 4, lag);
    e->d_c2fgp = (int32_t *)dmalloc((size_t)fnnz * 4, fnnz ? fast_gpos : NULL);
    e->d_c2sgp = (int32_t *)dmalloc((size_t)snnz * 4, snnz ? slow_gpos : NULL);
    if (!e->d_c2gain || !e->d_c2decay || !e->d_c2lag || !e->d_c2fgp || !e->d_c2sgp) {
        seterr(e->err, sizeof e->err, "coincidence list: allocation failed"); return -2;
    }
    e->coin2 = 1;
    return 0;
}

/* Design efe88957, Build 2: the seam's per-target accumulators after the
 * last window, 4 x n_coin floats in coin_tgt order: |product| max, |product|
 * sum, fast-trace max, delayed-slow max. */
extern "C" int cns_engine_coin_target_stats(cns_engine *e, uint32_t n_coin, float *out) {
    if (!e->P.n_coin || n_coin != e->P.n_coin) {
        seterr(e->err, sizeof e->err, "coincidence stats: no seam, or a count other than the descriptor's");
        return -1;
    }
    memcpy(out, e->h_cstats, (size_t)n_coin * 16);
    return 0;
}
