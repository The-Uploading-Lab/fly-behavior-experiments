/* cns_engine.m — Metal implementation of cns_engine.h.
 *
 * Per tick, in cns.py's order: conductance decay, ring consume, membrane
 * integrate (current-based, or conductance-based with optional true
 * shunting and the analytic one-step integrator), adaptation, refractory
 * clamp, threshold, drive candidates and forced spikes, refractory
 * countdown, reset; then the spike delivery into the delay rings in two
 * halves, an integer-atomic append of each edge to its target's bin
 * (k_scatter) and a per-target sorted fold into the ring (k_fold), so the
 * float additions at every ring slot happen in one fixed order, the C
 * reference's (tick, then source ascending), whatever the scheduler does;
 * and every g_every ticks the graded release of the designated non-spiking
 * cells, gathered per target in ascending source order within each arrival
 * slot and split by sign only after the sum, which is what the numpy engine
 * does (np.maximum on the summed vector, not per edge). No float atomic
 * remains: the 2026-09-03 six-wide replay of seed 201060 split into two
 * exact families because relaxed float atomics summed in scheduler order.
 *
 * Unsafe math optimizations and implicit contraction are off. Gap-current
 * dot products use explicit fma to match the ARM SciPy CSR matvec. Their
 * voltages are read before integration; spikelets arrive after all resets.
 *
 * K consecutive ticks fuse into one k_step dispatch when graded release,
 * gaps, coincidence and voltage recording do not require single ticks, and
 * K <= min(delay_slot), the bound under which no tick in the group can
 * read a ring slot the group's own scatters write. Barriers are explicit.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "cns_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SCATTER_TG
#define SCATTER_TG 512
#endif
#ifndef BIN_CAP
#define BIN_CAP 64   /* host copy of the MSL constant; keep both in step */
#endif
#ifndef FOLD_TG
#define FOLD_TG 256
#endif

static const char *MSL = R"MSL(
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

struct Params {
    uint  n, n_ring, words, fired_cap;
    int   cbased, split, analytic, use_adapt;
    float e_exc, e_inh, gscale, ggain;
    uint  g_every, n_graded, n_vrec, n_reltap;
    /* scalar copies used when the matching UNI_* function constant is set */
    float u_vrest, u_vreset, u_spike, u_decv;
    float u_dtt, u_decg, u_aa, u_ab;
    float u_dtw; int u_rt; uint tcnt_slot, _q1;
    uint n_coin, coin_lag; float coin_gain, coin_decay;
    uint kgroup, fbits_slot, n_gt, ftrip_slot;
    uint key_shift, has_tonic, n_gap, _q5;
    float gap_scale, gap_spikelet;
    uint gap_delay, gap_n_ap, gap_ap_off, gap_ap_slots;
};
/* per-target bin capacity in k_scatter; arrivals past it take k_fold's
 * transposed-graph path, which is exact but walks the whole in-degree in one
 * thread. Measured on the default net at 90 spikes per tick: the busiest
 * target receives about 32 arrivals per tick, so 64 leaves the slow path to
 * bursts. Entry = 32-bit key (slot << 20 | source) + float weight. */
#define BIN_CAP 64
#define FOLD_TG 256   /* k_fold threadgroup; k_scatter counts threadgroups in units of it */

/* Set by the host after scanning the per-neuron arrays: when every neuron
 * shares a value, the array read collapses to a scalar in Params and the
 * compiler removes the load. Same arithmetic, less traffic per tick. */
constant bool UNI_VREST  [[function_constant(0)]];
constant bool UNI_VRESET [[function_constant(1)]];
constant bool UNI_SPIKE  [[function_constant(2)]];
constant bool UNI_DECV   [[function_constant(3)]];
constant bool UNI_DTT    [[function_constant(4)]];
constant bool UNI_DECG   [[function_constant(5)]];
constant bool UNI_RT     [[function_constant(6)]];
constant bool UNI_ADAPT  [[function_constant(7)]];
constant bool SPARSE_VREST [[function_constant(8)]];   /* scalar v_rest except where ovr_bits is set */

kernel void k_step(constant Params &p            [[buffer(0)]],
                   device const float *v_rest      [[buffer(1)]],
                   device const float *v_reset     [[buffer(2)]],
                   device const float *spike_at    [[buffer(3)]],
                   device const float *decay_v     [[buffer(4)]],
                   device const float *dt_over_tau [[buffer(5)]],
                   device const float *decay_g     [[buffer(6)]],
                   device const short *ref_ticks   [[buffer(7)]],
                   device const uchar *graded      [[buffer(8)]],
                   device const float *adapt_a     [[buffer(9)]],
                   device const float *adapt_b     [[buffer(10)]],
                   device const float *dt_over_tauw[[buffer(11)]],
                   device float *v                 [[buffer(12)]],
                   device float *g                 [[buffer(13)]],
                   device float *g_i               [[buffer(14)]],
                   device short *refrac            [[buffer(15)]],
                   device float *w_adapt           [[buffer(16)]],
                   device uint  *counts            [[buffer(17)]],
                   device float *ring              [[buffer(18)]],
                   device float *ring_i            [[buffer(19)]],
                   device const uint *cand_bits    [[buffer(20)]],
                   device const uint *force_bits   [[buffer(21)]],
                   device uint *fired              [[buffer(22)]],
                   device atomic_uint *args        [[buffer(23)]],
                   constant uint &t0               [[buffer(24)]],
                   constant uint &K                [[buffer(25)]],
                   constant uint &kw0              [[buffer(26)]],
                   constant uint &have_bits        [[buffer(27)]],
                   device float *v_pre             [[buffer(28)]],
                   device const uint *clamp_bits   [[buffer(29)]],
                   device const float *clamp_val   [[buffer(30)]],
                   uint i [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    if (i >= p.n) return;
    {   /* the other parity's fire bits belong to the previous group, whose folds
         * are done; clear them for the next group (strided: kgroup*words may exceed n) */
        const uint fo = p.fbits_slot + (((have_bits >> 8) & 1u) ^ 1u) * p.kgroup * p.words;
        for (uint w = i; w < p.kgroup * p.words; w += p.n) atomic_store_explicit(&args[fo + w], 0u, memory_order_relaxed);
    }
    const uint word = i >> 5, bit = 1u << (i & 31u);
    const float vr   = UNI_VREST ? p.u_vrest
                     : (SPARSE_VREST ? ((clamp_bits[p.words + word] & bit) ? v_rest[i] : p.u_vrest) : v_rest[i]);   /* override mask lives in the 2nd half of clamp_bits */
    const float vrst = UNI_VRESET ? p.u_vreset : v_reset[i];
    const float sa   = UNI_SPIKE  ? p.u_spike  : spike_at[i];
    const float dv   = UNI_DECV   ? p.u_decv   : decay_v[i];
    const float dtt  = UNI_DTT    ? p.u_dtt    : dt_over_tau[i];
    const float dg   = UNI_DECG   ? p.u_decg   : decay_g[i];
    const short rt   = UNI_RT     ? (short)p.u_rt : ref_ticks[i];
    const bool isg = p.n_graded ? (graded[i] != 0) : false;
    const bool split = p.split != 0, cb = p.cbased != 0, ua = p.use_adapt != 0;
    const float aa  = !ua ? 0.0f : (UNI_ADAPT ? p.u_aa  : adapt_a[i]);
    const float ab  = !ua ? 0.0f : (UNI_ADAPT ? p.u_ab  : adapt_b[i]);
    const float dtw = !ua ? 0.0f : (UNI_ADAPT ? p.u_dtw : dt_over_tauw[i]);
    float vv = v[i], gg = g[i], ggi = split ? g_i[i] : 0.0f;
    float ww = ua ? w_adapt[i] : 0.0f;
    short r = refrac[i];
    uint cnt = 0;
    const bool cbits = (have_bits & 1u) != 0, fbits = (have_bits & 2u) != 0;
    const bool membits = (have_bits & 8u) != 0;
    /* coincidence seam: target mask in the 4th quarter of clamp_bits, the
     * per-tick product in the 3rd third of clamp_val (k_coin wrote it) */
    const bool coin = (have_bits & 16u) != 0 && (clamp_bits[3u * p.words + word] & bit) != 0;
    const bool clamped = (have_bits & 4u) != 0 && (clamp_bits[word] & bit) != 0;
    const float cval = clamped ? clamp_val[i] : 0.0f;
    for (uint k = 0; k < K; k++) {
        const uint tick = t0 + k, kw = kw0 + k;
        const uint slot = tick % p.n_ring;
        const ulong ri = (ulong)slot * p.n + i;
        gg = gg * dg;
        if (split) ggi = ggi * dg;
        gg = gg + ring[ri]; ring[ri] = 0.0f;
        if (split) { ggi = ggi + ring_i[ri]; ring_i[ri] = 0.0f; }
        if (coin) gg = gg + clamp_val[2u * p.n + i];
        float extra = 0.0f;
        if (p.n_gap && p.gap_scale != 0.0f)
            extra = extra + clamp_val[4u * p.n + i];
        if (ua) extra = extra - ww;
        float vi;
        if (cb) {
            float ge = max(gg, 0.0f) * p.gscale;
            float gi = split ? ggi * p.gscale : max(-gg, 0.0f) * p.gscale;
            /* tonic excitatory conductance, cns.py's `ge = ge + tonic_ge`,
             * here because it sits after ge and gi are formed and before
             * either integrator. Rides in the 4th section of clamp_val. */
            if (p.has_tonic) ge = ge + clamp_val[3u * p.n + i];
            if (p.analytic) {
                float total = 1.0f + ge + gi;
                float inv = 1.0f / total;
                float vinf = vr * inv + p.e_exc * (ge * inv)
                           + p.e_inh * (gi * inv) + extra * inv;
                float cdec = exp(-dtt * total);
                vi = vinf + (vv - vinf) * cdec;
            } else {
                vi = vr + (vv - vr) * dv
                   + (ge * (p.e_exc - vv) + gi * (p.e_inh - vv) + extra) * dtt;
            }
        } else {
            vi = vr + (vv - vr) * dv + (gg + extra) * dtt;
        }
        if (ua) ww = ww + dtw * (aa * (vi - vr) - ww);
        const bool active = r <= 0;
        if (!active) vi = vrst;
        const bool cand_event = cbits
            && (cand_bits[(ulong)kw * p.words + word] & bit);
        const bool mem_event = cand_event && membits
            && (clamp_bits[2u * p.words + word] & bit);
        bool fire = active && vi >= sa && !isg;
        if (cand_event && active && !mem_event) fire = true;
        if (fbits && (force_bits[(ulong)kw * p.words + word] & bit)) fire = true;
        if (clamped) vi = cval;   /* graded drive: set after detection, before release */
        /* The pre-reset membrane, kept for the voltage tap and, since
         * 2026-09-06, for graded release. cns.py computes release from
         * v[graded_idx] at its line 2825 and only resets fired cells at 2890,
         * so a graded cell that fired THIS tick (a graded row in the drive
         * set: threshold firing is barred for graded rows, a drive candidate
         * is not) releases at the voltage that crossed, not at v_reset. The
         * engine used to read the post-reset v in k_graded, which is why
         * cns_metal refused a graded cell in the drive set. */
        if (p.n_vrec) {
            /* The voltage tap also serves the default-off input recorder.
             * Section 0 preserves the historical post-integration,
             * pre-reset voltage. Sections 1-3 retain the exact membrane
             * voltage and split conductance accumulators used by this tick's
             * membrane equation. They are observational writes only. */
            v_pre[i] = vi;
            v_pre[p.n + i] = vv;
            v_pre[2u * p.n + i] = gg;
            v_pre[3u * p.n + i] = ggi;
        } else if (isg) {
            v_pre[i] = vi;   /* graded release reads the pre-reset membrane */
        }
        if (mem_event) vi = vi + clamp_val[p.n + i];   /* threshold reads it next tick */
        r = (r > 0) ? (short)(r - 1) : (short)0;
        if (fire) {
            uint s = atomic_fetch_add_explicit(&args[kw * 3], 1u, memory_order_relaxed);
            if (s < p.fired_cap) fired[(ulong)kw * p.fired_cap + s] = i;
            /* fire bit for k_fold's overflow path; parity in have_bits bit 8 */
            atomic_fetch_or_explicit(&args[p.fbits_slot + ((have_bits >> 8) & 1u) * p.kgroup * p.words
                                          + k * p.words + word], bit, memory_order_relaxed);
            vi = vrst; r = rt; cnt++;
            if (ua) ww = ww + ab;
        }
        vv = vi;
    }
    v[i] = vv; g[i] = gg; refrac[i] = r;
    if (split) g_i[i] = ggi;
    if (ua) w_adapt[i] = ww;
    if (cnt) counts[i] += cnt;
}

/* Spike delivery, first half: one threadgroup per spike, threads stride the
 * source's CSC column, and every edge is appended to its TARGET's bin with an
 * integer atomic. Integer addition commutes, so the bin's contents are the
 * same set in any schedule; k_fold puts them in order. The key sorts by
 * arrival slot, then source, which per ring slot is the C reference's order
 * (bins hold one tick, so tick order is the dispatch order). */
/* Delayed spikelets arrive before any membrane is read by gap coupling. */
kernel void k_gap_arrive(constant Params &p          [[buffer(0)]],
                         device const int *target   [[buffer(1)]],
                         device float *v            [[buffer(2)]],
                         device float *ring         [[buffer(3)]],
                         constant uint &tick        [[buffer(4)]],
                         uint j [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    if (j >= p.n_gap) return;
    const uint slot = (tick % (p.gap_delay + 1u)) * p.n_gap + j;
    v[target[j]] = v[target[j]] + ring[slot];
    ring[slot] = 0.0f;
}

/* All voltage reads complete before k_step writes any membrane. Compact
 * target rows avoid a full-CNS sparse matvec for a small electrical graph.
 * The dot product preserves G's stored CSR order, including zero weights. */
kernel void k_gap_current(constant Params &p          [[buffer(0)]],
                          device const int *target   [[buffer(1)]],
                          device const long *indptr  [[buffer(2)]],
                          device const int *source   [[buffer(3)]],
                          device const float *weight [[buffer(4)]],
                          device const float *degree [[buffer(5)]],
                          device const float *v      [[buffer(6)]],
                          device float *current      [[buffer(7)]],
                          device const float *ap     [[buffer(8)]],
                          constant uint &tick        [[buffer(9)]],
                          uint j [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    if (j >= p.n_gap) return;
    const uint i = (uint)target[j];
    float sum = 0.0f;
    for (long q = indptr[j]; q < indptr[j + 1]; q++) {
        /* SciPy's ARM CSR matvec fuses the product into the running sum.
         * Explicit fma preserves that rounding under contract(off). */
        const uint src = (uint)source[q];
        float voltage = v[target[src]];
        if (p.gap_n_ap) voltage = voltage + ap[(tick % p.gap_ap_slots) * p.n_gap + src];
        sum = fma(weight[q], voltage, sum);
    }
    current[4u * p.n + i] = p.gap_scale * (sum - degree[j] * v[i]);
}

/* Reset and spike detection finish first. Each unique target writes only its
 * own voltage and reads the complete firing mask, so simultaneous partners
 * cannot race. Chemical delays and refractory gates do not filter spikelets. */
kernel void k_gap_spikelet(constant Params &p         [[buffer(0)]],
                           device const int *target  [[buffer(1)]],
                           device const long *indptr [[buffer(2)]],
                           device const int *source  [[buffer(3)]],
                           device const float *weight[[buffer(4)]],
                           device float *v           [[buffer(5)]],
                           device const uint *args   [[buffer(6)]],
                           constant uint &par        [[buffer(7)]],
                           constant uint &kw         [[buffer(8)]],
                           device float *ring        [[buffer(9)]],
                           device float *ap          [[buffer(10)]],
                           device const float *wave  [[buffer(11)]],
                           constant uint &tick       [[buffer(12)]],
                           uint j [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    if (j >= p.n_gap) return;
    const uint i = (uint)target[j];
    // The reference consumes the waveform before threshold detection, then
    // schedules spikes. Clearing here preserves that order, including a
    // zero-delay first sample that stays in its slot until the ring wraps.
    if (p.gap_n_ap && p.gap_scale != 0.0f)
        ap[(tick % p.gap_ap_slots) * p.n_gap + j] = 0.0f;
    if (args[kw * 3] == 0) return;
    const uint fo = p.fbits_slot + par * p.kgroup * p.words;
    if (p.gap_n_ap && (args[fo + (i >> 5)] & (1u << (i & 31u)))) {
        for (uint k = 0; k < p.gap_n_ap; k++) {
            const uint a = ((tick + p.gap_ap_off + k) % p.gap_ap_slots) * p.n_gap + j;
            ap[a] = ap[a] + wave[k];
        }
    }
    if (p.gap_spikelet != 0.0f) {
        float sum = 0.0f;
        for (long q = indptr[j]; q < indptr[j + 1]; q++) {
            const uint src = (uint)target[source[q]];
            const float hot = (args[fo + (src >> 5)] & (1u << (src & 31u))) ? 1.0f : 0.0f;
            sum = sum + weight[q] * hot;
        }
        if (p.gap_delay) {
            const uint s = ((tick + p.gap_delay) % (p.gap_delay + 1u)) * p.n_gap + j;
            ring[s] = ring[s] + p.gap_spikelet * sum;
        } else {
            v[i] = v[i] + p.gap_spikelet * sum;
        }
    }
}

kernel void k_scatter(constant Params &p               [[buffer(0)]],
                      device const uint *fired          [[buffer(1)]],
                      device const long *indptr         [[buffer(2)]],
                      device const int  *indices        [[buffer(3)]],
                      device const float *weight        [[buffer(4)]],
                      device const int  *delay_slot     [[buffer(5)]],
                      device atomic_uint *bcount        [[buffer(6)]],
                      device uint2 *bin                 [[buffer(7)]],
                      device uint *touched              [[buffer(8)]],
                      constant uint &tick               [[buffer(9)]],
                      device uint *spk_tick             [[buffer(10)]],
                      constant uint &k                  [[buffer(11)]],
                      device atomic_uint *ovr           [[buffer(12)]],   /* [0] overflows, [1] largest bin */
                      device atomic_uint *args          [[buffer(13)]],
                      constant uint &wt                 [[buffer(14)]],
                      uint tg  [[threadgroup_position_in_grid]],
                      uint lid [[thread_position_in_threadgroup]],
                      uint tpt [[threads_per_threadgroup]]) {
#pragma clang fp contract(off)
    const uint src = fired[tg];
    /* the seam's "prev": which sources fired on this tick, read next tick */
    if (p.n_coin && lid == 0) spk_tick[src] = tick;
    const uint arr = (tick + (uint)delay_slot[src]) % p.n_ring;
    const uint key = (arr << p.key_shift) | src;   /* source in the low key_shift bits, arrival slot above; both checked at create */
    const long e0 = indptr[src], e1 = indptr[src + 1];
    for (long e = e0 + lid; e < e1; e += tpt) {
        const uint tgt = (uint)indices[e];
        const uint pos = atomic_fetch_add_explicit(&bcount[tgt], 1u, memory_order_relaxed);
        if (pos < BIN_CAP) {
            bin[(ulong)tgt * BIN_CAP + pos] = uint2(key, as_type<uint>(weight[e]));
        } else if (pos == BIN_CAP) {
            atomic_fetch_add_explicit(ovr, 1u, memory_order_relaxed);
        }
        if ((pos & 7u) == 7u) atomic_fetch_max_explicit(ovr + 1, pos + 1u, memory_order_relaxed);   /* largest bin, to 8 */
        if (pos == 3u || pos == 7u || pos == 15u || pos == 31u)   /* bin-size histogram: reached 4, 8, 16, 32 */
            atomic_fetch_add_explicit(ovr + 2 + (pos == 3u ? 0 : pos == 7u ? 1 : pos == 15u ? 2 : 3), 1u, memory_order_relaxed);
        if (pos == 0) {   /* first arrival at this target this tick: enlist it for k_fold */
            const uint t = atomic_fetch_add_explicit(&args[p.tcnt_slot + wt], 1u, memory_order_relaxed);
            touched[t] = tgt;
            /* exactly one thread sees each multiple of FOLD_TG, so the tick's fold
             * dispatch triple ends at ceil(touched / FOLD_TG) threadgroups */
            if ((t & (FOLD_TG - 1u)) == 0u)
                atomic_fetch_add_explicit(&args[p.ftrip_slot + wt * 3], 1u, memory_order_relaxed);
        }
    }
}

/* Knuth's two-sum: with round-to-nearest and contraction off, s = a + b and
 * e together represent a + b exactly, s = fl(a+b). The fold uses it to add a
 * tick's arrivals the way cns.py does: np.bincount sums a target's weights
 * for one arrival slot in float64 and .astype(np.float32) rounds that sum
 * ONCE, then `ring[slot] +=` rounds once more. Summing in float32 one edge at
 * a time rounds c times instead. Measured 2026-09-06: this is the LAST
 * arithmetic difference between the engines once FMA contraction is off. It
 * was built, measured as changing nothing and reverted earlier the same day,
 * when the contraction difference was still there and dominated it. */
static inline float2 twosum(float a, float b) {
    const float s = a + b;
    const float bb = s - a;
    return float2(s, (a - (s - bb)) + (b - bb));
}
static inline void dfadd(thread float &h, thread float &l, float w) {
    const float2 t = twosum(h, w);
    h = t.x; l += t.y;
}

/* Spike delivery, second half, once per TICK (a fused group runs scatter,
 * fold, scatter, fold: bins hold one tick, so overflow stays rare; folding
 * a whole 18-tick group overflowed most hubs and cost 50x, 2026-09-05).
 * One thread per TOUCHED target (k_scatter's list for the tick, dispatched
 * indirectly from the triple it counted): it sorts its bin on the key and
 * adds the weights into the ring in that order, so every ring slot is
 * summed in one fixed order, tick then source ascending, the C reference's
 * and scipy's. Only this thread writes ring[.][i] in the dispatch, so the
 * ring takes plain loads and stores and no float atomic is needed. A target
 * with more than BIN_CAP arrivals in the tick walks its incoming edges (the
 * transposed graph, sources ascending) against the tick's fire bits
 * instead: the same source order, exact, at the price of the whole
 * in-degree. Measured on the champion body, 2026-09-05: 7,523 targets hit
 * per tick, 97% with three entries or fewer, a bin of 16 or more once in
 * two ticks, 48 at most; three shapes that give the hubs more lanes (a
 * SIMD group per target with a bitonic network, the same with a counting
 * sort in threadgroup memory, and this kernel with bins above four handed
 * to the SIMD group) all cost more than this one, because the cost is the
 * second random pass over the ring and the dispatch, not the hub's chain.
 * See gpu/BENCH-LOG.md. */
kernel void k_fold(constant Params &p                [[buffer(0)]],
                   device uint *bcount                [[buffer(1)]],
                   device const uint2 *bin            [[buffer(2)]],
                   device const uint *touched         [[buffer(3)]],
                   device float *ring                 [[buffer(4)]],
                   device float *ring_i               [[buffer(5)]],
                   device atomic_uint *args           [[buffer(6)]],
                   device const long *tindptr         [[buffer(7)]],
                   device const int  *tsrc            [[buffer(8)]],
                   device const float *tw             [[buffer(9)]],
                   device const int  *delay_slot      [[buffer(10)]],
                   constant uint &t0                  [[buffer(11)]],
                   constant uint &k                   [[buffer(12)]],
                   constant uint &par                 [[buffer(13)]],
                   constant uint &wt                  [[buffer(14)]],
                   uint t [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    const uint fb = p.fbits_slot + par * p.kgroup * p.words;
    if (t >= atomic_load_explicit(&args[p.tcnt_slot + wt], memory_order_relaxed)) return;
    const uint i = touched[t];
    const uint c = bcount[i];   /* at least 1: every enlisted target had an arrival */
    bcount[i] = 0;
    const bool split = p.split != 0;
    if (c <= BIN_CAP) {
        uint kk[BIN_CAP]; float ww[BIN_CAP];
        for (uint j = 0; j < c; j++) {
            const uint2 b = bin[(ulong)i * BIN_CAP + j];
            kk[j] = b.x; ww[j] = as_type<float>(b.y);
        }
        for (uint j = 1; j < c; j++) {   /* insertion sort, at most BIN_CAP */
            const uint kj = kk[j]; const float wj = ww[j]; int m = (int)j - 1;
            while (m >= 0 && kk[m] > kj) { kk[m + 1] = kk[m]; ww[m + 1] = ww[m]; m--; }
            kk[m + 1] = kj; ww[m + 1] = wj;
        }
        /* The bin is sorted by (arrival slot, source), so a run of equal slot
         * is contiguous: sum it compensated, round the sum once, add once,
         * which is np.bincount(...).astype(np.float32) followed by `+=`. */
        for (uint j = 0; j < c; ) {
            const uint slot = kk[j] >> p.key_shift;
            float ph = 0.0f, pl = 0.0f, nh = 0.0f, nl = 0.0f;
            uint e = j;
            for (; e < c && (kk[e] >> p.key_shift) == slot; e++) {
                const float w = ww[e];
                if (split) { if (w > 0.0f) dfadd(ph, pl, w); else if (w < 0.0f) dfadd(nh, nl, -w); }
                else dfadd(ph, pl, w);
            }
            const ulong ri = (ulong)slot * p.n + i;
            const float sp = twosum(ph, pl).x;   /* = fl(exact sum), the reference's astype */
            if (sp != 0.0f) ring[ri] += sp;
            if (split) {
                const float sn = twosum(nh, nl).x;
                if (sn != 0.0f) ring_i[ri] += sn;
            }
            j = e;
        }
    } else {
        /* NOT compensated: a target's in-edges arrive here in source order,
         * not slot order, so the run of one arrival slot is not contiguous
         * and cannot be summed in one pass. This path rounds per edge, as
         * every path did before 2026-09-06. It runs only when a bin passes
         * BIN_CAP, which cns_engine_fold_overflows() counts and which is 0
         * on every body measured; a run reporting overflows has that many
         * target-ticks summed the old way. */
        const long e0 = tindptr[i], e1 = tindptr[i + 1];
        for (long e = e0; e < e1; e++) {
            const uint src = (uint)tsrc[e];
            const uint bits = atomic_load_explicit(&args[fb + k * p.words + (src >> 5)], memory_order_relaxed);
            if (!(bits & (1u << (src & 31u)))) continue;
            const uint arr = (t0 + k + (uint)delay_slot[src]) % p.n_ring;
            const ulong ri = (ulong)arr * p.n + i;
            const float w = tw[e];
            if (split) { if (w > 0.0f) ring[ri] += w; else if (w < 0.0f) ring_i[ri] += -w; }
            else ring[ri] += w;
        }
    }
}

/* Graded release, first half: one thread per graded cell computes its
 * release and the per-edge scale ggain * rel, the same two floats the
 * scatter multiplied by each weight before. */
kernel void k_graded(constant Params &p               [[buffer(0)]],
                     device const int *graded_idx      [[buffer(1)]],
                     device const float *v             [[buffer(2)]],   /* the PRE-reset membrane (b_vpre); see k_step */
                     device const float *gv50          [[buffer(3)]],
                     device const float *gslope        [[buffer(4)]],
                     device float *rel_cur             [[buffer(5)]],
                     device const float *ggain         [[buffer(6)]],
                     device float *gscale              [[buffer(7)]],
                     uint tg [[thread_position_in_grid]]) {
    if (tg >= p.n_graded) return;
    const uint src = (uint)graded_idx[tg];
    float x = -(v[src] - gv50[tg]) / gslope[tg];
    x = clamp(x, -60.0f, 60.0f);
    const float rel = 1.0f / (1.0f + exp(x));
    rel_cur[tg] = rel;
    gscale[tg] = ggain[tg] * rel;
}

/* Graded release, second half: one thread per target with graded input. Its
 * row is stored slot-major, source-ascending; the thread sums weight * scale
 * over each slot's run from zero, splits the sum by sign and adds it to the
 * live ring (numpy: csr @ rel per delay group, np.maximum after). Only this
 * thread writes ring[.][tgt] in the dispatch. */
kernel void k_gsum(constant Params &p                [[buffer(0)]],
                   device const int *gt_rows          [[buffer(1)]],
                   device const long *gt_indptr       [[buffer(2)]],
                   device const int *gt_gidx          [[buffer(3)]],
                   device const float *gt_w           [[buffer(4)]],
                   device const int *gt_slot          [[buffer(5)]],
                   device const float *gscale         [[buffer(6)]],
                   device float *ring                 [[buffer(7)]],
                   device float *ring_i               [[buffer(8)]],
                   constant uint &tick                [[buffer(9)]],
                   uint j [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    if (j >= p.n_gt) return;
    const uint tgt = (uint)gt_rows[j];
    const long e0 = gt_indptr[j], e1 = gt_indptr[j + 1];
    int cur = gt_slot[e0];
    float x = 0.0f;
    for (long e = e0; ; e++) {
        if (e == e1 || gt_slot[e] != cur) {
            if (x != 0.0f) {
                const ulong ri = (ulong)((tick + (uint)cur) % p.n_ring) * p.n + tgt;
                if (p.split) { if (x > 0.0f) ring[ri] += x; else ring_i[ri] += -x; }
                else ring[ri] += x;
            }
            if (e == e1) break;
            cur = gt_slot[e]; x = 0.0f;
        }
        const float t = gt_w[e] * gscale[gt_gidx[e]];
        x = x + t;
    }
}

/* The coincidence seam, one thread per designated target, once per tick
 * BEFORE k_step: two low-passed source-pool traces from last tick's spikes,
 * the slow one read back coin_lag ticks later, their product scaled by the
 * gain written where k_step adds it to the target's g. Every sum runs in
 * stored CSR order from zero, which is scipy's csr @ vector order, so the
 * arithmetic is cns.run()'s to the bit. spk_tick[src] == tick - 1 is
 * cns.run()'s prev vector (all zero on tick 0; the buffer starts at
 * 0xFFFFFFFF so an unfired source never matches). */
kernel void k_coin(constant Params &p                [[buffer(0)]],
                   device const int *tgt              [[buffer(1)]],
                   device const long *fip             [[buffer(2)]],
                   device const int *fsrc             [[buffer(3)]],
                   device const float *fw             [[buffer(4)]],
                   device const long *sip             [[buffer(5)]],
                   device const int *ssrc             [[buffer(6)]],
                   device const float *sw             [[buffer(7)]],
                   device const uint *spk_tick        [[buffer(8)]],
                   device float *fs                   [[buffer(9)]],
                   device float *ss                   [[buffer(10)]],
                   device float *cring                [[buffer(11)]],
                   device float *coin_add             [[buffer(12)]],   /* = clamp_val; section 2 */
                   device float *stats                [[buffer(13)]],
                   constant uint &tick                [[buffer(14)]],
                   uint j [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    if (j >= p.n_coin) return;
    const bool any = tick > 0u;
    const uint prev = tick - 1u;
    float df = 0.0f;
    for (long e = fip[j]; e < fip[j + 1]; e++)
        if (any && spk_tick[fsrc[e]] == prev) df = df + fw[e];
    float f = fs[j] * p.coin_decay;
    f = f + df;
    fs[j] = f;
    float ds = 0.0f;
    for (long e = sip[j]; e < sip[j + 1]; e++)
        if (any && spk_tick[ssrc[e]] == prev) ds = ds + sw[e];
    float s = ss[j] * p.coin_decay;
    s = s + ds;
    ss[j] = s;
    const ulong ri = (ulong)(tick % p.coin_lag) * p.n_coin + j;
    const float sd = cring[ri];
    cring[ri] = s;
    const float capp = (p.coin_gain * f) * sd;
    coin_add[2u * p.n + tgt[j]] = capp;   /* third section of clamp_val, where k_step reads it */
    const uint nc = p.n_coin;
    stats[j] = max(stats[j], fabs(capp));
    stats[nc + j] = stats[nc + j] + fabs(capp);
    stats[2u * nc + j] = max(stats[2u * nc + j], f);
    stats[3u * nc + j] = max(stats[3u * nc + j], sd);
}

kernel void k_reltap(device const int *pos          [[buffer(0)]],
                     device const float *rel_cur    [[buffer(1)]],
                     device float *reltap           [[buffer(2)]],
                     constant uint &n_tap           [[buffer(3)]],
                     constant uint &kw              [[buffer(4)]],
                     uint j [[thread_position_in_grid]]) {
    if (j >= n_tap) return;
    reltap[(ulong)kw * n_tap + j] = rel_cur[pos[j]];
}

kernel void k_vtap(device const int *vrec        [[buffer(0)]],
                   device const float *v_pre     [[buffer(1)]],
                   device float *vtap            [[buffer(2)]],
                   constant uint &n_vrec         [[buffer(3)]],
                   constant uint &kw             [[buffer(4)]],
                   constant uint &n              [[buffer(5)]],
                   uint j [[thread_position_in_grid]]) {
    if (j >= n_vrec) return;
    const uint row = (uint)vrec[j];
    const ulong out = ((ulong)kw * n_vrec + j) * 4u;
    vtap[out] = v_pre[row];
    vtap[out + 1u] = v_pre[n + row];
    vtap[out + 2u] = v_pre[2u * n + row];
    vtap[out + 3u] = v_pre[3u * n + row];
}
)MSL";

typedef struct {
    uint32_t n, n_ring, words, fired_cap;
    int32_t cbased, split, analytic, use_adapt;
    float e_exc, e_inh, gscale, ggain;
    uint32_t g_every, n_graded, n_vrec, n_reltap;
    float u_vrest, u_vreset, u_spike, u_decv;
    float u_dtt, u_decg, u_aa, u_ab;
    float u_dtw; int32_t u_rt; uint32_t tcnt_slot, _q1;
    uint32_t n_coin, coin_lag; float coin_gain, coin_decay;
    uint32_t kgroup, fbits_slot, n_gt, ftrip_slot;
    uint32_t key_shift, has_tonic, n_gap, _q5;
    float gap_scale, gap_spikelet;
    uint32_t gap_delay, gap_n_ap, gap_ap_off, gap_ap_slots;
} Params;

static int all_same_f(const float *a, uint32_t n) {
    for (uint32_t i = 1; i < n; i++) if (a[i] != a[0]) return 0;
    return 1;
}
static int all_same_i16(const int16_t *a, uint32_t n) {
    for (uint32_t i = 1; i < n; i++) if (a[i] != a[0]) return 0;
    return 1;
}

struct cns_engine {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> p_step, p_scatter, p_fold, p_graded, p_gsum, p_vtap, p_reltap, p_coin;
    id<MTLComputePipelineState> p_gap_current, p_gap_spikelet, p_gap_arrive;
    Params P;
    uint32_t window_ticks, kgroup, n_gslots, min_delay, n_gt, grp;
    id<MTLBuffer> bP, b_vrest, b_vreset, b_spike, b_decv, b_dtt, b_decg, b_rt,
        b_graded, b_aa, b_ab, b_dtw, b_v, b_g, b_gi, b_ref, b_w, b_counts,
        b_ring, b_ring_i, b_cand, b_force, b_fired, b_args, b_indptr,
        b_indices, b_weight, b_delay, b_gidx, b_gv50, b_gslope, b_ggain,
        b_gslots, b_vrec, b_vpre, b_vtap, b_relcur, b_relpos, b_reltap,
        b_clampbits, b_clampval, b_ovr,
        b_bcount, b_bin, b_touched, b_tindptr, b_tsrc, b_tw,
        b_gtrows, b_gtindptr, b_gtgidx, b_gtw, b_gtslot, b_gscale,
        b_ctgt, b_cfip, b_cfsrc, b_cfw, b_csip, b_cssrc, b_csw, b_spk,
        b_cfs, b_css, b_cring, b_cstats;
    id<MTLBuffer> b_gap_tgt, b_gap_ip, b_gap_src, b_gap_w, b_gap_deg;
    id<MTLBuffer> b_gap_ring, b_gap_ap, b_gap_wave;
    int clamp_on, membrane_drive_on;
    uint64_t touched_total, ticks_total;   /* gauge: targets hit per tick, summed */
    bool uniform[8];
    bool sparse_vrest;
    double gpu_s;
    /* optional per-kernel attribution (CNS_ENGINE_PROFILE=1): one encoder per
     * dispatch, stage-boundary timestamps, like bench-metal --profile. Slower
     * than the fast path (encoder overhead, no concurrency), so it measures
     * shares, not the wall the fast path achieves. */
    int profile;
    NSArray<id<MTLCounterSampleBuffer>> *sample_buffers;
    NSUInteger profile_capacity;
    double kns[7];
    uint64_t kcount[7];
    char err[512];
    char devname[128];
};

static id<MTLBuffer> mkbuf(id<MTLDevice> dev, size_t bytes, const void *src) {
    if (bytes == 0) bytes = 4;
    id<MTLBuffer> b = [dev newBufferWithLength:bytes
                                       options:MTLResourceStorageModeShared];
    if (!b) return nil;
    if (src) memcpy(b.contents, src, bytes); else memset(b.contents, 0, bytes);
    return b;
}

static void seterr(char *err, size_t n, const char *msg) {
    if (err && n) { strncpy(err, msg, n - 1); err[n - 1] = 0; }
}

cns_engine *cns_engine_create(const cns_engine_desc *d, char *err, size_t errlen) {
    if (d->n_gap && d->analytic) {
        seterr(err, errlen, "analytic integration with gap junctions lacks exact voltage parity");
        return NULL;
    }
    if (d->n_gap && (d->n_gap > d->n || !d->gap_tgt || !d->gap_indptr
                    || !d->gap_src || !d->gap_weight || !d->gap_deg)) {
        seterr(err, errlen, "invalid gap-junction descriptor");
        return NULL;
    }
    @autoreleasepool {
        cns_engine *e = calloc(1, sizeof *e);
        e->dev = MTLCreateSystemDefaultDevice();
        if (!e->dev) {
            seterr(err, errlen, "no Metal device (headless session, or the GPU is "
                   "unavailable to this process). Use the numpy backend for BOTH arms "
                   "of the comparison; the engine will not switch by itself.");
            free(e); return NULL;
        }
        strncpy(e->devname, e->dev.name.UTF8String, sizeof e->devname - 1);
        NSError *nserr = nil;
        MTLCompileOptions *opt = [MTLCompileOptions new];
        opt.mathMode = MTLMathModeSafe;
        id<MTLLibrary> lib = [e->dev newLibraryWithSource:
            [NSString stringWithUTF8String:MSL] options:opt error:&nserr];
        if (!lib) {
            seterr(err, errlen, nserr.localizedDescription.UTF8String);
            free(e); return NULL;
        }
        #define PSO(var, name) do { \
            id<MTLFunction> f = [lib newFunctionWithName:@name]; \
            var = f ? [e->dev newComputePipelineStateWithFunction:f error:&nserr] : nil; \
            if (!var) { seterr(err, errlen, name " pipeline failed"); free(e); return NULL; } \
        } while (0)
        const uint32_t n0 = d->n;
        bool uni[8] = {
            all_same_f(d->v_rest, n0), all_same_f(d->v_reset, n0),
            all_same_f(d->spike_at, n0), all_same_f(d->decay_v, n0),
            all_same_f(d->dt_over_tau, n0), all_same_f(d->decay_g, n0),
            all_same_i16(d->refrac_ticks, n0),
            d->use_adapt ? (all_same_f(d->adapt_a, n0) && all_same_f(d->adapt_b, n0)
                            && all_same_f(d->dt_over_tauw, n0)) : 1 };
        /* v_rest: uniform, or the mode value plus a bitmask of the few neurons
         * that differ (378 size-principle overrides on the champion), or dense */
        bool sparse_vr = false; float vr_mode = d->v_rest[0]; uint32_t *ovr = NULL;
        size_t words0 = ((size_t)n0 + 31) / 32;
        if (!uni[0]) {
            /* mode via a small hash of values; the override count is tiny */
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
                ovr = calloc(words0, 4);
                for (uint32_t i = 0; i < n0; i++) if (d->v_rest[i] != vr_mode) ovr[i >> 5] |= 1u << (i & 31u);
            }
        }
        MTLFunctionConstantValues *fcv = [MTLFunctionConstantValues new];
        for (int k = 0; k < 8; k++) [fcv setConstantValue:&uni[k] type:MTLDataTypeBool atIndex:k];
        [fcv setConstantValue:&sparse_vr type:MTLDataTypeBool atIndex:8];
        {
            id<MTLFunction> f = [lib newFunctionWithName:@"k_step" constantValues:fcv error:&nserr];
            e->p_step = f ? [e->dev newComputePipelineStateWithFunction:f error:&nserr] : nil;
            if (!e->p_step) { seterr(err, errlen, "k_step pipeline failed"); free(e); return NULL; }
        }
        memcpy(e->uniform, uni, sizeof uni);
        PSO(e->p_scatter, "k_scatter");
        PSO(e->p_fold, "k_fold");
        PSO(e->p_graded, "k_graded");
        PSO(e->p_gsum, "k_gsum");
        PSO(e->p_vtap, "k_vtap");
        PSO(e->p_reltap, "k_reltap");
        PSO(e->p_coin, "k_coin");
        if (d->n_gap) {
            PSO(e->p_gap_current, "k_gap_current");
            PSO(e->p_gap_spikelet, "k_gap_spikelet");
            PSO(e->p_gap_arrive, "k_gap_arrive");
        }
        e->queue = [e->dev newCommandQueue];
        const char *pf = getenv("CNS_ENGINE_PROFILE");
        e->profile = (pf && pf[0] == '1') ? 1 : 0;
        if (e->profile) {
            id<MTLCounterSet> tsset = nil;
            for (id<MTLCounterSet> cs in e->dev.counterSets)
                if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) tsset = cs;
            /* At most eleven encoders per tick with delayed gaps, taps, graded release
             * and coincidence all enabled. A fixed 2048-encoder table would
             * overflow for a 256-tick window carrying these mechanisms. */
            e->profile_capacity = (NSUInteger)(d->window_ticks ? d->window_ticks : 64) * 11;
            NSMutableArray<id<MTLCounterSampleBuffer>> *pages = [NSMutableArray new];
            for (NSUInteger off = 0; off < e->profile_capacity && tsset; off += 2048) {
                MTLCounterSampleBufferDescriptor *sd = [MTLCounterSampleBufferDescriptor new];
                sd.counterSet = tsset;
                /* This device caps one timestamp buffer at 4096 samples.
                 * Each encoder uses a start/end pair; page long windows. */
                sd.sampleCount = MIN(e->profile_capacity - off, 2048) * 2;
                sd.storageMode = MTLStorageModeShared;
                id<MTLCounterSampleBuffer> page = [e->dev newCounterSampleBufferWithDescriptor:sd error:&nserr];
                if (!page) { [pages removeAllObjects]; break; }
                [pages addObject:page];
            }
            e->sample_buffers = [pages copy];
            if (!pages.count) e->profile = 0;
        }

        const uint32_t n = d->n;
        uint32_t words = (n + 31) / 32;
        uint32_t cap = d->fired_cap ? d->fired_cap : 65536u;
        if (cap > n) cap = n;
        e->window_ticks = d->window_ticks ? d->window_ticks : 64;
        e->P = (Params){ n, d->n_ring, words, cap, d->cbased, d->split,
                         d->analytic, d->use_adapt, d->e_exc, d->e_inh,
                         d->gscale, d->ggain_scaled,
                         d->g_every ? d->g_every : 1, d->n_graded, d->n_vrec, d->n_reltap,
                         vr_mode, d->v_reset[0], d->spike_at[0], d->decay_v[0],
                         d->dt_over_tau[0], d->decay_g[0],
                         d->use_adapt ? d->adapt_a[0] : 0.0f, d->use_adapt ? d->adapt_b[0] : 0.0f,
                         d->use_adapt ? d->dt_over_tauw[0] : 0.0f, d->refrac_ticks[0],
                         (uint32_t)(e->window_ticks * 3), 0 };
        int32_t mind = d->delay_slot[0];
        for (uint32_t i = 1; i < n; i++) if (d->delay_slot[i] < mind) mind = d->delay_slot[i];
        if (mind < 1) { seterr(err, errlen, "delay_slot must be >= 1"); free(e); return NULL; }
        {   /* k_scatter packs (arrival slot, source) into one 32-bit key: the
             * source takes as many low bits as n needs and the slot gets the
             * rest, so a long-delay body is limited by its own graph size
             * rather than by a constant. 188,508 neurons need 18 bits, leaving
             * 14 for the slot: 16,384 ring slots, 1.6 s of delay at dt 0.1 ms.
             * Before 2026-09-06 the split was fixed at 20/12 and refused any
             * body past 4,096 slots, which the cord delay scale can reach. */
            uint32_t sh = 0; while ((1u << sh) < n) sh++;
            if (sh > 31 || d->n_ring > (1u << (32 - sh))) {
                char m[256];
                snprintf(m, sizeof m, "bin key packing: %u neurons need %u bits, "
                         "leaving %u for the arrival slot (%u max), but n_ring is %u",
                         n, sh, 32 - sh, 1u << (32 - sh), d->n_ring);
                seterr(err, errlen, m); free(e); return NULL;
            }
            e->P.key_shift = sh;
        }
        e->min_delay = (uint32_t)mind;
        uint32_t kg = d->kmax ? d->kmax : (uint32_t)mind;
        if (kg > (uint32_t)mind) kg = (uint32_t)mind;
        if (d->n_graded || d->n_vrec || d->n_coin || d->n_gap) kg = 1;
        e->kgroup = kg < 1 ? 1 : kg;

        /* distinct delay slots among graded sources: k_gsum's rows are
         * stored slot-major in this order, source-ascending within a slot */
        uint32_t *gsl = NULL; e->n_gslots = 0;
        if (d->n_graded) {
            uint8_t *seen = calloc(d->n_ring, 1);
            gsl = malloc(d->n_ring * sizeof(uint32_t));
            for (uint32_t k = 0; k < d->n_graded; k++) {
                uint32_t s = (uint32_t)d->delay_slot[d->graded_idx[k]];
                if (!seen[s]) { seen[s] = 1; gsl[e->n_gslots++] = s; }
            }
            free(seen);
        }

        size_t nf = (size_t)n * 4;
        e->P.n_coin = d->n_coin;
        e->P.coin_lag = d->coin_lag ? d->coin_lag : 1;
        e->P.coin_gain = d->coin_gain;
        e->P.coin_decay = d->coin_decay;
        e->P.n_gap = d->n_gap;
        e->P.gap_scale = d->gap_scale;
        e->P.gap_spikelet = d->gap_spikelet;
        e->P.gap_delay = d->gap_delay;
        e->P.gap_n_ap = d->gap_n_ap;
        e->P.gap_ap_off = d->gap_ap_off;
        e->P.gap_ap_slots = d->gap_ap_off + d->gap_n_ap + 1u;
        e->bP = mkbuf(e->dev, sizeof e->P, &e->P);
        e->b_vrest = mkbuf(e->dev, nf, d->v_rest);
        e->b_vreset = mkbuf(e->dev, nf, d->v_reset);
        e->b_spike = mkbuf(e->dev, nf, d->spike_at);
        e->b_decv = mkbuf(e->dev, nf, d->decay_v);
        e->b_dtt = mkbuf(e->dev, nf, d->dt_over_tau);
        e->b_decg = mkbuf(e->dev, nf, d->decay_g);
        e->b_rt = mkbuf(e->dev, (size_t)n * 2, d->refrac_ticks);
        e->b_graded = mkbuf(e->dev, (size_t)n, d->graded_mask);
        e->b_aa = mkbuf(e->dev, d->use_adapt ? nf : 0, d->use_adapt ? d->adapt_a : NULL);
        e->b_ab = mkbuf(e->dev, d->use_adapt ? nf : 0, d->use_adapt ? d->adapt_b : NULL);
        e->b_dtw = mkbuf(e->dev, d->use_adapt ? nf : 0, d->use_adapt ? d->dt_over_tauw : NULL);
        e->b_v = mkbuf(e->dev, nf, d->v_rest);
        e->b_g = mkbuf(e->dev, nf, NULL);
        e->b_gi = mkbuf(e->dev, d->split ? nf : 0, NULL);
        e->b_ref = mkbuf(e->dev, (size_t)n * 2, NULL);
        e->b_w = mkbuf(e->dev, d->use_adapt ? nf : 0, NULL);
        e->b_counts = mkbuf(e->dev, nf, NULL);
        size_t ringb = (size_t)d->n_ring * n * 4;
        e->b_ring = mkbuf(e->dev, ringb, NULL);
        e->b_ring_i = mkbuf(e->dev, d->split ? ringb : 0, NULL);
        e->b_gslots = mkbuf(e->dev, e->n_gslots * 4, gsl);
        {   /* the graph transposed: CSR by target, sources ascending (the
             * source loop below runs ascending), for k_fold's overflow path */
            int64_t *tip = calloc((size_t)n + 1, sizeof(int64_t));
            int32_t *tsrc = malloc((size_t)d->nnz * 4 + 4);
            float *tw = malloc((size_t)d->nnz * 4 + 4);
            int64_t *fill = malloc((size_t)n * sizeof(int64_t));
            for (uint64_t q = 0; q < d->nnz; q++) tip[d->indices[q] + 1]++;
            for (uint32_t i = 0; i < n; i++) tip[i + 1] += tip[i];
            memcpy(fill, tip, (size_t)n * sizeof(int64_t));
            for (uint32_t src = 0; src < n; src++)
                for (int64_t q = d->indptr[src]; q < d->indptr[src + 1]; q++) {
                    const int64_t at = fill[d->indices[q]]++;
                    tsrc[at] = (int32_t)src; tw[at] = d->weight[q];
                }
            e->b_tindptr = mkbuf(e->dev, (size_t)(n + 1) * 8, tip);
            e->b_tsrc = mkbuf(e->dev, (size_t)d->nnz * 4, tsrc);
            e->b_tw = mkbuf(e->dev, (size_t)d->nnz * 4, tw);
            free(tip); free(tsrc); free(tw); free(fill);
        }
        {   /* graded gather rows: one per target with graded input; entries
             * (graded position, weight, arrival slot) slot-major in gsl
             * order, source-ascending within a slot */
            uint32_t ng_ = d->n_graded;
            uint32_t *order = malloc((size_t)ng_ * 4 + 4);   /* graded positions by ascending source */
            for (uint32_t k = 0; k < ng_; k++) order[k] = k;
            for (uint32_t a = 1; a < ng_; a++) {   /* insertion sort; the list is normally sorted already */
                uint32_t x = order[a]; int64_t b = (int64_t)a - 1;
                while (b >= 0 && d->graded_idx[order[b]] > d->graded_idx[x]) { order[b + 1] = order[b]; b--; }
                order[b + 1] = x;
            }
            int32_t *rowid = malloc((size_t)n * 4);
            for (uint32_t i = 0; i < n; i++) rowid[i] = -1;
            int64_t *cnt = calloc((size_t)n, sizeof(int64_t));
            uint64_t tot = 0;
            for (uint32_t k = 0; k < ng_; k++) {
                int32_t src = d->graded_idx[k];
                for (int64_t q = d->indptr[src]; q < d->indptr[src + 1]; q++) { cnt[d->indices[q]]++; tot++; }
            }
            uint32_t ngt = 0;
            for (uint32_t i = 0; i < n; i++) if (cnt[i]) rowid[i] = (int32_t)ngt++;
            int32_t *rows = malloc((size_t)ngt * 4 + 4);
            int64_t *gip = calloc((size_t)ngt + 1, sizeof(int64_t));
            for (uint32_t i = 0; i < n; i++) if (cnt[i]) { rows[rowid[i]] = (int32_t)i; gip[rowid[i] + 1] = cnt[i]; }
            for (uint32_t r = 0; r < ngt; r++) gip[r + 1] += gip[r];
            int64_t *gfill = malloc((size_t)ngt * sizeof(int64_t) + 8);
            memcpy(gfill, gip, (size_t)ngt * sizeof(int64_t));
            int32_t *ggidx = malloc(tot * 4 + 4); float *gw = malloc(tot * 4 + 4); int32_t *gslot = malloc(tot * 4 + 4);
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
            e->b_gtrows = mkbuf(e->dev, (size_t)ngt * 4, rows);
            e->b_gtindptr = mkbuf(e->dev, (size_t)(ngt + 1) * 8, gip);
            e->b_gtgidx = mkbuf(e->dev, tot * 4, ggidx);
            e->b_gtw = mkbuf(e->dev, tot * 4, gw);
            e->b_gtslot = mkbuf(e->dev, tot * 4, gslot);
            e->b_gscale = mkbuf(e->dev, (size_t)ng_ * 4, NULL);
            free(order); free(rowid); free(cnt); free(rows); free(gip); free(gfill); free(ggidx); free(gw); free(gslot);
        }
        free(gsl);
        e->b_bcount = mkbuf(e->dev, nf, NULL);
        e->b_bin = mkbuf(e->dev, (size_t)n * BIN_CAP * 8, NULL);   /* (key, weight bits) per arrival */
        e->b_touched = mkbuf(e->dev, nf, NULL);                    /* targets hit this tick */
        size_t bitsb = (size_t)e->window_ticks * words * 4;
        e->b_cand = mkbuf(e->dev, bitsb, NULL);
        e->b_force = mkbuf(e->dev, bitsb, NULL);
        e->b_fired = mkbuf(e->dev, (size_t)e->window_ticks * cap * 4, NULL);
        /* args: per-tick dispatch triples, then two parities of fire bits
         * (kgroup ticks x words) for k_fold's overflow path */
        e->P.fbits_slot = e->window_ticks * 3 + 4;
        e->P.kgroup = e->kgroup;
        e->P.ftrip_slot = e->P.fbits_slot + 2u * e->kgroup * words;   /* k_fold's per-tick dispatch triples */
        e->P.tcnt_slot = e->P.ftrip_slot + e->window_ticks * 3;      /* per-tick touched-target counts */
        e->b_args = mkbuf(e->dev, ((size_t)e->P.tcnt_slot + e->window_ticks) * 4, NULL);
        e->b_indptr = mkbuf(e->dev, (size_t)(n + 1) * 8, d->indptr);
        e->b_indices = mkbuf(e->dev, (size_t)d->nnz * 4, d->indices);
        e->b_weight = mkbuf(e->dev, (size_t)d->nnz * 4, d->weight);
        e->b_delay = mkbuf(e->dev, nf, d->delay_slot);
        e->b_gidx = mkbuf(e->dev, (size_t)d->n_graded * 4, d->graded_idx);
        e->b_gv50 = mkbuf(e->dev, (size_t)d->n_graded * 4, d->gv50);
        e->b_gslope = mkbuf(e->dev, (size_t)d->n_graded * 4, d->gslope);
        {   /* per-row gain: the scalar broadcast to every row when absent,
             * so the kernel's ggain[tg] * rel is the same two floats the
             * scalar form multiplied. */
            float *gg = d->n_graded ? malloc((size_t)d->n_graded * 4) : NULL;
            for (uint32_t k = 0; k < d->n_graded; k++)
                gg[k] = d->ggain_rows ? d->ggain_rows[k] : d->ggain_scaled;
            e->b_ggain = mkbuf(e->dev, (size_t)d->n_graded * 4, gg);
            free(gg);
        }
        e->b_vrec = mkbuf(e->dev, (size_t)d->n_vrec * 4, d->vrec);
        e->b_vpre = mkbuf(e->dev, d->n_vrec ? nf * 4
                                   : d->n_graded ? nf : 0, NULL);
        e->b_vtap = mkbuf(e->dev,
                          (size_t)d->n_vrec * e->window_ticks * 4 * 4,
                          NULL);
        e->b_relcur = mkbuf(e->dev, (size_t)d->n_graded * 4, NULL);
        e->b_relpos = mkbuf(e->dev, (size_t)d->n_reltap * 4, d->reltap_pos);
        e->b_reltap = mkbuf(e->dev, (size_t)d->n_reltap * e->window_ticks * 4, NULL);
        /* Packed to stay within Metal's 31-buffer kernel limit:
         * bits = [graded clamp | v_rest override | membrane command | coincidence target],
         * values = [graded clamp voltage | membrane kick mV | coincidence product]. */
        e->b_clampbits = mkbuf(e->dev, (size_t)words * 16, NULL);
        e->b_clampval = mkbuf(e->dev, nf * (d->n_gap ? 5 : 4), NULL);   /* 0 clamp, 1 membrane drive, 2 seam product, 3 tonic ge, 4 gap current */
        if (d->n_gap) {
            const uint64_t gnz = (uint64_t)d->gap_indptr[d->n_gap];
            e->b_gap_tgt = mkbuf(e->dev, (size_t)d->n_gap * 4, d->gap_tgt);
            e->b_gap_ip = mkbuf(e->dev, ((size_t)d->n_gap + 1) * 8, d->gap_indptr);
            e->b_gap_src = mkbuf(e->dev, (size_t)gnz * 4, d->gap_src);
            e->b_gap_w = mkbuf(e->dev, (size_t)gnz * 4, d->gap_weight);
            e->b_gap_deg = mkbuf(e->dev, (size_t)d->n_gap * 4, d->gap_deg);
            e->b_gap_ring = mkbuf(e->dev, ((size_t)d->gap_delay + 1) * d->n_gap * 4, NULL);
            e->b_gap_ap = mkbuf(e->dev, (size_t)e->P.gap_ap_slots * d->n_gap * 4, NULL);
            e->b_gap_wave = mkbuf(e->dev, (size_t)d->gap_n_ap * 4, d->gap_ap_wave);
            if (!e->b_gap_tgt || !e->b_gap_ip || !e->b_gap_src
                    || !e->b_gap_w || !e->b_gap_deg || !e->b_gap_ring
                    || !e->b_gap_ap || !e->b_gap_wave) {
                seterr(err, errlen, "gap-junction buffer allocation failed");
                cns_engine_destroy(e); return NULL;
            }
        }
        if (d->tonic_ge) {
            memcpy((uint8_t *)e->b_clampval.contents + (size_t)n * 12, d->tonic_ge, nf);
            e->P.has_tonic = 1;
        }
        if (ovr) memcpy((uint8_t *)e->b_clampbits.contents + (size_t)words0 * 4, ovr, words0 * 4);
        {   /* coincidence seam buffers; 4-byte placeholders when off */
            const uint32_t nc = d->n_coin;
            const size_t fnnz = nc ? (size_t)d->coin_fast_indptr[nc] : 0;
            const size_t snnz = nc ? (size_t)d->coin_slow_indptr[nc] : 0;
            e->b_ctgt = mkbuf(e->dev, (size_t)nc * 4, nc ? d->coin_tgt : NULL);
            e->b_cfip = mkbuf(e->dev, (size_t)(nc + 1) * 8, nc ? d->coin_fast_indptr : NULL);
            e->b_cfsrc = mkbuf(e->dev, fnnz * 4, fnnz ? d->coin_fast_src : NULL);
            e->b_cfw = mkbuf(e->dev, fnnz * 4, fnnz ? d->coin_fast_w : NULL);
            e->b_csip = mkbuf(e->dev, (size_t)(nc + 1) * 8, nc ? d->coin_slow_indptr : NULL);
            e->b_cssrc = mkbuf(e->dev, snnz * 4, snnz ? d->coin_slow_src : NULL);
            e->b_csw = mkbuf(e->dev, snnz * 4, snnz ? d->coin_slow_w : NULL);
            e->b_spk = mkbuf(e->dev, nc ? nf : 0, NULL);
            if (nc && e->b_spk) memset(e->b_spk.contents, 0xFF, nf);   /* never fired */
            e->b_cfs = mkbuf(e->dev, (size_t)nc * 4, NULL);
            e->b_css = mkbuf(e->dev, (size_t)nc * 4, NULL);
            e->b_cring = mkbuf(e->dev, (size_t)e->P.coin_lag * nc * 4, NULL);
            e->b_cstats = mkbuf(e->dev, (size_t)nc * 16, NULL);
            if (nc && e->b_clampbits) {
                uint32_t *tm = (uint32_t *)e->b_clampbits.contents + (size_t)words * 3;
                for (uint32_t j = 0; j < nc; j++) {
                    uint32_t r = (uint32_t)d->coin_tgt[j];
                    tm[r >> 5] |= 1u << (r & 31u);
                }
            }
        }
        e->b_ovr = mkbuf(e->dev, 24, NULL);   /* [0] k_fold overflows (bin arrivals past BIN_CAP), [1] largest bin seen, rounded to 8, [2..5] target-ticks whose bin reached 4, 8, 16, 32 */
        e->P.n_gt = e->n_gt;
        if (e->bP) memcpy(e->bP.contents, &e->P, sizeof e->P);   /* fields set after bP was made */
        free(ovr);
        e->sparse_vrest = sparse_vr;
        e->clamp_on = 0;
        e->membrane_drive_on = 0;
        id<MTLBuffer> all[] = { e->bP, e->b_vrest, e->b_vreset, e->b_spike,
            e->b_decv, e->b_dtt, e->b_decg, e->b_rt, e->b_graded, e->b_aa,
            e->b_ab, e->b_dtw, e->b_v, e->b_g, e->b_gi, e->b_ref, e->b_w,
            e->b_counts, e->b_ring, e->b_ring_i, e->b_gslots,
            e->b_cand, e->b_force, e->b_fired, e->b_args, e->b_indptr,
            e->b_indices, e->b_weight, e->b_delay, e->b_gidx, e->b_gv50,
            e->b_gslope, e->b_ggain, e->b_vrec, e->b_vpre, e->b_vtap, e->b_relcur,
            e->b_relpos, e->b_reltap, e->b_clampbits, e->b_clampval,
            e->b_ovr, e->b_ctgt, e->b_cfip, e->b_cfsrc, e->b_cfw,
            e->b_csip, e->b_cssrc, e->b_csw, e->b_spk, e->b_cfs, e->b_css,
            e->b_cring, e->b_cstats, e->b_bcount, e->b_bin, e->b_touched,
            e->b_tindptr, e->b_tsrc, e->b_tw, e->b_gtrows, e->b_gtindptr,
            e->b_gtgidx, e->b_gtw, e->b_gtslot, e->b_gscale };
        for (size_t k = 0; k < sizeof all / sizeof all[0]; k++)
            if (!all[k]) {
                /* Unified memory: these buffers come out of the same pool as
                 * every process's RSS, so this fires when too many workers
                 * run at once, not when "the GPU is busy". The engine never
                 * falls back to numpy on its own: a silent engine switch
                 * would put two arms of a comparison on two engines, which
                 * is the one thing the fleet rule forbids. */
                char m[256];
                snprintf(m, sizeof m,
                         "buffer allocation failed: this engine needs about %llu MB of "
                         "unified memory (%u ring slots x %u neurons). Run fewer worker "
                         "processes, or use the numpy backend for BOTH arms of the "
                         "comparison; the engine will not switch by itself.",
                         (unsigned long long)(((uint64_t)d->n_ring * d->n * 4ull *
                             (d->split ? 2ull : 1ull) + (uint64_t)d->n * (BIN_CAP * 8ull + 8ull)
                             + d->nnz * 16ull) / 1000000ull),
                         d->n_ring, d->n);
                seterr(err, errlen, m);
                cns_engine_destroy(e); return NULL;
            }
        return e;
    }
}

void cns_engine_destroy(cns_engine *e) {
    if (!e) return;
    @autoreleasepool {
        e->bP = e->b_vrest = e->b_vreset = e->b_spike = e->b_decv = e->b_dtt =
        e->b_decg = e->b_rt = e->b_graded = e->b_aa = e->b_ab = e->b_dtw =
        e->b_v = e->b_g = e->b_gi = e->b_ref = e->b_w = e->b_counts =
        e->b_ring = e->b_ring_i = e->b_gslots = e->b_cand =
        e->b_force = e->b_fired = e->b_args = e->b_indptr = e->b_indices =
        e->b_weight = e->b_delay = e->b_gidx = e->b_gv50 = e->b_gslope = e->b_ggain = nil;
        e->b_vrec = e->b_vpre = e->b_vtap = e->b_relcur = e->b_relpos = e->b_reltap = nil;
        e->b_clampbits = e->b_clampval = e->b_ovr = nil;
        e->b_bcount = e->b_bin = e->b_touched = e->b_tindptr = e->b_tsrc = e->b_tw = nil;
        e->b_gtrows = e->b_gtindptr = e->b_gtgidx = e->b_gtw = e->b_gtslot = e->b_gscale = nil;
        e->b_ctgt = e->b_cfip = e->b_cfsrc = e->b_cfw = e->b_csip = e->b_cssrc = e->b_csw = nil;
        e->b_spk = e->b_cfs = e->b_css = e->b_cring = e->b_cstats = nil;
        e->b_gap_tgt = e->b_gap_ip = e->b_gap_src = e->b_gap_w = e->b_gap_deg = nil;
        e->b_gap_ring = e->b_gap_ap = e->b_gap_wave = nil;
        e->p_gap_current = e->p_gap_spikelet = e->p_gap_arrive = nil;
        e->sample_buffers = nil;
        e->p_step = e->p_scatter = e->p_fold = e->p_graded = e->p_gsum = e->p_vtap = e->p_reltap = e->p_coin = nil;
        e->queue = nil; e->dev = nil;
    }
    free(e);
}

int64_t cns_engine_run(cns_engine *e, uint32_t t0, uint32_t n_ticks,
                       const uint32_t *cand_bits, const uint32_t *force_bits,
                       uint32_t *out_tick, int32_t *out_id, int64_t cap) {
    if (n_ticks == 0) return 0;
    if (n_ticks > e->window_ticks) { seterr(e->err, sizeof e->err, "n_ticks > window_ticks"); return -2; }
    @autoreleasepool {
        const uint32_t n = e->P.n, words = e->P.words, fcap = e->P.fired_cap;
        uint32_t *args = (uint32_t *)e->b_args.contents;
        for (uint32_t k = 0; k < n_ticks; k++) {
            args[k * 3] = 0; args[k * 3 + 1] = 1; args[k * 3 + 2] = 1;
            const uint32_t f = e->P.ftrip_slot + k * 3;   /* k_fold's triple and touched count */
            args[f] = 0; args[f + 1] = 1; args[f + 2] = 1; args[e->P.tcnt_slot + k] = 0;
        }
        size_t bitsb = (size_t)n_ticks * words * 4;
        uint32_t have = 0;
        if (cand_bits) { memcpy(e->b_cand.contents, cand_bits, bitsb); have |= 1u; }
        if (force_bits) { memcpy(e->b_force.contents, force_bits, bitsb); have |= 2u; }
        if (e->clamp_on) have |= 4u;
        if (e->membrane_drive_on) have |= 8u;
        if (e->P.n_coin) have |= 16u;

        id<MTLCommandBuffer> cb = [e->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = nil;
        __block NSUInteger nenc = 0;
        NSMutableData *sequence = e->profile ? [NSMutableData dataWithLength:e->profile_capacity] : nil;
        uint8_t *kseq = sequence.mutableBytes;
        id<MTLComputeCommandEncoder> (^begin)(int) = ^id<MTLComputeCommandEncoder>(int kid) {
            if (!e->profile) return enc;
            MTLComputePassDescriptor *d = [MTLComputePassDescriptor computePassDescriptor];
            d.sampleBufferAttachments[0].sampleBuffer = e->sample_buffers[nenc / 2048];
            d.sampleBufferAttachments[0].startOfEncoderSampleIndex = (nenc % 2048) * 2;
            d.sampleBufferAttachments[0].endOfEncoderSampleIndex = (nenc % 2048) * 2 + 1;
            return [cb computeCommandEncoderWithDescriptor:d];
        };
        if (!e->profile)
            enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        MTLSize tgN = MTLSizeMake(256, 1, 1);
        MTLSize gridN = MTLSizeMake(((size_t)n + 255) / 256 * 256, 1, 1);
        const uint32_t ng = e->P.n_graded, gev = e->P.g_every;
        for (uint32_t kw = 0; kw < n_ticks; kw += e->kgroup) {
            uint32_t K = kw + e->kgroup <= n_ticks ? e->kgroup : n_ticks - kw;
            uint32_t tick0 = t0 + kw;
            const uint32_t par = e->grp & 1u; e->grp++;
            const uint32_t have_g = have | (par << 8);
            id<MTLComputeCommandEncoder> pe = enc;
            if (e->P.n_gap && e->P.gap_delay) {
                const uint32_t n_gap = e->P.n_gap;
                if (e->profile) { kseq[nenc] = 6; enc = begin(6); nenc++; }
                [enc setComputePipelineState:e->p_gap_arrive];
                [enc setBuffer:e->bP offset:0 atIndex:0];
                [enc setBuffer:e->b_gap_tgt offset:0 atIndex:1];
                [enc setBuffer:e->b_v offset:0 atIndex:2];
                [enc setBuffer:e->b_gap_ring offset:0 atIndex:3];
                [enc setBytes:&tick0 length:4 atIndex:4];
                [enc dispatchThreads:MTLSizeMake(n_gap, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(n_gap < 256 ? n_gap : 256, 1, 1)];
                if (e->profile) { [enc endEncoding]; enc = pe; }
                else [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            if (e->P.n_gap && e->P.gap_scale != 0.0f) {
                const uint32_t n_gap = e->P.n_gap;
                if (e->profile) { kseq[nenc] = 5; enc = begin(5); nenc++; }
                [enc setComputePipelineState:e->p_gap_current];
                [enc setBuffer:e->bP offset:0 atIndex:0];
                [enc setBuffer:e->b_gap_tgt offset:0 atIndex:1];
                [enc setBuffer:e->b_gap_ip offset:0 atIndex:2];
                [enc setBuffer:e->b_gap_src offset:0 atIndex:3];
                [enc setBuffer:e->b_gap_w offset:0 atIndex:4];
                [enc setBuffer:e->b_gap_deg offset:0 atIndex:5];
                [enc setBuffer:e->b_v offset:0 atIndex:6];
                [enc setBuffer:e->b_clampval offset:0 atIndex:7];
                [enc setBuffer:e->b_gap_ap offset:0 atIndex:8];
                [enc setBytes:&tick0 length:4 atIndex:9];
                [enc dispatchThreads:MTLSizeMake(n_gap, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(n_gap < 256 ? n_gap : 256, 1, 1)];
                if (e->profile) { [enc endEncoding]; enc = pe; }
                else [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            if (e->P.n_coin) {   /* K == 1 whenever n_coin > 0: reads last tick's stamps */
                const uint32_t nc = e->P.n_coin;
                if (e->profile) { kseq[nenc] = 4; enc = begin(4); nenc++; }
                [enc setComputePipelineState:e->p_coin];
                [enc setBuffer:e->bP offset:0 atIndex:0];
                [enc setBuffer:e->b_ctgt offset:0 atIndex:1];
                [enc setBuffer:e->b_cfip offset:0 atIndex:2];
                [enc setBuffer:e->b_cfsrc offset:0 atIndex:3];
                [enc setBuffer:e->b_cfw offset:0 atIndex:4];
                [enc setBuffer:e->b_csip offset:0 atIndex:5];
                [enc setBuffer:e->b_cssrc offset:0 atIndex:6];
                [enc setBuffer:e->b_csw offset:0 atIndex:7];
                [enc setBuffer:e->b_spk offset:0 atIndex:8];
                [enc setBuffer:e->b_cfs offset:0 atIndex:9];
                [enc setBuffer:e->b_css offset:0 atIndex:10];
                [enc setBuffer:e->b_cring offset:0 atIndex:11];
                [enc setBuffer:e->b_clampval offset:0 atIndex:12];
                [enc setBuffer:e->b_cstats offset:0 atIndex:13];
                [enc setBytes:&tick0 length:4 atIndex:14];
                [enc dispatchThreads:MTLSizeMake(nc, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(nc < 256 ? nc : 256, 1, 1)];
                if (e->profile) { [enc endEncoding]; enc = pe; }
                else [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            if (e->profile) { kseq[nenc] = 0; enc = begin(0); nenc++; }
            [enc setComputePipelineState:e->p_step];
            [enc setBuffer:e->bP offset:0 atIndex:0];
            [enc setBuffer:e->b_vrest offset:0 atIndex:1];
            [enc setBuffer:e->b_vreset offset:0 atIndex:2];
            [enc setBuffer:e->b_spike offset:0 atIndex:3];
            [enc setBuffer:e->b_decv offset:0 atIndex:4];
            [enc setBuffer:e->b_dtt offset:0 atIndex:5];
            [enc setBuffer:e->b_decg offset:0 atIndex:6];
            [enc setBuffer:e->b_rt offset:0 atIndex:7];
            [enc setBuffer:e->b_graded offset:0 atIndex:8];
            [enc setBuffer:e->b_aa offset:0 atIndex:9];
            [enc setBuffer:e->b_ab offset:0 atIndex:10];
            [enc setBuffer:e->b_dtw offset:0 atIndex:11];
            [enc setBuffer:e->b_v offset:0 atIndex:12];
            [enc setBuffer:e->b_g offset:0 atIndex:13];
            [enc setBuffer:e->b_gi offset:0 atIndex:14];
            [enc setBuffer:e->b_ref offset:0 atIndex:15];
            [enc setBuffer:e->b_w offset:0 atIndex:16];
            [enc setBuffer:e->b_counts offset:0 atIndex:17];
            [enc setBuffer:e->b_ring offset:0 atIndex:18];
            [enc setBuffer:e->b_ring_i offset:0 atIndex:19];
            [enc setBuffer:e->b_cand offset:0 atIndex:20];
            [enc setBuffer:e->b_force offset:0 atIndex:21];
            [enc setBuffer:e->b_fired offset:0 atIndex:22];
            [enc setBuffer:e->b_args offset:0 atIndex:23];
            [enc setBytes:&tick0 length:4 atIndex:24];
            [enc setBytes:&K length:4 atIndex:25];
            [enc setBytes:&kw length:4 atIndex:26];
            [enc setBytes:&have_g length:4 atIndex:27];
            [enc setBuffer:e->b_vpre offset:0 atIndex:28];
            [enc setBuffer:e->b_clampbits offset:0 atIndex:29];
            [enc setBuffer:e->b_clampval offset:0 atIndex:30];
            [enc dispatchThreads:gridN threadsPerThreadgroup:tgN];
            if (e->profile) { [enc endEncoding]; enc = pe; }
            else [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            if (e->P.n_vrec) {   /* K == 1 whenever n_vrec > 0 */
                uint32_t nv = e->P.n_vrec;
                if (e->profile) { kseq[nenc] = 0; enc = begin(0); nenc++; }
                [enc setComputePipelineState:e->p_vtap];
                [enc setBuffer:e->b_vrec offset:0 atIndex:0];
                [enc setBuffer:e->b_vpre offset:0 atIndex:1];
                [enc setBuffer:e->b_vtap offset:0 atIndex:2];
                [enc setBytes:&nv length:4 atIndex:3];
                [enc setBytes:&kw length:4 atIndex:4];
                [enc setBytes:&e->P.n length:4 atIndex:5];
                [enc dispatchThreads:MTLSizeMake(nv, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(nv < 256 ? nv : 256, 1, 1)];
                if (e->profile) { [enc endEncoding]; enc = pe; }
            }
            for (uint32_t k = 0; k < K; k++) {
                uint32_t tick = tick0 + k;
                const uint32_t wt = kw + k;   /* window tick: the args slot both kernels use */
                if (e->profile) { kseq[nenc] = 1; enc = begin(1); nenc++; }
                [enc setComputePipelineState:e->p_scatter];
                [enc setBuffer:e->bP offset:0 atIndex:0];
                [enc setBuffer:e->b_fired offset:(NSUInteger)(kw + k) * fcap * 4 atIndex:1];
                [enc setBuffer:e->b_indptr offset:0 atIndex:2];
                [enc setBuffer:e->b_indices offset:0 atIndex:3];
                [enc setBuffer:e->b_weight offset:0 atIndex:4];
                [enc setBuffer:e->b_delay offset:0 atIndex:5];
                [enc setBuffer:e->b_bcount offset:0 atIndex:6];
                [enc setBuffer:e->b_bin offset:0 atIndex:7];
                [enc setBuffer:e->b_touched offset:0 atIndex:8];
                [enc setBytes:&tick length:4 atIndex:9];
                [enc setBuffer:e->b_spk offset:0 atIndex:10];
                [enc setBytes:&k length:4 atIndex:11];
                [enc setBuffer:e->b_ovr offset:0 atIndex:12];
                [enc setBuffer:e->b_args offset:0 atIndex:13];
                [enc setBytes:&wt length:4 atIndex:14];
                [enc dispatchThreadgroupsWithIndirectBuffer:e->b_args
                    indirectBufferOffset:(NSUInteger)(kw + k) * 12
                    threadsPerThreadgroup:MTLSizeMake(SCATTER_TG, 1, 1)];
                if (e->profile) { [enc endEncoding]; enc = pe; }
                if (!e->profile) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                {   /* this tick's bins into the ring, one thread per touched target */
                    if (e->profile) { kseq[nenc] = 3; enc = begin(3); nenc++; }
                    [enc setComputePipelineState:e->p_fold];
                    [enc setBuffer:e->bP offset:0 atIndex:0];
                    [enc setBuffer:e->b_bcount offset:0 atIndex:1];
                    [enc setBuffer:e->b_bin offset:0 atIndex:2];
                    [enc setBuffer:e->b_touched offset:0 atIndex:3];
                    [enc setBuffer:e->b_ring offset:0 atIndex:4];
                    [enc setBuffer:e->b_ring_i offset:0 atIndex:5];
                    [enc setBuffer:e->b_args offset:0 atIndex:6];
                    [enc setBuffer:e->b_tindptr offset:0 atIndex:7];
                    [enc setBuffer:e->b_tsrc offset:0 atIndex:8];
                    [enc setBuffer:e->b_tw offset:0 atIndex:9];
                    [enc setBuffer:e->b_delay offset:0 atIndex:10];
                    [enc setBytes:&tick0 length:4 atIndex:11];
                    [enc setBytes:&k length:4 atIndex:12];
                    [enc setBytes:&par length:4 atIndex:13];
                    [enc setBytes:&wt length:4 atIndex:14];
                    [enc dispatchThreadgroupsWithIndirectBuffer:e->b_args
                        indirectBufferOffset:(NSUInteger)(e->P.ftrip_slot + wt * 3) * 4
                        threadsPerThreadgroup:MTLSizeMake(FOLD_TG, 1, 1)];
                    if (e->profile) { [enc endEncoding]; enc = pe; }
                    else [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
            }
            if (ng && (tick0 % gev) == 0) {   /* K == 1 whenever ng > 0 */
                uint32_t tick = tick0;
                if (e->profile) { kseq[nenc] = 2; enc = begin(2); nenc++; }
                [enc setComputePipelineState:e->p_graded];
                [enc setBuffer:e->bP offset:0 atIndex:0];
                [enc setBuffer:e->b_gidx offset:0 atIndex:1];
                [enc setBuffer:e->b_vpre offset:0 atIndex:2];   /* pre-reset membrane, cns.py's release order */
                [enc setBuffer:e->b_gv50 offset:0 atIndex:3];
                [enc setBuffer:e->b_gslope offset:0 atIndex:4];
                [enc setBuffer:e->b_relcur offset:0 atIndex:5];
                [enc setBuffer:e->b_ggain offset:0 atIndex:6];
                [enc setBuffer:e->b_gscale offset:0 atIndex:7];
                [enc dispatchThreads:MTLSizeMake(ng, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(ng < 256 ? ng : 256, 1, 1)];
                if (e->profile) { [enc endEncoding]; enc = pe; }
                else [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                if (e->P.n_reltap) {
                    uint32_t nt_ = e->P.n_reltap;
                    if (e->profile) { kseq[nenc] = 2; enc = begin(2); nenc++; }
                    [enc setComputePipelineState:e->p_reltap];
                    [enc setBuffer:e->b_relpos offset:0 atIndex:0];
                    [enc setBuffer:e->b_relcur offset:0 atIndex:1];
                    [enc setBuffer:e->b_reltap offset:0 atIndex:2];
                    [enc setBytes:&nt_ length:4 atIndex:3];
                    [enc setBytes:&kw length:4 atIndex:4];
                    [enc dispatchThreads:MTLSizeMake(nt_, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(nt_ < 256 ? nt_ : 256, 1, 1)];
                    if (e->profile) { [enc endEncoding]; enc = pe; }
                }
                if (e->n_gt) {
                    const uint32_t ngt = e->n_gt;
                    if (e->profile) { kseq[nenc] = 2; enc = begin(2); nenc++; }
                    [enc setComputePipelineState:e->p_gsum];
                    [enc setBuffer:e->bP offset:0 atIndex:0];
                    [enc setBuffer:e->b_gtrows offset:0 atIndex:1];
                    [enc setBuffer:e->b_gtindptr offset:0 atIndex:2];
                    [enc setBuffer:e->b_gtgidx offset:0 atIndex:3];
                    [enc setBuffer:e->b_gtw offset:0 atIndex:4];
                    [enc setBuffer:e->b_gtslot offset:0 atIndex:5];
                    [enc setBuffer:e->b_gscale offset:0 atIndex:6];
                    [enc setBuffer:e->b_ring offset:0 atIndex:7];
                    [enc setBuffer:e->b_ring_i offset:0 atIndex:8];
                    [enc setBytes:&tick length:4 atIndex:9];
                    [enc dispatchThreads:MTLSizeMake(ngt, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(ngt < 256 ? ngt : 256, 1, 1)];
                    if (e->profile) { [enc endEncoding]; enc = pe; }
                }
            }
            if (e->P.n_gap && (e->P.gap_spikelet != 0.0f || e->P.gap_n_ap)) {
                const uint32_t n_gap = e->P.n_gap;
                if (e->profile) { kseq[nenc] = 6; enc = begin(6); nenc++; }
                [enc setComputePipelineState:e->p_gap_spikelet];
                [enc setBuffer:e->bP offset:0 atIndex:0];
                [enc setBuffer:e->b_gap_tgt offset:0 atIndex:1];
                [enc setBuffer:e->b_gap_ip offset:0 atIndex:2];
                [enc setBuffer:e->b_gap_src offset:0 atIndex:3];
                [enc setBuffer:e->b_gap_w offset:0 atIndex:4];
                [enc setBuffer:e->b_v offset:0 atIndex:5];
                [enc setBuffer:e->b_args offset:0 atIndex:6];
                [enc setBytes:&par length:4 atIndex:7];
                [enc setBytes:&kw length:4 atIndex:8];
                [enc setBuffer:e->b_gap_ring offset:0 atIndex:9];
                [enc setBuffer:e->b_gap_ap offset:0 atIndex:10];
                [enc setBuffer:e->b_gap_wave offset:0 atIndex:11];
                [enc setBytes:&tick0 length:4 atIndex:12];
                [enc dispatchThreads:MTLSizeMake(n_gap, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(n_gap < 256 ? n_gap : 256, 1, 1)];
                if (e->profile) { [enc endEncoding]; enc = pe; }
            }
            if (!e->profile) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        if (!e->profile) [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            seterr(e->err, sizeof e->err, cb.error.localizedDescription.UTF8String);
            return -2;
        }
        e->gpu_s += cb.GPUEndTime - cb.GPUStartTime;
        if (e->profile && nenc) {
            for (NSUInteger off = 0; off < nenc; off += 2048) {
                NSUInteger cnt = MIN(nenc - off, 2048);
                NSData *data = [e->sample_buffers[off / 2048] resolveCounterRange:NSMakeRange(0, cnt * 2)];
                if (data) {
                    const MTLCounterResultTimestamp *ts = data.bytes;
                    NSUInteger have = data.length / sizeof(MTLCounterResultTimestamp);
                    for (NSUInteger i = 0; i < cnt && i * 2 + 1 < have; i++) {
                        uint64_t a = ts[i * 2].timestamp, b = ts[i * 2 + 1].timestamp;
                        if (a == MTLCounterErrorValue || b == MTLCounterErrorValue || b < a) continue;
                        e->kns[kseq[off + i]] += (double)(b - a); e->kcount[kseq[off + i]]++;
                    }
                }
            }
        }

        int64_t total = 0;
        for (uint32_t k = 0; k < n_ticks; k++) e->touched_total += args[e->P.tcnt_slot + k];
        e->ticks_total += n_ticks;
        const uint32_t *fired = (const uint32_t *)e->b_fired.contents;
        for (uint32_t k = 0; k < n_ticks; k++) {
            uint32_t c = args[k * 3];
            if (c > fcap) { seterr(e->err, sizeof e->err, "spike list overflow"); return -3; }
            if (total + c > cap) return -1;
            const uint32_t *lst = fired + (size_t)k * fcap;
            for (uint32_t s = 0; s < c; s++) { out_tick[total] = k; out_id[total] = (int32_t)lst[s]; total++; }
        }
        return total;
    }
}

const float *cns_engine_vtap(cns_engine *e) { return (const float *)e->b_vtap.contents; }
const float *cns_engine_reltap(cns_engine *e) { return (const float *)e->b_reltap.contents; }
void cns_engine_set_clamp(cns_engine *e, const uint32_t *clamp_bits, const float *clamp_val) {
    if (!clamp_bits || !clamp_val) { e->clamp_on = 0; return; }
    memcpy(e->b_clampbits.contents, clamp_bits, (size_t)e->P.words * 4);
    memcpy(e->b_clampval.contents, clamp_val, (size_t)e->P.n * 4);
    e->clamp_on = 1;
}

void cns_engine_set_membrane_drive(cns_engine *e, const uint32_t *row_bits,
                                   const float *kick_mv) {
    if (!row_bits || !kick_mv) { e->membrane_drive_on = 0; return; }
    memcpy((uint8_t *)e->b_clampbits.contents + (size_t)e->P.words * 8,
           row_bits, (size_t)e->P.words * 4);
    memcpy((uint8_t *)e->b_clampval.contents + (size_t)e->P.n * 4,
           kick_mv, (size_t)e->P.n * 4);
    e->membrane_drive_on = 1;
}

void cns_engine_read_state(cns_engine *e, float *v, float *g, float *g_i,
                           float *w_adapt, uint32_t *counts) {
    size_t nf = (size_t)e->P.n * 4;
    if (v) memcpy(v, e->b_v.contents, nf);
    if (g) memcpy(g, e->b_g.contents, nf);
    if (g_i && e->P.split) memcpy(g_i, e->b_gi.contents, nf);
    if (w_adapt && e->P.use_adapt) memcpy(w_adapt, e->b_w.contents, nf);
    if (counts) memcpy(counts, e->b_counts.contents, nf);
}
double cns_engine_gpu_seconds(cns_engine *e) { return e->gpu_s; }
/* per-kernel seconds (step, scatter, graded+gsum+reltap, fold, coin) and encoder counts; zeros unless CNS_ENGINE_PROFILE=1 */
void cns_engine_profile(cns_engine *e, double *secs, uint64_t *counts) {
    for (int k = 0; k < 5; k++) { if (secs) secs[k] = e->kns[k] / 1e9; if (counts) counts[k] = e->kcount[k]; }
}
void cns_engine_gap_profile(cns_engine *e, double *secs, uint64_t *counts) {
    for (int k = 0; k < 2; k++) { if (secs) secs[k] = e->kns[k + 5] / 1e9; if (counts) counts[k] = e->kcount[k + 5]; }
}
void cns_engine_coin_stats(cns_engine *e, double *out4) {
    const uint32_t nc = e->P.n_coin;
    const float *s = (const float *)e->b_cstats.contents;
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
uint32_t cns_engine_fold_overflows(cns_engine *e) { return *(const uint32_t *)e->b_ovr.contents; }
uint32_t cns_engine_bin_cap(void) { return BIN_CAP; }
uint32_t cns_engine_bin_max(cns_engine *e) { return ((const uint32_t *)e->b_ovr.contents)[1]; }
void cns_engine_bin_hist(cns_engine *e, uint32_t *out4) { const uint32_t *o = (const uint32_t *)e->b_ovr.contents; for (int q = 0; q < 4; q++) out4[q] = o[2 + q]; }
double cns_engine_touched_per_tick(cns_engine *e) { return e->ticks_total ? (double)e->touched_total / (double)e->ticks_total : 0.0; }
uint32_t cns_engine_uniform_mask(cns_engine *e) { uint32_t m = 0; for (int k = 0; k < 8; k++) if (e->uniform[k]) m |= 1u << k; if (e->sparse_vrest) m |= 1u << 8; return m; }
const char *cns_engine_device(cns_engine *e) { return e->devname; }
#ifndef CNS_ENGINE_SRC_SHA256
#define CNS_ENGINE_SRC_SHA256 "unset: built without the Makefile"
#endif
const char *cns_engine_source_sha256(void) { return CNS_ENGINE_SRC_SHA256; }
const char *cns_engine_last_error(cns_engine *e) { return e->err; }
