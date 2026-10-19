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
 *
 * Designs 7b0ad6b2 and 727bccb8: the PSI pair channel and the per-row state
 * tap, with cns_engine.cu's exported functions and semantics. A pair keeps
 * its own delayed exponential state x = r*x + inc*n, added to its target's
 * normalized excitatory conductance after tonic. k_step already binds 31
 * buffers, so k_pair updates the states before k_step and leaves each in
 * clamp_val, and k_pair_emit reads the fire bits after it; that split
 * needs single-tick groups, which configuration requires. Unconfigured,
 * both are absent and every earlier path runs unchanged.
 *
 * Design e05c0eef B1: the pair's recovery (dead time or depletion), with
 * cns_engine.cu's exported function, arithmetic and counters, carried by
 * k_pair in buffers 8 to 12; unconfigured (form 0) k_pair runs the pre-B1
 * arithmetic.
 *
 * Design d27f82df B8: B2's exact-update mask and B3's hold-down, with
 * cns_engine.cu's exported functions and semantics. They ride in the opsin
 * slots' buffers 25 to 27, so Metal refuses the opsin beside either: op_pos
 * holds a code per row (-2 neither, -1 exact, q >= 0 hold-down slot q),
 * op_f each slot's 16 floats and then the tap, op_u each slot's form,
 * latch_ticks and peak tick. Unconfigured, every path runs unchanged.
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
    /* pair channel and state tap: counts, then offsets into clamp_bits
     * (target mask, tap row per neuron), clamp_val (pair state and events
     * per target) and v_pre (the tap) */
    uint n_pairs, pair_nr, n_aux, aux_off;
    uint pmask_off, apos_off, px_off, na_off;
    /* design afc8fd03 B3: the injected current, one section of clamp_val */
    uint has_inj, inj_off;
    /* design 302c1bfb Build 8: bit 0 the listed rows' release factor, bit 1
     * its source-fed state; n_psi listed rows (cns_engine_psi_config) */
    uint has_psi, n_psi;
    /* design e2b1a20d B3: the per-row inhibitory reversal, one more section
     * of clamp_val */
    uint has_einh, einh_off;
    /* design 244fb801 Build 1: design 87a3ad35's opsin conductance, its
     * slots in k_step's buffers 25 to 27 (cns_engine_opsin_config) */
    uint n_op, xm_on;
    /* design d27f82df B8: xm_on the exact-update mask, n_hd hold-down slots,
     * hd_tap the per-tick tap, hd_dt the step the pool integrates over; all
     * in buffers 25 to 27 (cns_engine_row_exact_config) */
    uint n_hd, hd_tap; float hd_dt; uint _q7;
    /* design fcb48834 Build 0: design b1746728's receptor-scoped serotonin
     * state (cns_engine_serotonin_config). Bit 0 of has_ser counts the
     * source rows' spikes and runs k_ser, bit 1 gives the target rows the
     * current. ser_tmask and ser_smask are the target and source masks in
     * args; ser_s is s, its maximum, the tick's source count and the 64-bit
     * spike total, five more args slots. */
    uint has_ser, ser_tmask, ser_smask, ser_s;
    float ser_gain, ser_decay, ser_inc; uint ser_spb;
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
/* design 6d0042d5 Build 2: each row's coefficient below rest in the second
 * half of adapt_a (cns_engine_adapt_below_config); undefined is off */
constant bool ADAPT_SPLIT_C [[function_constant(9)]];
constant bool ADAPT_SPLIT = is_function_constant_defined(ADAPT_SPLIT_C) && ADAPT_SPLIT_C;

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
                   constant uint4 &tkw             [[buffer(24)]],   /* t0, K, kw0, have_bits */
                   device const int *op_pos        [[buffer(25)]],   /* per row: opsin slot, or -1; or design d27f82df's code */
                   device float *op_f              [[buffer(26)]],   /* per slot: a_on, a_off, g_max, e_rev, s; or the hold-down's */
                   device uint *op_u               [[buffer(27)]],   /* per slot: train start, stop, on, period; or the hold-down's */
                   device float *v_pre             [[buffer(28)]],
                   device const uint *clamp_bits   [[buffer(29)]],
                   device const float *clamp_val   [[buffer(30)]],
                   uint i [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    const uint t0 = tkw.x, K = tkw.y, kw0 = tkw.z, have_bits = tkw.w;
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
    /* design e2b1a20d B3: the row's own inhibitory reversal when a per-row
     * vector was set, as cns_engine.cu reads lm.e_inh[i] */
    const float ei = p.has_einh ? clamp_val[p.einh_off + i] : p.e_inh;
    /* pair channel target (k_pair left its state and events in clamp_val)
     * and state-tap row, 0xFFFFFFFF for none */
    const bool pq = p.n_pairs != 0 && (clamp_bits[p.pmask_off + word] & bit) != 0;
    const uint ap = p.n_aux ? clamp_bits[p.apos_off + i] : 0xFFFFFFFFu;
    /* design 244fb801 Build 1: the row's opsin slot and its gate */
    const int oq = p.n_op ? op_pos[i] : -1;
    float osv = oq >= 0 ? op_f[5u * (uint)oq + 4u] : 0.0f;
    /* design d27f82df B2 and B3: the row's code, its update and its slot;
     * per slot op_f holds e_rel, decay, dg, gmax, c_spk, k, va, tau_c,
     * latch_mv, then the state c, g_h, L, b, peak, and op_u form,
     * latch_ticks, peak_tick */
    const int xc = (p.xm_on || p.n_hd) ? op_pos[i] : -2;
    const bool exu = p.analytic != 0 || (p.xm_on && xc >= -1);
    const int hq = (p.n_hd && xc >= 0) ? xc : -1;
    const uint hf = hq >= 0 ? 16u * (uint)hq : 0u, hu = hq >= 0 ? 4u * (uint)hq : 0u;
    float hc = hq >= 0 ? op_f[hf + 9u] : 0.0f, hgh = hq >= 0 ? op_f[hf + 10u] : 0.0f;
    float hL = hq >= 0 ? op_f[hf + 11u] : 0.0f, hb = hq >= 0 ? op_f[hf + 12u] : 0.0f;
    float hpk = hq >= 0 ? op_f[hf + 13u] : 0.0f;
    uint hpt = hq >= 0 ? op_u[hu + 2u] : 0u;
    /* design fcb48834 Build 0: the row's serotonin roles, from the masks in args */
    const bool ser_tg = (p.has_ser & 2u) != 0
        && (atomic_load_explicit(&args[p.ser_tmask + word], memory_order_relaxed) & bit) != 0;
    const bool ser_src = (p.has_ser & 1u) != 0
        && (atomic_load_explicit(&args[p.ser_smask + word], memory_order_relaxed) & bit) != 0;
    for (uint k = 0; k < K; k++) {
        const uint tick = t0 + k, kw = kw0 + k;
        const uint slot = tick % p.n_ring;
        const ulong ri = (ulong)slot * p.n + i;
        gg = gg * dg;
        if (split) ggi = ggi * dg;
        /* state tap: fields written where they are formed, cns_engine.cu's order */
        const bool tap = ap != 0xFFFFFFFFu;
        const ulong tb = tap ? (ulong)p.aux_off + ((ulong)kw * p.n_aux + ap) * 16u : 0ul;
        const float arrival = ring[ri];
        const float arr_i = split ? ring_i[ri] : 0.0f;
        if (tap) { v_pre[tb] = vv; v_pre[tb + 2u] = arrival; v_pre[tb + 3u] = arr_i;
                   v_pre[tb + 13u] = ww; v_pre[tb + 14u] = (float)r; }
        gg = gg + arrival; ring[ri] = 0.0f;
        if (split) { ggi = ggi + arr_i; ring_i[ri] = 0.0f; }
        if (coin) gg = gg + clamp_val[2u * p.n + i];
        const float na = pq ? clamp_val[p.na_off + i] : 0.0f;
        const float pxv = pq ? clamp_val[p.px_off + i] : 0.0f;
        if (tap) { v_pre[tb + 4u] = gg; v_pre[tb + 5u] = ggi; v_pre[tb + 6u] = na; v_pre[tb + 7u] = pxv; }
        float extra = 0.0f;
        /* design fcb48834 Build 0: cns.py's serotonin term, set before the
         * gap current: the gain times the state k_ser left, decayed once
         * here as cns.py decays it at the top of the tick (k_ser forms the
         * same product for the state) */
        if (ser_tg)
            extra = p.ser_gain * (as_type<float>(atomic_load_explicit(&args[p.ser_s],
                                  memory_order_relaxed)) * p.ser_decay);
        if (p.n_gap && p.gap_scale != 0.0f)
            extra = extra + clamp_val[4u * p.n + i];
        /* design afc8fd03 B3: team-henri's injected current, where cns.py adds it */
        if (p.has_inj) extra = extra + clamp_val[p.inj_off + i];
        const float irow = extra;   /* design d27f82df: the hold-down tap's i_row */
        if (ua) extra = extra - ww;
        /* design 87a3ad35's gate, cns.py's opsin_lit and op_s: lit on an on
         * phase of the slot's pulse train, s = 1 + (s - 1) a_on, dark
         * s = s a_off, then g = g_max s. Windows, force and the threshold
         * window stay CUDA-only (cns_engine_opsin_config refuses them). */
        float og_op = 0.0f;
        if (oq >= 0) {
            const uint qf = 5u * (uint)oq, qu = 4u * (uint)oq;
            const bool olit = op_u[qu + 3u] != 0u && tick >= op_u[qu] && tick < op_u[qu + 1u]
                && (tick - op_u[qu]) % op_u[qu + 3u] < op_u[qu + 2u];
            osv = olit ? 1.0f + (osv - 1.0f) * op_f[qf] : osv * op_f[qf + 1u];
            og_op = op_f[qf + 2u] * osv;
        }
        float vi;
        if (cb) {
            float ge = max(gg, 0.0f) * p.gscale;
            float gi = split ? ggi * p.gscale : max(-gg, 0.0f) * p.gscale;
            /* tonic excitatory conductance, cns.py's `ge = ge + tonic_ge`,
             * here because it sits after ge and gi are formed and before
             * either integrator. Rides in the 4th section of clamp_val. */
            if (p.has_tonic) ge = ge + clamp_val[3u * p.n + i];
            if (tap) v_pre[tb + 8u] = ge;   /* ordinary excitatory conductance, tonic included */
            if (pq) ge = ge + pxv;
            if (tap) {   /* normalized drives at the tick's starting membrane */
                v_pre[tb + 9u] = gi;
                v_pre[tb + 10u] = pxv * (p.e_exc - vv);
                v_pre[tb + 11u] = ge * (p.e_exc - vv);
                v_pre[tb + 12u] = gi * (ei - vv);
            }
            if (exu) {   /* the analytic integrator, or design d27f82df's mask */
                float total = 1.0f + ge + gi;
                if (hq >= 0) total = total + hgh;   /* B3: last, as cns_engine.cu */
                float inv = 1.0f / total;
                float vinf = vr * inv + p.e_exc * (ge * inv)
                           + ei * (gi * inv) + extra * inv;
                if (hq >= 0) vinf = vinf + (vr + op_f[hf]) * (hgh * inv);
                float cdec = exp(-dtt * total);
                vi = vinf + (vv - vinf) * cdec;
            } else if (oq < 0) {
                vi = vr + (vv - vr) * dv
                   + (ge * (p.e_exc - vv) + gi * (ei - vv) + extra) * dtt;
            } else {   /* cns.py's op_brk: the same bracket, the opsin current last */
                float brk = ge * (p.e_exc - vv) + gi * (ei - vv) + extra;
                brk = brk + og_op * (op_f[5u * (uint)oq + 3u] - vv);
                vi = vr + (vv - vr) * dv + brk * dtt;
            }
        } else {
            if (tap) { v_pre[tb + 8u] = 0.0f; v_pre[tb + 9u] = 0.0f; v_pre[tb + 10u] = 0.0f;
                       v_pre[tb + 11u] = 0.0f; v_pre[tb + 12u] = 0.0f; }
            vi = vr + (vv - vr) * dv + (gg + extra) * dtt;
        }
        if (ua) {
            const float xr = vi - vr;
            const float ae = (ADAPT_SPLIT && xr < 0.0f) ? adapt_a[p.n + i] : aa;
            ww = ww + dtw * (ae * xr - ww);
        }
        float hx = 0.0f;
        if (hq >= 0) {   /* design d27f82df B3: x before the refractory clamp */
            hx = vi - vr;
            if (op_u[hu] == 0u) hgh = hgh * op_f[hf + 1u];
            else {
                const float load = (op_f[hf + 5u] * max(hx - op_f[hf + 6u], 0.0f)) * hL;
                hc = max(hc + p.hd_dt * (load - hc / op_f[hf + 7u]), 0.0f);
                hgh = (op_f[hf + 3u] * hc) / (1.0f + hc);
            }
        }
        const bool active = r <= 0;
        if (!active) vi = vrst;
        const bool cand_event = cbits
            && (cand_bits[(ulong)kw * p.words + word] & bit);
        const bool mem_event = cand_event && membits
            && (clamp_bits[2u * p.words + word] & bit);
        const bool thr = active && vi >= sa && !isg;
        bool fire = thr;
        if (cand_event && active && !mem_event) fire = true;
        const bool forced = fbits && (force_bits[(ulong)kw * p.words + word] & bit) != 0;
        if (forced) fire = true;
        if (clamped) vi = cval;   /* graded drive: set after detection, before release */
        if (tap) {   /* bit 0 active, 1 threshold, 2 candidate, 3 membrane kick,
                      * 4 forced, 5 fired, 6 clamped, 7 graded */
            v_pre[tb + 1u] = vi;
            v_pre[tb + 15u] = (float)((active ? 1u : 0u) | (thr ? 2u : 0u) | (cand_event ? 4u : 0u)
                                      | (mem_event ? 8u : 0u) | (forced ? 16u : 0u) | (fire ? 32u : 0u)
                                      | (clamped ? 64u : 0u) | (isg ? 128u : 0u));
        }
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
            /* design fcb48834 Build 0: a source spike for k_ser, every fire
             * counted, driven ones included, as cns.py's merged `fired` */
            if (ser_src) atomic_fetch_add_explicit(&args[p.ser_s + 2u], 1u, memory_order_relaxed);
            vi = vrst; r = rt; cnt++;
            if (ua) ww = ww + ab;
            if (hq >= 0) {   /* design d27f82df B3 */
                if (op_u[hu] == 0u) hgh = hgh + op_f[hf + 2u];
                else { hc = hc + op_f[hf + 4u]; hgh = (op_f[hf + 3u] * hc) / (1.0f + hc); }
                hL = 1.0f; hb = 0.0f;
            }
        }
        if (hq >= 0) {   /* B3: the latch count on an active tick that did not fire */
            if (active && !fire) {
                hb = hx < op_f[hf + 8u] ? hb + 1.0f : 0.0f;
                if (hb >= (float)op_u[hu + 1u]) hL = 0.0f;
            }
            if (hgh > hpk) { hpk = hgh; hpt = tick; }
            if (p.hd_tap) {
                const uint ht = 16u * p.n_hd + (kw * p.n_hd + (uint)hq) * 5u;
                op_f[ht] = irow; op_f[ht + 1u] = hc; op_f[ht + 2u] = hgh;
                op_f[ht + 3u] = hL; op_f[ht + 4u] = hb;
            }
        }
        vv = vi;
    }
    v[i] = vv; g[i] = gg; refrac[i] = r;
    if (split) g_i[i] = ggi;
    if (ua) w_adapt[i] = ww;
    if (cnt) counts[i] += cnt;
    if (oq >= 0) op_f[5u * (uint)oq + 4u] = osv;
    if (hq >= 0) {
        op_f[hf + 9u] = hc; op_f[hf + 10u] = hgh; op_f[hf + 11u] = hL;
        op_f[hf + 12u] = hb; op_f[hf + 13u] = hpk; op_u[hu + 2u] = hpt;
    }
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

/* Synchronous electrical substeps over the compact junction block only.
 * One thread preserves pair accumulation order. Membrane/AP state is read
 * before k_step; scratch writes never alter the AP or chemical delay rings. */
struct GapLawParams { uint law, pairs, steps, tap, stride; float rho; };
kernel void k_gap_law(constant Params &p [[buffer(0)]],
                     constant GapLawParams &g [[buffer(1)]],
                     device const int *row [[buffer(2)]],
                     device const int2 *pair [[buffer(3)]],
                     device const float *weight [[buffer(4)]],
                     device const float *v [[buffer(5)]],
                     device const float *ap [[buffer(6)]],
                     device const float *dtt [[buffer(7)]],
                     device const short *ref [[buffer(8)]],
                     device float *scratch [[buffer(9)]],
                     device float *current [[buffer(10)]],
                     device float *tap [[buffer(11)]],
                     constant uint &tick [[buffer(12)]],
                     constant uint &kw [[buffer(13)]],
                     device const long *ip [[buffer(14)]],
                     device const int *src [[buffer(15)]],
                     device const float *csr_w [[buffer(16)]],
                     device const float *degree [[buffer(17)]],
                     uint tid [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    if (tid) return;
    const uint m = p.n_gap, abase = (tick % p.gap_ap_slots) * m;
    const uint base = kw * g.stride;
    // Above one substep, accumulate voltage changes rather than repeatedly
    // rounding a small increment onto a roughly -50 mV absolute voltage.
    // The one-step path keeps the qualified legacy arithmetic unchanged.
    const bool refined = g.steps > 1;
    for (uint j = 0; j < m; j++) {
        scratch[j] = refined ? 0.0f : v[row[j]];
        scratch[2*m+j] = 0.0f;
    }
    if (g.tap) {
        for (uint k = 0; k < g.pairs; k++) {
            int2 q = pair[k];
            tap[base + 4*k] = v[row[q.x]];
            tap[base + 4*k+1] = p.gap_n_ap ? ap[abase+q.x] : 0.0f;
            tap[base + 4*k+2] = v[row[q.y]];
            tap[base + 4*k+3] = p.gap_n_ap ? ap[abase+q.y] : 0.0f;
            tap[base + 4*g.pairs+2*k] = 0.0f;
            tap[base + 4*g.pairs+2*k+1] = 0.0f;
        }
    }
    for (uint step = 0; step < g.steps; step++) {
        for (uint j = 0; j < m; j++) scratch[m+j] = 0.0f;
        for (uint k = 0; k < g.pairs; k++) {
            int2 q = pair[k];
            float a = p.gap_n_ap ? ap[abase+q.x] : 0.0f;
            float b = p.gap_n_ap ? ap[abase+q.y] : 0.0f;
            float delta;
            if (refined) {
                delta = (v[row[q.x]]-v[row[q.y]]) + (scratch[q.x]-scratch[q.y]);
                delta = g.law == 1 ? delta + (a-b) : delta + a;
            } else {
                delta = g.law == 1 ? (scratch[q.x]+a)-(scratch[q.y]+b)
                                  : (scratch[q.x]+a)-scratch[q.y];
            }
            float gain = g.law == 1 && delta <= 0.0f ? g.rho*weight[k] : weight[k];
            float c = gain*delta;
            if (g.law == 1) {
                scratch[m+q.y] = scratch[m+q.y] + c;
                scratch[m+q.x] = scratch[m+q.x] - c;
            }
            if (g.tap) {
                tap[base+4*g.pairs+2*k] += c / float(g.steps);
                tap[base+4*g.pairs+2*k+1] += g.law == 1 ? -c / float(g.steps) : 0.0f;
            }
        }
        if (g.law == 2) {
            for (uint j = 0; j < m; j++) {
                float sum = 0.0f;
                for (long k = ip[j]; k < ip[j+1]; k++) {
                    uint a = (uint)src[k];
                    float vv = refined ? v[row[a]] + scratch[a] : scratch[a];
                    if (p.gap_n_ap) vv = vv + ap[abase+a];
                    sum = fma(csr_w[k], vv, sum);
                }
                float vv = refined ? v[row[j]] + scratch[j] : scratch[j];
                scratch[m+j] = p.gap_scale*(sum - degree[j]*vv);
            }
        }
        for (uint j = 0; j < m; j++) {
            if (ref[row[j]] <= 0) {
                float increment = (dtt[row[j]] / float(g.steps))*scratch[m+j];
                if (refined) {
                    float corrected = increment - scratch[2*m+j];
                    float next = scratch[j] + corrected;
                    scratch[2*m+j] = (next-scratch[j]) - corrected;
                    scratch[j] = next;
                } else {
                    scratch[j] = scratch[j] + increment;
                }
            }
        }
    }
    for (uint j = 0; j < m; j++) {
        // At one step, the old one-way kernel has already written its exact sum.
        if (!(g.law == 2 && g.steps == 1))
            current[4*p.n+row[j]] = g.steps == 1 ? scratch[m+j]
                : scratch[j] / dtt[row[j]];
        if (g.tap) tap[base+6*g.pairs+j] = current[4*p.n+row[j]];
    }
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
                      device const float *fac           [[buffer(15)]],   /* has_psi: this tick's release factor per row */
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
    /* design 302c1bfb Build 8: every synapse of the spike delivers its weight
     * times the source's factor, cns.py's data * release_snap in float32;
     * 1.0 for a row k_psi does not list */
    const bool psi = (p.has_psi & 1u) != 0;
    const float fs = psi ? fac[src] : 1.0f;
    for (long e = e0 + lid; e < e1; e += tpt) {
        const uint tgt = (uint)indices[e];
        const uint pos = atomic_fetch_add_explicit(&bcount[tgt], 1u, memory_order_relaxed);
        if (pos < BIN_CAP) {
            const float wv = psi ? weight[e] * fs : weight[e];
            bin[(ulong)tgt * BIN_CAP + pos] = uint2(key, as_type<uint>(wv));
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
                   device const float *fac            [[buffer(15)]],
                   device const int *ppos             [[buffer(16)]],   /* has_psi bit 1: listed position, -1 elsewhere */
                   device const uchar *p9src          [[buffer(17)]],   /* has_psi bit 1: 1 on a source row */
                   device float *ring9                [[buffer(18)]],   /* has_psi bit 1: n_ring x n_psi */
                   uint t [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    const uint fb = p.fbits_slot + par * p.kgroup * p.words;
    if (t >= atomic_load_explicit(&args[p.tcnt_slot + wt], memory_order_relaxed)) return;
    const uint i = touched[t];
    const uint c = bcount[i];   /* at least 1: every enlisted target had an arrival */
    bcount[i] = 0;
    const bool split = p.split != 0;
    /* design 302c1bfb Build 8: a listed target also sums the inhibitory part
     * that comes from the source rows, into its own ring (k_psi reads it) */
    const int hp = (p.has_psi & 2u) ? ppos[i] : -1;
    const uint smask = (1u << p.key_shift) - 1u;
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
            float ph = 0.0f, pl = 0.0f, nh = 0.0f, nl = 0.0f, h9 = 0.0f, l9 = 0.0f;
            uint e = j;
            for (; e < c && (kk[e] >> p.key_shift) == slot; e++) {
                const float w = ww[e];
                if (split) {
                    if (w > 0.0f) dfadd(ph, pl, w);
                    else if (w < 0.0f) {
                        dfadd(nh, nl, -w);
                        if (hp >= 0 && p9src[kk[e] & smask]) dfadd(h9, l9, -w);
                    }
                }
                else dfadd(ph, pl, w);
            }
            const ulong ri = (ulong)slot * p.n + i;
            const float sp = twosum(ph, pl).x;   /* = fl(exact sum), the reference's astype */
            if (sp != 0.0f) ring[ri] += sp;
            if (split) {
                const float sn = twosum(nh, nl).x;
                if (sn != 0.0f) ring_i[ri] += sn;
            }
            if (hp >= 0) {
                const float s9 = twosum(h9, l9).x;
                if (s9 != 0.0f) ring9[(ulong)slot * p.n_psi + (uint)hp] += s9;
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
            const float w = (p.has_psi & 1u) ? tw[e] * fac[src] : tw[e];
            if (split) {
                if (w > 0.0f) ring[ri] += w;
                else if (w < 0.0f) {
                    ring_i[ri] += -w;
                    if (hp >= 0 && p9src[src]) ring9[(ulong)arr * p.n_psi + (uint)hp] += -w;
                }
            }
            else ring[ri] += w;
        }
    }
}

/* Design 302c1bfb, Build 8: the release factor of the listed rows, once per
 * tick between k_step and k_scatter (kgroup 1). With bit 1 each listed row's
 * source-fed state first decays and takes this tick's arrivals, cns.py's
 * `psi_g9 *= psi_r9` and `psi_g9 += psi_c9 * s9`; the state G is that, or
 * the row's g_i after k_step. A listed row that fired this tick gets
 * f = 1/(1 + G*kg) for k_scatter and one log entry (f, G, g_i, 1) at its
 * window tick; the others log a 0 flag. */
kernel void k_psi(constant Params &p            [[buffer(0)]],
                  device const int *prow         [[buffer(1)]],
                  device const float *g_i        [[buffer(2)]],
                  device float *g9               [[buffer(3)]],
                  device float *ring9            [[buffer(4)]],
                  device const float *r9         [[buffer(5)]],
                  device const float *c9         [[buffer(6)]],
                  device float *fac              [[buffer(7)]],
                  device float *plog             [[buffer(8)]],
                  device atomic_uint *args       [[buffer(9)]],
                  constant uint &tick            [[buffer(10)]],
                  constant uint &par             [[buffer(11)]],
                  constant uint &wt              [[buffer(12)]],
                  constant float &kg             [[buffer(13)]],
                  uint j [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    if (j >= p.n_psi) return;
    const uint i = (uint)prow[j];
    float G;
    if (p.has_psi & 2u) {
        const ulong ri = (ulong)(tick % p.n_ring) * p.n_psi + j;
        float s = g9[j] * r9[j];
        s = s + c9[j] * ring9[ri];
        ring9[ri] = 0.0f;
        g9[j] = s;
        G = s;
    } else {
        G = g_i[i];
    }
    const uint bits = atomic_load_explicit(&args[p.fbits_slot + par * p.kgroup * p.words + (i >> 5)],
                                           memory_order_relaxed);
    const ulong lb = ((ulong)wt * p.n_psi + j) * 4u;
    if (bits & (1u << (i & 31u))) {
        const float f = 1.0f / (1.0f + G * kg);
        fac[i] = f;
        plog[lb] = f; plog[lb + 1u] = G; plog[lb + 2u] = g_i[i]; plog[lb + 3u] = 1.0f;
    } else {
        plog[lb + 3u] = 0.0f;
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

/* Design e05c0eef B1: the recovery's form, dead-time ticks, per-tick
 * resource decay and per-event keep (cns_engine_pair_recovery_config). */
struct PairRec { uint form, ticks; float decay, keep; };

/* The pair channel, one thread per pair. Before k_step: the events arriving
 * this tick, then x = r*x + inc*n, cns_engine.cu's arithmetic, with the state
 * and the event count left in clamp_val for the target's k_step thread.
 * Design e05c0eef B1: with a recovery form, n is replaced by the delivered
 * multiplier of cns_engine.cu's pair_recover, the same arithmetic in the
 * same order, its clock, resource, counters and area kept per pair. */
kernel void k_pair(constant Params &p         [[buffer(0)]],
                   device const int *ptgt     [[buffer(1)]],
                   device const float *pr     [[buffer(2)]],
                   device const float *pinc   [[buffer(3)]],
                   device float *px           [[buffer(4)]],
                   device uint *pring         [[buffer(5)]],
                   device float *clamp_val    [[buffer(6)]],
                   constant uint &tick        [[buffer(7)]],
                   constant PairRec &rc       [[buffer(8)]],
                   device uint *pclk          [[buffer(9)]],
                   device float *pres         [[buffer(10)]],
                   device uint *pcnt          [[buffer(11)]],
                   device float *parea        [[buffer(12)]],
                   uint q [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    if (q >= p.n_pairs) return;
    const ulong at = (ulong)(tick % p.pair_nr) * p.n_pairs + q;
    const uint na = pring[at];
    pring[at] = 0u;
    float nd = (float)na;
    if (rc.form != 0u) {
        if (na > 1u) pcnt[4u * q + 3u] += 1u;
        if (rc.form == 1u) {
            uint c = pclk[q];
            if (c > 0u) c -= 1u;
            if (!na) nd = 0.0f;
            else if (c) { pcnt[4u * q + 1u] += na; nd = 0.0f; }
            else { c = rc.ticks; pcnt[4u * q] += na; parea[q] = parea[q] + (float)na; }
            pclk[q] = c;
        } else {
            float r = pres[q];
            r = 1.0f - (1.0f - r) * rc.decay;
            float sm = 0.0f;
            for (uint k = 0; k < na; k++) {
                if (r < 1.0f) pcnt[4u * q + 2u] += 1u;
                sm = sm + r;
                r = r * rc.keep;
            }
            if (na) { pcnt[4u * q] += na; parea[q] = parea[q] + sm; }
            pres[q] = r;
            nd = sm;
        }
    }
    const float x = px[q] * pr[q] + pinc[q] * nd;
    px[q] = x;
    const uint t = (uint)ptgt[q];
    clamp_val[p.px_off + t] = x;
    clamp_val[p.na_off + t] = (float)na;
}

/* After k_step: a pair whose source fired this tick counts one event into
 * the slot tick + delay. Configuration keeps delay in [1, nr - 1], so that
 * slot is neither this tick's nor one read before the arrival tick. */
kernel void k_pair_emit(constant Params &p         [[buffer(0)]],
                        device const int *psrc     [[buffer(1)]],
                        device const uint *pdelay  [[buffer(2)]],
                        device uint *pring         [[buffer(3)]],
                        device const uint *args    [[buffer(4)]],
                        constant uint &tick        [[buffer(5)]],
                        constant uint &par         [[buffer(6)]],
                        uint q [[thread_position_in_grid]]) {
    if (q >= p.n_pairs) return;
    const uint s = (uint)psrc[q];
    if (args[p.fbits_slot + par * p.kgroup * p.words + (s >> 5)] & (1u << (s & 31u))) {
        const ulong at = (ulong)((tick + pdelay[q]) % p.pair_nr) * p.n_pairs + q;
        pring[at] = pring[at] + 1u;
    }
}

/* Design fcb48834 Build 0: design b1746728's serotonin state after k_step,
 * one thread, cns_engine.cu's k_ser in cns.py's order. s is decayed, the
 * decayed value goes to sd[kw] for the host's double applied sum (Metal has
 * no double; the host adds the window's values in tick order, as cns.py
 * adds them), the source spikes k_step counted raise s by inc each as one
 * product and one sum, the maximum and the spike total follow, and the
 * state at a bin end is kept. tw: the tick, the window tick, the bin count. */
kernel void k_ser(constant Params &p     [[buffer(0)]],
                  device uint *args      [[buffer(1)]],
                  device float *sd_out   [[buffer(2)]],
                  device float *bins     [[buffer(3)]],
                  constant uint4 &tw     [[buffer(4)]],
                  uint i [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    if (i) return;
    const uint tick = tw.x, kw = tw.y, n_bins = tw.z;
    const float sd = as_type<float>(args[p.ser_s]) * p.ser_decay;
    sd_out[kw] = sd;
    const uint n = args[p.ser_s + 2u];
    float s = sd;
    if (n) {
        const float t = p.ser_inc * (float)n;
        s = sd + t;
        const ulong sp = ((ulong)args[p.ser_s + 4u] << 32 | (ulong)args[p.ser_s + 3u]) + (ulong)n;
        args[p.ser_s + 3u] = (uint)sp;
        args[p.ser_s + 4u] = (uint)(sp >> 32);
        args[p.ser_s + 1u] = as_type<uint>(max(as_type<float>(args[p.ser_s + 1u]), s));
    }
    args[p.ser_s] = as_type<uint>(s);
    args[p.ser_s + 2u] = 0u;
    if ((tick + 1u) % p.ser_spb == 0u) {
        const uint b = tick / p.ser_spb;
        if (b < n_bins) bins[b] = s;
    }
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
    uint32_t n_pairs, pair_nr, n_aux, aux_off;
    uint32_t pmask_off, apos_off, px_off, na_off;
    uint32_t has_inj, inj_off;
    uint32_t has_psi, n_psi;
    uint32_t has_einh, einh_off;
    uint32_t n_op, xm_on;
    uint32_t n_hd, hd_tap; float hd_dt; uint32_t _q7;
    uint32_t has_ser, ser_tmask, ser_smask, ser_s;
    float ser_gain, ser_decay, ser_inc; uint32_t ser_spb;
} Params;

static int all_same_f(const float *a, uint32_t n) {
    for (uint32_t i = 1; i < n; i++) if (a[i] != a[0]) return 0;
    return 1;
}
static int all_same_i16(const int16_t *a, uint32_t n) {
    for (uint32_t i = 1; i < n; i++) if (a[i] != a[0]) return 0;
    return 1;
}

typedef struct { uint32_t law, pairs, steps, tap, stride; float rho; } GapLawParams;

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
    GapLawParams gap_law;
    id<MTLComputePipelineState> p_gap_law;
    id<MTLBuffer> b_gap_law, b_gap_pairs, b_gap_pairs_w, b_gap_scratch, b_gap_tap;
    /* pair channel and state tap (designs 7b0ad6b2, 727bccb8) */
    id<MTLComputePipelineState> p_pair, p_pair_emit;
    id<MTLBuffer> b_psrc, b_ptgt, b_pdelay, b_pr, b_pinc, b_px, b_pring;
    /* design e05c0eef B1: the pair recovery's parameters, clock, resource,
     * counters and area */
    id<MTLBuffer> b_prc, b_pclk, b_pres, b_pcnt, b_parea;
    /* design 302c1bfb: Build 8's release factor (b_fac is 1.0 on every row
     * it does not list) and Build 10's threshold switch, which rebuilds
     * k_step from the kept library when spike_at stops being uniform */
    id<MTLLibrary> lib;
    id<MTLComputePipelineState> p_psi;
    id<MTLBuffer> b_fac, b_prow, b_g9, b_ring9, b_r9, b_c9, b_plog, b_ppos, b_p9src;
    float psi_kg;
    /* design 244fb801 Build 1: the opsin slots k_step binds at 25 to 27,
     * 4-byte placeholders until cns_engine_opsin_config */
    id<MTLBuffer> b_oppos, b_opf, b_opu;
    /* design fcb48834 Build 0: k_ser, the window's decayed states for the
     * double applied sum, the bin readback */
    id<MTLComputePipelineState> p_ser;
    id<MTLBuffer> b_ser_sd, b_ser_bins;
    uint32_t ser_nbins;
    double ser_applied;
    int op_train;
    int ext_on;   /* clamp_bits and clamp_val carry the pair and tap sections */
    uint64_t nnz, n_gt_entries;   /* for cns_engine_read_buffer */
    int clamp_on, membrane_drive_on;
    uint64_t touched_total, ticks_total;   /* gauge: targets hit per tick, summed */
    bool uniform[8];
    bool sparse_vrest;
    bool adapt_split;   /* design 6d0042d5 Build 2 */
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
        e->lib = lib;
        PSO(e->p_psi, "k_psi");
        PSO(e->p_scatter, "k_scatter");
        PSO(e->p_fold, "k_fold");
        PSO(e->p_graded, "k_graded");
        PSO(e->p_gsum, "k_gsum");
        PSO(e->p_vtap, "k_vtap");
        PSO(e->p_reltap, "k_reltap");
        PSO(e->p_coin, "k_coin");
        PSO(e->p_pair, "k_pair");
        PSO(e->p_pair_emit, "k_pair_emit");
        PSO(e->p_ser, "k_ser");
        if (d->n_gap) {
            PSO(e->p_gap_law, "k_gap_law");
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
            /* At most fourteen encoders per tick with delayed gaps, taps, graded release,
             * coincidence, the pair channel and the serotonin state all enabled. A fixed 2048-encoder table
             * would overflow for a 256-tick window carrying these mechanisms. */
            e->profile_capacity = (NSUInteger)(d->window_ticks ? d->window_ticks : 64) * 14;
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
            e->n_gt_entries = tot;
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
        e->nnz = d->nnz;
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
        e->b_oppos = mkbuf(e->dev, 4, NULL);
        e->b_opf = mkbuf(e->dev, 4, NULL);
        e->b_opu = mkbuf(e->dev, 4, NULL);
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
        e->b_fac = mkbuf(e->dev, nf, NULL);   /* Build 8's factor: 1.0 until k_psi writes a listed row */
        if (e->b_fac) { float *fac = (float *)e->b_fac.contents; for (uint32_t i = 0; i < n; i++) fac[i] = 1.0f; }
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
            e->b_gtgidx, e->b_gtw, e->b_gtslot, e->b_gscale, e->b_fac,
            e->b_oppos, e->b_opf, e->b_opu };
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
        e->p_gap_law = nil;
        e->b_gap_law = e->b_gap_pairs = e->b_gap_pairs_w = e->b_gap_scratch = e->b_gap_tap = nil;
        e->p_gap_current = e->p_gap_spikelet = e->p_gap_arrive = nil;
        e->b_psrc = e->b_ptgt = e->b_pdelay = e->b_pr = e->b_pinc = e->b_px = e->b_pring = nil;
        e->b_prc = e->b_pclk = e->b_pres = e->b_pcnt = e->b_parea = nil;
        e->p_pair = e->p_pair_emit = nil;
        e->b_fac = e->b_prow = e->b_g9 = e->b_ring9 = e->b_r9 = e->b_c9 = e->b_plog = nil;
        e->b_ppos = e->b_p9src = nil;
        e->b_oppos = e->b_opf = e->b_opu = nil;
        e->b_ser_sd = e->b_ser_bins = nil; e->p_ser = nil;
        e->p_psi = nil; e->lib = nil;
        e->sample_buffers = nil;
        e->p_step = e->p_scatter = e->p_fold = e->p_graded = e->p_gsum = e->p_vtap = e->p_reltap = e->p_coin = nil;
        e->queue = nil; e->dev = nil;
    }
    free(e);
}

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
    e->gap_law = (GapLawParams){law, pairs, steps, tap, 6*pairs+e->P.n_gap, rho};
    e->b_gap_law = mkbuf(e->dev, sizeof e->gap_law, &e->gap_law);
    e->b_gap_pairs = mkbuf(e->dev, (size_t)pairs*8, ends);
    e->b_gap_pairs_w = mkbuf(e->dev, (size_t)pairs*4, weights);
    e->b_gap_scratch = mkbuf(e->dev, (size_t)e->P.n_gap*12, NULL);
    e->b_gap_tap = mkbuf(e->dev, tap ? (size_t)e->window_ticks*e->gap_law.stride*4 : 4, NULL);
    if (!e->b_gap_law || !e->b_gap_pairs || !e->b_gap_pairs_w || !e->b_gap_scratch || !e->b_gap_tap) return -2;
    return 0;
}
const float *cns_engine_gap_tap(cns_engine *e) { return (const float *)e->b_gap_tap.contents; }

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
            if (e->P.n_gap && e->P.gap_scale != 0.0f
                    && (!e->gap_law.law || (e->gap_law.law == 2 && e->gap_law.steps == 1))) {
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
            if (e->gap_law.law && (e->gap_law.law == 1 || e->gap_law.steps > 1 || e->gap_law.tap)) {
                if (e->profile) { kseq[nenc] = 5; enc = begin(5); nenc++; }
                [enc setComputePipelineState:e->p_gap_law];
                id<MTLBuffer> buffers[] = {e->bP, e->b_gap_law, e->b_gap_tgt,
                    e->b_gap_pairs, e->b_gap_pairs_w, e->b_v, e->b_gap_ap,
                    e->b_dtt, e->b_ref, e->b_gap_scratch, e->b_clampval, e->b_gap_tap};
                for (int k = 0; k < 12; k++) [enc setBuffer:buffers[k] offset:0 atIndex:k];
                [enc setBytes:&tick0 length:4 atIndex:12];
                [enc setBytes:&kw length:4 atIndex:13];
                [enc setBuffer:e->b_gap_ip offset:0 atIndex:14];
                [enc setBuffer:e->b_gap_src offset:0 atIndex:15];
                [enc setBuffer:e->b_gap_w offset:0 atIndex:16];
                [enc setBuffer:e->b_gap_deg offset:0 atIndex:17];
                [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
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
            if (e->P.n_pairs) {   /* K == 1 whenever n_pairs > 0: this tick's pair states */
                const uint32_t np = e->P.n_pairs;
                if (e->profile) { kseq[nenc] = 0; enc = begin(0); nenc++; }
                [enc setComputePipelineState:e->p_pair];
                [enc setBuffer:e->bP offset:0 atIndex:0];
                [enc setBuffer:e->b_ptgt offset:0 atIndex:1];
                [enc setBuffer:e->b_pr offset:0 atIndex:2];
                [enc setBuffer:e->b_pinc offset:0 atIndex:3];
                [enc setBuffer:e->b_px offset:0 atIndex:4];
                [enc setBuffer:e->b_pring offset:0 atIndex:5];
                [enc setBuffer:e->b_clampval offset:0 atIndex:6];
                [enc setBytes:&tick0 length:4 atIndex:7];
                [enc setBuffer:e->b_prc offset:0 atIndex:8];
                [enc setBuffer:e->b_pclk offset:0 atIndex:9];
                [enc setBuffer:e->b_pres offset:0 atIndex:10];
                [enc setBuffer:e->b_pcnt offset:0 atIndex:11];
                [enc setBuffer:e->b_parea offset:0 atIndex:12];
                [enc dispatchThreads:MTLSizeMake(np, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(np < 256 ? np : 256, 1, 1)];
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
            const uint32_t tkw[4] = {tick0, K, kw, have_g};
            [enc setBytes:tkw length:sizeof tkw atIndex:24];
            [enc setBuffer:e->b_oppos offset:0 atIndex:25];
            [enc setBuffer:e->b_opf offset:0 atIndex:26];
            [enc setBuffer:e->b_opu offset:0 atIndex:27];
            [enc setBuffer:e->b_vpre offset:0 atIndex:28];
            [enc setBuffer:e->b_clampbits offset:0 atIndex:29];
            [enc setBuffer:e->b_clampval offset:0 atIndex:30];
            [enc dispatchThreads:gridN threadsPerThreadgroup:tgN];
            if (e->profile) { [enc endEncoding]; enc = pe; }
            else [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            if (e->P.has_ser) {   /* K == 1 whenever has_ser: the state after this tick's
                                   * spikes; the group's closing barrier orders it before
                                   * the next k_step reads it */
                const uint32_t tw[4] = {tick0, kw, e->ser_nbins, 0u};
                if (e->profile) { kseq[nenc] = 0; enc = begin(0); nenc++; }
                [enc setComputePipelineState:e->p_ser];
                [enc setBuffer:e->bP offset:0 atIndex:0];
                [enc setBuffer:e->b_args offset:0 atIndex:1];
                [enc setBuffer:e->b_ser_sd offset:0 atIndex:2];
                [enc setBuffer:e->b_ser_bins offset:0 atIndex:3];
                [enc setBytes:tw length:sizeof tw atIndex:4];
                [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                if (e->profile) { [enc endEncoding]; enc = pe; }
            }
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
            if (e->P.n_pairs) {   /* this tick's pair events, from k_step's fire bits */
                const uint32_t np = e->P.n_pairs;
                if (e->profile) { kseq[nenc] = 0; enc = begin(0); nenc++; }
                [enc setComputePipelineState:e->p_pair_emit];
                [enc setBuffer:e->bP offset:0 atIndex:0];
                [enc setBuffer:e->b_psrc offset:0 atIndex:1];
                [enc setBuffer:e->b_pdelay offset:0 atIndex:2];
                [enc setBuffer:e->b_pring offset:0 atIndex:3];
                [enc setBuffer:e->b_args offset:0 atIndex:4];
                [enc setBytes:&tick0 length:4 atIndex:5];
                [enc setBytes:&par length:4 atIndex:6];
                [enc dispatchThreads:MTLSizeMake(np, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(np < 256 ? np : 256, 1, 1)];
                if (e->profile) { [enc endEncoding]; enc = pe; }
            }
            if (e->P.has_psi) {   /* K == 1 whenever has_psi: this tick's factors, before the scatter */
                const uint32_t nps = e->P.n_psi;
                id<MTLBuffer> q9 = e->b_g9 ? e->b_g9 : e->b_fac;
                id<MTLBuffer> r9 = e->b_ring9 ? e->b_ring9 : e->b_fac;
                if (e->profile) { kseq[nenc] = 0; enc = begin(0); nenc++; }
                [enc setComputePipelineState:e->p_psi];
                [enc setBuffer:e->bP offset:0 atIndex:0];
                [enc setBuffer:e->b_prow offset:0 atIndex:1];
                [enc setBuffer:e->b_gi offset:0 atIndex:2];
                [enc setBuffer:q9 offset:0 atIndex:3];
                [enc setBuffer:r9 offset:0 atIndex:4];
                [enc setBuffer:(e->b_r9 ? e->b_r9 : e->b_fac) offset:0 atIndex:5];
                [enc setBuffer:(e->b_c9 ? e->b_c9 : e->b_fac) offset:0 atIndex:6];
                [enc setBuffer:e->b_fac offset:0 atIndex:7];
                [enc setBuffer:e->b_plog offset:0 atIndex:8];
                [enc setBuffer:e->b_args offset:0 atIndex:9];
                [enc setBytes:&tick0 length:4 atIndex:10];
                [enc setBytes:&par length:4 atIndex:11];
                [enc setBytes:&kw length:4 atIndex:12];
                [enc setBytes:&e->psi_kg length:4 atIndex:13];
                [enc dispatchThreads:MTLSizeMake(nps, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(nps < 256 ? nps : 256, 1, 1)];
                if (e->profile) { [enc endEncoding]; enc = pe; }
                else [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
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
                [enc setBuffer:e->b_fac offset:0 atIndex:15];
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
                    [enc setBuffer:e->b_fac offset:0 atIndex:15];
                    [enc setBuffer:(e->b_ppos ? e->b_ppos : e->b_fac) offset:0 atIndex:16];
                    [enc setBuffer:(e->b_p9src ? e->b_p9src : e->b_fac) offset:0 atIndex:17];
                    [enc setBuffer:(e->b_ring9 ? e->b_ring9 : e->b_fac) offset:0 atIndex:18];
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
        if (e->P.has_ser) {   /* design fcb48834 Build 0: cns.py's double applied sum, in tick order */
            const float *sd = (const float *)e->b_ser_sd.contents;
            for (uint32_t k = 0; k < n_ticks; k++) e->ser_applied += (double)sd[k];
        }
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

/* Design afc8fd03 B3: team-henri's current injection (cns_engine.cu there).
 * inj is one float per row, added to k_step's membrane current after the gap
 * current and before adaptation from the next window on; NULL turns it off.
 * The first vector appends one section to clamp_val, whose earlier contents
 * are copied, so every other section stays where k_step reads it. */
int cns_engine_set_inject(cns_engine *e, const float *inj) {
    const size_t n = e->P.n;
    if (!inj) {
        e->P.has_inj = 0;
        memcpy(e->bP.contents, &e->P, sizeof e->P);
        return 0;
    }
    if (!e->P.inj_off) {
        const size_t used = e->ext_on ? 7 : (e->P.n_gap ? 5 : 4);
        const size_t ein = e->P.einh_off ? 1 : 0;   /* a reversal section moves past it */
        id<MTLBuffer> vals = mkbuf(e->dev, n * (used + 1 + ein) * 4, NULL);
        if (!vals) { seterr(e->err, sizeof e->err, "inject: allocation failed"); return -2; }
        memcpy(vals.contents, e->b_clampval.contents, n * used * 4);
        if (ein) {
            memcpy((float *)vals.contents + n * (used + 1),
                   (const float *)e->b_clampval.contents + e->P.einh_off, n * 4);
            e->P.einh_off = (uint32_t)(n * (used + 1));
        }
        e->b_clampval = vals;
        e->P.inj_off = (uint32_t)(n * used);
    }
    memcpy((float *)e->b_clampval.contents + e->P.inj_off, inj, n * 4);
    e->P.has_inj = 1;
    memcpy(e->bP.contents, &e->P, sizeof e->P);
    return 0;
}

/* Design e2b1a20d B3 on Metal: the per-row inhibitory reversal, one float
 * per row, which k_step reads in place of the scalar e_inh in the recorded
 * inhibitory drive and in both integrators, as cns_engine.cu reads its
 * lamina e_inh. Set once, before the first window. The vector takes one
 * section at the end of clamp_val, whose earlier contents are copied;
 * set_inject and ensure_ext carry it when they grow the buffer. Refused, as
 * cns_engine.cu refuses it, when the synapses are current-based, whose
 * update has no reversal. */
int cns_engine_set_inh_rows(cns_engine *e, const float *rows) {
    const size_t n = e->P.n;
    if (!rows || e->P.einh_off) {
        seterr(e->err, sizeof e->err, "inhibitory reversal: NULL or already set");
        return -1;
    }
    if (!e->P.cbased) {
        seterr(e->err, sizeof e->err, "inhibitory reversal: current-based synapses have none");
        return -2;
    }
    const size_t used = (e->ext_on ? 7 : (e->P.n_gap ? 5 : 4)) + (e->P.inj_off ? 1 : 0);
    id<MTLBuffer> vals = mkbuf(e->dev, n * (used + 1) * 4, NULL);
    if (!vals) { seterr(e->err, sizeof e->err, "inhibitory reversal: allocation failed"); return -3; }
    memcpy(vals.contents, e->b_clampval.contents, n * used * 4);
    memcpy((float *)vals.contents + n * used, rows, n * 4);
    e->b_clampval = vals;
    e->P.einh_off = (uint32_t)(n * used);
    e->P.has_einh = 1;
    memcpy(e->bP.contents, &e->P, sizeof e->P);
    return 0;
}

/* Design 6d0042d5, Build 2: the subthreshold coefficient below rest, one
 * float per row; cns_engine.cu exports the same name and signature. After
 * creation and before the first tick. Row i's update uses below[i] in place
 * of adapt_a[i] while v < v_rest, so (a, 0) is a max(v - v_rest, 0), (0, a)
 * a min(v - v_rest, 0) and (a, a) the symmetric law. adapt_a grows to 2n
 * floats with these in its second half, and k_step is rebuilt with
 * ADAPT_SPLIT on and UNI_ADAPT off, so each row reads its own coefficients,
 * which hold the values the scalar path used. Refused: NULL, a repeated or
 * late call, an engine without adaptation, or a non-finite value. */
int cns_engine_adapt_below_config(cns_engine *e, const float *below) {
    const size_t n = e->P.n;
    const char *bad = NULL;
    if (!below || e->adapt_split || e->ticks_total)
        bad = "adapt below: NULL, repeated or late configuration";
    else if (!e->P.use_adapt)
        bad = "adapt below: the engine carries no adaptation";
    for (size_t i = 0; !bad && i < n; i++)
        if (!isfinite(below[i])) bad = "adapt below: a non-finite value";
    if (bad) { seterr(e->err, sizeof e->err, bad); return -1; }
    @autoreleasepool {
        id<MTLBuffer> aa = mkbuf(e->dev, n * 8, NULL);
        if (!aa) { seterr(e->err, sizeof e->err, "adapt below: allocation failed"); return -2; }
        memcpy(aa.contents, e->b_aa.contents, n * 4);
        memcpy((float *)aa.contents + n, below, n * 4);
        bool uni[8];
        memcpy(uni, e->uniform, sizeof uni);
        uni[7] = false;
        bool sparse_vr = e->sparse_vrest, split = true;
        MTLFunctionConstantValues *fcv = [MTLFunctionConstantValues new];
        for (int k = 0; k < 8; k++) [fcv setConstantValue:&uni[k] type:MTLDataTypeBool atIndex:k];
        [fcv setConstantValue:&sparse_vr type:MTLDataTypeBool atIndex:8];
        [fcv setConstantValue:&split type:MTLDataTypeBool atIndex:9];
        NSError *nserr = nil;
        id<MTLFunction> f = [e->lib newFunctionWithName:@"k_step" constantValues:fcv error:&nserr];
        id<MTLComputePipelineState> ps = f ? [e->dev newComputePipelineStateWithFunction:f error:&nserr] : nil;
        if (!ps) { seterr(e->err, sizeof e->err, "adapt below: k_step pipeline failed"); return -3; }
        e->p_step = ps;
        e->b_aa = aa;
        e->uniform[7] = false;
        e->adapt_split = true;
    }
    return 0;
}

/* The reversal each row's update reads: its section entry, or the scalar.
 * cns_engine.cu exports the same name and signature. */
int cns_engine_inh_readback(cns_engine *e, float *out) {
    const size_t n = e->P.n;
    if (e->P.has_einh)
        memcpy(out, (const float *)e->b_clampval.contents + e->P.einh_off, n * 4);
    else
        for (size_t i = 0; i < n; i++) out[i] = e->P.e_inh;
    return 0;
}

/* Design 244fb801 Build 1: design 87a3ad35's opsin conductance on Metal,
 * under cns_engine.cu's names and arguments. One slot per row in
 * cns.opsin_arrays' order; k_step updates slot q's gate once per tick, lit
 * s = 1 + (s - 1) a_on and dark s = s a_off, and appends g_max s (e_rev - v)
 * last to the Euler bracket, as cns.py does. The lit test reads the pulse
 * train alone (cns_engine_opsin_train, design 2163ba7d), the one light
 * walk_search's pulse_exc keys set. Refused, and left to CUDA: windows,
 * force, the threshold window (kind 1), the tap, the analytic integrator,
 * and the current-based law, which has no reversal. Set once, before the
 * first tick. The gate starts at 0, as cns.py's op_s does. */
int cns_engine_opsin_config(cns_engine *e, uint32_t n_slot, const int32_t *rows,
        const float *a_on, const float *a_off, const float *g_max, const float *e_rev,
        const float *thr, const uint8_t *force, const uint8_t *kind, const uint32_t *win,
        const uint32_t *n_win, uint32_t tap_on) {
    (void)thr; (void)win;
    const uint32_t n = e->P.n;
    const char *bad = NULL;
    if (e->P.n_op || e->ticks_total || !n_slot || n_slot > n || tap_on > 1u)
        bad = "opsin: repeated, late or empty configuration";
    else if (!e->P.cbased)
        bad = "opsin: the conductance needs the conductance-based law";
    else if (e->P.analytic)
        bad = "opsin: Metal carries the opsin under the Euler integrator only";
    else if (e->P.xm_on || e->P.n_hd)
        bad = "opsin: design d27f82df's mask and hold-down hold its buffers on Metal";
    else if (tap_on)
        bad = "opsin: the tap is CUDA-only";
    int32_t *pos = bad ? NULL : malloc((size_t)n * 4);
    float *f = bad ? NULL : malloc((size_t)n_slot * 5 * 4);
    if (!bad && (!pos || !f)) {
        free(pos); free(f);
        seterr(e->err, sizeof e->err, "opsin: allocation failed"); return -2;
    }
    if (pos) for (uint32_t i = 0; i < n; i++) pos[i] = -1;
    for (uint32_t q = 0; !bad && q < n_slot; q++) {
        if (rows[q] < 0 || (uint32_t)rows[q] >= n || pos[rows[q]] >= 0)
            bad = "opsin: row out of range or repeated";
        else if (!isfinite(a_on[q]) || !isfinite(a_off[q]) || !isfinite(g_max[q])
                 || !isfinite(e_rev[q]) || g_max[q] < 0.0f)
            bad = "opsin: a non-finite value or a negative g_max";
        else if (force[q] || kind[q])
            bad = "opsin: force and the threshold window are CUDA-only";
        else if (n_win[q])
            bad = "opsin: windows are CUDA-only; Metal lights a slot by its pulse train";
        else {
            pos[rows[q]] = (int32_t)q;
            f[5u * q] = a_on[q]; f[5u * q + 1u] = a_off[q];
            f[5u * q + 2u] = g_max[q]; f[5u * q + 3u] = e_rev[q];
            f[5u * q + 4u] = 0.0f;
        }
    }
    if (bad) { free(pos); free(f); seterr(e->err, sizeof e->err, bad); return -1; }
    id<MTLBuffer> bp = mkbuf(e->dev, (size_t)n * 4, pos);
    id<MTLBuffer> bf = mkbuf(e->dev, (size_t)n_slot * 5 * 4, f);
    id<MTLBuffer> bu = mkbuf(e->dev, (size_t)n_slot * 4 * 4, NULL);   /* period 0: dark */
    free(pos); free(f);
    if (!bp || !bf || !bu) { seterr(e->err, sizeof e->err, "opsin: allocation failed"); return -2; }
    e->b_oppos = bp; e->b_opf = bf; e->b_opu = bu;
    e->P.n_op = n_slot;
    memcpy(e->bP.contents, &e->P, sizeof e->P);
    return 0;
}

/* One (start, stop, on, period) step row per slot, cns_engine.cu's check:
 * period 0 with every field 0 is no train, otherwise start < stop and
 * 0 < on < period. */
int cns_engine_opsin_train(cns_engine *e, const uint32_t *train) {
    const char *bad = NULL;
    if (!e->P.n_op || e->op_train || e->ticks_total)
        bad = "opsin train: no configuration, or a repeated or late call";
    for (uint32_t q = 0; !bad && q < e->P.n_op; q++) {
        const uint32_t *t = train + (size_t)q * 4u;
        if (t[3] == 0u ? (t[0] || t[1] || t[2]) : (t[0] >= t[1] || !t[2] || t[2] >= t[3]))
            bad = "opsin train: start >= stop, or on outside (0, period)";
    }
    if (bad) { seterr(e->err, sizeof e->err, bad); return -1; }
    memcpy(e->b_opu.contents, train, (size_t)e->P.n_op * 4 * 4);
    e->op_train = 1;
    return 0;
}

/* The tap is refused on Metal, so there is nothing to read; the width is
 * cns.OPSIN_TAP_FIELDS', which cns_metal checks on binding. */
const float *cns_engine_opsin_tap(cns_engine *e) { (void)e; return NULL; }
uint32_t cns_engine_opsin_width(void) { return 4u; }

/* Design d27f82df B8: B2's exact-update mask, one byte per row, after
 * creation and before the first tick, with cns_engine.cu's refusals: a
 * repeated or late call, NULL, the current-based membrane, a mask with no
 * row set. Metal also refuses it beside the opsin, whose buffers it takes:
 * op_pos becomes one code per row, -2 off and -1 exact, and a hold-down
 * configured first under the analytic integrator keeps its slots. */
int cns_engine_row_exact_config(cns_engine *e, const uint8_t *mask) {
    const uint32_t n = e->P.n;
    const char *bad = NULL;
    if (!mask || e->P.xm_on || e->ticks_total)
        bad = "row exact: NULL, repeated or late configuration";
    else if (!e->P.cbased)
        bad = "row exact: needs the conductance-based membrane";
    else if (e->P.n_op)
        bad = "row exact: the opsin holds its buffers on Metal";
    uint32_t on = 0;
    for (uint32_t i = 0; !bad && i < n; i++) on += mask[i] != 0;
    if (!bad && !on) bad = "row exact: no row set";
    if (bad) { seterr(e->err, sizeof e->err, bad); return -1; }
    if (e->P.n_hd) {
        int32_t *pos = (int32_t *)e->b_oppos.contents;
        for (uint32_t i = 0; i < n; i++) if (pos[i] == -2 && mask[i]) pos[i] = -1;
    } else {
        int32_t *pos = malloc((size_t)n * 4);
        if (!pos) { seterr(e->err, sizeof e->err, "row exact: allocation failed"); return -2; }
        for (uint32_t i = 0; i < n; i++) pos[i] = mask[i] ? -1 : -2;
        id<MTLBuffer> bp = mkbuf(e->dev, (size_t)n * 4, pos);
        free(pos);
        if (!bp) { seterr(e->err, sizeof e->err, "row exact: allocation failed"); return -2; }
        e->b_oppos = bp;
    }
    e->P.xm_on = 1;
    memcpy(e->bP.contents, &e->P, sizeof e->P);
    return 0;
}

/* Design d27f82df B8: B3's n_slot hold-down rows, cns_engine.cu's arguments
 * and refusals (see cns_engine_holddown_config there), and the opsin's, as
 * above. Each slot's 16 floats in op_f are e_rel, decay, dg, gmax, c_spk, k,
 * va, tau_c, latch_mv, then c, g_h, L, b and peak, zeroed here; the tap,
 * window_ticks x n_slot x 5, follows them. op_u holds form, latch_ticks and
 * the peak tick per slot. */
int cns_engine_holddown_config(cns_engine *e, uint32_t n_slot,
        const int32_t *rows, const uint32_t *form, const float *e_rel,
        const float *decay, const float *dg, const float *gmax, const float *c_spk,
        const float *k, const float *va, const float *tau_c, const float *latch_mv,
        const uint32_t *latch_ticks, float dt, uint32_t tap) {
    const uint32_t n = e->P.n;
    const char *bad = NULL;
    if (e->P.n_hd || e->ticks_total || !n_slot)
        bad = "hold-down: repeated, late or empty configuration";
    else if (!e->P.cbased)
        bad = "hold-down: needs the conductance-based membrane";
    else if (!(dt > 0.0f) || !isfinite(dt))
        bad = "hold-down: dt must be finite and above zero";
    else if (e->P.n_op)
        bad = "hold-down: the opsin holds its buffers on Metal";
    else if (!e->P.analytic && !e->P.xm_on)
        bad = "hold-down: no row is on the exact update";
    int32_t *pos = bad ? NULL : malloc((size_t)n * 4);
    if (!bad && !pos) { seterr(e->err, sizeof e->err, "hold-down: allocation failed"); return -2; }
    if (pos) {
        if (e->P.xm_on) memcpy(pos, e->b_oppos.contents, (size_t)n * 4);
        else for (uint32_t i = 0; i < n; i++) pos[i] = -2;
    }
    for (uint32_t q = 0; !bad && q < n_slot; q++) {
        if (rows[q] < 0 || (uint32_t)rows[q] >= n) bad = "hold-down: row out of range";
        else if (pos[rows[q]] >= 0) bad = "hold-down: row repeated";
        else if (!e->P.analytic && pos[rows[q]] != -1) bad = "hold-down: a row not on the exact update";
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
    if (bad) { free(pos); seterr(e->err, sizeof e->err, bad); return -1; }
    const size_t nf = (size_t)n_slot * 16 + (tap ? (size_t)e->window_ticks * n_slot * 5 : 0);
    float *f = calloc(nf, 4);
    uint32_t *u = calloc((size_t)n_slot * 4, 4);
    if (!f || !u) {
        free(pos); free(f); free(u);
        seterr(e->err, sizeof e->err, "hold-down: allocation failed"); return -2;
    }
    for (uint32_t q = 0; q < n_slot; q++) {
        float *x = f + 16u * q;
        x[0] = e_rel[q]; x[1] = decay[q]; x[2] = dg[q]; x[3] = gmax[q]; x[4] = c_spk[q];
        x[5] = k[q]; x[6] = va[q]; x[7] = tau_c[q]; x[8] = latch_mv[q];
        u[4u * q] = form[q]; u[4u * q + 1u] = latch_ticks[q];
    }
    id<MTLBuffer> bp = mkbuf(e->dev, (size_t)n * 4, pos);
    id<MTLBuffer> bf = mkbuf(e->dev, nf * 4, f);
    id<MTLBuffer> bu = mkbuf(e->dev, (size_t)n_slot * 4 * 4, u);
    free(pos); free(f); free(u);
    if (!bp || !bf || !bu) { seterr(e->err, sizeof e->err, "hold-down: allocation failed"); return -2; }
    e->b_oppos = bp; e->b_opf = bf; e->b_opu = bu;
    e->P.n_hd = n_slot; e->P.hd_tap = tap ? 1u : 0u; e->P.hd_dt = dt;
    memcpy(e->bP.contents, &e->P, sizeof e->P);
    return 0;
}

/* Design d27f82df B8: the state after the last window, per slot, float32 c,
 * g_h, L, b and peak, and uint32 peak_tick. */
int cns_engine_holddown_read(cns_engine *e, float *c, float *g_h, float *L,
                             float *b, float *peak, uint32_t *peak_tick) {
    const uint32_t nh = e->P.n_hd;
    if (!nh) { seterr(e->err, sizeof e->err, "hold-down: not configured"); return -1; }
    const float *f = (const float *)e->b_opf.contents;
    const uint32_t *u = (const uint32_t *)e->b_opu.contents;
    for (uint32_t q = 0; q < nh; q++) {
        c[q] = f[16u * q + 9u]; g_h[q] = f[16u * q + 10u]; L[q] = f[16u * q + 11u];
        b[q] = f[16u * q + 12u]; peak[q] = f[16u * q + 13u]; peak_tick[q] = u[4u * q + 2u];
    }
    return 0;
}

/* The last window's hold-down tap, window_ticks x n x 5 float32, or NULL. */
const float *cns_engine_holddown_tap(cns_engine *e) {
    return e->P.n_hd && e->P.hd_tap ? (const float *)e->b_opf.contents + 16u * e->P.n_hd : NULL;
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

/* Designs 7b0ad6b2 and 727bccb8: the pair and tap sections. clamp_bits gains
 * the pair target mask (words) and the tap row per neuron (n, 0xFFFFFFFF for
 * none) after its four sections; clamp_val gains the pair state and event
 * count per target after its five. Contents are copied, so clamps, v_rest
 * overrides, coincidence targets and tonic conductance set at creation stay
 * where k_step reads them. */
static int ensure_ext(cns_engine *e) {
    if (e->ext_on) return 0;
    const size_t n = e->P.n, words = e->P.words;
    const size_t inj = e->P.inj_off ? 1 : 0;   /* an injection section moves to the end */
    const size_t ein = e->P.einh_off ? 1 : 0;  /* and a reversal section after it */
    id<MTLBuffer> bits = mkbuf(e->dev, (words * 5 + n) * 4, NULL);
    id<MTLBuffer> vals = mkbuf(e->dev, n * (7 + inj + ein) * 4, NULL);
    if (!bits || !vals) return -2;
    memcpy(bits.contents, e->b_clampbits.contents, words * 16);
    memset((uint32_t *)bits.contents + words * 5, 0xFF, n * 4);
    memcpy(vals.contents, e->b_clampval.contents, n * 4 * (e->P.n_gap ? 5 : 4));
    if (inj) {
        memcpy((float *)vals.contents + n * 7,
               (const float *)e->b_clampval.contents + e->P.inj_off, n * 4);
        e->P.inj_off = (uint32_t)(n * 7);
    }
    if (ein) {
        memcpy((float *)vals.contents + n * (7 + inj),
               (const float *)e->b_clampval.contents + e->P.einh_off, n * 4);
        e->P.einh_off = (uint32_t)(n * (7 + inj));
    }
    e->b_clampbits = bits; e->b_clampval = vals;
    e->P.pmask_off = (uint32_t)(words * 4);
    e->P.apos_off = (uint32_t)(words * 5);
    e->P.px_off = (uint32_t)(n * 5);
    e->P.na_off = (uint32_t)(n * 6);
    e->ext_on = 1;
    return 0;
}

/* np pairs (src[q], tgt[q]), each with its own transport delay[q] in ticks,
 * decay r[q] per tick and increment inc[q] per arriving event, refused
 * exactly as cns_engine.cu refuses them: a repeated or late configuration;
 * the current-based law or unsplit inhibition; a row out of range; a
 * repeated target; a graded source or target; a delay outside
 * [kgroup, nr - kgroup]; r outside [0, 1); inc not finite or negative; a pair
 * whose ordinary weight in the uploaded graph is not exactly zero, because
 * the channel replaces that contribution. Metal also refuses kgroup > 1:
 * k_pair and k_pair_emit run between single-tick groups. */
int cns_engine_pair_config(cns_engine *e, uint32_t np, uint32_t nr,
        const int32_t *src, const int32_t *tgt, const uint32_t *delay,
        const float *r, const float *inc) {
    const uint32_t n = e->P.n;
    const char *bad = NULL;
    if (e->P.n_pairs || e->ticks_total) bad = "pair channel: repeated or late configuration";
    else if (!np || nr < 2) bad = "pair channel: no pairs, or a ring under two slots";
    else if (!e->P.cbased || !e->P.split)
        bad = "pair channel: needs the conductance law with split inhibition";
    else if (e->kgroup != 1)
        bad = "pair channel: the Metal engine updates pair states between single-tick groups (kgroup 1)";
    if (bad) { seterr(e->err, sizeof e->err, bad); return -1; }
    const uint8_t *graded = (const uint8_t *)e->b_graded.contents;
    const int64_t *ip = (const int64_t *)e->b_indptr.contents;
    const int32_t *idx = (const int32_t *)e->b_indices.contents;
    const float *w = (const float *)e->b_weight.contents;
    int32_t *tgt_of = malloc((size_t)n * 4);
    if (!tgt_of) { seterr(e->err, sizeof e->err, "pair channel: out of host memory"); return -2; }
    for (uint32_t i = 0; i < n; i++) tgt_of[i] = -1;
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
    for (uint32_t q = 0; q < np && !bad; q++)   /* the ordinary weight must already be zero */
        for (int64_t j = ip[src[q]]; j < ip[src[q] + 1] && !bad; j++)
            if (idx[j] == tgt[q] && w[j] != 0.0f)
                bad = "pair channel: the pair's ordinary weight is not zero (it would be delivered twice)";
    free(tgt_of);
    if (bad) { seterr(e->err, sizeof e->err, bad); return -1; }
    if (ensure_ext(e)) { seterr(e->err, sizeof e->err, "pair channel: allocation failed"); return -2; }
    e->b_psrc = mkbuf(e->dev, (size_t)np * 4, src);
    e->b_ptgt = mkbuf(e->dev, (size_t)np * 4, tgt);
    e->b_pdelay = mkbuf(e->dev, (size_t)np * 4, delay);
    e->b_pr = mkbuf(e->dev, (size_t)np * 4, r);
    e->b_pinc = mkbuf(e->dev, (size_t)np * 4, inc);
    e->b_px = mkbuf(e->dev, (size_t)np * 4, NULL);
    e->b_pring = mkbuf(e->dev, (size_t)nr * np * 4, NULL);
    const uint32_t rc0[4] = {0u, 0u, 0u, 0x3f800000u};   /* form 0, keep 1.0f */
    e->b_prc = mkbuf(e->dev, 16, rc0);
    e->b_pclk = mkbuf(e->dev, (size_t)np * 4, NULL);
    e->b_pres = mkbuf(e->dev, (size_t)np * 4, NULL);
    e->b_pcnt = mkbuf(e->dev, (size_t)np * 16, NULL);
    e->b_parea = mkbuf(e->dev, (size_t)np * 4, NULL);
    if (!e->b_psrc || !e->b_ptgt || !e->b_pdelay || !e->b_pr || !e->b_pinc || !e->b_px || !e->b_pring
            || !e->b_prc || !e->b_pclk || !e->b_pres || !e->b_pcnt || !e->b_parea) {
        seterr(e->err, sizeof e->err, "pair channel: allocation failed"); return -2;
    }
    for (uint32_t q = 0; q < np; q++) ((float *)e->b_pres.contents)[q] = 1.0f;
    uint32_t *mask = (uint32_t *)e->b_clampbits.contents + e->P.pmask_off;
    for (uint32_t q = 0; q < np; q++) mask[(uint32_t)tgt[q] >> 5] |= 1u << ((uint32_t)tgt[q] & 31u);
    e->P.n_pairs = np; e->P.pair_nr = nr;
    memcpy(e->bP.contents, &e->P, sizeof e->P);
    return 0;
}

/* Design e05c0eef B1: cns_engine.cu's cns_engine_pair_recovery_config, the
 * same arguments and refusals; the four values go to k_pair's buffer 8. */
int cns_engine_pair_recovery_config(cns_engine *e, uint32_t form, uint32_t rticks,
        float rdecay, float rkeep) {
    const char *bad = NULL;
    const uint32_t *cur = e->P.n_pairs ? (const uint32_t *)e->b_prc.contents : NULL;
    if (!e->P.n_pairs) bad = "pair recovery: no pair channel is configured";
    else if (cur[0] || e->ticks_total) bad = "pair recovery: repeated or late configuration";
    else if (form != 1u && form != 2u) bad = "pair recovery: form must be 1 (dead time) or 2 (depletion)";
    else if (!(rdecay >= 0.0f && rdecay <= 1.0f) || !(rkeep >= 0.0f && rkeep <= 1.0f))
        bad = "pair recovery: decay or keep outside [0, 1]";
    if (bad) { seterr(e->err, sizeof e->err, bad); return -1; }
    uint32_t *rc = (uint32_t *)e->b_prc.contents;
    rc[0] = form; rc[1] = rticks;
    memcpy(&rc[2], &rdecay, 4); memcpy(&rc[3], &rkeep, 4);
    return 0;
}

/* Design fcb48834 Build 0: design b1746728's receptor-scoped serotonin state
 * on Metal, with cns_engine.cu's entry points, arguments and refusals. Called
 * after creation and before the first tick; src and tgt list n_src and n_tgt
 * distinct rows; gain (mV/Hz), decay and inc are cns.py's float32 values;
 * init is s and its maximum at t = 0; the readback has n_bins bins of
 * steps_per_bin ticks. Refused: a repeated or late call, an engine with
 * kgroup above 1, the injected current (cns_engine.cu refuses its held
 * current; cns_metal.py refuses both before the engine sees them), an empty,
 * repeated or out-of-range row list, a gain or init that is negative or not
 * finite, a decay outside (0, 1], an inc that is not finite and positive, and
 * a zero bin. k_step binds all 31 buffer slots Metal allows, so the target
 * mask (only when gain > 0, as cns.py adds nothing at gain 0), the source
 * mask and the state go into args past the touched counts: args is remade
 * that much longer, its earlier contents copied. */
int cns_engine_serotonin_config(cns_engine *e, uint32_t n_src, const int32_t *src,
        uint32_t n_tgt, const int32_t *tgt, float gain, float decay, float inc, float init,
        uint32_t steps_per_bin, uint32_t n_bins) {
    const uint32_t n = e->P.n, words = e->P.words;
    const char *bad = NULL;
    if (e->P.has_ser || e->ticks_total) bad = "serotonin: repeated or late configuration";
    else if (e->kgroup != 1)
        bad = "serotonin: k_step reads the state k_ser leaves each tick (kgroup 1)";
    else if (e->P.has_inj) bad = "serotonin: not defined with the injected current";
    else if (!n_src || !n_tgt || n_src > n || n_tgt > n || !src || !tgt)
        bad = "serotonin: no source or no target rows";
    else if (!isfinite(gain) || !(gain >= 0.0f) || !isfinite(init) || !(init >= 0.0f))
        bad = "serotonin: gain or init negative or not finite";
    else if (!isfinite(decay) || !(decay > 0.0f) || !(decay <= 1.0f))
        bad = "serotonin: decay outside (0, 1]";
    else if (!isfinite(inc) || !(inc > 0.0f)) bad = "serotonin: inc not finite and positive";
    else if (!steps_per_bin) bad = "serotonin: a zero readback bin";
    if (bad) { seterr(e->err, sizeof e->err, bad); return -1; }
    uint32_t *ms = calloc(words, 4), *mt = calloc(words, 4);
    if (!ms || !mt) {
        free(ms); free(mt); seterr(e->err, sizeof e->err, "serotonin: out of host memory"); return -2;
    }
    for (uint32_t q = 0; !bad && q < n_src; q++) {
        const uint32_t r = (uint32_t)src[q];
        if (src[q] < 0 || r >= n) bad = "serotonin: a source row out of range";
        else if (ms[r >> 5] & (1u << (r & 31u))) bad = "serotonin: a source row named twice";
        else ms[r >> 5] |= 1u << (r & 31u);
    }
    for (uint32_t q = 0; !bad && q < n_tgt; q++) {
        const uint32_t r = (uint32_t)tgt[q];
        if (tgt[q] < 0 || r >= n) bad = "serotonin: a target row out of range";
        else if (mt[r >> 5] & (1u << (r & 31u))) bad = "serotonin: a target row named twice";
        else mt[r >> 5] |= 1u << (r & 31u);
    }
    if (bad) { free(ms); free(mt); seterr(e->err, sizeof e->err, bad); return -1; }
    const uint32_t base = e->P.tcnt_slot + e->window_ticks;   /* args' length in slots */
    const uint32_t ss = base + 2u * words;
    id<MTLBuffer> na = mkbuf(e->dev, ((size_t)ss + 5u) * 4, NULL);
    id<MTLBuffer> sd = mkbuf(e->dev, (size_t)e->window_ticks * 4, NULL);
    id<MTLBuffer> bb = mkbuf(e->dev, (size_t)n_bins * 4, NULL);
    if (!na || !sd || !bb) {
        free(ms); free(mt); seterr(e->err, sizeof e->err, "serotonin: allocation failed"); return -2;
    }
    uint32_t *a = (uint32_t *)na.contents;
    memcpy(a, e->b_args.contents, (size_t)base * 4);
    if (gain > 0.0f) memcpy(a + base, mt, (size_t)words * 4);
    memcpy(a + base + words, ms, (size_t)words * 4);
    memcpy(a + ss, &init, 4);        /* s */
    memcpy(a + ss + 1u, &init, 4);   /* its maximum; the count and the spike total stay 0 */
    free(ms); free(mt);
    e->b_args = na; e->b_ser_sd = sd; e->b_ser_bins = bb;
    e->ser_nbins = n_bins; e->ser_applied = 0.0;
    e->P.has_ser = 1u | (gain > 0.0f ? 2u : 0u);
    e->P.ser_tmask = base; e->P.ser_smask = base + words; e->P.ser_s = ss;
    e->P.ser_gain = gain; e->P.ser_decay = decay; e->P.ser_inc = inc;
    e->P.ser_spb = steps_per_bin;
    memcpy(e->bP.contents, &e->P, sizeof e->P);
    return 0;
}

/* Design fcb48834 Build 0: the state after the last window, as
 * cns_engine.cu's cns_engine_serotonin_read: s and its maximum, the double
 * applied sum, the source spikes and the n_bins readback, which must be the
 * configured count. Refused without a configuration. */
int cns_engine_serotonin_read(cns_engine *e, float *state2, double *applied,
        uint64_t *spikes, float *bins, uint32_t n_bins) {
    if (!e->P.has_ser || n_bins != e->ser_nbins) {
        seterr(e->err, sizeof e->err, "serotonin: not configured, or the bin count differs");
        return -1;
    }
    const uint32_t *a = (const uint32_t *)e->b_args.contents + e->P.ser_s;
    memcpy(state2, a, 8);
    *applied = e->ser_applied;
    *spikes = (uint64_t)a[3] | ((uint64_t)a[4] << 32);
    if (n_bins) memcpy(bins, e->b_ser_bins.contents, (size_t)n_bins * 4);
    return 0;
}

/* Design 302c1bfb, Build 8: n_psi listed rows (distinct, ungraded) release
 * each spike's synapses at f = 1/(1 + G*kg), kg = float32(k * gscale). With
 * src9 (one byte per row, 1 on a source) G is a second state per listed row:
 * decay r9, and each inhibitory arrival from a source row adds c9 times its
 * g_i increment; without it G is the row's g_i. Refused: a late or repeated
 * configuration, kgroup > 1 (the factor is set between single-tick groups),
 * the source state without split inhibition, a row out of range, repeated or
 * graded. Metal only; it is not in cns_engine.h. */
int cns_engine_psi_config(cns_engine *e, uint32_t n_psi, const int32_t *rows,
        const uint8_t *src9, const float *r9, const float *c9, float kg) {
    const uint32_t n = e->P.n;
    const char *bad = NULL;
    if (e->P.has_psi || e->ticks_total) bad = "release factor: repeated or late configuration";
    else if (!n_psi || n_psi > n || !rows) bad = "release factor: no rows";
    else if (e->kgroup != 1)
        bad = "release factor: the Metal engine sets it between single-tick groups (kgroup 1)";
    else if (src9 && (!e->P.split || !r9 || !c9))
        bad = "release factor: the source-fed state needs split inhibition, r9 and c9";
    else if (!isfinite(kg) || !(kg >= 0.0f)) bad = "release factor: kg not finite and nonnegative";
    if (bad) { seterr(e->err, sizeof e->err, bad); return -1; }
    const uint8_t *graded = (const uint8_t *)e->b_graded.contents;
    int32_t *pos = malloc((size_t)n * 4);
    if (!pos) { seterr(e->err, sizeof e->err, "release factor: out of host memory"); return -2; }
    for (uint32_t i = 0; i < n; i++) pos[i] = -1;
    for (uint32_t j = 0; j < n_psi && !bad; j++) {
        if (rows[j] < 0 || (uint32_t)rows[j] >= n) bad = "release factor: row out of range";
        else if (pos[rows[j]] >= 0) bad = "release factor: repeated row";
        else if (graded[rows[j]]) bad = "release factor: graded row";
        else pos[rows[j]] = (int32_t)j;
    }
    if (bad) { free(pos); seterr(e->err, sizeof e->err, bad); return -1; }
    @autoreleasepool {
        e->b_prow = mkbuf(e->dev, (size_t)n_psi * 4, rows);
        e->b_plog = mkbuf(e->dev, (size_t)e->window_ticks * n_psi * 16, NULL);
        int ok = e->b_prow && e->b_plog;
        if (ok && src9) {
            e->b_g9 = mkbuf(e->dev, (size_t)n_psi * 4, NULL);
            e->b_ring9 = mkbuf(e->dev, (size_t)e->P.n_ring * n_psi * 4, NULL);
            e->b_r9 = mkbuf(e->dev, (size_t)n_psi * 4, r9);
            e->b_c9 = mkbuf(e->dev, (size_t)n_psi * 4, c9);
            e->b_ppos = mkbuf(e->dev, (size_t)n * 4, pos);
            e->b_p9src = mkbuf(e->dev, (size_t)n, src9);
            ok = e->b_g9 && e->b_ring9 && e->b_r9 && e->b_c9 && e->b_ppos && e->b_p9src;
        }
        free(pos);
        if (!ok) { seterr(e->err, sizeof e->err, "release factor: allocation failed"); return -2; }
    }
    e->psi_kg = kg;
    e->P.has_psi = 1u | (src9 ? 2u : 0u);
    e->P.n_psi = n_psi;
    memcpy(e->bP.contents, &e->P, sizeof e->P);
    return 0;
}

/* The last window's release-factor log, window_ticks x n_psi x (f, G, g_i,
 * fired flag); a fired flag of 0 marks a tick the row did not spike. */
const float *cns_engine_psi_log(cns_engine *e) { return e->b_plog ? (const float *)e->b_plog.contents : NULL; }

/* The listed rows' source-fed state after the last window (n_psi floats). */
const float *cns_engine_psi_state(cns_engine *e) { return e->b_g9 ? (const float *)e->b_g9.contents : NULL; }

/* Design 302c1bfb, Build 10: set spike_at on n_rows rows between windows.
 * k_step reads one scalar threshold while every row's is equal (UNI_SPIKE),
 * so the first call on such an engine rebuilds k_step with UNI_SPIKE off,
 * which reads the same per-row values it held. Metal only. */
int cns_engine_set_spike_at(cns_engine *e, uint32_t n_rows, const int32_t *rows, const float *vals) {
    const uint32_t n = e->P.n;
    for (uint32_t q = 0; q < n_rows; q++)
        if (rows[q] < 0 || (uint32_t)rows[q] >= n) {
            seterr(e->err, sizeof e->err, "set_spike_at: row out of range"); return -1;
        }
    if (e->uniform[2]) {
        @autoreleasepool {
            bool uni[8];
            memcpy(uni, e->uniform, sizeof uni);
            uni[2] = false;
            bool sparse_vr = e->sparse_vrest, split = e->adapt_split;
            MTLFunctionConstantValues *fcv = [MTLFunctionConstantValues new];
            for (int k = 0; k < 8; k++) [fcv setConstantValue:&uni[k] type:MTLDataTypeBool atIndex:k];
            [fcv setConstantValue:&sparse_vr type:MTLDataTypeBool atIndex:8];
            if (split) [fcv setConstantValue:&split type:MTLDataTypeBool atIndex:9];
            NSError *nserr = nil;
            id<MTLFunction> f = [e->lib newFunctionWithName:@"k_step" constantValues:fcv error:&nserr];
            id<MTLComputePipelineState> ps = f ? [e->dev newComputePipelineStateWithFunction:f error:&nserr] : nil;
            if (!ps) { seterr(e->err, sizeof e->err, "set_spike_at: k_step pipeline failed"); return -2; }
            e->p_step = ps;
            e->uniform[2] = false;
        }
    }
    float *sa = (float *)e->b_spike.contents;
    for (uint32_t q = 0; q < n_rows; q++) sa[rows[q]] = vals[q];
    return 0;
}

/* Per-tick state of n_aux named rows, 16 floats each, window-major, in
 * cns_engine.cu's field order (see k_step). The tap lives in v_pre after its
 * four voltage-tap sections. */
int cns_engine_aux_config(cns_engine *e, uint32_t n_aux, const int32_t *rows) {
    const uint32_t n = e->P.n;
    if (e->P.n_aux || e->ticks_total || !n_aux || n_aux > n) {
        seterr(e->err, sizeof e->err, "state tap: repeated, late or empty configuration"); return -1;
    }
    uint32_t *apos = malloc((size_t)n * 4);
    if (!apos) { seterr(e->err, sizeof e->err, "state tap: allocation failed"); return -2; }
    memset(apos, 0xFF, (size_t)n * 4);
    for (uint32_t j = 0; j < n_aux; j++) {
        if (rows[j] < 0 || (uint32_t)rows[j] >= n || apos[rows[j]] != 0xFFFFFFFFu) {
            free(apos);
            seterr(e->err, sizeof e->err, "state tap: row out of range or repeated"); return -1;
        }
        apos[rows[j]] = j;
    }
    const size_t off = (size_t)n * 4;   /* floats: v_pre's four voltage-tap sections */
    id<MTLBuffer> vpre = ensure_ext(e) ? nil
        : mkbuf(e->dev, (off + (size_t)e->window_ticks * n_aux * 16) * 4, NULL);
    if (!vpre) { free(apos); seterr(e->err, sizeof e->err, "state tap: allocation failed"); return -2; }
    memcpy(vpre.contents, e->b_vpre.contents,
           e->P.n_vrec ? (size_t)n * 16 : e->P.n_graded ? (size_t)n * 4 : 0);
    memcpy((uint32_t *)e->b_clampbits.contents + e->P.apos_off, apos, (size_t)n * 4);
    free(apos);
    e->b_vpre = vpre;
    e->P.n_aux = n_aux; e->P.aux_off = (uint32_t)off;
    memcpy(e->bP.contents, &e->P, sizeof e->P);
    return 0;
}
const float *cns_engine_aux_tap(cns_engine *e) {
    return e->P.n_aux ? (const float *)e->b_vpre.contents + e->P.aux_off : NULL;
}
uint32_t cns_engine_aux_width(void) { return 16; }

/* A copy of one engine buffer as the kernels read it, cns_engine.cu's
 * numbering: 0 weight, 1 indptr (int64), 2 indices, 3 delay slot, 4-6 the
 * transposed graph's weight, source and indptr (int64), 7-11 the graded
 * gather's weight, target rows, indptr (int64), source position and arrival
 * slot, 12 decay_g, 13 refractory counter (int16), 14-20 the pair channel's
 * source, target, delay (uint32), decay, increment, state and event ring
 * (nr x pairs, uint32), 21 v_rest, 22 v_reset, 23 spike_at, 24 decay_v,
 * 25 dt_over_tau, 26 refractory ticks (int16), 27-29 adapt_a, adapt_b,
 * dt_over_tauw, 30 inhibitory state, 31-32 excitatory and inhibitory rings
 * (n_ring x n), 33 membrane, 34 excitatory state, 35 adaptation, 36 spike
 * counts (uint32). Returns the byte length, and with out NULL only the
 * length; -1 unknown buffer, -2 capacity too small, -3 missing buffer. */
int64_t cns_engine_read_buffer(cns_engine *e, uint32_t which, void *out, uint64_t capacity) {
    const uint64_t n = e->P.n, nnz = e->nnz, ngt = e->n_gt, tot = e->n_gt_entries;
    const uint64_t np = e->P.n_pairs, ua = e->P.use_adapt ? 1 : 0, sp = e->P.split ? 1 : 0;
    const uint64_t ringn = (uint64_t)e->P.n_ring * n;
    id<MTLBuffer> b = nil; uint64_t bytes = 0;
    switch (which) {
    case 0: b = e->b_weight; bytes = nnz * 4; break;
    case 1: b = e->b_indptr; bytes = (n + 1) * 8; break;
    case 2: b = e->b_indices; bytes = nnz * 4; break;
    case 3: b = e->b_delay; bytes = n * 4; break;
    case 4: b = e->b_tw; bytes = nnz * 4; break;
    case 5: b = e->b_tsrc; bytes = nnz * 4; break;
    case 6: b = e->b_tindptr; bytes = (n + 1) * 8; break;
    case 7: b = e->b_gtw; bytes = tot * 4; break;
    case 8: b = e->b_gtrows; bytes = ngt * 4; break;
    case 9: b = e->b_gtindptr; bytes = (ngt + 1) * 8; break;
    case 10: b = e->b_gtgidx; bytes = tot * 4; break;
    case 11: b = e->b_gtslot; bytes = tot * 4; break;
    case 12: b = e->b_decg; bytes = n * 4; break;
    case 13: b = e->b_ref; bytes = n * 2; break;
    case 14: b = e->b_psrc; bytes = np * 4; break;
    case 15: b = e->b_ptgt; bytes = np * 4; break;
    case 16: b = e->b_pdelay; bytes = np * 4; break;
    case 17: b = e->b_pr; bytes = np * 4; break;
    case 18: b = e->b_pinc; bytes = np * 4; break;
    case 19: b = e->b_px; bytes = np * 4; break;
    case 20: b = e->b_pring; bytes = (uint64_t)e->P.pair_nr * np * 4; break;
    case 21: b = e->b_vrest; bytes = n * 4; break;
    case 22: b = e->b_vreset; bytes = n * 4; break;
    case 23: b = e->b_spike; bytes = n * 4; break;
    case 24: b = e->b_decv; bytes = n * 4; break;
    case 25: b = e->b_dtt; bytes = n * 4; break;
    case 26: b = e->b_rt; bytes = n * 2; break;
    case 27: b = e->b_aa; bytes = ua * n * 4; break;
    case 28: b = e->b_ab; bytes = ua * n * 4; break;
    case 29: b = e->b_dtw; bytes = ua * n * 4; break;
    case 30: b = e->b_gi; bytes = sp * n * 4; break;
    case 31: b = e->b_ring; bytes = ringn * 4; break;
    case 32: b = e->b_ring_i; bytes = sp * ringn * 4; break;
    case 33: b = e->b_v; bytes = n * 4; break;
    case 34: b = e->b_g; bytes = n * 4; break;
    case 35: b = e->b_w; bytes = ua * n * 4; break;
    case 36: b = e->b_counts; bytes = n * 4; break;
    /* design 02e57559's resource, replacement, recorder and tap selectors are
     * CUDA only: cns_metal.py refuses to configure them on this engine, so they
     * read as the CUDA engine reads them unconfigured, zero bytes */
    case 37: case 38: case 39: case 40: case 41: case 42:
    case 43: case 44: case 45: case 46: case 47: bytes = 0; break;
    /* design e05c0eef B1: the pair recovery's clock, resource, counters
     * (delivered, skipped, scaled, multi-event ticks per pair) and area */
    case 72: b = e->b_pclk; bytes = np * 4; break;
    case 73: b = e->b_pres; bytes = np * 4; break;
    case 74: b = e->b_pcnt; bytes = np * 16; break;
    case 75: b = e->b_parea; bytes = np * 4; break;
    default: return -1;
    }
    if (!out) return (int64_t)bytes;
    if (capacity < bytes) return -2;
    if (bytes && (!b || b.length < bytes)) { seterr(e->err, sizeof e->err, "read_buffer: buffer missing"); return -3; }
    if (bytes) memcpy(out, b.contents, bytes);
    return (int64_t)bytes;
}
