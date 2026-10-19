/* cns_engine.h — the CNS step loop as a Metal library the Python engine calls.
 *
 * cns.run() keeps every decision that is not per-tick arithmetic (drive
 * draws, callbacks, bookkeeping) in Python and hands this library windows
 * of ticks. The library owns the state (v, g, refractory, adaptation, the
 * delay rings) on the GPU for the life of the engine and returns the spike
 * list of each window. Pure C ABI so ctypes loads it; no Python inside.
 */
#pragma once
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct cns_engine cns_engine;

typedef struct {
    uint32_t n;              /* neurons */
    uint32_t n_ring;         /* max(delay_slot) + 1 */
    uint32_t window_ticks;   /* upper bound on n_ticks per run call */
    uint32_t kmax;           /* max ticks fused per k_step dispatch; 0 = auto */
    uint32_t fired_cap;      /* spike-list capacity per tick; 0 = 65536 */
    int32_t cbased, split, analytic, use_adapt;
    float e_exc, e_inh, gscale;
    /* per-neuron constants, length n, float32 / int32 / uint8 */
    const float *v_rest, *v_reset, *spike_at, *decay_v, *dt_over_tau, *decay_g;
    const int16_t *refrac_ticks; const int32_t *delay_slot;   /* refractory ticks fit int16 */
    const uint8_t *graded_mask;
    const float *adapt_a, *adapt_b, *dt_over_tauw;   /* NULL when !use_adapt */
    /* graph, CSC on the presynaptic axis: column j = neuron j's outgoing edges */
    const int64_t *indptr; const int32_t *indices; const float *weight;
    uint64_t nnz;
    /* graded release: sources, their release-curve parameters, cadence */
    uint32_t n_graded; const int32_t *graded_idx;
    const float *gv50, *gslope;      /* length n_graded */
    float ggain_scaled;              /* graded_gain * graded_every, float32 */
    uint32_t g_every;
    /* voltage tap: rows whose membrane potential is recorded every tick,
     * taken after the refractory clamp and before the spike reset, as
     * cns.run() records it. Forces K = 1. */
    uint32_t n_vrec; const int32_t *vrec;
    /* graded feedback: positions (into graded_idx) whose release is tapped
     * on every graded tick, n_ticks x n_reltap per window. */
    uint32_t n_reltap; const int32_t *reltap_pos;
    /* per-graded-row gain, length n_graded, already scaled by g_every;
     * NULL means every row uses ggain_scaled (lane A, 2026-09-02). */
    const float *ggain_rows;
    /* coincidence seam (lane C, 2026-09-04; ported 2026-09-05): n_coin
     * designated targets, each with two source pools as CSR rows over the
     * source axis, weights already |w| as cns.run() builds them. Per tick,
     * per target: fs = fs*decay + sum(fast_w over sources that fired last
     * tick), ss likewise, the slow trace read back coin_lag ticks later,
     * and gain*fs*slow_delayed added to the target's g after the ring
     * consume. Sums run in the stored CSR order, so the float order is the
     * numpy engine's csr @ vector. Forces K = 1. n_coin = 0 disables it. */
    uint32_t n_coin; const int32_t *coin_tgt;
    const int64_t *coin_fast_indptr; const int32_t *coin_fast_src; const float *coin_fast_w;
    const int64_t *coin_slow_indptr; const int32_t *coin_slow_src; const float *coin_slow_w;
    float coin_gain, coin_decay; uint32_t coin_lag;
    /* Per-neuron tonic excitatory conductance, added to ge every tick in the
     * conductance-based branch exactly where cns.py adds it (`ge = ge + tonic_ge`
     * after ge and gi are formed and before either integrator). Requires
     * conductance_based; NULL disables it and the engine is bit-identical. */
    const float *tonic_ge;
    /* Electrical coupling, CSR over the n_gap participating source/target rows.
     * gap_tgt maps compact rows to neuron IDs; gap_src indexes gap_tgt.
     * Each CSR row preserves the NumPy G matrix's source order and weights,
     * including directed pairs. gap_deg is its precomputed weighted degree.
     * Before each tick: extra += gap_scale * (G @ v - gap_deg * v).
     * After all resets: v += gap_spikelet * (G @ fired_mask).
     * A positive gap_delay delivers spikelets at the top of the arrival tick.
     * gap_ap_wave adds the reference's presynaptic waveform inside G @ v,
     * with its exact schedule and ring wrap. Rings cover only compact rows.
     * n_gap = 0 disables both; otherwise K = 1.
     * Supported by Metal with Euler integration. The analytic combination
     * is refused because its exact voltage gate fails. CUDA refuses gaps. */
    uint32_t n_gap; const int32_t *gap_tgt;
    const int64_t *gap_indptr; const int32_t *gap_src;
    const float *gap_weight, *gap_deg;
    float gap_scale, gap_spikelet;
    uint32_t gap_delay, gap_n_ap, gap_ap_off;
    const float *gap_ap_wave;
} cns_engine_desc;

/* Returns NULL and fills err on failure. Copies every array; the caller may
 * free them afterwards. */
cns_engine *cns_engine_create(const cns_engine_desc *d, char *err, size_t errlen);
void cns_engine_destroy(cns_engine *e);

/* Optional Metal junction law, configured before the first tick. Pair endpoints
 * index the compact gap rows. law=1 conserves current, law=2 is one-way.
 * The historical descriptor and default dispatch remain unchanged. */
int cns_engine_gap_law_config(cns_engine *e, uint32_t law, uint32_t pairs,
    const int32_t *ends, const float *weights, float rho,
    uint32_t steps, uint32_t tap);
const float *cns_engine_gap_tap(cns_engine *e);

/* Optional CUDA per-row release resource (design 0e0587be). Configure once
 * before advancing: pos has n entries, -1 off and a unique [0,n_resource)
 * index on each selected row. Coefficients have n_resource float32 entries.
 * The log records the spike-time value used by scatter and overflow delivery.
 * Capacity exhaustion is an error, including forced spikes beyond the normal
 * refractory bound. The Metal host refuses this mechanism until ported. */
int cns_engine_resource_config(cns_engine *e, uint32_t n_resource,
    const int32_t *pos, const float *d, const float *c, const float *keep,
    uint32_t log_cap);
int64_t cns_engine_resource_log(cns_engine *e, int32_t *tick, int32_t *row,
    float *value, uint32_t cap);

/* Optional CUDA compass observation and arrival interruption. group is 0
 * outside, 1 E, 2 Delta7; pos is -1 outside the recorded target set. The
 * ordinary descriptor and disabled path are unchanged. A negative cut_tick
 * records a sham. Native taps have 29 float32 fields per target and tick:
 * 12 signed-class arrivals, 12 decayed class accumulators, adaptation before
 * and after integration/reset, ungated positive E-E arrivals, gate flag and
 * delivered positive E-E edge count. Classes: positive E-E, negative D-D,
 * E-D, D-E, other ring edges, outside-ring sources; each has E/I channels. */
int cns_engine_compass_config(cns_engine *e, uint32_t targets,
    const int32_t *pos, const uint8_t *group, int32_t cut_tick);
const float *cns_engine_compass_tap(cns_engine *e);
/* Complete dynamical state immediately before the first positive E-E arrival
 * at or after tick 17000, on sham and interruption alike. Return bytes, or
 * zero if there was no such arrival. A NULL buffer queries size and tick. */
uint64_t cns_engine_compass_snapshot(cns_engine *e, uint8_t *buffer,
    uint64_t capacity, uint32_t *tick);

/* Advance n_ticks from absolute tick t0. cand_bits / force_bits are
 * n_ticks x ceil(n/32) uint32 bitmasks (window-relative rows) or NULL:
 * a candidate fires if not refractory, a forced neuron fires regardless.
 * Spikes are written tick-ordered as (tick offset, neuron). Returns the
 * count, -1 if cap was too small, -2 on a GPU error, -3 if a tick's spike
 * list overflowed fired_cap. */
int64_t cns_engine_run(cns_engine *e, uint32_t t0, uint32_t n_ticks,
                       const uint32_t *cand_bits, const uint32_t *force_bits,
                       uint32_t *out_tick, int32_t *out_id, int64_t cap);

/* After a run: n_ticks x n_vrec x 4 float32 state tap. Fields are
 * post-integration/pre-reset voltage, voltage entering the membrane equation,
 * post-decay/arrival excitatory accumulator, and the inhibitory accumulator.
 * The Python vtap() compatibility method returns field 0 only. */
const float *cns_engine_vtap(cns_engine *e);
/* After a run: tapped graded release, n_ticks x n_reltap float32; rows of
 * ticks that were not graded ticks are left as they were. */
const float *cns_engine_reltap(cns_engine *e);
/* Graded drive clamp for the next run: bitmask (ceil(n/32) words) of
 * neurons whose v is SET to clamp_val[i] every tick after threshold
 * detection and before graded release, as cns.run() does for driven
 * graded cells. NULL clears it. */
void cns_engine_set_clamp(cns_engine *e, const uint32_t *clamp_bits, const float *clamp_val);

/* Event-driven command membrane delivery. row_bits marks the subset of
 * cand_bits that are membrane kicks rather than forced spike candidates;
 * kick_mv is a length-n per-row vector. NULL clears it. The stochastic event
 * tape remains cand_bits, so enabling this changes delivery, not RNG draws. */
void cns_engine_set_membrane_drive(cns_engine *e, const uint32_t *row_bits,
                                   const float *kick_mv);

/* Copies of live state (any pointer may be NULL). */
void cns_engine_read_state(cns_engine *e, float *v, float *g, float *g_i,
                           float *w_adapt, uint32_t *counts);
double cns_engine_gpu_seconds(cns_engine *e);
/* The seam's receipt, reduced over targets: applied |gain*fs*slow| max and
 * running sum, fs max, delayed-slow max (cns.run()'s four scalars; the sum
 * is accumulated per target in float32 and reduced in double, so it agrees
 * with numpy's per-tick float32 sum to rounding, not bit for bit). */
void cns_engine_coin_stats(cns_engine *e, double *out4);
/* Per-kernel GPU seconds and dispatch counts (step, scatter, graded, flush, coin).
 * Filled only when the engine was created with CNS_ENGINE_PROFILE=1 in the
 * environment; that mode is slower (one encoder per dispatch). */
void cns_engine_profile(cns_engine *e, double *secs, uint64_t *counts);
/* Metal-only extra profile buckets: continuous gap current and spikelets.
 * Each output array has two entries. The existing five-bucket ABI is unchanged. */
void cns_engine_gap_profile(cns_engine *e, double *secs, uint64_t *counts);
uint32_t cns_engine_kgroup(cns_engine *e);
/* Spike-delivery bins that overflowed BIN_CAP so far (each took the exact,
 * slower transposed-graph path in k_fold); a diagnostic, not an error. */
uint32_t cns_engine_fold_overflows(cns_engine *e);
uint32_t cns_engine_bin_cap(void);
/* Largest per-target bin any tick filled so far, rounded down to a multiple of 8
 * (0 when no bin reached 8): the number that sizes BIN_CAP. */
uint32_t cns_engine_bin_max(cns_engine *e);
/* Mean number of targets that received at least one spike per tick so far:
 * the size of k_fold's dispatch. */
double cns_engine_touched_per_tick(cns_engine *e);
/* Target-ticks whose spike bin reached 4, 8, 16 and 32 entries (four counters):
 * the bin-size histogram that sets k_fold's hub threshold. */
void cns_engine_bin_hist(cns_engine *e, uint32_t *out4);
/* bit k set when per-neuron array k was found uniform and specialised away:
 * 0 v_rest 1 v_reset 2 spike_at 3 decay_v 4 dt_over_tau 5 decay_g 6 refrac 7 adaptation
 * 8 v_rest read as a scalar plus an override bitmask (few neurons differ) */
uint32_t cns_engine_uniform_mask(cns_engine *e);
const char *cns_engine_device(cns_engine *e);
/* SHA-256 of cns_engine.m followed by cns_engine.h as they were when this
 * library was compiled (the Makefile passes it in). cns_metal.py hashes the
 * same two files at load and refuses a mismatch: a dylib built before a
 * kernel-signature change ran the network SILENT with no error on
 * 2026-09-02 (int16 arrays into an int32 kernel), twice in five hours. */
const char *cns_engine_source_sha256(void);
const char *cns_engine_last_error(cns_engine *e);

#ifdef __cplusplus
}
#endif
