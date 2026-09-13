"""Metal backend for CNS.run(): the step loop on the laptop's own GPU.

Default off. `SimParams.backend = "metal"` (theta key `backend`) routes
CNS.run() here; every other run is the numpy path, byte for byte.

Division of labour. Everything that is not per-tick arithmetic over the
188,508 neurons stays in this file, in Python, replicating cns.run()'s
logic line for line: the Poisson and renewal drive draws (so the random
stream is the numpy engine's, same seed, same order), the drive_fn
callback cadence, forced spikes, the trace bins, first-spike times and
the feedback counts. The GPU library (gpu/cns_engine.m, loaded through
ctypes) holds the state for the life of the network and advances windows
of ticks, returning each window's spike list.

Supported: current-based and conductance-based membranes (euler and
analytic), true shunting (split conductances), adaptation, per-neuron
membrane / threshold / refractory / synaptic-decay vectors, per-neuron
delays of either delivery flavour, graded release with per-neuron or
scalar curves, graded feedback rows and graded drive (the visual
hs_graded_opponent relay), Poisson and Erlang-renewal drive, drive_fn with
feedback rows, force_spikes, first_spike, record, record_trace_every and
record_voltage (voltage after the refractory clamp, before thresholding),
absolute or cell-relative command membrane drive, and the coincidence seam
(coincidence_gain with its target / fast / slow pools; ported 2026-09-05,
gate `gpu/gate_v0.py --coincidence`, per-target deterministic gather so the
sums run in the numpy engine's order).

Refused loudly (NotImplementedError), never silently approximated: gap
junctions, short-term depression, exponential integrate-and-fire,
graded_phasic and force_spikes_fn. Add them to the kernels with their own fingerprint before
lifting a refusal. A graded cell in the drive set was refused until 2026-09-06
and now runs: see the note at the drive clamp.

Parity: the per-neuron arithmetic is the numpy engine's, in float32 with
fast math off, so the first ticks agree exactly; the synaptic sums run in
float32 in ascending source order per target and arrival slot (the C
reference's order) where np.bincount sums in float64 and rounds once, so
the two rasters separate after the first neuron within one ulp of
threshold tips, and agreement after that is statistical. gpu/gate_v0.py
measures both.

Determinism (2026-09-05): no float atomic remains in the engine. Spike
delivery appends each edge to its target's bin with an integer atomic and a
per-target fold adds the bin in sorted order; graded release is a
per-target gather in stored order. The same network, seed and drive give
the same spike train to the bit on any number of concurrent Metal workers
(gpu/gate_replay.py), so the shared advisory lock is about memory and
throughput, not results.
"""
from __future__ import annotations

import ctypes
try:
    import fcntl
except ImportError:  # Portable CPU reproduction on Windows.
    fcntl = None
import hashlib
import json
import os
import time
import weakref
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

_LIB = None
_MAC = os.uname().sysname == "Darwin"
_LIB_PATH = (Path(__file__).resolve().parent / "gpu"
             / ("libcnsengine.dylib" if _MAC else "libcnsengine.so"))
# The two files the library is built from, in the build's hashing order.
_SRC_PATHS = [_LIB_PATH.parent / ("cns_engine.m" if _MAC else "cns_engine.cu"),
              _LIB_PATH.parent / "cns_engine.h"]
_LOCK_PATH = Path("/tmp/fly-wbe-metal.lock")


@contextmanager
def _metal_run_lock():
    """Coordinate Metal use across the repository's worktrees.

    Routine BUILD runs use a shared lock and may contend. Receipts intended to
    support a title, champion, registry change, or scientific claim set
    ``FLY_WBE_METAL_LOCK_MODE=exclusive`` and run without another Metal worker.
    Kernel crashes release the advisory lock with the process.
    """
    mode = os.environ.get("FLY_WBE_METAL_LOCK_MODE", "shared")
    if mode not in {"shared", "exclusive"}:
        raise ValueError(
            "FLY_WBE_METAL_LOCK_MODE must be 'shared' or 'exclusive'")
    try:
        timeout_s = float(os.environ.get(
            "FLY_WBE_METAL_LOCK_TIMEOUT_S", "900"))
    except ValueError as exc:
        raise ValueError(
            "FLY_WBE_METAL_LOCK_TIMEOUT_S must be a finite positive number"
        ) from exc
    if not np.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError(
            "FLY_WBE_METAL_LOCK_TIMEOUT_S must be a finite positive number")

    lock_path = Path(os.environ.get("FLY_WBE_METAL_LOCK_PATH", str(_LOCK_PATH)))
    operation = fcntl.LOCK_EX if mode == "exclusive" else fcntl.LOCK_SH
    started = time.monotonic()
    with lock_path.open("a+") as lock_file:
        while True:
            try:
                fcntl.flock(lock_file.fileno(), operation | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - started >= timeout_s:
                    raise TimeoutError(
                        f"timed out after {timeout_s:g}s waiting for {mode} "
                        f"Metal lock {lock_path}") from None
                time.sleep(0.1)
        try:
            yield {"mode": mode, "wait_s": time.monotonic() - started}
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _source_sha256():
    h = hashlib.sha256()
    for p in _SRC_PATHS:
        h.update(p.read_bytes())
    return h.hexdigest()


def _check_build_stamp(L):
    """Refuse a dylib whose compiled-in source hash differs from the files on
    disk. A stale dylib fed int16 arrays to an int32 kernel on 2026-09-02 and
    the network ran silent with no error; this makes that a load-time
    refusal with the remedy in the message."""
    try:
        f = L.cns_engine_source_sha256
    except AttributeError:
        raise RuntimeError(
            f"{_LIB_PATH} predates the build stamp (no cns_engine_source_sha256 "
            f"symbol) and may not match gpu/cns_engine.m: rebuild with "
            f"`make -C {_LIB_PATH.parent} libcnsengine.dylib`") from None
    f.restype = ctypes.c_char_p
    f.argtypes = []
    built = f().decode()
    on_disk = _source_sha256()
    if built != on_disk:
        raise RuntimeError(
            f"{_LIB_PATH} is stale: built from source {built[:16]}, "
            f"gpu/cns_engine.m + cns_engine.h on disk are {on_disk[:16]}: "
            f"rebuild with `make -C {_LIB_PATH.parent} libcnsengine.dylib`")
    return built


class _Desc(ctypes.Structure):
    _fields_ = [
        ("n", ctypes.c_uint32), ("n_ring", ctypes.c_uint32),
        ("window_ticks", ctypes.c_uint32), ("kmax", ctypes.c_uint32),
        ("fired_cap", ctypes.c_uint32),
        ("cbased", ctypes.c_int32), ("split", ctypes.c_int32),
        ("analytic", ctypes.c_int32), ("use_adapt", ctypes.c_int32),
        ("e_exc", ctypes.c_float), ("e_inh", ctypes.c_float),
        ("gscale", ctypes.c_float),
        ("v_rest", ctypes.c_void_p), ("v_reset", ctypes.c_void_p),
        ("spike_at", ctypes.c_void_p), ("decay_v", ctypes.c_void_p),
        ("dt_over_tau", ctypes.c_void_p), ("decay_g", ctypes.c_void_p),
        ("refrac_ticks", ctypes.c_void_p), ("delay_slot", ctypes.c_void_p),
        ("graded_mask", ctypes.c_void_p),
        ("adapt_a", ctypes.c_void_p), ("adapt_b", ctypes.c_void_p),
        ("dt_over_tauw", ctypes.c_void_p),
        ("indptr", ctypes.c_void_p), ("indices", ctypes.c_void_p),
        ("weight", ctypes.c_void_p), ("nnz", ctypes.c_uint64),
        ("n_graded", ctypes.c_uint32), ("graded_idx", ctypes.c_void_p),
        ("gv50", ctypes.c_void_p), ("gslope", ctypes.c_void_p),
        ("ggain_scaled", ctypes.c_float), ("g_every", ctypes.c_uint32),
        ("n_vrec", ctypes.c_uint32), ("vrec", ctypes.c_void_p),
        ("n_reltap", ctypes.c_uint32), ("reltap_pos", ctypes.c_void_p),
        ("ggain_rows", ctypes.c_void_p),
        ("n_coin", ctypes.c_uint32), ("coin_tgt", ctypes.c_void_p),
        ("coin_fast_indptr", ctypes.c_void_p), ("coin_fast_src", ctypes.c_void_p),
        ("coin_fast_w", ctypes.c_void_p),
        ("coin_slow_indptr", ctypes.c_void_p), ("coin_slow_src", ctypes.c_void_p),
        ("coin_slow_w", ctypes.c_void_p),
        ("coin_gain", ctypes.c_float), ("coin_decay", ctypes.c_float),
        ("coin_lag", ctypes.c_uint32),
        ("tonic_ge", ctypes.c_void_p),
    ]


def _lib():
    global _LIB
    if _LIB is None:
        if not _LIB_PATH.exists():
            raise RuntimeError(
                f"{_LIB_PATH} is missing: build it with "
                f"`make -C {_LIB_PATH.parent} libcnsengine.dylib`")
        L = ctypes.CDLL(str(_LIB_PATH))
        global _ENGINE_SHA
        _ENGINE_SHA = _check_build_stamp(L)[:16]
        L.cns_engine_create.restype = ctypes.c_void_p
        L.cns_engine_create.argtypes = [ctypes.POINTER(_Desc), ctypes.c_char_p,
                                        ctypes.c_size_t]
        L.cns_engine_destroy.argtypes = [ctypes.c_void_p]
        L.cns_engine_run.restype = ctypes.c_int64
        L.cns_engine_run.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                     ctypes.c_uint32, ctypes.c_void_p,
                                     ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_void_p, ctypes.c_int64]
        L.cns_engine_read_state.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 5
        L.cns_engine_vtap.restype = ctypes.POINTER(ctypes.c_float)
        L.cns_engine_vtap.argtypes = [ctypes.c_void_p]
        L.cns_engine_reltap.restype = ctypes.POINTER(ctypes.c_float)
        L.cns_engine_reltap.argtypes = [ctypes.c_void_p]
        L.cns_engine_set_clamp.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        L.cns_engine_set_clamp.restype = None
        L.cns_engine_set_membrane_drive.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        L.cns_engine_set_membrane_drive.restype = None
        L.cns_engine_gpu_seconds.restype = ctypes.c_double
        L.cns_engine_gpu_seconds.argtypes = [ctypes.c_void_p]
        L.cns_engine_kgroup.restype = ctypes.c_uint32
        L.cns_engine_kgroup.argtypes = [ctypes.c_void_p]
        L.cns_engine_fold_overflows.restype = ctypes.c_uint32
        L.cns_engine_fold_overflows.argtypes = [ctypes.c_void_p]
        L.cns_engine_bin_cap.restype = ctypes.c_uint32
        L.cns_engine_bin_cap.argtypes = []
        L.cns_engine_bin_max.restype = ctypes.c_uint32
        L.cns_engine_bin_max.argtypes = [ctypes.c_void_p]
        L.cns_engine_touched_per_tick.restype = ctypes.c_double
        L.cns_engine_touched_per_tick.argtypes = [ctypes.c_void_p]
        L.cns_engine_bin_hist.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        L.cns_engine_uniform_mask.restype = ctypes.c_uint32
        L.cns_engine_uniform_mask.argtypes = [ctypes.c_void_p]
        L.cns_engine_profile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        L.cns_engine_coin_stats.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        L.cns_engine_coin_stats.restype = None
        L.cns_engine_device.restype = ctypes.c_char_p
        L.cns_engine_device.argtypes = [ctypes.c_void_p]
        L.cns_engine_last_error.restype = ctypes.c_char_p
        L.cns_engine_last_error.argtypes = [ctypes.c_void_p]
        _LIB = L
    return _LIB


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p) if a is not None else None


def _vec(x, n, dtype):
    """Broadcast a scalar or (n,) parameter to a contiguous (n,) array."""
    a = np.asarray(x, dtype=dtype)
    if a.ndim == 0:
        return np.full(n, a, dtype=dtype)
    if a.shape != (n,):
        raise ValueError(f"expected scalar or ({n},), got {a.shape}")
    return np.ascontiguousarray(a, dtype=dtype)


# At most one engine's GPU buffers per process. walk_search.evaluate caches
# one CNS object per distinct network config (`net_cache[key]`), and each of
# those keeps the engine it last ran with, so a sweep over K arms holds K
# engines' buffers while running one. Measured 2026-09-07 12:0x from the live
# fleet: one engine is about 670 MB of GPU memory on the champion network (a
# 170.4 MB delay ring per polarity, four 52.0 MB edge arrays, a 92.0 MB spike
# bin), lane A's worker carried four of them and lane C two, 2.7 GB and 1.3 GB
# of a 25.8 GB machine that was swapping out 30,084 pages every twenty seconds.
# Nothing is lost by closing them: run_metal builds a fresh engine for every
# run in any case, the graph upload being about 0.1 s against the run.
_LIVE_ENGINES: "weakref.WeakSet[MetalEngine]" = weakref.WeakSet()


class MetalEngine:
    """GPU-resident state for one CNS object and one parameter set."""

    def __init__(self, net, window_ticks, kmax=0, vrec=None, reltap_pos=None):
        p = net.p
        N = net.N
        self.net = net
        self.N = N
        self.window_ticks = int(window_ticks)
        # --- refusals: mechanisms the kernels do not carry ------------------
        if p.delta_T > 0:
            raise NotImplementedError("metal backend: exponential IF (delta_T)")
        if net.G is not None:
            raise NotImplementedError("metal backend: gap junctions")
        if p.std_U > 0:
            raise NotImplementedError("metal backend: short-term depression")
        if getattr(p, "spike_reg_per_neuron", None) is not None:
            raise NotImplementedError(
                "metal backend: per-neuron spike regularity not yet ported")
        if getattr(p, "presynaptic_inhibition_rows", None) is not None:
            raise NotImplementedError(
                "metal backend: presynaptic inhibition not yet ported")
        if float(getattr(p, "graded_phasic", 0.0) or 0.0) > 0.0:
            raise NotImplementedError("metal backend: graded_phasic")
        # --- per-neuron vectors, derived exactly as cns.run() derives them ---
        dt = float(p.dt)
        dpn = getattr(p, "delay_per_neuron", None)
        if dpn is None:
            delay_slot = np.full(N, max(1, int(round(p.delay / dt))), dtype=np.int32)
        else:
            _dpn = np.asarray(dpn, dtype=np.float64)
            if _dpn.shape != (N,):
                raise ValueError("delay_per_neuron needs one value per neuron")
            if not np.all(np.isfinite(_dpn)) or np.any(_dpn < 0):
                raise ValueError("delay_per_neuron must be finite and >= 0")
            delay_slot = np.maximum(1, np.rint(_dpn / dt)).astype(np.int32)
        n_ring = int(delay_slot.max()) + 1
        n_refrac = np.broadcast_to(
            np.rint(np.asarray(p.t_refrac, dtype=np.float64) / dt
                    ).astype(np.int32), (N,)).copy()
        if n_refrac.max() > 32000 or n_refrac.min() < 0:
            raise NotImplementedError("metal backend: refractory ticks must fit int16")
        tau_mem = np.asarray(p.tau_mem, dtype=np.float32)
        v_rest = np.asarray(p.v_rest, dtype=np.float32)
        if p.v_rest_override:
            v_rest = np.broadcast_to(v_rest, (N,)).copy()
            pos_vo = pd.Series(np.arange(N), index=net.ids)
            for bid, mv in p.v_rest_override.items():
                r_ = pos_vo.get(bid)
                if r_ is not None and not np.isnan(r_):
                    v_rest[int(r_)] = mv
        v_reset = _vec(p.v_reset, N, np.float32)
        decay_v = _vec(np.exp(-dt / tau_mem).astype(np.float32), N, np.float32)
        dt_over_tau = _vec(np.float32(dt) / tau_mem, N, np.float32)
        tau_syn = np.asarray(p.tau_syn, dtype=np.float32)
        if tau_syn.ndim == 0:
            decay_g = np.full(N, np.float32(np.exp(-dt / tau_syn)), dtype=np.float32)
        else:
            decay_g = np.exp(-np.float32(dt) / tau_syn).astype(np.float32)
        gscale = 1.0 / abs(p.e_exc - float(np.mean(np.asarray(p.v_rest))))
        spike_at = _vec(p.v_threshold, N, np.float32)
        adapt_a = np.asarray(p.adapt_a, dtype=np.float32)
        adapt_b = np.asarray(p.adapt_b, dtype=np.float32)
        use_adapt = bool(np.any(adapt_a != 0) or np.any(adapt_b != 0))
        dt_over_tauw = np.float32(dt) / np.asarray(p.tau_w, dtype=np.float32)
        graded_mask = np.ascontiguousarray(net.graded, dtype=np.uint8)
        graded_idx = np.ascontiguousarray(
            getattr(net, "graded_idx", np.empty(0, np.int64)), dtype=np.int32)
        ng = int(len(graded_idx))
        gv50 = np.asarray(p.graded_v50, dtype=np.float32)
        gslope = np.asarray(p.graded_slope, dtype=np.float32)
        gv50 = (np.full(ng, gv50, np.float32) if gv50.ndim == 0
                else np.ascontiguousarray(gv50[graded_idx], dtype=np.float32))
        gslope = (np.full(ng, gslope, np.float32) if gslope.ndim == 0
                  else np.ascontiguousarray(gslope[graded_idx], dtype=np.float32))
        g_every = max(1, int(p.graded_every))
        _gg = np.asarray(p.graded_gain, dtype=np.float32)
        if _gg.ndim == 0:
            ggain_scaled = np.float32(_gg) * g_every
            ggain_rows = None
        else:
            if (_gg.shape != (N,) or not np.all(np.isfinite(_gg))
                    or np.any(_gg < 0.0)):
                raise ValueError("graded_gain must be a finite scalar or a "
                                 "finite non-negative (N,) array")
            ggain_scaled = np.float32(0.0)
            ggain_rows = np.ascontiguousarray(
                _gg[graded_idx] * np.float32(g_every), dtype=np.float32)
        # coincidence seam: the two pool matrices exactly as cns.run() builds
        # them (|w| of the graph's edges from each pool onto the targets, CSR
        # over the source axis in scipy's canonical order), the same refusal
        # of an empty pool, and the same float32 gain and decay.
        self.n_coin = 0
        self.coin_receipt = None
        coin_arrays = {}
        if (p.coincidence_gain and p.coincidence_targets is not None
                and p.coincidence_fast_rows is not None
                and p.coincidence_slow_rows is not None):
            _ctgt = np.asarray(p.coincidence_targets, dtype=np.int64)
            if _ctgt.size:
                from scipy import sparse
                _tpos = np.full(N, -1, dtype=np.int64)
                _tpos[_ctgt] = np.arange(len(_ctgt))
                _pre_all = np.repeat(np.arange(N, dtype=np.int64),
                                     np.diff(net._indptr))
                _post_all = net._indices
                _in_t = _tpos[_post_all] >= 0
                _mats = []
                for _srcs in (p.coincidence_fast_rows,
                              p.coincidence_slow_rows):
                    _m = _in_t & np.isin(_pre_all,
                                         np.asarray(_srcs, dtype=np.int64))
                    _mats.append(sparse.csr_matrix(
                        (np.abs(net._data[_m]).astype(np.float32),
                         (_tpos[_post_all[_m]], _pre_all[_m])),
                        shape=(len(_ctgt), N)))
                for _nm, _mat in (("fast", _mats[0]), ("slow", _mats[1])):
                    if _mat.nnz == 0:
                        raise ValueError(
                            f"coincidence seam: the {_nm} source pool has NO "
                            f"edges onto the {len(_ctgt)} designated targets, so "
                            f"its product term is identically zero at any gain. "
                            f"Check coincidence_{_nm}_rows against the target "
                            f"set; the other pool carries {_mats[1 - (_nm == 'fast')].nnz} edges.")
                _clag = max(1, int(round(float(p.coincidence_lag_ms) / dt)))
                coin_arrays = dict(
                    coin_tgt=np.ascontiguousarray(_ctgt, dtype=np.int32),
                    coin_fast_indptr=np.ascontiguousarray(_mats[0].indptr, dtype=np.int64),
                    coin_fast_src=np.ascontiguousarray(_mats[0].indices, dtype=np.int32),
                    coin_fast_w=np.ascontiguousarray(_mats[0].data, dtype=np.float32),
                    coin_slow_indptr=np.ascontiguousarray(_mats[1].indptr, dtype=np.int64),
                    coin_slow_src=np.ascontiguousarray(_mats[1].indices, dtype=np.int32),
                    coin_slow_w=np.ascontiguousarray(_mats[1].data, dtype=np.float32))
                self.n_coin = int(len(_ctgt))
                self.coin_lag = int(_clag)
                self.coin_gain = np.float32(p.coincidence_gain)
                self.coin_decay = np.float32(np.exp(
                    -dt / max(1e-6, float(p.coincidence_tau_ms))))
                self.coin_receipt = {"fast_nnz": int(_mats[0].nnz),
                                     "slow_nnz": int(_mats[1].nnz),
                                     "n_targets": int(len(_ctgt)),
                                     "gain": float(self.coin_gain)}
        tonic_arr = None
        _tg = getattr(p, "tonic_exc_conductance_per_neuron", None)
        if _tg is not None:
            # cns.py validates shape, finiteness, non-negativity and that
            # conductance_based is on before it reaches us; re-checked here
            # because the engine reads the buffer with no further test.
            tonic_arr = np.ascontiguousarray(np.asarray(_tg, dtype=np.float32))
            if (tonic_arr.shape != (N,) or not np.all(np.isfinite(tonic_arr))
                    or np.any(tonic_arr < 0.0)):
                raise ValueError("tonic_exc_conductance_per_neuron must be a "
                                 "finite nonnegative (N,) vector")
            if not p.conductance_based:
                raise ValueError("tonic_exc_conductance_per_neuron requires "
                                 "conductance_based=True")
        self.tonic_arr = tonic_arr
        W = net.W
        if W.format != "csc":
            raise RuntimeError("net.W must be CSC on the presynaptic axis")
        indptr = np.ascontiguousarray(W.indptr, dtype=np.int64)
        indices = np.ascontiguousarray(W.indices, dtype=np.int32)
        weight = np.ascontiguousarray(W.data, dtype=np.float32)
        self.n_ring = n_ring
        self.split = bool(p.conductance_based and p.split_conductances)
        self.use_adapt = use_adapt
        self.delay_slot = np.ascontiguousarray(delay_slot, dtype=np.int32)
        self.decay_g = np.ascontiguousarray(decay_g, dtype=np.float32)
        self.graded_idx = graded_idx
        self.graded_gain_scaled = np.ascontiguousarray(
            (np.full(ng, ggain_scaled, dtype=np.float32)
             if ggain_rows is None else ggain_rows), dtype=np.float32)
        self.gscale = np.float32(gscale)
        self.e_exc = np.float32(p.e_exc)
        self.e_inh = np.float32(p.e_inh)
        keep = dict(v_rest=np.ascontiguousarray(_vec(v_rest, N, np.float32)),
                    v_reset=v_reset, spike_at=spike_at, decay_v=decay_v,
                    dt_over_tau=dt_over_tau, decay_g=decay_g,
                    refrac_ticks=np.ascontiguousarray(n_refrac, dtype=np.int16),
                    delay_slot=np.ascontiguousarray(delay_slot, dtype=np.int32),
                    graded_mask=graded_mask,
                    adapt_a=_vec(adapt_a, N, np.float32) if use_adapt else None,
                    adapt_b=_vec(adapt_b, N, np.float32) if use_adapt else None,
                    dt_over_tauw=_vec(dt_over_tauw, N, np.float32) if use_adapt else None,
                    indptr=indptr, indices=indices, weight=weight,
                    graded_idx=graded_idx, gv50=gv50, gslope=gslope,
                    ggain_rows=ggain_rows)
        d = _Desc()
        d.n, d.n_ring = N, n_ring
        d.window_ticks, d.kmax, d.fired_cap = self.window_ticks, int(kmax), 0
        d.cbased = int(bool(p.conductance_based))
        d.split = int(self.split)
        d.analytic = int(getattr(p, "conductance_integrator", "euler") == "analytic")
        d.use_adapt = int(use_adapt)
        d.e_exc, d.e_inh, d.gscale = float(p.e_exc), float(p.e_inh), float(gscale)
        for k in ("v_rest", "v_reset", "spike_at", "decay_v", "dt_over_tau",
                  "decay_g", "refrac_ticks", "delay_slot", "graded_mask",
                  "adapt_a", "adapt_b", "dt_over_tauw", "indptr", "indices",
                  "weight", "graded_idx", "gv50", "gslope"):
            setattr(d, k, _ptr(keep[k]))
        d.nnz = int(W.nnz)
        d.n_graded = ng
        d.ggain_scaled = float(ggain_scaled)
        d.ggain_rows = _ptr(keep["ggain_rows"])
        d.g_every = g_every
        self.vrec = (None if vrec is None
                     else np.ascontiguousarray(vrec, dtype=np.int32))
        keep["vrec"] = self.vrec
        d.n_vrec = 0 if self.vrec is None else int(len(self.vrec))
        d.vrec = _ptr(self.vrec)
        self.reltap_pos = (None if reltap_pos is None
                           else np.ascontiguousarray(reltap_pos, dtype=np.int32))
        keep["reltap_pos"] = self.reltap_pos
        d.n_reltap = 0 if self.reltap_pos is None else int(len(self.reltap_pos))
        d.reltap_pos = _ptr(self.reltap_pos)
        keep.update(coin_arrays)
        d.n_coin = self.n_coin
        for k in ("coin_tgt", "coin_fast_indptr", "coin_fast_src", "coin_fast_w",
                  "coin_slow_indptr", "coin_slow_src", "coin_slow_w"):
            setattr(d, k, _ptr(coin_arrays.get(k)))
        if self.n_coin:
            d.coin_gain = float(self.coin_gain)
            d.coin_decay = float(self.coin_decay)
            d.coin_lag = self.coin_lag
        if self.tonic_arr is not None:
            keep["tonic_ge"] = self.tonic_arr
            d.tonic_ge = _ptr(self.tonic_arr)
        self._keep = keep
        # kept for the graded-drive clamp values (rest + level*(upper-rest))
        self.v_rest = keep["v_rest"]
        self.spike_at = keep["spike_at"]
        err = ctypes.create_string_buffer(512)
        L = _lib()
        self.h = L.cns_engine_create(ctypes.byref(d), err, 512)
        if not self.h:
            raise RuntimeError(f"cns_engine_create: {err.value.decode()}")
        self._L = L
        self.device = L.cns_engine_device(self.h).decode()
        self.kgroup = int(L.cns_engine_kgroup(self.h))
        self.uniform_mask = int(L.cns_engine_uniform_mask(self.h))
        self._snap: dict = {}
        _LIVE_ENGINES.add(self)
        self.words = (N + 31) // 32
        cap = self.window_ticks * min(N, 65536)
        self._out_tick = np.empty(cap, dtype=np.uint32)
        self._out_id = np.empty(cap, dtype=np.int32)

    def run(self, t0, n_ticks, cand_bits=None, force_bits=None):
        """Advance n_ticks; returns (tick_offsets uint32, neuron ids int32)."""
        got = self._L.cns_engine_run(
            self.h, int(t0), int(n_ticks), _ptr(cand_bits), _ptr(force_bits),
            _ptr(self._out_tick), _ptr(self._out_id), len(self._out_tick))
        if got < 0:
            raise RuntimeError(
                f"cns_engine_run returned {got}: "
                f"{self._L.cns_engine_last_error(self.h).decode()}")
        return self._out_tick[:got].copy(), self._out_id[:got].copy()

    def reltap(self, n_ticks):
        """Tapped graded release, (n_ticks, n_reltap) float32 copy; only the
        rows of graded ticks (t % g_every == 0) are meaningful."""
        nt = len(self.reltap_pos)
        buf = np.ctypeslib.as_array(self._L.cns_engine_reltap(self.h),
                                    shape=(self.window_ticks, nt))
        return buf[:n_ticks].copy()

    def set_clamp(self, bits, vals):
        """Graded drive for the next window: neurons in `bits` get v set to
        `vals[i]` every tick after threshold detection; None clears."""
        self._L.cns_engine_set_clamp(self.h, _ptr(bits), _ptr(vals))

    def set_membrane_drive(self, bits, kick_mv):
        """Set command-row membership and each row's event kick in mV."""
        self._L.cns_engine_set_membrane_drive(
            self.h, _ptr(bits), _ptr(kick_mv))

    def state_tap(self, n_ticks):
        """Native-tick state for tapped rows, shape (ticks, rows, 4).

        Fields are post-integration/pre-reset voltage, voltage entering the
        membrane equation, and the post-decay/arrival excitatory and
        inhibitory accumulators. The tap writes no live state.
        """
        nv = len(self.vrec)
        buf = np.ctypeslib.as_array(self._L.cns_engine_vtap(self.h),
                                    shape=(self.window_ticks, nv, 4))
        return buf[:n_ticks].copy()

    def vtap(self, n_ticks):
        """Compatibility view of the historical post-integration voltage."""
        return self.state_tap(n_ticks)[..., 0]

    def read_state(self):
        N = self.N
        v = np.empty(N, np.float32); g = np.empty(N, np.float32)
        gi = np.empty(N, np.float32) if self.split else None
        w = np.empty(N, np.float32) if self.use_adapt else None
        c = np.empty(N, np.uint32)
        self._L.cns_engine_read_state(self.h, _ptr(v), _ptr(g), _ptr(gi),
                                      _ptr(w), _ptr(c))
        return {"v": v, "g": g, "g_i": gi, "w_adapt": w, "counts": c}

    def profile(self):
        """Per-kernel GPU seconds and dispatch counts; zeros unless the
        process set CNS_ENGINE_PROFILE=1 before the engine was created."""
        names = ("step", "scatter", "graded", "fold", "coin")
        secs = np.zeros(5); cnt = np.zeros(5, dtype=np.uint64)
        self._L.cns_engine_profile(self.h, _ptr(secs), _ptr(cnt))
        return dict(zip(names, secs)), dict(zip(names, cnt))

    def coin_stats(self):
        """The seam's receipt so far: applied |product| max and sum, fs max,
        delayed-slow max, reduced over targets."""
        out = np.zeros(4)
        self._L.cns_engine_coin_stats(self.h, _ptr(out))
        return out

    def gpu_seconds(self):
        if self.h is None:
            return self._snap["gpu_seconds"]
        return float(self._L.cns_engine_gpu_seconds(self.h))

    def fold_overflows(self):
        """Targets whose spike bin overflowed in some tick group and took the
        exact transposed-graph path instead (a cost note, not an error)."""
        if self.h is None:
            return self._snap["fold_overflows"]
        return int(self._L.cns_engine_fold_overflows(self.h))

    def bin_max(self):
        """Largest per-target spike bin any tick filled, rounded down to 8; the
        gauge that sizes BIN_CAP (a run with bin_max near bin_cap is about to
        overflow into the slow exact path)."""
        if self.h is None:
            return self._snap["bin_max"]
        return int(self._L.cns_engine_bin_max(self.h))

    def touched_per_tick(self):
        """Mean targets receiving at least one spike per tick: k_fold's dispatch size."""
        if self.h is None:
            return self._snap["touched_per_tick"]
        return float(self._L.cns_engine_touched_per_tick(self.h))

    def bin_hist(self):
        """Target-ticks whose spike bin reached 4, 8, 16 and 32 entries."""
        out = np.zeros(4, dtype=np.uint32)
        self._L.cns_engine_bin_hist(self.h, _ptr(out))
        return [int(x) for x in out]

    def close(self):
        if getattr(self, "h", None):
            # A reader that asks for this run's GPU seconds after the buffers
            # are freed gets the recorded value rather than a call through a
            # dead handle: 39 of the 40 uses of `_metal_engine` in the tree
            # read scalars like these, and only one reads live device state.
            self._snap = {
                "gpu_seconds": float(self._L.cns_engine_gpu_seconds(self.h)),
                "fold_overflows": int(self._L.cns_engine_fold_overflows(self.h)),
                "bin_max": int(self._L.cns_engine_bin_max(self.h)),
                "touched_per_tick": float(
                    self._L.cns_engine_touched_per_tick(self.h)),
            }
            self._L.cns_engine_destroy(self.h)
            self.h = None
        _LIVE_ENGINES.discard(self)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class _InputRecorder:
    """Read-only native-tick attribution for a small target set.

    Spike and graded events are taken from the same Metal run that advances
    the network. The engine tap supplies the exact target state used by the
    membrane equation. Source routes only partition observation; they never
    alter the graph, state, drive, or random stream.
    """

    _SPEC_KEYS = {"target_rows", "route_by_source", "route_names", "window_ms"}

    def __init__(self, net, spec, n_steps, dt):
        if not isinstance(spec, dict) or set(spec) != self._SPEC_KEYS:
            raise ValueError(
                "record_input requires exactly target_rows, route_by_source, "
                "route_names, and window_ms")
        if not (net.p.conductance_based and net.p.split_conductances):
            raise ValueError(
                "record_input requires conductance_based and split_conductances")

        targets = np.asarray(spec["target_rows"])
        if (targets.ndim != 1 or not np.issubdtype(targets.dtype, np.integer)
                or len(targets) == 0
                or len(np.unique(targets)) != len(targets)
                or np.any(targets < 0) or np.any(targets >= net.N)):
            raise ValueError("record_input target_rows must be unique valid rows")
        self.target_rows = np.ascontiguousarray(targets, dtype=np.int64)

        names = tuple(spec["route_names"])
        if (not names or len(set(names)) != len(names)
                or any(not isinstance(name, str) or not name for name in names)):
            raise ValueError("record_input route_names must be unique nonempty strings")
        self.route_names = names
        routes = np.asarray(spec["route_by_source"])
        if (routes.shape != (net.N,)
                or not np.issubdtype(routes.dtype, np.integer)
                or np.any(routes < 0) or np.any(routes >= len(names))):
            raise ValueError(
                "record_input route_by_source must assign every neuron a valid route")
        self.route_by_source = np.ascontiguousarray(routes, dtype=np.int16)

        window_ms = spec["window_ms"]
        if (not isinstance(window_ms, (list, tuple)) or len(window_ms) != 2
                or not all(type(value) in (int, float) and np.isfinite(value)
                           for value in window_ms)):
            raise ValueError("record_input window_ms must contain two finite numbers")
        start = int(round(float(window_ms[0]) / dt))
        stop = int(round(float(window_ms[1]) / dt))
        if (not np.isclose(start * dt, float(window_ms[0]), rtol=0.0, atol=1e-9)
                or not np.isclose(stop * dt, float(window_ms[1]), rtol=0.0,
                                  atol=1e-9)
                or not 0 <= start < stop <= n_steps):
            raise ValueError(
                "record_input window_ms must align to native ticks inside the run")
        self.window_ticks = (start, stop)
        self.window_ms = (float(window_ms[0]), float(window_ms[1]))
        self.n_steps = int(n_steps)
        self.dt = float(dt)

        weights = net.W[self.target_rows, :].toarray().T
        self.weights = np.ascontiguousarray(weights, dtype=np.float32)
        self.incoming = np.any(self.weights != 0.0, axis=1)
        self.incoming_source_rows = np.flatnonzero(self.incoming).astype(np.int64)
        graded_idx = np.asarray(
            getattr(net, "graded_idx", np.empty(0, np.int64)), dtype=np.int64)
        self.graded_positions = np.flatnonzero(
            self.incoming[graded_idx]).astype(np.int64)
        self.graded_source_rows = graded_idx[self.graded_positions]

        shape = (self.n_steps, len(self.target_rows), len(self.route_names))
        self.spike_exc = np.zeros(shape, dtype=np.float64)
        self.spike_inh = np.zeros(shape, dtype=np.float64)
        self.graded_exc = np.zeros(shape, dtype=np.float64)
        self.graded_inh = np.zeros(shape, dtype=np.float64)
        self.graded_release = np.full(
            (self.n_steps, len(self.graded_positions)), np.nan,
            dtype=np.float32)
        self.state = np.full(
            (self.n_steps, len(self.target_rows), 4), np.nan, dtype=np.float32)
        self.state_columns = None
        self.reltap_columns = None
        self.eng = None
        self.spike_source_events = 0
        self.graded_source_updates = 0
        self.arrivals_after_run = 0

    def bind(self, eng, state_columns, reltap_columns):
        if not eng.split:
            raise ValueError("record_input requires the Metal split-conductance path")
        if eng.n_coin:
            raise NotImplementedError(
                "record_input cannot route-separate coincidence conductance")
        if (eng.tonic_arr is not None
                and np.any(eng.tonic_arr[self.target_rows] != 0.0)):
            raise NotImplementedError(
                "record_input cannot assign tonic target conductance to a source route")
        self.state_columns = np.ascontiguousarray(state_columns, dtype=np.int64)
        self.reltap_columns = np.ascontiguousarray(reltap_columns, dtype=np.int64)
        if self.state_columns.shape != self.target_rows.shape:
            raise ValueError("record_input target tap mapping changed shape")
        if self.reltap_columns.shape != self.graded_positions.shape:
            raise ValueError("record_input graded tap mapping changed shape")
        self.eng = eng

    @staticmethod
    def _add_by_sign(exc, inh, ticks, routes, weights, target):
        for route in np.unique(routes):
            selected = routes == route
            positive = selected & (weights > 0.0)
            negative = selected & (weights < 0.0)
            if positive.any():
                np.add.at(exc[:, target, int(route)], ticks[positive],
                          weights[positive].astype(np.float64))
            if negative.any():
                np.add.at(inh[:, target, int(route)], ticks[negative],
                          (-weights[negative]).astype(np.float64))

    def record_spikes(self, absolute_ticks, ids):
        ids = np.asarray(ids, dtype=np.int64)
        if not len(ids):
            return
        ticks = np.asarray(absolute_ticks, dtype=np.int64)
        selected = self.incoming[ids]
        if not selected.any():
            return
        sources = ids[selected]
        arrivals = ticks[selected] + self.eng.delay_slot[sources].astype(np.int64)
        inside = arrivals < self.n_steps
        self.arrivals_after_run += int((~inside).sum())
        sources = sources[inside]
        arrivals = arrivals[inside]
        if not len(sources):
            return
        routes = self.route_by_source[sources]
        self.spike_source_events += int(len(sources))
        for target in range(len(self.target_rows)):
            weights = self.weights[sources, target]
            nonzero = weights != 0.0
            if nonzero.any():
                self._add_by_sign(
                    self.spike_exc, self.spike_inh, arrivals[nonzero],
                    routes[nonzero], weights[nonzero], target)

    def record_graded(self, window_start, releases):
        if not len(self.graded_positions):
            return
        releases = np.asarray(releases, dtype=np.float32)
        if releases.ndim != 2 or releases.shape[1] != len(self.graded_positions):
            raise ValueError("record_input graded release tap changed shape")
        ticks = window_start + np.arange(len(releases), dtype=np.int64)
        live = (ticks % max(1, int(self.eng.net.p.graded_every))) == 0
        ticks = ticks[live]
        releases = releases[live]
        if not len(ticks):
            return
        self.graded_release[ticks] = releases
        for position, source in enumerate(self.graded_source_rows):
            arrivals = ticks + int(self.eng.delay_slot[source])
            inside = arrivals < self.n_steps
            self.arrivals_after_run += int((~inside).sum())
            if not inside.any():
                continue
            at = arrivals[inside]
            release = releases[inside, position]
            scale = np.float32(
                release * self.eng.graded_gain_scaled[
                    self.graded_positions[position]])
            route = int(self.route_by_source[source])
            self.graded_source_updates += int(len(at))
            for target in range(len(self.target_rows)):
                weight = self.weights[source, target]
                if weight == 0.0:
                    continue
                contribution = np.float32(scale * weight).astype(np.float64)
                if weight > 0.0:
                    np.add.at(self.graded_exc[:, target, route], at, contribution)
                else:
                    np.add.at(self.graded_inh[:, target, route], at, -contribution)

    def record_state(self, window_start, state_tap):
        state_tap = np.asarray(state_tap, dtype=np.float32)
        selected = state_tap[:, self.state_columns, :]
        stop = window_start + len(selected)
        self.state[window_start:stop] = selected

    def finish(self):
        if self.eng is None or np.isnan(self.state).any():
            raise RuntimeError("record_input did not retain every native-tick state")
        spike_exc = self.spike_exc.astype(np.float32)
        spike_inh = self.spike_inh.astype(np.float32)
        graded_exc = self.graded_exc.astype(np.float32)
        graded_inh = self.graded_inh.astype(np.float32)
        increment_exc = np.float32(spike_exc + graded_exc)
        increment_inh = np.float32(spike_inh + graded_inh)

        route_exc = np.zeros_like(increment_exc)
        route_inh = np.zeros_like(increment_inh)
        previous_exc = np.zeros(increment_exc.shape[1:], dtype=np.float32)
        previous_inh = np.zeros(increment_inh.shape[1:], dtype=np.float32)
        decay = self.eng.decay_g[self.target_rows, None]
        for tick in range(self.n_steps):
            previous_exc = np.float32(previous_exc * decay)
            previous_inh = np.float32(previous_inh * decay)
            previous_exc = np.float32(previous_exc + increment_exc[tick])
            previous_inh = np.float32(previous_inh + increment_inh[tick])
            route_exc[tick] = previous_exc
            route_inh[tick] = previous_inh

        voltage_after = self.state[..., 0]
        voltage_before = self.state[..., 1]
        engine_exc_raw = self.state[..., 2]
        engine_inh_raw = self.state[..., 3]
        gscale = np.float32(self.eng.gscale)
        engine_exc = np.float32(np.maximum(engine_exc_raw, 0.0) * gscale)
        engine_inh = np.float32(engine_inh_raw * gscale)
        route_exc_g = np.float32(route_exc * gscale)
        route_inh_g = np.float32(route_inh * gscale)
        exc_drive = np.float32(self.eng.e_exc - voltage_before)
        inh_drive = np.float32(self.eng.e_inh - voltage_before)
        route_current_exc = np.float32(route_exc_g * exc_drive[..., None])
        route_current_inh = np.float32(route_inh_g * inh_drive[..., None])
        engine_current_exc = np.float32(engine_exc * exc_drive)
        engine_current_inh = np.float32(engine_inh * inh_drive)
        engine_current = np.float32(engine_current_exc + engine_current_inh)
        route_current = np.float32(route_current_exc + route_current_inh)

        reconstructed_exc = route_exc.sum(axis=2, dtype=np.float32)
        reconstructed_inh = route_inh.sum(axis=2, dtype=np.float32)
        exc_error = np.abs(reconstructed_exc - engine_exc_raw)
        inh_error = np.abs(reconstructed_inh - engine_inh_raw)
        start, stop = self.window_ticks
        sl = slice(start, stop)
        return {
            "tick_ms": self.dt,
            "window_ms": list(self.window_ms),
            "target_rows": self.target_rows.copy(),
            "route_names": list(self.route_names),
            "incoming_source_rows": self.incoming_source_rows.copy(),
            "incoming_source_routes": self.route_by_source[
                self.incoming_source_rows].copy(),
            "incoming_source_delay_ticks": self.eng.delay_slot[
                self.incoming_source_rows].copy(),
            "incoming_source_is_graded": np.isin(
                self.incoming_source_rows,
                self.graded_source_rows).astype(np.uint8),
            "incoming_weights": self.weights[
                self.incoming_source_rows].copy(),
            "graded_source_rows": self.graded_source_rows.copy(),
            "spike_source_events": int(self.spike_source_events),
            "graded_source_updates": int(self.graded_source_updates),
            "arrivals_after_run": int(self.arrivals_after_run),
            "conductance_scale": float(gscale),
            "e_exc_mv": float(self.eng.e_exc),
            "e_inh_mv": float(self.eng.e_inh),
            "current_units": (
                "leak-normalized mV-equivalent; sign is the engine's "
                "physical reversal-potential convention, but absolute "
                "amperes are unavailable because the model has no membrane "
                "capacitance or leak conductance"),
            "graded_route_split_convention": (
                "each source is assigned before the engine's signed graded "
                "sum; this run has one graded source into the recorded targets"),
            "max_abs_raw_exc_reconstruction_error": float(exc_error.max()),
            "max_abs_raw_inh_reconstruction_error": float(inh_error.max()),
            "spike_exc_increment": np.float32(spike_exc[sl] * gscale),
            "spike_inh_increment": np.float32(spike_inh[sl] * gscale),
            "graded_exc_increment": np.float32(graded_exc[sl] * gscale),
            "graded_inh_increment": np.float32(graded_inh[sl] * gscale),
            "graded_release": self.graded_release[sl].copy(),
            "exc_increment": np.float32(increment_exc[sl] * gscale),
            "inh_increment": np.float32(increment_inh[sl] * gscale),
            "route_exc_conductance": route_exc_g[sl],
            "route_inh_conductance": route_inh_g[sl],
            "route_current_exc": route_current_exc[sl],
            "route_current_inh": route_current_inh[sl],
            "route_current_total": route_current[sl],
            "voltage_before_mv": voltage_before[sl],
            "voltage_after_mv": voltage_after[sl],
            "engine_exc_conductance": engine_exc[sl],
            "engine_inh_conductance": engine_inh[sl],
            "engine_current_exc_mv": engine_current_exc[sl],
            "engine_current_inh_mv": engine_current_inh[sl],
            "engine_current_total_mv": engine_current[sl],
        }


_RUN_LOG = Path.home() / "Library/Logs/fly-wbe/metal-runs.jsonl"
_ENGINE_SHA = None   # the running engine's compiled-in source hash, for the receipt


def _log_run(net, args, kwargs, out, lock, t0, load1, ru0):
    """One JSON line per Metal run, best effort, to _RUN_LOG: the fleet's
    compute-versus-wait accounting (Robin, 2026-09-05: absolute compute time
    against inference time against GPU wait time is the number that decides
    when to buy cloud compute). Never raises: a receipt must not fail a run."""
    try:
        import datetime
        import resource
        ru1 = resource.getrusage(resource.RUSAGE_SELF)
        dur = args[0] if args else kwargs.get("duration_ms")
        seed = args[2] if len(args) > 2 else kwargs.get("seed")
        cwd = os.getcwd()
        row = {
            "ts": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "pid": os.getpid(),
            "lane": os.path.basename(cwd),
            "cwd": cwd,
            "seed": seed,
            "duration_ms": dur,
            "n": int(getattr(net, "N", 0)),
            "wall_s": round(time.monotonic() - t0, 3),
            "gpu_s": round(float(out.get("gpu_seconds", 0.0)), 3),
            "cpu_s": round((ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime), 3),
            "lock_wait_s": round(float(lock.get("wait_s", 0.0)), 3),
            "lock_mode": lock.get("mode"),
            "load1": round(load1, 1),
            "n_spikes": int(out.get("n_spikes", 0)),
            # WHICH engine produced this run. A receipt that cannot say is
            # useless across an engine change, and the 2026-09-07 re-freeze
            # is exactly that: pre- and post-merge runs of one seed differ.
            "engine_sha16": _ENGINE_SHA,
        }
        _RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _RUN_LOG.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
    except Exception:
        pass


def run_metal(net, *args, **kwargs):
    """CNS.run() on Metal, under the process-wide shared/exclusive GPU gate,
    with one receipt line per run appended to ~/Library/Logs/fly-wbe/metal-runs.jsonl."""
    import resource
    t0 = time.monotonic()
    load1 = os.getloadavg()[0]
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    with _metal_run_lock() as lock:
        out = _run_metal_unlocked(net, *args, **kwargs)
    _log_run(net, args, kwargs, out, lock, t0, load1, ru0)
    return out


def _run_metal_unlocked(net, duration_ms, drive=None, seed=0, record=None,
                        record_trace_every=None, record_voltage=None,
                        drive_window=None, drive_fn=None, feedback_rows=None,
                        drive_every_ms=1.0, force_spikes=None,
                        first_spike=False, force_spikes_fn=None,
                        feedback_graded_release_rows=None,
                        graded_drive_max_rate_hz=None,
                        graded_drive_signed=False, membrane_drive_rows=None,
                        membrane_drive_mv=0.0,
                        membrane_drive_fraction=None, record_input=None):
    """CNS.run() on the Metal engine. Same arguments, same return shape."""
    p = net.p
    N = net.N
    if (getattr(p, "tau_syn_exc", None) is not None
            or getattr(p, "tau_syn_inh", None) is not None):
        raise NotImplementedError(
            "Metal does not yet implement split excitatory/inhibitory "
            "synaptic decay; use the NumPy engine")
    requested_vrec = (None if record_voltage is None
                      else np.asarray(record_voltage, dtype=np.int64))
    if force_spikes_fn is not None:
        raise NotImplementedError("metal backend: force_spikes_fn")
    graded = net.graded
    graded_idx = getattr(net, "graded_idx", np.empty(0, np.int64))
    fb_graded_rows = (None if feedback_graded_release_rows is None
                      else np.asarray(feedback_graded_release_rows, dtype=np.int64))
    if graded_drive_max_rate_hz is not None:
        if (type(graded_drive_max_rate_hz) not in (int, float)
                or not np.isfinite(graded_drive_max_rate_hz)
                or float(graded_drive_max_rate_hz) <= 0.0):
            raise ValueError("graded_drive_max_rate_hz must be finite and positive")
        graded_drive_max_rate_hz = float(graded_drive_max_rate_hz)
    if type(graded_drive_signed) is not bool:
        raise ValueError("graded_drive_signed must be Boolean")
    if graded_drive_signed and graded_drive_max_rate_hz is None:
        raise ValueError(
            "graded_drive_signed requires graded_drive_max_rate_hz")
    rng = np.random.default_rng(seed)
    dt = float(p.dt)
    n_steps = int(round(duration_ms / dt))
    input_recorder = (None if record_input is None
                      else _InputRecorder(net, record_input, n_steps, dt))
    if input_recorder is None:
        engine_vrec = requested_vrec
    elif requested_vrec is None:
        engine_vrec = np.unique(input_recorder.target_rows).astype(np.int64)
    else:
        engine_vrec = np.unique(np.concatenate(
            [requested_vrec, input_recorder.target_rows])).astype(np.int64)
    fb_every = max(1, int(round(drive_every_ms / dt)))
    window = fb_every if drive_fn is not None else min(n_steps, 256)
    # every run starts from rest, like run(); the engine holds state, so a
    # fresh one is built per run (graph upload is ~0.1 s against the run)
    # Every engine this process still holds, not only the one on this net:
    # walk_search's net_cache keeps one CNS per config and each keeps its last
    # engine, so on a multi-arm sweep the others are pure GPU memory. Closing
    # them here rather than in __del__ is what bounds the process, because the
    # cache holds a strong reference to every net it has built.
    for _other in list(_LIVE_ENGINES):
        _other.close()
    fb_rows = (None if feedback_rows is None
               else np.asarray(feedback_rows, dtype=np.int64))
    fb_graded_callback_pos = np.empty(0, dtype=np.int64)
    fb_graded_release_pos = np.empty(0, dtype=np.int64)
    if fb_graded_rows is not None:
        if fb_rows is None:
            raise ValueError("feedback_graded_release_rows requires feedback_rows")
        if (fb_graded_rows.ndim != 1
                or len(np.unique(fb_graded_rows)) != len(fb_graded_rows)
                or np.any(fb_graded_rows < 0) or np.any(fb_graded_rows >= N)):
            raise ValueError("feedback_graded_release_rows must be unique valid rows")
        row_to_callback = {int(row): pos for pos, row in enumerate(fb_rows)}
        graded_to_release = {int(row): pos for pos, row in enumerate(graded_idx)}
        if any(int(row) not in row_to_callback for row in fb_graded_rows):
            raise ValueError("graded feedback rows must be present in feedback_rows")
        if any(int(row) not in graded_to_release for row in fb_graded_rows):
            raise ValueError("graded feedback rows must designate graded neurons")
        fb_graded_callback_pos = np.asarray(
            [row_to_callback[int(row)] for row in fb_graded_rows], dtype=np.int64)
        fb_graded_release_pos = np.asarray(
            [graded_to_release[int(row)] for row in fb_graded_rows], dtype=np.int64)
    recorder_release_pos = (np.empty(0, dtype=np.int64)
                            if input_recorder is None
                            else input_recorder.graded_positions)
    engine_reltap_pos = np.unique(np.concatenate(
        [fb_graded_release_pos, recorder_release_pos])).astype(np.int64)
    eng = MetalEngine(
        net, window_ticks=max(window, 1), vrec=engine_vrec,
        reltap_pos=(engine_reltap_pos if len(engine_reltap_pos) else None))
    net._metal_engine = eng
    feedback_reltap_columns = (
        np.searchsorted(engine_reltap_pos, fb_graded_release_pos)
        if len(fb_graded_release_pos) else np.empty(0, dtype=np.int64))
    if input_recorder is not None:
        state_columns = np.searchsorted(engine_vrec, input_recorder.target_rows)
        if not np.array_equal(
                engine_vrec[state_columns], input_recorder.target_rows):
            raise RuntimeError("record_input target rows are absent from state tap")
        recorder_reltap_columns = (
            np.searchsorted(engine_reltap_pos, recorder_release_pos)
            if len(recorder_release_pos) else np.empty(0, dtype=np.int64))
        input_recorder.bind(eng, state_columns, recorder_reltap_columns)
    g_every = max(1, int(p.graded_every))

    use_reg = float(getattr(p, "spike_reg", 1.0)) > 1.0
    if use_reg:
        _kreg = int(round(float(p.spike_reg)))
        reg_acc = None
        reg_thr = None
    force_ms = ({round(float(k), 3): np.asarray(v, dtype=np.int64)
                 for k, v in force_spikes.items()} if force_spikes else None)
    t_first = np.full(N, -1, dtype=np.int32) if first_spike else None

    drive_idx = np.empty(0, dtype=np.int64)
    drive_rate = np.empty(0, dtype=np.float64)
    drive_prob = np.empty(0, dtype=np.float64)
    if drive:
        di, dp = [], []
        for k, rate in drive.items():
            k = np.asarray(k, dtype=np.int64)
            di.append(k)
            dp.append(np.full(len(k), rate * dt / 1000.0))
        drive_idx = np.concatenate(di)
        drive_prob = np.concatenate(dp)
        drive_rate = drive_prob * (1000.0 / dt)
    if drive_window is None:
        step_on, step_off = 0, n_steps
    else:
        step_on = int(round(drive_window[0] / dt))
        step_off = int(round(drive_window[1] / dt))
    fb_accum = (None if fb_rows is None else np.zeros(
        len(fb_rows), dtype=(np.float64 if fb_graded_rows is not None
                             else np.int64)))

    rec = np.arange(N) if record is None else np.asarray(record)
    trace = None
    if record_trace_every:
        n_bins = int(np.ceil(duration_ms / record_trace_every))
        trace = np.zeros((n_bins, len(rec)), dtype=np.int32)
        steps_per_bin = int(round(record_trace_every / dt))
    counts = np.zeros(N, dtype=np.int64)
    total_spikes = 0
    # row -> callback position, -1 elsewhere: lets the per-window tally touch
    # only the spikes instead of allocating an N-length bincount every window
    fb_map = None
    if fb_rows is not None:
        fb_map = np.full(N, -1, dtype=np.int64)
        fb_map[fb_rows] = np.arange(len(fb_rows))
    rec_map = None
    if trace is not None:
        rec_map = np.full(N, -1, dtype=np.int64)
        rec_map[rec] = np.searchsorted(rec, rec)
    vtrace = (None if requested_vrec is None
              else np.full((n_steps, len(requested_vrec)), np.nan,
                           dtype=np.float32))
    voltage_columns = (
        np.empty(0, dtype=np.int64) if requested_vrec is None
        else np.arange(len(requested_vrec), dtype=np.int64)
        if input_recorder is None
        else np.searchsorted(engine_vrec, requested_vrec))

    words = eng.words
    cand = np.zeros((window, words), dtype=np.uint32)
    force = np.zeros((window, words), dtype=np.uint32) if force_ms else None

    def _set_bits(bits, ks, ids):
        ids = np.asarray(ids, dtype=np.int64)
        np.bitwise_or.at(bits, (np.asarray(ks, dtype=np.int64), ids >> 5),
                         (np.uint32(1) << (ids & 31).astype(np.uint32)))

    membrane_drive_receipt = None
    membrane_drive_mask = None
    membrane_drive_events = 0
    if membrane_drive_rows is not None and len(membrane_drive_rows):
        membrane_drive_rows = np.asarray(
            membrane_drive_rows, dtype=np.int64)
        if (membrane_drive_rows.ndim != 1
                or len(np.unique(membrane_drive_rows))
                != len(membrane_drive_rows)
                or membrane_drive_rows.min() < 0
                or membrane_drive_rows.max() >= N):
            raise IndexError(
                "membrane_drive_rows must be unique valid rows")
        threshold = np.asarray(eng.spike_at, dtype=np.float32)
        rest = np.asarray(eng.v_rest, dtype=np.float32)
        margin = (threshold[membrane_drive_rows]
                  - rest[membrane_drive_rows]).astype(np.float32)
        if membrane_drive_fraction is not None:
            if (not np.all(np.isfinite(margin))
                    or np.any(margin <= 0.0)):
                raise ValueError(
                    "cell-relative membrane-driven rows need finite positive "
                    "v_threshold - effective_v_rest margins")
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
            drive_mode = "threshold_margin_fraction"
            requested = fraction
        else:
            if (np.asarray(membrane_drive_mv).ndim != 0
                    or not np.isfinite(membrane_drive_mv)
                    or membrane_drive_mv <= 0.0):
                raise ValueError(
                    "membrane_drive_mv must be a finite scalar > 0")
            selected_kicks = np.full(
                len(membrane_drive_rows), np.float32(membrane_drive_mv),
                dtype=np.float32)
            drive_mode = "absolute_mv"
            requested = float(np.float32(membrane_drive_mv))
        membership = np.zeros(words, dtype=np.uint32)
        _set_bits(
            membership[None, :],
            np.zeros(len(membrane_drive_rows), dtype=np.int64),
            membrane_drive_rows)
        kick_vector = np.zeros(N, dtype=np.float32)
        kick_vector[membrane_drive_rows] = selected_kicks
        membrane_drive_mask = np.zeros(N, dtype=bool)
        membrane_drive_mask[membrane_drive_rows] = True
        eng.set_membrane_drive(membership, kick_vector)
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
            "backend": "metal",
        }

    gd_mask = None
    if graded_drive_max_rate_hz is not None and len(drive_idx) and drive_fn is None:
        raise NotImplementedError(
            "metal backend: graded_drive_max_rate_hz without drive_fn")
    for w0 in range(0, n_steps, window):
        nt = min(window, n_steps - w0)
        cand[:nt] = 0
        cand_k, cand_id = [], []
        any_force = False
        if force is not None:
            force[:nt] = 0
        if drive_fn is not None and w0 % fb_every == 0:
            # windows are fb_every ticks, so the callback lands on tick w0
            # and the rates hold for the whole window, as in run()
            idx_new, rate_new = drive_fn(w0 * dt, fb_accum)
            drive_idx = np.asarray(idx_new, dtype=np.int64)
            drive_rate = np.asarray(rate_new, dtype=np.float64)
            if drive_idx.shape != drive_rate.shape:
                raise ValueError(
                    "drive_fn indices and rates must have matching shapes")
            drive_prob = drive_rate * dt / 1000.0
            if not np.all(np.isfinite(drive_prob)):
                bad = drive_idx[~np.isfinite(drive_prob)]
                raise FloatingPointError(
                    f"non-finite afferent rate for {len(bad)} cells "
                    f"(first ids: {bad[:5].tolist()}) at t={w0 * dt} ms")
            if fb_accum is not None:
                fb_accum[:] = 0
            gd_mask = None
            if graded_drive_max_rate_hz is not None and len(drive_idx):
                gd_mask = graded[drive_idx]
                if (graded_drive_signed
                        and np.any(drive_rate[~gd_mask] < 0.0)):
                    raise ValueError(
                        "negative signed drive requires graded neurons")
                if not gd_mask.any():
                    gd_mask = None
            if gd_mask is not None:
                # graded drive: v of each driven graded cell is SET every
                # tick to rest + level*(upper - rest), after threshold
                # detection and before release, exactly as run() does. Signed
                # mode mirrors that existing cell-relative span below rest;
                # it adds no threshold or gain. The window holds one command
                # vector, so one clamp per window.
                driven = drive_idx[gd_mask]
                level = np.clip(
                    drive_rate[gd_mask] / graded_drive_max_rate_hz,
                    -1.0 if graded_drive_signed else 0.0, 1.0)
                vals = np.zeros(N, dtype=np.float32)
                vals[driven] = (eng.v_rest[driven]
                                + level * (eng.spike_at[driven] - eng.v_rest[driven]))
                cb = np.zeros(words, dtype=np.uint32)
                _set_bits(cb[None, :], np.zeros(len(driven), dtype=np.int64), driven)
                eng.set_clamp(cb, vals)
            else:
                eng.set_clamp(None, None)
            # A graded cell in the drive set without graded_drive_max_rate_hz
            # was refused until 2026-09-06 because release ordering differed:
            # cns.py computes graded release from v BEFORE resetting the cells
            # that fired this tick (its lines 2825 and 2890), while k_graded
            # read the post-reset v. k_step now keeps the pre-reset membrane
            # in v_pre whenever there are graded rows and k_graded reads that,
            # so a graded row that a drive candidate fired releases at the
            # voltage that crossed, as the reference does. Gated by
            # gpu/gate_v0.py --drive-graded.
        if force_ms is not None:
            for k in range(nt):
                f = force_ms.get(round((w0 + k) * dt, 3))
                if f is not None and len(f):
                    _set_bits(force, np.full(len(f), k), f)
                    any_force = True
        if len(drive_idx):
            k0, k1 = max(0, step_on - w0), min(nt, step_off - w0)
            if k1 > k0:
                # A clamped graded row cannot spike. Remove it before random
                # draws so changing only its analog command cannot perturb
                # the renewal stream of the real spiking afferents.
                spiking_mask = None if gd_mask is None else ~gd_mask
                spiking_idx = (drive_idx if spiking_mask is None
                               else drive_idx[spiking_mask])
                spiking_prob = (drive_prob if spiking_mask is None
                                else drive_prob[spiking_mask])
                n_e = len(spiking_idx)
                if not use_reg:
                    # one draw for the window: Generator.random fills a
                    # (ticks, entries) block in the order the per-tick
                    # calls would consume it, so the stream is run()'s
                    u = rng.random((k1 - k0, n_e))
                    crossed = u < spiking_prob[None, :]
                    kk, ee = np.nonzero(crossed)
                    if len(kk):
                        cand_k.append(kk + k0)
                        cand_id.append(spiking_idx[ee])
                else:
                    if reg_acc is None or len(reg_acc) != n_e:
                        reg_acc = np.zeros(n_e)
                        reg_thr = rng.gamma(_kreg, 1.0 / _kreg, n_e)
                    for k in range(k0, k1):
                        reg_acc += spiking_prob
                        crossed = reg_acc >= reg_thr
                        if crossed.any():
                            reg_acc[crossed] = 0.0
                            reg_thr[crossed] = rng.gamma(
                                _kreg, 1.0 / _kreg, int(crossed.sum()))
                            hit = spiking_idx[crossed]
                            if len(hit):
                                cand_k.append(np.full(len(hit), k))
                                cand_id.append(hit)
        any_cand = bool(cand_k)
        if any_cand:
            cand_k_all = np.concatenate(cand_k)
            cand_id_all = np.concatenate(cand_id)
            _set_bits(cand, cand_k_all, cand_id_all)
            if membrane_drive_mask is not None:
                membrane_pairs = np.column_stack((
                    cand_k_all[membrane_drive_mask[cand_id_all]],
                    cand_id_all[membrane_drive_mask[cand_id_all]],
                ))
                if len(membrane_pairs):
                    # cand_bits collapses repeated entries for the same row
                    # and tick. This counts draws presented to the membrane
                    # route; a reset or subsequent refractory clamp can still
                    # erase their voltage effect.
                    membrane_drive_events += int(
                        len(np.unique(membrane_pairs, axis=0)))
        ticks, ids = eng.run(w0, nt, cand if any_cand else None,
                             force if any_force else None)
        state_tap = (None if engine_vrec is None
                     else eng.state_tap(nt))
        if vtrace is not None:
            vtrace[w0:w0 + nt] = state_tap[:, voltage_columns, 0]
        if input_recorder is not None:
            input_recorder.record_state(w0, state_tap)
        release_tap = (eng.reltap(nt) if len(engine_reltap_pos) else None)
        if fb_graded_rows is not None and len(fb_graded_release_pos):
            gk = [k for k in range(nt) if (w0 + k) % g_every == 0]
            if gk:
                # mean release over the callback window, scaled as run() does
                fb_accum[fb_graded_callback_pos] += (
                    release_tap[gk][:, feedback_reltap_columns].sum(axis=0)
                    * (float(g_every) / float(fb_every)))
        if input_recorder is not None and len(recorder_release_pos):
            input_recorder.record_graded(
                w0, release_tap[:, input_recorder.reltap_columns])
        if input_recorder is not None:
            input_recorder.record_spikes(w0 + ticks.astype(np.int64), ids)
        if len(ids):
            np.add.at(counts, ids, 1)
            total_spikes += int(len(ids))
            if fb_accum is not None:
                fp = fb_map[ids]
                hit = fp >= 0
                if hit.any():
                    np.add.at(fb_accum, fp[hit], 1)
            if t_first is not None:
                u, first = np.unique(ids, return_index=True)
                new = t_first[u] < 0
                t_first[u[new]] = w0 + ticks[first[new]]
            if trace is not None:
                pos = rec_map[ids]
                sel = pos >= 0
                if sel.any():
                    b = (w0 + ticks[sel]) // steps_per_bin
                    ok = b < len(trace)
                    np.add.at(trace, (b[ok], pos[sel][ok]), 1)

    out = {
        "counts": counts[rec],
        **({"t_first_ms": np.where(t_first >= 0, t_first * dt, np.nan)}
           if t_first is not None else {}),
        "rates_hz": counts[rec] / (duration_ms / 1000.0),
        "n_spikes": total_spikes,
        "duration_ms": duration_ms,
        "recorded": rec,
        "backend": "metal",
        "gpu_seconds": eng.gpu_seconds(),
        "fold_overflows": eng.fold_overflows(),
        "bin_max": eng.bin_max(),
        "touched_per_tick": eng.touched_per_tick(),
        "bin_hist": eng.bin_hist(),
    }
    if membrane_drive_receipt is not None:
        membrane_drive_receipt["drawn_unique_row_tick_events"] = int(
            membrane_drive_events)
        out["membrane_drive"] = membrane_drive_receipt
    if eng.n_coin:
        # the seam's own receipt, cns.run()'s four scalars from the kernel's
        # per-target accumulators (max exact; the sum to float rounding)
        _cs = eng.coin_stats()
        out["coincidence"] = {
            **eng.coin_receipt,
            "applied_abs_max": float(_cs[0]),
            "applied_abs_sum": float(_cs[1]),
            "fs_max": float(_cs[2]),
            "slow_delayed_max": float(_cs[3]),
        }
    if trace is not None:
        out["trace"] = trace
    if vtrace is not None:
        out["voltage"] = vtrace
        out["voltage_idx"] = requested_vrec
    if input_recorder is not None:
        out["input_recording"] = input_recorder.finish()
    return out
