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
record_voltage (voltage after the refractory clamp, before thresholding), and
absolute or cell-relative command membrane drive.

Refused loudly (NotImplementedError), never silently approximated: gap
junctions, short-term depression, exponential integrate-and-fire,
graded_phasic and force_spikes_fn. Add them to the kernels with their own fingerprint before
lifting a refusal.

Parity: the per-neuron arithmetic is the numpy engine's, in float32 with
fast math off, so the first ticks agree exactly; the synaptic sums are
accumulated by atomics in an order that differs from np.bincount, so the
two rasters separate after the first neuron within one ulp of threshold
tips, and agreement after that is statistical. gpu/gate_v0.py measures
both.
"""
from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

_LIB = None
_LIB_PATH = Path(__file__).resolve().parent / "gpu" / "libcnsengine.dylib"
# The two files the dylib is built from, in the Makefile's hashing order.
_SRC_PATHS = [_LIB_PATH.parent / "cns_engine.m", _LIB_PATH.parent / "cns_engine.h"]
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
            yield mode
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
    ]


def _lib():
    global _LIB
    if _LIB is None:
        if not _LIB_PATH.exists():
            raise RuntimeError(
                f"{_LIB_PATH} is missing: build it with "
                f"`make -C {_LIB_PATH.parent} libcnsengine.dylib`")
        L = ctypes.CDLL(str(_LIB_PATH))
        _check_build_stamp(L)
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
        L.cns_engine_uniform_mask.restype = ctypes.c_uint32
        L.cns_engine_uniform_mask.argtypes = [ctypes.c_void_p]
        L.cns_engine_profile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
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
        W = net.W
        if W.format != "csc":
            raise RuntimeError("net.W must be CSC on the presynaptic axis")
        indptr = np.ascontiguousarray(W.indptr, dtype=np.int64)
        indices = np.ascontiguousarray(W.indices, dtype=np.int32)
        weight = np.ascontiguousarray(W.data, dtype=np.float32)
        self.n_ring = n_ring
        self.split = bool(p.conductance_based and p.split_conductances)
        self.use_adapt = use_adapt
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

    def vtap(self, n_ticks):
        """The window's tapped voltages, (n_ticks, n_vrec) float32 copy."""
        nv = len(self.vrec)
        buf = np.ctypeslib.as_array(self._L.cns_engine_vtap(self.h),
                                    shape=(self.window_ticks, nv))
        return buf[:n_ticks].copy()

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
        secs = np.zeros(4); cnt = np.zeros(4, dtype=np.uint64)
        self._L.cns_engine_profile(self.h, _ptr(secs), _ptr(cnt))
        return dict(zip(("step", "scatter", "graded", "flush"), secs)), dict(zip(("step", "scatter", "graded", "flush"), cnt))

    def gpu_seconds(self):
        return float(self._L.cns_engine_gpu_seconds(self.h))

    def close(self):
        if getattr(self, "h", None):
            self._L.cns_engine_destroy(self.h)
            self.h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def run_metal(net, *args, **kwargs):
    """CNS.run() on Metal, under the process-wide shared/exclusive GPU gate."""
    with _metal_run_lock():
        return _run_metal_unlocked(net, *args, **kwargs)


def _run_metal_unlocked(net, duration_ms, drive=None, seed=0, record=None,
                        record_trace_every=None, record_voltage=None,
                        drive_window=None, drive_fn=None, feedback_rows=None,
                        drive_every_ms=1.0, force_spikes=None,
                        first_spike=False, force_spikes_fn=None,
                        feedback_graded_release_rows=None,
                        graded_drive_max_rate_hz=None,
                        graded_drive_signed=False, membrane_drive_rows=None,
                        membrane_drive_mv=0.0,
                        membrane_drive_fraction=None):
    """CNS.run() on the Metal engine. Same arguments, same return shape."""
    p = net.p
    N = net.N
    if (getattr(p, "tau_syn_exc", None) is not None
            or getattr(p, "tau_syn_inh", None) is not None):
        raise NotImplementedError(
            "Metal does not yet implement split excitatory/inhibitory "
            "synaptic decay; use the NumPy engine")
    vrec = None if record_voltage is None else np.asarray(record_voltage)
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
    fb_every = max(1, int(round(drive_every_ms / dt)))
    window = fb_every if drive_fn is not None else min(n_steps, 256)
    # every run starts from rest, like run(); the engine holds state, so a
    # fresh one is built per run (graph upload is ~0.1 s against the run)
    eng = getattr(net, "_metal_engine", None)
    if eng is not None:
        eng.close()
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
    eng = MetalEngine(net, window_ticks=max(window, 1), vrec=vrec,
                      reltap_pos=(fb_graded_release_pos
                                  if fb_graded_rows is not None else None))
    net._metal_engine = eng
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
    vtrace = (None if vrec is None
              else np.full((n_steps, len(vrec)), np.nan, dtype=np.float32))

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
            if len(drive_idx) and graded[drive_idx].any() and gd_mask is None:
                raise NotImplementedError(
                    "metal backend: a graded cell in the drive set without "
                    "graded_drive_max_rate_hz (release ordering differs)")
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
        if vtrace is not None:
            vtrace[w0:w0 + nt] = eng.vtap(nt)
        if fb_graded_rows is not None and len(fb_graded_release_pos):
            rt = eng.reltap(nt)
            gk = [k for k in range(nt) if (w0 + k) % g_every == 0]
            if gk:
                # mean release over the callback window, scaled as run() does
                fb_accum[fb_graded_callback_pos] += (
                    rt[gk].sum(axis=0) * (float(g_every) / float(fb_every)))
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
    }
    if membrane_drive_receipt is not None:
        membrane_drive_receipt["drawn_unique_row_tick_events"] = int(
            membrane_drive_events)
        out["membrane_drive"] = membrane_drive_receipt
    if trace is not None:
        out["trace"] = trace
    if vtrace is not None:
        out["voltage"] = vtrace
        out["voltage_idx"] = vrec
    return out
