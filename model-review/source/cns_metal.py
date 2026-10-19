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
feedback rows, force_spikes, force_spikes_fn, first_spike, record, record_trace_every and
record_voltage (voltage after the refractory clamp, before thresholding),
absolute or cell-relative command membrane drive, and the coincidence seam
(coincidence_gain with its target / fast / slow pools; ported 2026-09-05,
gate `gpu/gate_v0.py --coincidence`, per-target deterministic gather so the
sums run in the numpy engine's order).

Metal also carries electrical coupling with Euler integration: symmetric or
directed G, scalar or per-pair weights, delayed spikelets and the reference's
presynaptic AP waveform. The gate is gpu/gate_timing.py. Callback forcing uses
the same read-only feedback snapshot, row validation and tick cadence as NumPy.

Refused loudly (NotImplementedError), never silently approximated: analytic
integration with gaps, CUDA gap spikelets, AP waveforms, substeps and every gap_law but
one_way (gpu/GAP-JUNCTIONS.md), short-term depression, exponential integrate-and-fire,
and graded_phasic. Add them to the kernels with their own fingerprint before
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

import copy
import ctypes
import fcntl
import hashlib
import json
import os
import time
import weakref
from contextlib import contextmanager
from fractions import Fraction
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
# cns_engine_read_buffer's index and element type per device buffer (0c25fab3).
READ_BUFFERS = {
    "weight": (0, np.float32), "indptr": (1, np.int64),
    "indices": (2, np.int32), "delay_slot": (3, np.int32),
    "t_weight": (4, np.float32), "t_source": (5, np.int32),
    "t_indptr": (6, np.int64), "gt_weight": (7, np.float32),
    "gt_rows": (8, np.int32), "gt_indptr": (9, np.int64),
    "gt_position": (10, np.int32), "gt_slot": (11, np.int32),
    # design 7b0ad6b2: per-neuron arrays, live state and the PSI pair channel
    "decay_g": (12, np.float32), "refrac_count": (13, np.int16),
    "pair_src": (14, np.int32), "pair_tgt": (15, np.int32),
    "pair_delay": (16, np.uint32), "pair_decay": (17, np.float32),
    "pair_increment": (18, np.float32), "pair_state": (19, np.float32),
    "pair_ring": (20, np.uint32), "v_rest": (21, np.float32),
    "v_reset": (22, np.float32), "spike_at": (23, np.float32),
    "decay_v": (24, np.float32), "dt_over_tau": (25, np.float32),
    "refrac_ticks": (26, np.int16), "adapt_a": (27, np.float32),
    "adapt_b": (28, np.float32), "dt_over_tauw": (29, np.float32),
    "g_i": (30, np.float32), "ring": (31, np.float32),
    "ring_i": (32, np.float32), "v": (33, np.float32),
    "g": (34, np.float32), "w_adapt": (35, np.float32),
    "counts": (36, np.uint32),
    # design 02e57559, CUDA only: the resource, replacement, recorder and tap
    # selectors as the kernels read them; zero length when unconfigured
    "rr_pos": (37, np.int32), "rr_tmask": (38, np.uint8),
    "rr_d": (39, np.float32), "rr_c": (40, np.float32),
    "rr_keep": (41, np.float32), "replacement_bits": (42, np.uint32),
    "deliv_src": (43, np.uint8), "deliv_tgt": (44, np.uint8),
    "ptap_dsrc": (45, np.uint8), "rr_x": (46, np.float32),
    "ptap_gdir": (47, np.float32),
    # design 81427d47, CUDA only: the occupancy channel and its recorder as the
    # kernels read them; zero length when unconfigured
    "occ_src": (48, np.int32), "occ_tgt": (49, np.int32),
    "occ_delay": (50, np.uint32), "occ_quanta": (51, np.uint64),
    "occ_t_begin": (52, np.uint32), "occ_alpha": (53, np.float64),
    "occ_gmax": (54, np.float64), "occ_beta": (55, np.float64),
    "occ_dec": (56, np.float64), "occ_scale": (57, np.float64),
    "occ_state": (58, np.float64), "occ_ring": (59, np.uint32),
    "occ_a_count": (60, np.uint32), "occ_s_count": (61, np.uint32),
    "occ_a_tick": (62, np.int32), "occ_a_edge": (63, np.uint32),
    "occ_a_n": (64, np.uint32), "occ_s_tick": (65, np.int32),
    "occ_s_n": (66, np.uint32), "occ_s_val": (67, np.float64),
    "occ_s_g": (68, np.float32), "occ_tgt_of": (69, np.int32),
    "occ_src_first": (70, np.int32), "occ_src_next": (71, np.int32),
    # design e05c0eef B1: the pair recovery's clock, resource, counters
    # (delivered, skipped, scaled, multi-event ticks per pair) and area
    "pair_rec_clock": (72, np.uint32), "pair_rec_resource": (73, np.float32),
    "pair_rec_count": (74, np.uint32), "pair_rec_area": (75, np.float32),
}
# design 7b0ad6b2: the state tap's per-row fields, in cns_engine.cu's order
AUX_FIELDS = ("v_start", "v_tested", "arrival_exc", "arrival_inh", "g_exc_state",
              "g_inh_state", "pair_events", "pair_g", "ge_ordinary", "gi_used",
              "pair_drive", "exc_drive", "inh_drive", "w_adapt", "refrac", "flags")
# Design 02e57559: cns_engine_state_read's sections, in the engine's order.
STATE_SECTIONS = (
    "v", "g", "g_i", "refrac", "w_adapt", "counts", "ring", "ring_i", "v_pre",
    "graded_relcur", "graded_scale", "ovr", "bin_count", "args", "clamp_bits",
    "clamp_val", "rr_x", "rr_snapshot", "rr_count", "rr_tick", "rr_row",
    "rr_value", "dl_count", "dl_dir_tick", "dl_dir_edge", "dl_dir_w",
    "dl_ev_tick", "dl_ev_src", "dl_ev_pos", "dl_ev_neg", "dl_ev_nother",
    "dl_ev_ndir", "ptap_ring", "ptap_gdir")
# The tap's sixteen channels per native tick (gpu/cns_engine.cu, k_step).
PTAP_CHANNELS = (
    "v_in", "v_detect", "w_in", "w_out", "refrac_after", "g_exc_raw",
    "g_inh_raw", "g_exc_norm", "g_inh_norm", "i_inh", "i_exc", "g_direct_raw",
    "arrival_direct", "fired", "arrival_exc", "arrival_inh")
# Design 81427d47: the occupancy tap's twenty channels per target per native
# tick (gpu/cns_engine.cu, k_step), float64. State is o (mode 1) or h (mode 2);
# the conductances are normalized and the currents are g*(E - V) at the tick's
# starting membrane.
OCC_CHANNELS = (
    "v_in", "v_detect", "w_in", "refrac_in", "arrivals", "A", "state_before",
    "state_decayed", "state_after", "g_occ", "gmax_or_beta", "ge_before_occ",
    "ge_total", "gi_total", "i_occ", "i_exc", "i_inh", "arrival_exc_ordinary",
    "arrival_inh_ordinary", "flags")
_STATE_MAGIC = 0x3232353565373230


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
        ("n_gap", ctypes.c_uint32), ("gap_tgt", ctypes.c_void_p),
        ("gap_indptr", ctypes.c_void_p), ("gap_src", ctypes.c_void_p),
        ("gap_weight", ctypes.c_void_p), ("gap_deg", ctypes.c_void_p),
        ("gap_scale", ctypes.c_float), ("gap_spikelet", ctypes.c_float),
        ("gap_delay", ctypes.c_uint32), ("gap_n_ap", ctypes.c_uint32),
        ("gap_ap_off", ctypes.c_uint32), ("gap_ap_wave", ctypes.c_void_p),
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
        if _MAC:
            L.cns_engine_gap_profile.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            L.cns_engine_gap_profile.restype = None
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


_BOUND_02E57559 = False


def _lib_02e57559():
    """The CUDA library's design 02e57559 entry points, bound on first use."""
    global _BOUND_02E57559
    L = _lib()
    if _MAC or not hasattr(L, "cns_engine_state_save"):
        raise NotImplementedError(
            "design 02e57559's recorders and state slots need the CUDA library")
    if not _BOUND_02E57559:
        c = ctypes
        vp, u32 = c.c_void_p, c.c_uint32
        for name, res, args in (
                ("cns_engine_resource_targets", c.c_int, [vp, vp]),
                ("cns_engine_set_replacement", c.c_int, [vp, vp]),
                ("cns_engine_deliv_config", c.c_int, [vp, vp, vp, u32, u32]),
                ("cns_engine_deliv_counts", c.c_int, [vp, vp]),
                ("cns_engine_deliv_read", c.c_int,
                 [vp, vp, vp, vp, u32, vp, vp, vp, vp, vp, vp, u32]),
                ("cns_engine_ptap_config", c.c_int, [vp, u32, vp, vp]),
                ("cns_engine_ptap", c.POINTER(c.c_float), [vp]),
                ("cns_engine_state_sections", u32, [vp, vp, u32]),
                ("cns_engine_state_bytes", c.c_uint64, [vp]),
                ("cns_engine_state_save", c.c_int, [vp, u32]),
                ("cns_engine_state_load", c.c_int, [vp, u32]),
                ("cns_engine_state_read", c.c_int64, [vp, c.c_int32, vp, c.c_uint64])):
            f = getattr(L, name)
            f.restype, f.argtypes = res, args
        _BOUND_02E57559 = True
    return L


_BOUND_81427D47 = False


def _lib_81427d47():
    """The CUDA library's design 81427d47 entry points, bound on first use."""
    global _BOUND_81427D47
    L = _lib_02e57559()
    if not hasattr(L, "cns_engine_occ_config"):
        raise NotImplementedError("design 81427d47's occupancy channel needs the CUDA library")
    if not _BOUND_81427D47:
        c = ctypes
        vp, u32 = c.c_void_p, c.c_uint32
        for name, res, args in (
                ("cns_engine_occ_config", c.c_int,
                 [vp, u32, u32, u32, u32, vp, vp, vp, vp, u32, vp, vp, vp, vp, vp, vp]),
                ("cns_engine_occ_record_config", c.c_int, [vp, u32, vp, vp, u32]),
                ("cns_engine_occ_tap", c.POINTER(c.c_double), [vp]),
                ("cns_engine_occ_width", u32, [])):
            f = getattr(L, name)
            f.restype, f.argtypes = res, args
        if int(L.cns_engine_occ_width()) != len(OCC_CHANNELS):
            raise RuntimeError("occupancy tap width differs from OCC_CHANNELS")
        _BOUND_81427D47 = True
    return L


_BOUND_87A3AD35 = False


def _lib_87a3ad35():
    """The library's design 87a3ad35 entry points, bound on first use. The
    Metal library carries them since design 244fb801 Build 1, for the
    conductance on its pulse train under the Euler integrator."""
    global _BOUND_87A3AD35
    L = _lib()
    if not hasattr(L, "cns_engine_opsin_config"):
        raise NotImplementedError("design 87a3ad35's opsin channel: this library does not carry it")
    if not _BOUND_87A3AD35:
        c = ctypes
        vp, u32 = c.c_void_p, c.c_uint32
        for name, res, args in (
                ("cns_engine_opsin_config", c.c_int, [vp, u32] + [vp] * 10 + [u32]),
                ("cns_engine_opsin_train", c.c_int, [vp, vp]),
                ("cns_engine_opsin_tap", c.POINTER(c.c_float), [vp]),
                ("cns_engine_opsin_width", u32, [])):
            f = getattr(L, name)
            f.restype, f.argtypes = res, args
        from cns import OPSIN_TAP_FIELDS
        if int(L.cns_engine_opsin_width()) != len(OPSIN_TAP_FIELDS):
            raise RuntimeError("opsin tap width differs from cns.OPSIN_TAP_FIELDS")
        _BOUND_87A3AD35 = True
    return L


def _opsin_present(p):
    """Whether SimParams names an opsin or threshold-window row (design 87a3ad35)."""
    a = getattr(p, "opsin_rows", None)
    b = getattr(p, "threshold_window_rows", None)
    return (a is not None and len(a) > 0) or (b is not None and len(b) > 0)


_BOUND_2D12A3B3 = False


def _lib_2d12a3b3():
    """The CUDA library's design 2d12a3b3 entry point, bound on first use."""
    global _BOUND_2D12A3B3
    L = _lib()
    if _MAC or not hasattr(L, "cns_engine_row_current_config"):
        raise NotImplementedError("design 2d12a3b3's row_current needs the CUDA library")
    if not _BOUND_2D12A3B3:
        c = ctypes
        f = L.cns_engine_row_current_config
        f.restype = c.c_int
        f.argtypes = [c.c_void_p, c.c_uint32] + [c.c_void_p] * 5 + [c.c_uint32]
        _BOUND_2D12A3B3 = True
    return L


_BOUND_2163BA7D_B2 = False


def _lib_2163ba7d_b2():
    """The CUDA library's design 2163ba7d Build 2 entry points (the slow
    inhibitory pool), bound on first use."""
    global _BOUND_2163BA7D_B2
    L = _lib()
    if _MAC or not hasattr(L, "cns_engine_slow_config"):
        raise NotImplementedError(
            "design 2163ba7d's slow inhibitory pool (Build 2) needs the CUDA library")
    if not _BOUND_2163BA7D_B2:
        c = ctypes
        vp, u32, f32 = c.c_void_p, c.c_uint32, c.c_float
        for name, res, args in (
                ("cns_engine_slow_config", c.c_int,
                 [vp, u32, vp, vp, vp, u32, f32, f32, f32, u32, u32]),
                ("cns_engine_slow_read", c.c_int, [vp, u32, u32] + [vp] * 6)):
            f = getattr(L, name)
            f.restype, f.argtypes = res, args
        _BOUND_2163BA7D_B2 = True
    return L


_BOUND_EFE88957_B2 = False


def _lib_efe88957_b2():
    """Design efe88957 Build 2's entry points (the coincidence seam's list
    form), bound on first use. The CUDA library carries them."""
    global _BOUND_EFE88957_B2
    L = _lib()
    if _MAC or not hasattr(L, "cns_engine_coin2_config"):
        raise NotImplementedError(
            "coincidence_triples (design efe88957, Build 2) needs the CUDA library")
    if not _BOUND_EFE88957_B2:
        c = ctypes
        vp, u32, u64 = c.c_void_p, c.c_uint32, c.c_uint64
        for name, res, args in (
                ("cns_engine_coin2_config", c.c_int, [vp, u32, vp, vp, vp, u64, vp, u64, vp]),
                ("cns_engine_coin_target_stats", c.c_int, [vp, u32, vp])):
            f = getattr(L, name)
            f.restype, f.argtypes = res, args
        _BOUND_EFE88957_B2 = True
    return L


_BOUND_B1746728 = False


def _lib_b1746728():
    """Design b1746728 Build 1's entry points (the receptor-scoped serotonin
    state), bound on first use. The CUDA library carries them, and
    cns_engine.m since design fcb48834's Build 0, with the same arguments."""
    global _BOUND_B1746728
    L = _lib()
    if not hasattr(L, "cns_engine_serotonin_config"):
        raise NotImplementedError(
            "the receptor-scoped serotonin state (design b1746728, Build 1) needs "
            "a library built from it (CUDA, or cns_engine.m from design fcb48834's "
            "Build 0); refusing a silent zero-effect run")
    if not _BOUND_B1746728:
        c = ctypes
        vp, u32, f32 = c.c_void_p, c.c_uint32, c.c_float
        for name, res, args in (
                ("cns_engine_serotonin_config", c.c_int,
                 [vp, u32, vp, u32, vp, f32, f32, f32, f32, u32, u32]),
                ("cns_engine_serotonin_read", c.c_int, [vp] * 5 + [u32])):
            f = getattr(L, name)
            f.restype, f.argtypes = res, args
        _BOUND_B1746728 = True
    return L


def _serotonin_plan(p, n, dt, n_steps):
    """Design b1746728, Build 1: cns.py's serotonin configuration, validated
    with its messages and given its float32 constants, or None when off."""
    src = getattr(p, "serotonin_source_rows", None)
    tgt = getattr(p, "serotonin_target_rows", None)
    if src is None and tgt is None:
        return None
    _lib_b1746728()
    if src is None or tgt is None:
        raise ValueError("serotonin source and target rows must be configured together")
    plan = {}
    for name, rows in (("source", src), ("target", tgt)):
        r = np.asarray(rows, dtype=np.int64)
        if (r.ndim != 1 or not len(r) or len(np.unique(r)) != len(r)
                or np.any(r < 0) or np.any(r >= n)):
            raise ValueError(f"serotonin_{name}_rows must be unique valid rows")
        plan[name] = r
    tau = float(p.serotonin_tau_ms)
    gain = float(p.serotonin_gain_mv_per_hz)
    init = float(getattr(p, "serotonin_init_hz", 0.0))
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("serotonin_tau_ms must be finite and positive")
    if not np.isfinite(gain) or gain < 0.0:
        raise ValueError("serotonin_gain_mv_per_hz must be finite and non-negative")
    if not np.isfinite(init) or init < 0.0:
        raise ValueError("serotonin_init_hz must be finite and non-negative")
    spb = max(1, int(round(5.0 / dt)))
    plan.update(tau=tau, gain=gain, init=init, spb=spb, n_bins=n_steps // spb,
                decay=np.float32(np.exp(-dt / tau)),
                inc=np.float32(1000.0 / (len(plan["source"]) * tau)))
    return plan


_BOUND_8B6E0441_B1 = False


def _lib_8b6e0441_b1():
    """Design 8b6e0441 Build 1's entry points (the octopamine state), bound on
    first use. The CUDA library carries them; cns_engine.m does not yet, so a
    Metal run refuses the state rather than run it as a silent zero."""
    global _BOUND_8B6E0441_B1
    L = _lib()
    if not hasattr(L, "cns_engine_octopamine_state_config"):
        raise NotImplementedError(
            "the octopamine state (design 8b6e0441, Build 1) needs a library built "
            "from it (the CUDA library; cns_engine.m does not carry it yet); "
            "refusing a silent zero-effect run")
    if not _BOUND_8B6E0441_B1:
        c = ctypes
        vp, u32, f32 = c.c_void_p, c.c_uint32, c.c_float
        for name, res, args in (
                ("cns_engine_octopamine_state_config", c.c_int,
                 [vp, u32, vp, u32, vp, f32, f32, f32, f32, u32, u32]),
                ("cns_engine_octopamine_state_read", c.c_int, [vp] * 5 + [u32])):
            f = getattr(L, name)
            f.restype, f.argtypes = res, args
        _BOUND_8B6E0441_B1 = True
    return L


_BOUND_D27F82DF = False


def _lib_d27f82df():
    """Design d27f82df's entry points (B2's exact-update mask, B3's
    hold-down), bound on first use. The CUDA library carries them, and
    cns_engine.m since B8, where they hold the opsin's buffers, so Metal
    refuses the opsin beside either."""
    global _BOUND_D27F82DF
    if not hasattr(_lib(), "cns_engine_holddown_config"):
        raise NotImplementedError(
            "design d27f82df's exact-update mask and hold-down need a library "
            "built after its B8")
    L = _lib()
    if not _BOUND_D27F82DF:
        c = ctypes
        vp, u32, f32 = c.c_void_p, c.c_uint32, c.c_float
        for name, res, args in (
                ("cns_engine_row_exact_config", c.c_int, [vp, vp]),
                ("cns_engine_holddown_config", c.c_int, [vp, u32] + [vp] * 12 + [f32, u32]),
                ("cns_engine_holddown_read", c.c_int, [vp] * 7),
                ("cns_engine_holddown_tap", c.POINTER(c.c_float), [vp])):
            f = getattr(L, name)
            f.restype, f.argtypes = res, args
        _BOUND_D27F82DF = True
    return L


HOLDDOWN_SLOT_FIELDS = ("rows", "form", "e_rel", "decay", "dg", "gmax", "c_spk",
                        "k", "va", "tau_c", "latch_mv", "latch_ticks")
HOLDDOWN_TAP_CHANNELS = ("i_row", "hd_c", "hd_g", "hd_latch", "hd_below")


ROW_CURRENT_FIELDS = ("rows", "i0_mv", "i1_mv", "on_tick", "off_tick")


def row_current_arrays(spec, n, n_steps):
    """Design 2d12a3b3 (B2): validate CNS.run's row_current argument and return
    its five contiguous arrays, one entry per window, with the sha256 of each.
    Refused: a Darwin host, a missing or extra field, arrays of unequal length
    or not one-dimensional, a row outside the network, a non-finite current,
    off before on, and a step outside [0, n_steps]. Design 6d0042d5, Build 3:
    a row may be listed once per window when its windows share one i0 and the
    non-empty ones do not overlap; the receipt counts rows and windows."""
    import hashlib
    if _MAC:
        raise NotImplementedError(
            "row_current (design 2d12a3b3) is CUDA-only; the Metal host refuses it")
    if not isinstance(spec, dict) or set(spec) != set(ROW_CURRENT_FIELDS):
        raise ValueError(f"row_current must be a dict with exactly {ROW_CURRENT_FIELDS}")
    raw = {k: np.asarray(spec[k]) for k in ROW_CURRENT_FIELDS}
    m = len(raw["rows"])
    if not m or any(a.ndim != 1 or len(a) != m for a in raw.values()):
        raise ValueError("row_current arrays must be one-dimensional, non-empty and equal in length")
    for k in ("rows", "on_tick", "off_tick"):
        if not np.issubdtype(raw[k].dtype, np.integer):
            raise ValueError(f"row_current {k} must be integers")
    rows = raw["rows"].astype(np.int64)
    if rows.min() < 0 or rows.max() >= n:
        raise ValueError("row_current: a row outside the network")
    i0 = np.ascontiguousarray(raw["i0_mv"], dtype=np.float32)
    i1 = np.ascontiguousarray(raw["i1_mv"], dtype=np.float32)
    if not (np.all(np.isfinite(i0)) and np.all(np.isfinite(i1))):
        raise ValueError("row_current: a non-finite current")
    on = raw["on_tick"].astype(np.int64)
    off = raw["off_tick"].astype(np.int64)
    if np.any(off < on):
        raise ValueError("row_current: off before on")
    if on.min() < 0 or off.max() > n_steps:
        raise ValueError(f"row_current: a step outside the run's {n_steps} steps")
    order = np.lexsort((np.arange(m), on, rows))          # by row, then on, as the CUDA config orders them
    for a, b in zip(order[:-1], order[1:]):
        if rows[a] != rows[b]:
            continue
        if i0[a] != i0[b]:
            raise ValueError("row_current: one row's windows differ in their held current")
    for r in np.unique(rows):
        sl = order[rows[order] == r]
        live = sl[on[sl] < off[sl]]
        if np.any(on[live][1:] < off[live][:-1]):
            raise ValueError("row_current: one row's windows overlap")
    arr = {"rows": np.ascontiguousarray(rows, dtype=np.int32), "i0_mv": i0, "i1_mv": i1,
           "on_tick": np.ascontiguousarray(on, dtype=np.uint32),
           "off_tick": np.ascontiguousarray(off, dtype=np.uint32)}
    receipt = {"n_rows": int(len(np.unique(rows))), "n_windows": int(m),
               "sha256": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arr.items()}}
    return arr, receipt


_BOUND_336CACB4 = False


def _lib_336cacb4():
    """The CUDA library's design 336cacb4 entry points, bound on first use."""
    global _BOUND_336CACB4
    L = _lib()
    if _MAC or not hasattr(L, "cns_engine_set_graded_gain"):
        raise NotImplementedError(
            "design 336cacb4's time-gated graded release gain needs the CUDA library")
    if not _BOUND_336CACB4:
        vp, u32 = ctypes.c_void_p, ctypes.c_uint32
        for name in ("cns_engine_set_graded_gain", "cns_engine_get_graded_gain"):
            f = getattr(L, name)
            f.restype, f.argtypes = ctypes.c_int, [vp, vp, u32]
        _BOUND_336CACB4 = True
    return L


_BOUND_AFC8FD03 = False


def _lib_afc8fd03():
    """Design afc8fd03 B3: the Metal library's current injection, bound on
    first use. Team's CUDA library does not carry it (team-henri's does)."""
    global _BOUND_AFC8FD03
    L = _lib()
    if not hasattr(L, "cns_engine_set_inject"):
        raise NotImplementedError(
            "inject needs cns_engine_set_inject, which this engine library lacks "
            "(on team it is ported to Metal only)")
    if not _BOUND_AFC8FD03:
        f = L.cns_engine_set_inject
        f.restype, f.argtypes = ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p]
        _BOUND_AFC8FD03 = True
    return L


def _run_windows(start, stop, window, inj_segs, light_every=None):
    """The (w0, nt) windows of a run: range(start, stop, window) without
    inject; with it (design afc8fd03, team-henri's split) windows also end
    where the injected vector changes, so one window holds one vector. The
    drive draws fill the same random stream whatever the window length.
    With a per-row light level (design e2f5f784 P4-3) windows also end at
    every renewal, so one window holds one level vector."""
    edges = [start, stop]
    if inj_segs is not None:
        edges = sorted({start, stop} | {a for a, _, _ in inj_segs if start < a < stop})
    if light_every is not None:
        edges = sorted(set(edges) | {a for a in range(0, stop, light_every) if start < a < stop})
    for a, b in zip(edges[:-1], edges[1:]):
        for w in range(a, b, window):
            yield w, min(window, b - w)


def _row_mask(n, rows, what):
    rows = np.asarray(rows, dtype=np.int64)
    if (rows.ndim != 1 or not len(rows) or len(np.unique(rows)) != len(rows)
            or rows.min() < 0 or rows.max() >= n):
        raise ValueError(f"{what} must be unique valid rows")
    m = np.zeros(n, dtype=np.uint8)
    m[rows] = 1
    return m


def state_canonical(buf, section_bytes):
    """(header, [(name, bytes)]) of a cns_engine_state_read host form, with
    each event log cut to its count and sorted by its unique key, so that two
    executions that appended the same entries in another atomic order read
    alike. Every other section is kept byte for byte."""
    buf = np.asarray(buf, dtype=np.uint8)
    hdr = buf[:64].view(np.uint64)
    if int(hdr[0]) != _STATE_MAGIC or int(hdr[1]) != 1:
        raise ValueError("not a version-1 engine state")
    if int(hdr[2]) != len(STATE_SECTIONS) or len(section_bytes) != len(STATE_SECTIONS):
        raise ValueError("state section table does not match STATE_SECTIONS")
    if int(hdr[7]) != len(buf) or 64 + int(np.sum(section_bytes)) != len(buf):
        raise ValueError("state length does not match its section table")
    raw, at = {}, 64
    for name, b in zip(STATE_SECTIONS, section_bytes):
        raw[name] = buf[at:at + int(b)]
        at += int(b)
    out = dict(raw)

    def cut(names_dtypes, count, keys):
        cols = [raw[nm].view(dt) for nm, dt in names_dtypes]
        n = min(int(count), len(cols[0]))
        cols = [c[:n] for c in cols]
        order = np.lexsort(tuple(cols[k] for k in reversed(keys)))
        for (nm, _), col in zip(names_dtypes, cols):
            out[nm] = np.ascontiguousarray(col[order]).view(np.uint8)

    if len(raw["rr_count"]):
        cut([("rr_tick", np.int32), ("rr_row", np.int32), ("rr_value", np.float32)],
            raw["rr_count"].view(np.uint32)[0], (0, 1))
    if len(raw["dl_count"]):
        c = raw["dl_count"].view(np.uint32)
        cut([("dl_dir_tick", np.int32), ("dl_dir_edge", np.int64),
             ("dl_dir_w", np.float32)], c[0], (0, 1))
        cut([("dl_ev_tick", np.int32), ("dl_ev_src", np.int32),
             ("dl_ev_pos", np.float64), ("dl_ev_neg", np.float64),
             ("dl_ev_nother", np.uint32), ("dl_ev_ndir", np.uint32)], c[1], (0, 1))
    return hdr.copy(), [(nm, out[nm]) for nm in STATE_SECTIONS]


def state_digest(buf, section_bytes):
    """sha256 of the canonical state: header words, then each section's name,
    length and bytes."""
    hdr, secs = state_canonical(buf, section_bytes)
    h = hashlib.sha256(hdr.tobytes())
    for name, b in secs:
        h.update(name.encode() + b"\0" + np.uint64(len(b)).tobytes())
        h.update(np.ascontiguousarray(b).tobytes())
    return h.hexdigest()


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
        if net.G is not None and not _MAC:
            # gpu-7, note 10906d21: CUDA carries the plain coupling term, and
            # of the junction laws the one Metal runs on that kernel
            if p.gap_spikelet != 0 or float(getattr(p, "gap_ap_mv", 0.0)) > 0.0:
                raise NotImplementedError(
                    "CUDA backend: gap spikelets, delays and action-potential waveforms")
            if p.gap_law not in (None, "one_way") or p.gap_current_tap:
                raise NotImplementedError(
                    "CUDA backend: the conserving gap law and the gap current tap")
        if net.G is not None and getattr(p, "conductance_integrator", "euler") == "analytic":
            raise NotImplementedError(
                "metal backend: analytic integration with gap junctions lacks exact voltage parity")
        if p.std_U > 0:
            raise NotImplementedError("metal backend: short-term depression")
        if _MAC and getattr(p, "release_resource_rows", None) is not None:
            raise NotImplementedError("Metal backend: per-row release resource is CUDA-only")
        if _MAC and getattr(p, "adapt_w_init_rows", None) is not None:
            raise NotImplementedError(
                "Metal backend: the adaptation warm start (design fbcd0f61, B6) is CUDA-only")
        if getattr(p, "spike_reg_per_neuron", None) is not None:
            raise NotImplementedError(
                "metal backend: per-neuron spike regularity not yet ported")
        psi_rows = getattr(p, "presynaptic_inhibition_rows", None)
        if float(getattr(p, "graded_phasic", 0.0) or 0.0) > 0.0:
            raise NotImplementedError("metal backend: graded_phasic")
        psi_pairs = getattr(net, "psi_pairs", None)
        aux_rows = getattr(net, "aux_rows", None)
        shared_occ = getattr(net, "shared_occupancy", None)
        if shared_occ is not None and _MAC:
            raise NotImplementedError(
                "Metal backend: shared_occupancy (design 81427d47) is CUDA-only")
        if _MAC and _opsin_present(p):
            # design 244fb801, Build 1: Metal carries the opsin conductance on
            # its pulse train under the Euler integrator. The library refuses
            # windows, force and the tap; these two are refused before a build.
            rows = getattr(p, "threshold_window_rows", None)
            if rows is not None and len(rows) > 0:
                raise NotImplementedError(
                    "Metal backend: the threshold window (design 87a3ad35) is CUDA-only")
            if getattr(p, "conductance_integrator", "euler") == "analytic":
                raise NotImplementedError(
                    "Metal backend: the opsin conductance under the analytic "
                    "integrator is CUDA-only")
        if _MAC and getattr(p, "slow_inh_rows", None) is not None:
            raise NotImplementedError(
                "Metal backend: the slow inhibitory pool (design 2163ba7d, "
                "Build 2) is CUDA-only")
        # run() refuses until configure_slow() has uploaded the pool, so an
        # engine built outside run_metal cannot drop it silently
        self._slow_wanted = getattr(p, "slow_inh_rows", None) is not None
        # design d27f82df: run() refuses until configure_row_exact() and
        # configure_holddown() have uploaded what p carries
        self._exact_wanted = getattr(p, "row_exact_mask", None) is not None
        self._holddown_wanted = getattr(p, "holddown_slots", None) is not None
        self.holddown_n = 0
        self.slow_shape = None
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
        # Design 6d0042d5, Build 2: the coefficient below rest, set right after
        # creation (cns_engine_adapt_below_config) on both engines. None builds
        # the engine exactly as before.
        self._adapt_below = None
        if getattr(p, "adapt_a_below", None) is not None:
            self._adapt_below = np.ascontiguousarray(np.broadcast_to(
                np.asarray(p.adapt_a_below, dtype=np.float32), (N,)))
            if not np.isfinite(self._adapt_below).all():
                raise ValueError("adapt_a_below must be finite")
        use_adapt = bool(np.any(adapt_a != 0) or np.any(adapt_b != 0)
                         or (self._adapt_below is not None
                             and np.any(self._adapt_below != 0)))
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
        # Design efe88957, Build 2: the list form rides the single key's
        # descriptor slots, targets and pool CSRs concatenated in triple order
        # and the ring as deep as the longest lag; k_coin2 reads each target's
        # own triple values and each pool edge's read mode. CUDA only.
        self.coin2 = None
        if getattr(p, "coincidence_triples", ()):
            _lib_efe88957_b2()
            if self.n_coin:
                raise ValueError("coincidence_gain and coincidence_triples are "
                                 "two forms of one seam; give one")
            from cns import coincidence_triple_seam
            _seam = coincidence_triple_seam(
                N, net._indptr, net._indices, net._data, p.coincidence_triples,
                np.asarray(net.graded, dtype=bool), dt)
            _gpos = np.full(N, -1, dtype=np.int32)
            _gpos[graded_idx] = np.arange(len(graded_idx), dtype=np.int32)
            self.coin2 = {"seam": _seam}
            for _nm in ("fast", "slow"):
                _ip, _off = [np.zeros(1, dtype=np.int64)], 0
                for _q in _seam:
                    _ip.append(_q[_nm].indptr[1:].astype(np.int64) + _off)
                    _off += int(_q[_nm].nnz)
                _src = np.concatenate([_q[_nm].indices for _q in _seam]).astype(np.int32)
                coin_arrays[f"coin_{_nm}_indptr"] = np.ascontiguousarray(
                    np.concatenate(_ip), dtype=np.int64)
                coin_arrays[f"coin_{_nm}_src"] = np.ascontiguousarray(_src)
                coin_arrays[f"coin_{_nm}_w"] = np.ascontiguousarray(
                    np.concatenate([_q[_nm].data for _q in _seam]), dtype=np.float32)
                self.coin2[_nm + "_gpos"] = np.ascontiguousarray(_gpos[_src])
            coin_arrays["coin_tgt"] = np.ascontiguousarray(
                np.concatenate([_q["tgt"] for _q in _seam]), dtype=np.int32)
            for _f, _ty in (("gain", np.float32), ("decay", np.float32), ("lag", np.uint32)):
                self.coin2[_f] = np.ascontiguousarray(np.concatenate(
                    [np.full(len(_q["tgt"]), _q[_f], dtype=_ty) for _q in _seam]))
            self.n_coin = int(len(coin_arrays["coin_tgt"]))
            self.coin_lag = int(max(_q["lag"] for _q in _seam))
            self.coin_gain = np.float32(0.0)    # k_coin2 reads coin2's gains
            self.coin_decay = np.float32(0.0)
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
        self.n_gap = 0
        self.gap_n_sub = 1
        self.gap_tap_enabled = p.gap_current_tap
        self.gap_rows = np.empty(0, dtype=np.int32)
        gap_arrays = {}
        if net.G is not None:
            if net.G.format != "csr":
                raise RuntimeError("net.G must be CSR on the target axis")
            # Include source-only rows of a directed junction so its AP can
            # be stored without allocating a full-CNS voltage-history ring.
            targets = np.union1d(np.flatnonzero(np.diff(net.G.indptr)), net.G.indices)
            gap = net.G[targets, :]
            self.n_gap = len(targets)
            self.gap_rows = targets.astype(np.int32)
            gap_arrays = dict(
                gap_tgt=np.ascontiguousarray(targets, dtype=np.int32),
                gap_indptr=np.ascontiguousarray(gap.indptr, dtype=np.int64),
                gap_src=np.ascontiguousarray(np.searchsorted(targets, gap.indices), dtype=np.int32),
                gap_weight=np.ascontiguousarray(gap.data, dtype=np.float32),
                gap_deg=np.ascontiguousarray(net.gap_deg[targets], dtype=np.float32))
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
        # the release factor is set between single-tick groups
        d.window_ticks, d.kmax, d.fired_cap = (
            self.window_ticks, 1 if psi_rows is not None else int(kmax), 0)
        d.cbased = int(bool(p.conductance_based))
        d.split = int(self.split)
        d.analytic = int(getattr(p, "conductance_integrator", "euler") == "analytic")
        d.use_adapt = int(use_adapt)
        einh = np.asarray(p.e_inh, dtype=np.float32)
        # Design e2b1a20d B3: a per-row reversal. CUDA takes it through
        # gpu/optic/lamina_record.configure; Metal sets it right after
        # creation (cns_engine_set_inh_rows), so every engine built here
        # carries it. The scalar below is then read by no update.
        self._einh_rows = None
        if einh.ndim and _MAC:
            if einh.shape != (N,) or not np.isfinite(einh).all():
                raise ValueError("e_inh must be scalar or a finite per-row vector")
            self._einh_rows = np.ascontiguousarray(einh)
        d.e_exc, d.e_inh, d.gscale = float(p.e_exc), float(einh.flat[0]), float(gscale)
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
        keep.update(gap_arrays)
        d.n_gap = self.n_gap
        for k in ("gap_tgt", "gap_indptr", "gap_src", "gap_weight", "gap_deg"):
            setattr(d, k, _ptr(gap_arrays.get(k)))
        if self.n_gap:
            d.gap_scale = float(net.gap_scale)
            d.gap_spikelet = float(p.gap_spikelet)
            delay_ms = float(getattr(p, "gap_delay_ms", 0.0))
            if d.gap_spikelet and delay_ms > 0.0:
                delay = int(round(delay_ms / p.dt))
                if delay < 1:
                    raise ValueError("gap_delay_ms must be at least one timestep")
                if delay >= 2**32 - 1:
                    raise ValueError("gap delay exceeds the native tick range")
                d.gap_delay = delay
            if float(getattr(p, "gap_ap_mv", 0.0)) > 0.0:
                n_ap = max(int(round(float(p.gap_ap_ms) / p.dt)), 2)
                n_rise = max(n_ap // 3, 1)
                up = np.linspace(0.0, 1.0, n_rise + 1)[1:]
                down = np.linspace(1.0, 0.0, n_ap - n_rise + 1)[1:]
                wave = (np.concatenate([up, down]) * float(p.gap_ap_mv)).astype(np.float32)
                ap_off = int(round(delay_ms / p.dt)) if delay_ms > 0.0 else 0
                if ap_off + len(wave) >= 2**32 - 1:
                    raise ValueError("gap AP history exceeds the native tick range")
                keep["gap_ap_wave"] = wave
                d.gap_n_ap, d.gap_ap_off, d.gap_ap_wave = len(wave), ap_off, _ptr(wave)
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
        if self.coin2 is not None:
            c2 = self.coin2
            status = _lib_efe88957_b2().cns_engine_coin2_config(
                self.h, self.n_coin, _ptr(c2["gain"]), _ptr(c2["decay"]), _ptr(c2["lag"]),
                len(c2["fast_gpos"]), _ptr(c2["fast_gpos"]),
                len(c2["slow_gpos"]), _ptr(c2["slow_gpos"]))
            if status:
                msg = L.cns_engine_last_error(self.h).decode()
                L.cns_engine_destroy(self.h)
                self.h = None
                raise RuntimeError(f"coincidence list configuration {status}: {msg}")
        if self._einh_rows is not None:
            status = None
            if hasattr(L, "cns_engine_set_inh_rows"):
                f = L.cns_engine_set_inh_rows
                f.restype, f.argtypes = ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p]
                status = f(self.h, _ptr(self._einh_rows))
            if status != 0:
                msg = ("this engine library lacks cns_engine_set_inh_rows" if status is None
                       else f"{status}: " + L.cns_engine_last_error(self.h).decode())
                L.cns_engine_destroy(self.h)
                self.h = None
                raise RuntimeError("per-row inhibitory reversal: " + msg)
        # With every coefficient zero (use_adapt false) there is no w to split
        # and the engines refuse the call, so the zero law runs unsplit.
        if self._adapt_below is not None and use_adapt:
            status = None
            if hasattr(L, "cns_engine_adapt_below_config"):
                f = L.cns_engine_adapt_below_config
                f.restype, f.argtypes = ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p]
                status = f(self.h, _ptr(self._adapt_below))
            if status != 0:
                msg = ("this engine library lacks cns_engine_adapt_below_config"
                       if status is None
                       else f"{status}: " + L.cns_engine_last_error(self.h).decode())
                L.cns_engine_destroy(self.h)
                self.h = None
                raise RuntimeError("adaptation below rest: " + msg)
        if p.gap_law is not None:
            try:
                pairs = np.asarray(p.gap_pairs, dtype=np.int64)
                ends = np.ascontiguousarray(np.searchsorted(self.gap_rows, pairs), dtype=np.int32)
                weights = np.ascontiguousarray(net.gap_pair_weights, dtype=np.float32)
                from cns import gap_law_substeps
                self.gap_n_sub = gap_law_substeps(p, N)
                if self.gap_n_sub >= 2**32:
                    raise ValueError("gap substep count exceeds uint32")
                if not _MAC and self.gap_n_sub != 1:
                    raise NotImplementedError("CUDA backend: electrical substeps")
                self.gap_pair_count = len(pairs)
                self.gap_tap_stride = 6*len(pairs) + self.n_gap
                L.cns_engine_gap_law_config.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                    ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float,
                    ctypes.c_uint32, ctypes.c_uint32]
                L.cns_engine_gap_law_config.restype = ctypes.c_int
                L.cns_engine_gap_tap.argtypes = [ctypes.c_void_p]
                L.cns_engine_gap_tap.restype = ctypes.POINTER(ctypes.c_float)
                status = L.cns_engine_gap_law_config(self.h, 1 if p.gap_law == "conserving" else 2,
                    len(pairs), _ptr(ends), _ptr(weights), float(p.gap_reverse_ratio or 0),
                    self.gap_n_sub, int(self.gap_tap_enabled))
                if status:
                    raise RuntimeError(f"native gap law configuration failed: {status}")
            except Exception:
                L.cns_engine_destroy(self.h)
                self.h = None
                raise
        self.device = L.cns_engine_device(self.h).decode()
        self.kgroup = int(L.cns_engine_kgroup(self.h))
        self.uniform_mask = int(L.cns_engine_uniform_mask(self.h))
        self.pair_ring = 0
        self.aux_rows = None
        self.occ = None
        self.psi = None
        self.pair_recovery = None
        try:
            if psi_pairs is not None:
                self._configure_pairs(psi_pairs)
            if getattr(p, "psi_pair_recovery", ()):
                if psi_pairs is None:
                    raise ValueError("psi_pair_recovery needs the pair channel")
                self._configure_pair_recovery(p)
            if psi_rows is not None:
                self._configure_psi(p, psi_rows, gscale)
            if aux_rows is not None:
                self._configure_aux(aux_rows)
            if shared_occ is not None:
                self._configure_occ(shared_occ)
        except Exception:
            L.cns_engine_destroy(self.h)
            self.h = None
            raise
        self._snap: dict = {}
        _LIVE_ENGINES.add(self)
        self.words = (N + 31) // 32
        cap = self.window_ticks * min(N, 65536)
        self._out_tick = np.empty(cap, dtype=np.uint32)
        self._out_id = np.empty(cap, dtype=np.int32)

    def _configure_pairs(self, pp):
        """Design 7b0ad6b2: upload the pair channel cns.py built. pp carries
        src and tgt rows, delay_ticks (source event to first used tick), and
        per-pair float32 decay r and increment inc = G*Q*(1-r)/dt. The ring
        holds max delay + kgroup slots, the least the native refusal accepts."""
        src = np.ascontiguousarray(pp["src"], dtype=np.int32)
        tgt = np.ascontiguousarray(pp["tgt"], dtype=np.int32)
        delay = np.ascontiguousarray(pp["delay_ticks"], dtype=np.uint32)
        r = np.ascontiguousarray(pp["r"], dtype=np.float32)
        inc = np.ascontiguousarray(pp["inc"], dtype=np.float32)
        if not (len(src) == len(tgt) == len(delay) == len(r) == len(inc) > 0):
            raise ValueError("psi_pairs arrays must be nonempty and of one length")
        nr = int(delay.max()) + self.kgroup
        L = self._L
        L.cns_engine_pair_config.restype = ctypes.c_int
        L.cns_engine_pair_config.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                             ctypes.c_uint32] + [ctypes.c_void_p] * 5
        status = L.cns_engine_pair_config(self.h, len(src), nr, _ptr(src), _ptr(tgt),
                                          _ptr(delay), _ptr(r), _ptr(inc))
        if status:
            raise RuntimeError(f"pair channel configuration {status}: "
                               + L.cns_engine_last_error(self.h).decode())
        self.pair_ring = nr
        self._keep.update(pair_src=src, pair_tgt=tgt, pair_delay=delay,
                          pair_r=r, pair_inc=inc)

    def _configure_pair_recovery(self, p):
        """Design e05c0eef B1: the pair channel's recovery at this engine's
        step, recomputed from p.psi_pair_recovery and p.dt (a refined copy of
        the network keeps the spec and gets its own ticks and decay)."""
        from cns import psi_pair_recovery_values
        v = psi_pair_recovery_values(p.psi_pair_recovery, float(p.dt))
        L = self._L
        if not hasattr(L, "cns_engine_pair_recovery_config"):
            raise NotImplementedError("native library predates the pair recovery (design e05c0eef B1)")
        L.cns_engine_pair_recovery_config.restype = ctypes.c_int
        L.cns_engine_pair_recovery_config.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                                      ctypes.c_uint32, ctypes.c_float,
                                                      ctypes.c_float]
        status = L.cns_engine_pair_recovery_config(self.h, v["form_code"], v["ticks"],
                                                   v["decay"], v["keep"])
        if status:
            raise RuntimeError(f"pair recovery configuration {status}: "
                               + L.cns_engine_last_error(self.h).decode())
        self.pair_recovery = v

    def pair_recovery_readback(self):
        """Design e05c0eef B1: per pair, events delivered, skipped (dead
        time), delivered with R < 1 (depletion), ticks with more than one
        arrival, the delivered fraction summed, and the clock and resource
        now; None without a recovery."""
        if self.pair_recovery is None:
            return None
        cnt = self.read_buffer("pair_rec_count").reshape(-1, 4)
        return {**self.pair_recovery,
                "delivered": [int(x) for x in cnt[:, 0]], "skipped": [int(x) for x in cnt[:, 1]],
                "scaled": [int(x) for x in cnt[:, 2]], "multi_event_ticks": [int(x) for x in cnt[:, 3]],
                "area_fraction": [float(x) for x in self.read_buffer("pair_rec_area")],
                "clock": [int(x) for x in self.read_buffer("pair_rec_clock")],
                "resource": [float(x) for x in self.read_buffer("pair_rec_resource")]}

    def _configure_psi(self, p, rows, gscale):
        """Design 302c1bfb, Build 8: the listed rows release each spike's
        synapses at f = 1/(1 + G*kg), with the arrays cns.run() derives: the
        rows, kg = float32(k * gscale) from run()'s float64 gscale, and with
        source rows the sources of inhibitory sign, the state's decay r9 and
        the increment scale c9 = (1 - r9)/(1 - r_i), r_i the rows' own
        inhibitory decay (one decay on Metal, which refuses a split one)."""
        net, N = self.net, self.N
        if not self.split:
            raise ValueError("presynaptic inhibition requires conductance_based=True "
                             "and split_conductances=True")
        rows = np.asarray(rows, dtype=np.int64)
        if (rows.ndim != 1 or not len(rows) or len(np.unique(rows)) != len(rows)
                or np.any(rows < 0) or np.any(rows >= N)):
            raise ValueError("presynaptic_inhibition_rows must be unique valid rows")
        gain = getattr(p, "presynaptic_inhibition_gain", None)
        if gain is None:
            kg = np.float32(gscale)
        else:
            if not np.isfinite(float(gain)) or float(gain) < 0.0:
                raise ValueError("presynaptic_inhibition_gain must be finite and >= 0")
            kg = np.float32(float(gain) * gscale)
        src = getattr(p, "presynaptic_inhibition_source_rows", None)
        tau = getattr(p, "presynaptic_inhibition_tau_ms", None)
        if src is None and tau is not None:
            raise ValueError("presynaptic_inhibition_tau_ms is the decay of the source "
                             "state and needs presynaptic_inhibition_source_rows")
        rows32 = np.ascontiguousarray(rows, dtype=np.int32)
        src9 = r9 = c9 = None
        n_src = n_pairs = None
        if src is not None:
            if (getattr(p, "delay_delivery", "bucket") != "queue"
                    or getattr(p, "delay_per_neuron", None) is None):
                raise ValueError("the source-fed release state needs delay_delivery "
                                 "'queue', where each arrival tick is one delivery")
            src = np.asarray(src, dtype=np.int64)
            if (src.ndim != 1 or not len(src) or len(np.unique(src)) != len(src)
                    or np.any(src < 0) or np.any(src >= N)):
                raise ValueError("presynaptic_inhibition_source_rows must be unique valid rows")
            if np.any(np.asarray(net.graded, dtype=bool)[src]):
                raise ValueError("a graded source releases without a spike arrival")
            src = src[net.sign[src] < 0]
            src9 = np.zeros(N, dtype=np.uint8)
            src9[src] = 1
            pos = np.full(N, -1, dtype=np.int64)
            pos[rows] = np.arange(len(rows))
            ip = np.asarray(net._indptr, dtype=np.int64)
            n_out = ip[src + 1] - ip[src]
            off = (np.repeat(ip[src], n_out) + np.arange(int(n_out.sum()))
                   - np.repeat(np.cumsum(n_out) - n_out, n_out))
            n_pairs = int(np.count_nonzero(
                (pos[net._indices[off]] >= 0) & (net._data[off] != 0)))
            n_src = int(len(src))
            r_i = self.decay_g[rows].astype(np.float32)
            if tau is None:
                r9 = r_i.copy()
            else:
                if not np.isfinite(float(tau)) or float(tau) <= 0.0:
                    raise ValueError("presynaptic_inhibition_tau_ms must be finite and > 0")
                r9 = np.full(len(rows), np.float32(np.exp(
                    -p.dt / np.asarray(float(tau), dtype=np.float32))), dtype=np.float32)
            c9 = ((1.0 - r9.astype(np.float64))
                  / (1.0 - r_i.astype(np.float64))).astype(np.float32)
        L = self._L
        L.cns_engine_psi_config.restype = ctypes.c_int
        L.cns_engine_psi_config.argtypes = [ctypes.c_void_p, ctypes.c_uint32] + \
            [ctypes.c_void_p] * 4 + [ctypes.c_float]
        L.cns_engine_psi_log.restype = ctypes.POINTER(ctypes.c_float)
        L.cns_engine_psi_log.argtypes = [ctypes.c_void_p]
        L.cns_engine_psi_state.restype = ctypes.POINTER(ctypes.c_float)
        L.cns_engine_psi_state.argtypes = [ctypes.c_void_p]
        status = L.cns_engine_psi_config(self.h, len(rows32), _ptr(rows32), _ptr(src9),
                                         _ptr(r9), _ptr(c9), float(kg))
        if status:
            raise RuntimeError(f"release factor configuration {status}: "
                               + L.cns_engine_last_error(self.h).decode())
        self._keep.update(psi_rows=rows32, psi_src9=src9, psi_r9=r9, psi_c9=c9)
        self.psi = {"rows": rows, "kg": kg, "gain": None if gain is None else float(gain),
                    "state": "source" if src is not None else "g_i",
                    "source_rows": n_src, "pairs": n_pairs,
                    "tau_ms": None if tau is None else float(tau),
                    "r9": None if r9 is None else r9, "c9": None if c9 is None else c9}

    def psi_log(self, n_ticks):
        """The last window's factor log as (tick in window, position, f, G,
        g_i) for the listed rows that fired, in tick then position order."""
        n = len(self.psi["rows"])
        raw = np.ctypeslib.as_array(self._L.cns_engine_psi_log(self.h),
                                    shape=(self.window_ticks, n, 4))[:n_ticks]
        k, j = np.nonzero(raw[:, :, 3] != 0.0)
        return k, j, raw[k, j, 0].copy(), raw[k, j, 1].copy(), raw[k, j, 2].copy()

    def _configure_occ(self, oc):
        """Design 81427d47: upload the occupancy channel cns.py built
        (cns.shared_occupancy_arrays). Each connection's delay is its source's
        own delivery slot, which the native refusal checks, so its arrival
        reaches the target on the tick the ordinary event would have; the ring
        holds max delay + kgroup slots, the least the refusal accepts."""
        L = _lib_81427d47()
        src = np.ascontiguousarray(oc["src"], dtype=np.int32)
        tgt = np.ascontiguousarray(oc["tgt"], dtype=np.int32)
        delay = np.ascontiguousarray(self.delay_slot[src], dtype=np.uint32)
        quanta = np.ascontiguousarray(oc["quanta"], dtype=np.uint64)
        rows = np.ascontiguousarray(oc["tgt_rows"], dtype=np.int32)
        f64 = {k: np.ascontiguousarray(np.asarray(oc[k], dtype=np.float64))
               for k in ("alpha", "gmax", "beta", "dec", "scale")}
        if not (len(src) == len(tgt) == len(quanta) > 0) or any(
                len(v) != len(rows) for v in f64.values()):
            raise ValueError("shared_occupancy arrays have inconsistent lengths")
        nr = int(delay.max()) + self.kgroup
        status = L.cns_engine_occ_config(
            self.h, len(src), nr, int(oc["mode"]), int(oc["interrupt"]), _ptr(src), _ptr(tgt),
            _ptr(delay), _ptr(quanta), len(rows), _ptr(rows), _ptr(f64["alpha"]),
            _ptr(f64["gmax"]), _ptr(f64["beta"]), _ptr(f64["dec"]), _ptr(f64["scale"]))
        if status:
            raise RuntimeError(f"occupancy configuration {status}: "
                               + L.cns_engine_last_error(self.h).decode())
        self.occ = {"n_edge": len(src), "n_tgt": len(rows), "nr": nr, "rows": rows,
                    "src": src, "tgt": tgt, "delay": delay, "mode": int(oc["mode"]),
                    "interrupt": int(oc["interrupt"])}
        self._keep.update(occ_src=src, occ_tgt=tgt, occ_delay=delay, occ_quanta=quanta,
                          occ_rows=rows, **{"occ_" + k: v for k, v in f64.items()})

    def occ_record(self, log_on, a_cap, s_cap, tap_on):
        """Design 81427d47: the occupancy logs (per-target capacities) and the
        dense tap, before the first run."""
        L = _lib_81427d47()
        a_cap = np.ascontiguousarray(a_cap, dtype=np.uint32)
        s_cap = np.ascontiguousarray(s_cap, dtype=np.uint32)
        if self.occ is None or len(a_cap) != self.occ["n_tgt"] or len(s_cap) != len(a_cap):
            raise ValueError("occupancy recorder needs the channel and one capacity per target")
        if L.cns_engine_occ_record_config(self.h, int(bool(log_on)), _ptr(a_cap), _ptr(s_cap),
                                          int(bool(tap_on))):
            raise RuntimeError("occupancy recorder: " + L.cns_engine_last_error(self.h).decode())
        self.occ.update(log_on=bool(log_on), tap_on=bool(tap_on), a_cap=a_cap.copy(),
                        s_cap=s_cap.copy())
        self._keep.update(occ_a_cap=a_cap, occ_s_cap=s_cap)

    def occ_tap_view(self):
        """The last window's occupancy tap, (window_ticks, n_tgt, OCC_CHANNELS)
        float64, a view the next run overwrites."""
        return np.ctypeslib.as_array(_lib_81427d47().cns_engine_occ_tap(self.h),
                                     shape=(self.window_ticks, self.occ["n_tgt"],
                                            len(OCC_CHANNELS)))

    def _configure_aux(self, rows):
        """Design 7b0ad6b2: a per-tick state tap on the named rows."""
        rows = np.ascontiguousarray(rows, dtype=np.int32)
        L = self._L
        L.cns_engine_aux_config.restype = ctypes.c_int
        L.cns_engine_aux_config.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
        L.cns_engine_aux_tap.restype = ctypes.POINTER(ctypes.c_float)
        L.cns_engine_aux_tap.argtypes = [ctypes.c_void_p]
        L.cns_engine_aux_width.restype = ctypes.c_uint32
        L.cns_engine_aux_width.argtypes = []
        if int(L.cns_engine_aux_width()) != len(AUX_FIELDS):
            raise RuntimeError("state tap width differs from AUX_FIELDS")
        status = L.cns_engine_aux_config(self.h, len(rows), _ptr(rows))
        if status:
            raise RuntimeError(f"state tap configuration {status}: "
                               + L.cns_engine_last_error(self.h).decode())
        self.aux_rows = rows
        self._keep["aux_rows"] = rows

    def aux_tap(self, n_ticks):
        """The last window's state tap, (n_ticks, rows, AUX_FIELDS) float32 copy."""
        buf = np.ctypeslib.as_array(self._L.cns_engine_aux_tap(self.h),
                                    shape=(self.window_ticks, len(self.aux_rows),
                                           len(AUX_FIELDS)))
        return buf[:n_ticks].copy()

    def configure_opsin(self, op, tap_on):
        """Design 87a3ad35 (B2): upload the slots of cns.opsin_arrays and, with
        tap_on, the per-step recorder, before the first run."""
        L = _lib_87a3ad35()
        n = len(op["rows"])
        win = np.asarray(op["win"], dtype=np.int64)
        if win.shape != (n, 4, 2) or win.min() < 0 or win.max() >= 2 ** 32:
            raise ValueError("opsin windows must be (slots, 4, 2) steps inside uint32")
        arr = {"rows": np.ascontiguousarray(op["rows"], dtype=np.int32),
               "win": np.ascontiguousarray(win, dtype=np.uint32),
               "n_win": np.ascontiguousarray(op["n_win"], dtype=np.uint32),
               **{k: np.ascontiguousarray(op[k], dtype=np.float32)
                  for k in ("a_on", "a_off", "g_max", "e_rev", "thr_mv")},
               **{k: np.ascontiguousarray(op[k], dtype=np.uint8) for k in ("force", "kind")}}
        status = L.cns_engine_opsin_config(
            self.h, n, _ptr(arr["rows"]), _ptr(arr["a_on"]), _ptr(arr["a_off"]),
            _ptr(arr["g_max"]), _ptr(arr["e_rev"]), _ptr(arr["thr_mv"]), _ptr(arr["force"]),
            _ptr(arr["kind"]), _ptr(arr["win"]), _ptr(arr["n_win"]), int(bool(tap_on)))
        if status:
            raise RuntimeError(f"opsin configuration {status}: "
                               + L.cns_engine_last_error(self.h).decode())
        if op.get("train") is not None:
            # design 2163ba7d, Build 1: the pulse train, one (start, stop, on,
            # period) step row per slot
            tr = np.asarray(op["train"], dtype=np.int64)
            if tr.shape != (n, 4) or tr.min() < 0 or tr.max() >= 2 ** 32:
                raise ValueError("opsin trains must be (slots, 4) steps inside uint32")
            arr["train"] = np.ascontiguousarray(tr, dtype=np.uint32)
            status = L.cns_engine_opsin_train(self.h, _ptr(arr["train"]))
            if status:
                raise RuntimeError(f"opsin train {status}: "
                                   + L.cns_engine_last_error(self.h).decode())
        self.opsin_slots = n
        self._keep.update({"opsin_" + k: v for k, v in arr.items()})

    def configure_row_current(self, arr, n_steps):
        """Design 2d12a3b3 (B2): upload row_current_arrays' slots before the
        first run."""
        L = _lib_2d12a3b3()
        status = L.cns_engine_row_current_config(
            self.h, len(arr["rows"]), _ptr(arr["rows"]), _ptr(arr["i0_mv"]),
            _ptr(arr["i1_mv"]), _ptr(arr["on_tick"]), _ptr(arr["off_tick"]), int(n_steps))
        if status:
            raise RuntimeError(f"row current configuration {status}: "
                               + L.cns_engine_last_error(self.h).decode())
        self._keep.update({"row_current_" + k: v for k, v in arr.items()})

    def configure_row_exact(self, mask):
        """Design d27f82df B2: put the rows mask marks on the exact update,
        before the first run."""
        L = _lib_d27f82df()
        m = np.ascontiguousarray(mask, dtype=np.uint8)
        if m.shape != (self.N,):
            raise ValueError("row_exact_mask needs one byte per row")
        status = L.cns_engine_row_exact_config(self.h, _ptr(m))
        if status:
            raise RuntimeError(f"row exact configuration {status}: "
                               + L.cns_engine_last_error(self.h).decode())
        self._keep["row_exact_mask"] = m

    def configure_holddown(self, slots, dt, tap=False):
        """Design d27f82df B3: upload walk_search.holddown_by_id_apply's
        per-slot arrays before the first run; tap keeps the per-tick
        hold-down tap (HOLDDOWN_TAP_CHANNELS)."""
        L = _lib_d27f82df()
        if set(slots) != set(HOLDDOWN_SLOT_FIELDS):
            raise ValueError(f"holddown_slots must carry exactly {HOLDDOWN_SLOT_FIELDS}")
        dts = {"rows": np.int32, "form": np.uint32, "latch_ticks": np.uint32}
        arr = {k: np.ascontiguousarray(slots[k], dtype=dts.get(k, np.float32))
               for k in HOLDDOWN_SLOT_FIELDS}
        n = len(arr["rows"])
        if any(a.shape != (n,) for a in arr.values()):
            raise ValueError("holddown_slots arrays must share one length")
        status = L.cns_engine_holddown_config(
            self.h, n, *(_ptr(arr[k]) for k in HOLDDOWN_SLOT_FIELDS), float(dt), int(bool(tap)))
        if status:
            raise RuntimeError(f"hold-down configuration {status}: "
                               + L.cns_engine_last_error(self.h).decode())
        self.holddown_n = n
        self.holddown_tapped = bool(tap)
        self._keep.update({"holddown_" + k: v for k, v in arr.items()})

    def holddown_state(self):
        """Design d27f82df B3: per slot c, g_h, L, b and the run's peak g_h as
        float32, and the peak's tick, after the last window."""
        L = _lib_d27f82df()
        n = self.holddown_n
        got = {k: np.zeros(n, dtype=np.float32) for k in ("c", "g_h", "L", "b", "peak")}
        got["peak_tick"] = np.zeros(n, dtype=np.uint32)
        status = L.cns_engine_holddown_read(
            self.h, *(_ptr(got[k]) for k in ("c", "g_h", "L", "b", "peak", "peak_tick")))
        if status:
            raise RuntimeError(f"hold-down read {status}: "
                               + L.cns_engine_last_error(self.h).decode())
        return got

    def holddown_tap(self, n_ticks):
        """The last window's hold-down tap, (n_ticks, slots, 5) float32."""
        L = _lib_d27f82df()
        buf = np.ctypeslib.as_array(L.cns_engine_holddown_tap(self.h),
                                    shape=(self.window_ticks, self.holddown_n,
                                           len(HOLDDOWN_TAP_CHANNELS)))
        return buf[:n_ticks].copy()

    def configure_slow(self, sl):
        """Design 2163ba7d (Build 2): upload cns.slow_inh_arrays' pool before
        the first run. The engine's fast decay must be the one the pool's area
        factor was computed from."""
        L = _lib_2163ba7d_b2()
        if not (np.all(self.decay_g == self.decay_g[0]) and self.decay_g[0] == sl["d_i"]):
            raise ValueError("slow_inh: the engine's fast decay differs from the pool's d_i")
        nt = len(sl["targets"])
        rp = np.asarray(sl["rec_pos"], dtype=np.int64)
        cols = np.flatnonzero(rp >= 0)
        rec = np.full(nt, -1, dtype=np.int32)
        rec[rp[cols]] = cols
        arr = {"src": np.ascontiguousarray(sl["mask"], dtype=np.uint8),
               "pos": np.ascontiguousarray(sl["tpos"], dtype=np.int32),
               "rec": rec}
        status = L.cns_engine_slow_config(
            self.h, nt, _ptr(arr["src"]), _ptr(arr["pos"]), _ptr(arr["rec"]), len(rp),
            float(sl["keep"]), float(sl["sfac"]), float(sl["d_s"]), int(sl["spb"]),
            int(sl["n_bins"]))
        if status:
            raise RuntimeError(f"slow pool configuration {status}: "
                               + L.cns_engine_last_error(self.h).decode())
        self.slow_shape = (nt, len(rp), int(sl["n_bins"]))
        self._keep.update({"slow_" + k: v for k, v in arr.items()})

    def read_slow(self):
        """Design 2163ba7d (Build 2): the slow pool after the last window, as
        float64 acc_fast, acc_slow and intg and float32 g per target slot, the
        ring (n_ring, slots) and the record (n_bins, record rows)."""
        L = _lib_2163ba7d_b2()
        nt, n_rec, n_bins = self.slow_shape
        got = {"acc_fast": np.zeros(nt), "acc_slow": np.zeros(nt), "intg": np.zeros(nt),
               "g": np.zeros(nt, dtype=np.float32),
               "ring": np.zeros((int(self.n_ring), nt), dtype=np.float32),
               "rec_acc": np.zeros((n_bins, n_rec))}
        status = L.cns_engine_slow_read(
            self.h, int(self.n_ring), nt,
            *(_ptr(got[k]) for k in ("acc_fast", "acc_slow", "intg", "g", "ring", "rec_acc")))
        if status:
            raise RuntimeError(f"slow pool read {status}: "
                               + L.cns_engine_last_error(self.h).decode())
        return got

    def configure_serotonin(self, se):
        """Design b1746728, Build 1: _serotonin_plan's state before the first
        run, on an engine built with kmax=1."""
        L = _lib_b1746728()
        arr = {"src": np.ascontiguousarray(se["source"], dtype=np.int32),
               "tgt": np.ascontiguousarray(se["target"], dtype=np.int32)}
        status = L.cns_engine_serotonin_config(
            self.h, len(arr["src"]), _ptr(arr["src"]), len(arr["tgt"]), _ptr(arr["tgt"]),
            float(np.float32(se["gain"])), float(se["decay"]), float(se["inc"]),
            float(np.float32(se["init"])), int(se["spb"]), int(se["n_bins"]))
        if status:
            raise RuntimeError(f"serotonin configuration {status}: "
                               + L.cns_engine_last_error(self.h).decode())
        self.serotonin_bins = int(se["n_bins"])
        self._keep.update({"serotonin_" + k: v for k, v in arr.items()})

    def configure_octopamine_state(self, os_):
        """Design 8b6e0441, Build 1: cns.octopamine_state_plan's state before
        the first run, on an engine built with kmax=1. The engine counts the
        `counted` rows' spikes; the increment carries the given set's size."""
        L = _lib_8b6e0441_b1()
        arr = {"src": np.ascontiguousarray(os_["counted"], dtype=np.int32),
               "tgt": np.ascontiguousarray(os_["target"], dtype=np.int32)}
        status = L.cns_engine_octopamine_state_config(
            self.h, len(arr["src"]), _ptr(arr["src"]), len(arr["tgt"]), _ptr(arr["tgt"]),
            float(np.float32(os_["gain"])), float(os_["decay"]), float(os_["inc"]),
            float(np.float32(os_["init"])), int(os_["spb"]), int(os_["n_bins"]))
        if status:
            raise RuntimeError(f"octopamine state configuration {status}: "
                               + L.cns_engine_last_error(self.h).decode())
        self.octopamine_bins = int(os_["n_bins"])
        self._keep.update({"octopamine_state_" + k: v for k, v in arr.items()})

    def read_octopamine_state(self):
        """Design 8b6e0441, Build 1: read_serotonin's fields for the octopamine
        state."""
        L = _lib_8b6e0441_b1()
        s2 = np.zeros(2, dtype=np.float32)
        applied = np.zeros(1)
        spikes = np.zeros(1, dtype=np.uint64)
        bins = np.zeros(self.octopamine_bins, dtype=np.float32)
        status = L.cns_engine_octopamine_state_read(
            self.h, _ptr(s2), _ptr(applied), _ptr(spikes), _ptr(bins), len(bins))
        if status:
            raise RuntimeError(f"octopamine state read {status}: "
                               + L.cns_engine_last_error(self.h).decode())
        return {"state": s2[0], "max": s2[1], "applied": float(applied[0]),
                "spikes": int(spikes[0]), "bins": bins}

    def configure_octopamine_offset(self, oa):
        """Design 8b6e0441, Build 2: cns.octopamine_offset_plan's (rows,
        offset) before the first window. A CUDA library built from Build 2
        adds the offset as its own term; on Metal the offset vector is written
        into the injection section (design afc8fd03 B3), which k_step adds at
        the same point, after the serotonin and gap terms and before the
        adaptation, so cns_engine.m is unchanged. Returns the route taken."""
        rows, off = oa
        L = self._L
        if hasattr(L, "cns_engine_octopamine_offset_config"):
            f = L.cns_engine_octopamine_offset_config
            f.restype = ctypes.c_int
            f.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_float]
            tgt = np.ascontiguousarray(rows, dtype=np.int32)
            if f(self.h, len(tgt), _ptr(tgt), float(off)):
                raise RuntimeError("octopamine offset configuration: "
                                   + L.cns_engine_last_error(self.h).decode())
            self._keep["octopamine_offset_tgt"] = tgt
            return "own term (cns_engine_octopamine_offset_config)"
        if _MAC:
            vec = np.zeros(self.N, dtype=np.float32)
            vec[rows] = off
            self.set_inject(vec)
            return "injection section (cns_engine_set_inject)"
        raise NotImplementedError(
            "the octopamine offset (design 8b6e0441, Build 2) needs a CUDA library "
            "built from it; refusing a silent zero-effect run")

    def read_serotonin(self):
        """Design b1746728, Build 1: the state after the last window: s, its
        maximum, the double applied sum, the source spikes and the readback."""
        L = _lib_b1746728()
        s2 = np.zeros(2, dtype=np.float32)
        applied = np.zeros(1)
        spikes = np.zeros(1, dtype=np.uint64)
        bins = np.zeros(self.serotonin_bins, dtype=np.float32)
        status = L.cns_engine_serotonin_read(
            self.h, _ptr(s2), _ptr(applied), _ptr(spikes), _ptr(bins), len(bins))
        if status:
            raise RuntimeError(f"serotonin read {status}: "
                               + L.cns_engine_last_error(self.h).decode())
        return {"state": s2[0], "max": s2[1], "applied": float(applied[0]),
                "spikes": int(spikes[0]), "bins": bins}

    def opsin_tap(self, n_ticks):
        """The last window's opsin tap, (n_ticks, slots, 4) float32 copy, the
        channels in cns.OPSIN_TAP_FIELDS order."""
        from cns import OPSIN_TAP_FIELDS
        buf = np.ctypeslib.as_array(_lib_87a3ad35().cns_engine_opsin_tap(self.h),
                                    shape=(self.window_ticks, self.opsin_slots,
                                           len(OPSIN_TAP_FIELDS)))
        return buf[:n_ticks].copy()

    def run(self, t0, n_ticks, cand_bits=None, force_bits=None):
        """Advance n_ticks; returns (tick_offsets uint32, neuron ids int32)."""
        if self._slow_wanted and self.slow_shape is None:
            raise RuntimeError("slow_inh_rows is set but configure_slow() has not run")
        if self._exact_wanted and "row_exact_mask" not in self._keep:
            raise RuntimeError("row_exact_mask is set but configure_row_exact() has not run")
        if self._holddown_wanted and not self.holddown_n:
            raise RuntimeError("holddown_slots is set but configure_holddown() has not run")
        got = self._L.cns_engine_run(
            self.h, int(t0), int(n_ticks), _ptr(cand_bits), _ptr(force_bits),
            _ptr(self._out_tick), _ptr(self._out_id), len(self._out_tick))
        if got < 0:
            raise RuntimeError(
                f"cns_engine_run returned {got}: "
                f"{self._L.cns_engine_last_error(self.h).decode()}")
        return self._out_tick[:got].copy(), self._out_id[:got].copy()

    def configure_resource(self, n_steps, extra=None):
        """Configure the optional native resource and bound its actual release log.

        `extra`, design 02e57559: per-row counts of prescribed events added to the
        refractory bound, since a prescribed event fires while refractory. The
        target mask of `release_resource_target_rows` is applied here too."""
        from cns import release_resource_arrays
        rr = release_resource_arrays(self.net.p, self.N, self.net.graded)
        self.resource_cap = 0
        targets = getattr(self.net.p, "release_resource_target_rows", None)
        if rr is None:
            if targets is not None:
                raise ValueError("release_resource_target_rows needs release_resource_rows")
            return
        L = self._L
        if not hasattr(L, "cns_engine_resource_config"):
            raise NotImplementedError("native library has no per-row release resource")
        L.cns_engine_resource_config.restype = ctypes.c_int
        L.cns_engine_resource_config.argtypes = [ctypes.c_void_p, ctypes.c_uint32] + [ctypes.c_void_p] * 4 + [ctypes.c_uint32]
        L.cns_engine_resource_log.restype = ctypes.c_int64
        L.cns_engine_resource_log.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint32]
        ref = np.broadcast_to(np.rint(np.asarray(self.net.p.t_refrac, dtype=np.float64) /
                                     self.net.p.dt).astype(np.int64), (self.N,))[rr["rows"]]
        ref = np.maximum(1, ref)
        cap = int(np.sum((int(n_steps) + ref - 1) // ref + 1))
        if extra is not None:
            extra = np.asarray(extra, dtype=np.int64)
            if extra.shape != ref.shape or np.any(extra < 0):
                raise ValueError("extra needs one nonnegative count per resource row")
            cap += int(extra.sum())
        if cap >= 2**32:
            raise ValueError("release log capacity exceeds uint32")
        pos = np.full(self.N, -1, np.int32)
        pos[rr["rows"]] = np.arange(len(rr["rows"]), dtype=np.int32)
        status = L.cns_engine_resource_config(self.h, len(rr["rows"]), _ptr(pos),
                    _ptr(rr["d"]), _ptr(rr["c"]), _ptr(rr["k"]), cap)
        if status:
            raise RuntimeError(f"release resource configuration {status}: " + L.cns_engine_last_error(self.h).decode())
        self.resource_cap = cap
        if targets is not None:
            mask = _row_mask(self.N, targets, "release_resource_target_rows")
            if np.any(np.diff(np.asarray(targets, dtype=np.int64)) <= 0):
                raise ValueError("release_resource_target_rows must be sorted")
            status = _lib_02e57559().cns_engine_resource_targets(self.h, _ptr(mask))
            if status:
                raise RuntimeError("resource targets: " + L.cns_engine_last_error(self.h).decode())

    def state_sections(self):
        """Design 02e57559: the byte length of each STATE_SECTIONS entry."""
        L = _lib_02e57559()
        b = np.zeros(64, dtype=np.uint64)
        n = int(L.cns_engine_state_sections(self.h, _ptr(b), len(b)))
        if n != len(STATE_SECTIONS):
            raise RuntimeError(f"engine reports {n} state sections, not {len(STATE_SECTIONS)}")
        return b[:n].copy()

    def state_read(self, slot=-1):
        """The host form of the live state (slot -1) or of a saved slot."""
        L = _lib_02e57559()
        size = int(L.cns_engine_state_bytes(self.h))
        buf = np.empty(size, dtype=np.uint8)
        got = L.cns_engine_state_read(self.h, int(slot), _ptr(buf), size)
        if got != size:
            raise RuntimeError(f"state read {got}: " + self._L.cns_engine_last_error(self.h).decode())
        return buf

    def state_save(self, slot):
        if _lib_02e57559().cns_engine_state_save(self.h, int(slot)):
            raise RuntimeError("state save: " + self._L.cns_engine_last_error(self.h).decode())

    def state_load(self, slot):
        if _lib_02e57559().cns_engine_state_load(self.h, int(slot)):
            raise RuntimeError("state load: " + self._L.cns_engine_last_error(self.h).decode())

    def resource_log(self):
        from cns import release_log_pack
        cap = self.resource_cap
        t, r, x = np.empty(cap, np.int32), np.empty(cap, np.int32), np.empty(cap, np.float32)
        n = self._L.cns_engine_resource_log(self.h, _ptr(t), _ptr(r), _ptr(x), cap)
        if n < 0:
            raise RuntimeError(f"release log {n}: " + self._L.cns_engine_last_error(self.h).decode())
        return release_log_pack(t[:n], r[:n], x[:n])

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

    def set_inject(self, vec):
        """Design afc8fd03 B3: a constant per-neuron current for the next
        windows, one float32 per row, added to the membrane current after the
        gap current and before adaptation; None turns it off."""
        L = _lib_afc8fd03()
        if vec is not None:
            vec = np.ascontiguousarray(vec, dtype=np.float32)
            if vec.shape != (self.N,):
                raise ValueError("set_inject needs one float32 per row")
        self._keep_inject = vec
        if L.cns_engine_set_inject(self.h, _ptr(vec)):
            raise RuntimeError("set_inject: " + L.cns_engine_last_error(self.h).decode())

    def inh_readback(self):
        """Design e2b1a20d B3: the inhibitory reversal each row's update
        reads, one float32 per row (the scalar on every row when unset)."""
        f = self._L.cns_engine_inh_readback
        f.restype, f.argtypes = ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p]
        out = np.empty(self.N, dtype=np.float32)
        if f(self.h, _ptr(out)):
            raise RuntimeError("inhibitory reversal readback failed")
        return out

    def set_graded_gain(self, gain):
        """Design 336cacb4, CUDA only: replace the per-row graded gain that
        k_graded reads, in graded_idx order with graded_every folded in as the
        build uploads it (graded_gain_scaled), from the next window on."""
        L = _lib_336cacb4()
        g = np.ascontiguousarray(gain, dtype=np.float32)
        if g.shape != self.graded_gain_scaled.shape:
            raise ValueError("set_graded_gain needs one float32 per graded row")
        if L.cns_engine_set_graded_gain(self.h, _ptr(g), len(g)):
            raise RuntimeError("set_graded_gain: "
                               + L.cns_engine_last_error(self.h).decode())

    def graded_gain(self):
        """Design 336cacb4, CUDA only: the gains k_graded reads next."""
        L = _lib_336cacb4()
        out = np.empty(len(self.graded_gain_scaled), dtype=np.float32)
        if L.cns_engine_get_graded_gain(self.h, _ptr(out), len(out)):
            raise RuntimeError("graded_gain: "
                               + L.cns_engine_last_error(self.h).decode())
        return out

    def set_membrane_drive(self, bits, kick_mv):
        """Set command-row membership and each row's event kick in mV."""
        self._L.cns_engine_set_membrane_drive(
            self.h, _ptr(bits), _ptr(kick_mv))

    def configure_adapt_w(self):
        """Design fbcd0f61, Build B6: w at t = 0 on p.adapt_w_init_rows, before
        the first tick; the native call refuses an engine without adaptation."""
        from cns import adapt_w_init_arrays
        rows, vals = adapt_w_init_arrays(self.net.p, self.N)
        rows = np.ascontiguousarray(rows, dtype=np.int32)
        vals = np.ascontiguousarray(vals, dtype=np.float32)
        L = self._L
        if not hasattr(L, "cns_engine_set_adapt_w"):
            raise NotImplementedError("native library has no adaptation warm start")
        L.cns_engine_set_adapt_w.restype = ctypes.c_int
        L.cns_engine_set_adapt_w.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                             ctypes.c_void_p, ctypes.c_void_p]
        status = L.cns_engine_set_adapt_w(self.h, len(rows), _ptr(rows), _ptr(vals))
        if status:
            raise RuntimeError(f"set_adapt_w {status}: " + L.cns_engine_last_error(self.h).decode())

    def set_spike_at(self, rows, values):
        """Design 302c1bfb, Build 10: the listed rows' spike thresholds, set
        between windows."""
        rows = np.ascontiguousarray(rows, dtype=np.int32)
        values = np.ascontiguousarray(values, dtype=np.float32)
        L = self._L
        L.cns_engine_set_spike_at.restype = ctypes.c_int
        L.cns_engine_set_spike_at.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                              ctypes.c_void_p, ctypes.c_void_p]
        status = L.cns_engine_set_spike_at(self.h, len(rows), _ptr(rows), _ptr(values))
        if status:
            raise RuntimeError(f"set_spike_at {status}: " + L.cns_engine_last_error(self.h).decode())
        self.spike_at = self.spike_at.copy()
        self.spike_at[rows] = values

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

    def gap_tap(self, n_ticks):
        """Copy the optional native tap; no buffer is written by this read."""
        if not self.gap_tap_enabled:
            raise ValueError("gap_current_tap was not enabled")
        a = np.ctypeslib.as_array(self._L.cns_engine_gap_tap(self.h),
            shape=(self.window_ticks, self.gap_tap_stride))[:n_ticks].copy()
        k = self.gap_pair_count
        return {"gap_operands": a[:, :4*k].reshape(n_ticks, k, 4),
                "gap_current": a[:, 4*k:6*k].reshape(n_ticks, k, 2),
                "gap_extra_eff": a[:, 6*k:], "gap_rows": self.gap_rows.copy()}

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

    def read_buffer(self, name):
        """A copy of one device-resident graph buffer as the kernels read it
        (design 0c25fab3; on Metal from design 727bccb8, where design
        02e57559's selectors read zero bytes). Names are READ_BUFFERS' keys."""
        which, dtype = READ_BUFFERS[name]
        L = self._L
        if not hasattr(L, "cns_engine_read_buffer"):
            raise NotImplementedError("native library has no buffer readback")
        L.cns_engine_read_buffer.restype = ctypes.c_int64
        L.cns_engine_read_buffer.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                             ctypes.c_void_p, ctypes.c_uint64]
        size = L.cns_engine_read_buffer(self.h, which, None, 0)
        item = np.dtype(dtype).itemsize
        if size < 0 or size % item:
            raise RuntimeError(f"cns_engine_read_buffer {name}: size {size}")
        out = np.empty(size // item, dtype=dtype)
        got = L.cns_engine_read_buffer(self.h, which, _ptr(out), size)
        if got != size:
            raise RuntimeError(f"cns_engine_read_buffer {name} returned {got}: "
                               + L.cns_engine_last_error(self.h).decode())
        return out

    def profile(self):
        """Per-kernel GPU seconds and dispatch counts; zeros unless the
        process set CNS_ENGINE_PROFILE=1 before the engine was created."""
        names = ("step", "scatter", "graded", "fold", "coin")
        secs = np.zeros(5); cnt = np.zeros(5, dtype=np.uint64)
        self._L.cns_engine_profile(self.h, _ptr(secs), _ptr(cnt))
        if _MAC:
            gap_secs = np.zeros(2)
            gap_cnt = np.zeros(2, dtype=np.uint64)
            self._L.cns_engine_gap_profile(self.h, _ptr(gap_secs), _ptr(gap_cnt))
            names += ("gap_current", "gap_spikelet")
            secs, cnt = np.r_[secs, gap_secs], np.r_[cnt, gap_cnt]
        return dict(zip(names, secs)), dict(zip(names, cnt))

    def coin_stats(self):
        """The seam's receipt so far: applied |product| max and sum, fs max,
        delayed-slow max, reduced over targets."""
        out = np.zeros(4)
        self._L.cns_engine_coin_stats(self.h, _ptr(out))
        return out

    def coin2_receipt(self):
        """Design efe88957, Build 2: one receipt per triple from k_coin2's
        per-target accumulators, reduced over the triple's targets as
        cns_engine_coin_stats reduces the single key's (max exact, the sum
        in float64 in target order)."""
        from cns import coincidence_triple_receipt
        L = _lib_efe88957_b2()
        buf = np.zeros(4 * self.n_coin, dtype=np.float32)
        status = L.cns_engine_coin_target_stats(self.h, self.n_coin, _ptr(buf))
        if status:
            raise RuntimeError(f"coincidence list stats {status}: "
                               + L.cns_engine_last_error(self.h).decode())
        acc = buf.reshape(4, self.n_coin)
        stats, j = [], 0
        for q in self.coin2["seam"]:
            sl = slice(j, j + len(q["tgt"]))
            j += len(q["tgt"])
            stats.append((float(acc[0, sl].max()), sum(float(x) for x in acc[1, sl]),
                          float(acc[2, sl].max()), float(acc[3, sl].max())))
        return coincidence_triple_receipt(self.coin2["seam"], stats)

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
        if net.G is not None:
            raise NotImplementedError(
                "record_input cannot attribute gap-junction current to chemical source routes")
        if not isinstance(spec, dict) or set(spec) != self._SPEC_KEYS:
            raise ValueError(
                "record_input requires exactly target_rows, route_by_source, "
                "route_names, and window_ms")
        if not (net.p.conductance_based and net.p.split_conductances):
            raise ValueError(
                "record_input requires conductance_based and split_conductances")
        if np.ndim(net.p.e_inh):
            raise NotImplementedError(
                "record_input reports one inhibitory reversal; the per-row "
                "vector (design e2b1a20d B3) is not ported to it")

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
        # k_gsum sums weight * release over a target's graded sources IN ONE DELAY
        # SLOT and splits that sum by sign, so the netting group is (target, slot)
        # and not the single source. Splitting each source by its own weight sign
        # puts a positive source into the excitatory channel on a tick where its
        # group's sum is negative and the engine delivered to the inhibitory ring
        # instead; on a target holding graded weights of both signs the two
        # conventions differ by the smaller side's whole contribution. Sources in
        # one slot share their arrival ticks, so the group is accumulated signed
        # per route and folded once by the sign of its route-sum, which leaves the
        # sum over routes equal to the engine's delivery at every tick.
        by_slot = {}
        for position, source in enumerate(self.graded_source_rows):
            by_slot.setdefault(int(self.eng.delay_slot[source]), []).append(
                (int(position), int(source)))
        n_routes = self.graded_exc.shape[2]
        for slot, members in by_slot.items():
            arrivals = ticks + slot
            inside = arrivals < self.n_steps
            self.arrivals_after_run += int((~inside).sum()) * len(members)
            if not inside.any():
                continue
            at = arrivals[inside]
            signed = np.zeros(
                (len(at), len(self.target_rows), n_routes), dtype=np.float64)
            for position, source in members:
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
                    signed[:, target, route] += np.float32(
                        scale * weight).astype(np.float64)
            net = signed.sum(axis=2)
            to_exc = net > 0.0
            to_inh = net < 0.0
            for route in range(n_routes):
                part = signed[:, :, route]
                np.add.at(self.graded_exc[:, :, route], at,
                          np.where(to_exc, part, 0.0))
                np.add.at(self.graded_inh[:, :, route], at,
                          np.where(to_inh, -part, 0.0))

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
                "the engine's netting group, one target and one delay slot, is "
                "summed over its sources first and the sum's sign picks the "
                "channel, as k_gsum does; each route carries its own signed part "
                "of that group, so the sum over routes is the engine's delivery"),
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


def _sorted_rows(rows, n, what, limit=None):
    rows = np.asarray(rows, dtype=np.int64)
    if (rows.ndim != 1 or not len(rows) or np.any(np.diff(rows) <= 0)
            or rows[0] < 0 or rows[-1] >= n):
        raise ValueError(f"{what} must be sorted unique valid rows")
    if limit is not None and len(rows) > limit:
        raise ValueError(f"{what} takes at most {limit} rows")
    return rows


class _NativeRecord:
    """Design 02e57559's native recorders, event replacement and state branch
    for one run; CUDA only. The spec is a dict of:

      replace_rows      sorted rows whose events are replaced: each fires only
                        on its forced events (force_spikes), even while
                        refractory, and its membrane and inputs still evolve
      deliv_src/tgt     sorted rows: every output of a src row onto a tgt row
                        is logged (tick, CSC edge, weight as delivered); its
                        other outputs are summed by sign per source event
      ptap_rows         sorted rows, at most four, whose sixteen native-tick
                        channels (PTAP_CHANNELS) are kept on ptap_keep
      ptap_direct_src   sorted rows whose positive arrivals feed the tap's
                        direct conductance
      ptap_keep         [[t0, t1), ...] absolute ticks; absent keeps all
      event_rows        sorted rows whose native-tick events are kept
      population        {"labels": (N,) int category per row, "n_categories",
                        "bin_ticks"}: whole-network spike sums per bin
      row_bins          {"rows": sorted rows, "bin_ticks"}: per-row sums
      window_ticks      the window length (runs without callbacks)
      capacity_ticks    ticks the release and delivery logs are sized for
      forced_counts     {"rows", "counts"}: prescribed events per row that the
                        logs must also hold, at least this call's own; a prefix
                        states the most any of its continuations prescribes
      final_state       read the final engine state and its digest
      branch            {"save_at_tick": T} runs [0, T) and returns a handle;
                        {"resume": handle} continues that handle's state
      occ_log           design 81427d47: every arrival (tick, connection,
                        count) and every update with arrivals (tick, count,
                        A, state before decay, after decay, after arrivals,
                        delivered conductance) of the occupancy channel
      occ_keep          [[t0, t1), ...] absolute ticks of the channel's dense
                        tap (OCC_CHANNELS per target per tick)
      holddown_tap      design d27f82df B5: true keeps HOLDDOWN_TAP_CHANNELS
                        per native tick and hold-down row, on ptap_keep's
                        ticks; needs holddown_by_id

    Recorders read only; replacement and the branch are the assay itself."""

    KEYS = frozenset((
        "replace_rows", "deliv_src", "deliv_tgt", "ptap_rows", "ptap_direct_src",
        "ptap_keep", "event_rows", "population", "row_bins", "window_ticks",
        "capacity_ticks", "forced_counts", "final_state", "branch", "occ_log", "occ_keep",
        "holddown_tap"))

    def __init__(self, net, spec, n_steps, force_ms, dt):
        bad = set(spec) - self.KEYS
        if bad:
            raise ValueError(f"native_record: unknown keys {sorted(bad)}")
        self.L = _lib_02e57559()
        N = net.N
        self.N, self.n_steps, self.spec = N, int(n_steps), spec
        self.window = spec.get("window_ticks")
        if self.window is not None:
            self.window = int(self.window)
            if self.window < 1:
                raise ValueError("window_ticks must be positive")
        self.capacity = int(spec.get("capacity_ticks") or n_steps)
        if self.capacity < n_steps:
            raise ValueError("capacity_ticks must cover the run")
        # prescribed events per row, for the log capacities; a key that is no
        # native tick of this run would never fire, so it is refused here
        self.forced = np.zeros(N, dtype=np.int64)
        for ms, ids in (force_ms or {}).items():
            t = int(round(ms / dt))
            if round(t * dt, 3) != ms or not 0 <= t < n_steps:
                raise ValueError(f"prescribed event at {ms} ms is no tick of the run")
            ids = np.asarray(ids, dtype=np.int64)
            if len(np.unique(ids)) != len(ids) or (len(ids) and (ids.min() < 0 or ids.max() >= N)):
                raise ValueError(f"prescribed events at {ms} ms repeat a row or leave the graph")
            np.add.at(self.forced, ids, 1)
        fc = spec.get("forced_counts")
        if fc is not None:
            rows = _sorted_rows(fc["rows"], N, "forced_counts rows")
            cnt = np.asarray(fc["counts"], dtype=np.int64)
            if cnt.shape != rows.shape or np.any(cnt < 0):
                raise ValueError("forced_counts needs one nonnegative count per row")
            bound = np.zeros(N, dtype=np.int64)
            bound[rows] = cnt
            self.forced = np.maximum(self.forced, bound)
        from cns import release_resource_arrays
        rr = release_resource_arrays(net.p, N, net.graded)
        self.rr_rows = (np.empty(0, np.int64) if rr is None
                        else np.asarray(rr["rows"], dtype=np.int64))
        ref = np.broadcast_to(np.rint(np.asarray(net.p.t_refrac, dtype=np.float64)
                                      / dt).astype(np.int64), (N,))
        self.bound = (self.capacity + np.maximum(1, ref) - 1) // np.maximum(1, ref) + 1
        r = spec.get("replace_rows")
        self.replace = None if r is None else _sorted_rows(r, N, "replace_rows")
        s, t = spec.get("deliv_src"), spec.get("deliv_tgt")
        if (s is None) != (t is None):
            raise ValueError("deliv_src and deliv_tgt go together")
        self.deliv_src = None if s is None else _sorted_rows(s, N, "deliv_src")
        self.deliv_tgt = None if t is None else _sorted_rows(t, N, "deliv_tgt")
        pr, pd_ = spec.get("ptap_rows"), spec.get("ptap_direct_src")
        if (pr is None) != (pd_ is None):
            raise ValueError("ptap_rows and ptap_direct_src go together")
        self.ptap_rows = None if pr is None else _sorted_rows(pr, N, "ptap_rows", 4)
        self.ptap_dsrc = None if pd_ is None else _sorted_rows(pd_, N, "ptap_direct_src")
        keep = spec.get("ptap_keep")
        if keep is None:
            keep = [[0, n_steps]]
        self.keep = [(int(a), int(b)) for a, b in keep]
        for a, b in self.keep:
            if not 0 <= a < b <= n_steps:
                raise ValueError("ptap_keep intervals must lie inside the run")
        if any(b > a2 for (_, b), (a2, _) in zip(self.keep, self.keep[1:])):
            raise ValueError("ptap_keep intervals must be sorted and disjoint")
        er = spec.get("event_rows")
        self.ev_map = None
        if er is not None:
            er = _sorted_rows(er, N, "event_rows")
            self.ev_map = np.full(N, -1, dtype=np.int64)
            self.ev_map[er] = np.arange(len(er))
        pop = spec.get("population")
        self.pop = None
        if pop is not None:
            lab = np.asarray(pop["labels"], dtype=np.int64)
            C, B = int(pop["n_categories"]), int(pop["bin_ticks"])
            if lab.shape != (N,) or lab.min() < 0 or lab.max() >= C or B < 1:
                raise ValueError("population needs one label per row below n_categories")
            self.pop_lab, self.pop_C, self.pop_B = lab, C, B
            self.pop = np.zeros((-(-n_steps // B), C), dtype=np.int64)
        rb = spec.get("row_bins")
        self.rb = None
        if rb is not None:
            rows = _sorted_rows(rb["rows"], N, "row_bins rows")
            self.rb_B = int(rb["bin_ticks"])
            self.rb_rows = rows
            self.rb_map = np.full(N, -1, dtype=np.int64)
            self.rb_map[rows] = np.arange(len(rows))
            self.rb = np.zeros((-(-n_steps // self.rb_B), len(rows)), dtype=np.int32)
        self.occ_log = bool(spec.get("occ_log"))
        ok = spec.get("occ_keep")
        self.occ_keep = None if ok is None else [(int(a), int(b)) for a, b in ok]
        for a, b in self.occ_keep or ():
            if not 0 <= a < b <= n_steps:
                raise ValueError("occ_keep intervals must lie inside the run")
        if self.occ_keep and any(b > a2 for (_, b), (a2, _) in
                                 zip(self.occ_keep, self.occ_keep[1:])):
            raise ValueError("occ_keep intervals must be sorted and disjoint")
        hd = spec.get("holddown_tap")
        if hd is not None and type(hd) is not bool:
            raise ValueError("holddown_tap must be true or false")
        self.hd_tap = bool(hd)
        self.ev_t, self.ev_r, self.tap, self.otap, self.hdtap = [], [], [], [], []
        self.eng = None

    def configure(self, eng):
        """Before the first window of a fresh engine."""
        L, N = self.L, self.N
        err = lambda what: RuntimeError(f"{what}: " + eng._L.cns_engine_last_error(eng.h).decode())
        if self.replace is not None:
            bits = np.zeros(eng.words, dtype=np.uint32)
            np.bitwise_or.at(bits, self.replace >> 5,
                             (np.uint32(1) << (self.replace & 31).astype(np.uint32)))
            if L.cns_engine_set_replacement(eng.h, _ptr(bits)):
                raise err("replacement")
        if self.deliv_src is not None:
            src = _row_mask(N, self.deliv_src, "deliv_src")
            tgt = _row_mask(N, self.deliv_tgt, "deliv_tgt")
            ind = np.asarray(eng.net.W.indices)
            ptr = np.asarray(eng.net.W.indptr)
            per = np.asarray([int(tgt[ind[ptr[j]:ptr[j + 1]]].sum()) for j in self.deliv_src])
            events = self.bound[self.deliv_src] + self.forced[self.deliv_src]
            cap_e = int(events.sum())
            cap_d = max(1, int((events * per).sum()))
            if cap_e >= 2**32 or cap_d >= 2**32:
                raise ValueError("delivery log capacity exceeds uint32")
            if L.cns_engine_deliv_config(eng.h, _ptr(src), _ptr(tgt), cap_d, cap_e):
                raise err("delivery recorder")
            self.cap_d, self.cap_e = cap_d, cap_e
            eng.deliv_caps = (cap_d, cap_e)
        if self.ptap_rows is not None:
            rows = np.ascontiguousarray(self.ptap_rows, dtype=np.int32)
            dsrc = _row_mask(N, self.ptap_dsrc, "ptap_direct_src")
            if L.cns_engine_ptap_config(eng.h, len(rows), _ptr(rows), _ptr(dsrc)):
                raise err("state tap")
        if self.occ_log or self.occ_keep is not None:
            oc = eng.occ
            if oc is None:
                raise ValueError("occ_log and occ_keep need the occupancy channel")
            # one arrival entry per connection per source event; an update
            # needs at least one arrival and at most one per tick
            ev = self.bound[oc["src"]] + self.forced[oc["src"]]
            a_cap = np.asarray([int(ev[oc["tgt"] == r].sum()) for r in oc["rows"]], np.int64)
            s_cap = np.minimum(a_cap, self.capacity)
            if a_cap.sum() >= 2**31:
                raise ValueError("occupancy log capacity exceeds int32")
            eng.occ_record(self.occ_log, a_cap, s_cap, self.occ_keep is not None)
        if self.hd_tap and not getattr(eng, "holddown_tapped", False):
            raise ValueError("holddown_tap needs the hold-down (holddown_by_id)")
        self.eng = eng

    def attach(self, eng):
        """A continuation: the engine is configured and its logs restored."""
        self.eng = eng
        if self.deliv_src is not None:
            self.cap_d, self.cap_e = eng.deliv_caps

    def consume(self, w0, nt, ticks, ids):
        at = w0 + ticks.astype(np.int64)
        if self.ev_map is not None and len(ids):
            sel = self.ev_map[ids] >= 0
            if sel.any():
                self.ev_t.append(at[sel].astype(np.int32))
                self.ev_r.append(ids[sel].astype(np.int32))
        if self.pop is not None and len(ids):
            b = at // self.pop_B
            lo = int(w0 // self.pop_B)
            hi = int((w0 + nt - 1) // self.pop_B) + 1
            flat = (b - lo) * self.pop_C + self.pop_lab[ids]
            self.pop[lo:hi] += np.bincount(flat, minlength=(hi - lo) * self.pop_C
                                           ).reshape(hi - lo, self.pop_C)
        if self.rb is not None and len(ids):
            pos = self.rb_map[ids]
            sel = pos >= 0
            if sel.any():
                np.add.at(self.rb, (at[sel] // self.rb_B, pos[sel]), 1)
        if self.ptap_rows is not None:
            buf = np.ctypeslib.as_array(self.L.cns_engine_ptap(self.eng.h),
                                        shape=(self.eng.window_ticks, 4, len(PTAP_CHANNELS)))
            for a, b in self.keep:
                lo, hi = max(a, w0), min(b, w0 + nt)
                if lo < hi:
                    self.tap.append((lo, buf[lo - w0:hi - w0, :len(self.ptap_rows)].copy()))
        if self.occ_keep is not None:
            buf = self.eng.occ_tap_view()
            for a, b in self.occ_keep:
                lo, hi = max(a, w0), min(b, w0 + nt)
                if lo < hi:
                    self.otap.append((lo, buf[lo - w0:hi - w0].copy()))
        if self.hd_tap:
            buf = self.eng.holddown_tap(nt)
            for a, b in self.keep:
                lo, hi = max(a, w0), min(b, w0 + nt)
                if lo < hi:
                    self.hdtap.append((lo, buf[lo - w0:hi - w0]))

    def snapshot(self):
        return {"ev_t": list(self.ev_t), "ev_r": list(self.ev_r), "tap": list(self.tap),
                "otap": list(self.otap), "hdtap": list(self.hdtap),
                "pop": None if self.pop is None else self.pop.copy(),
                "rb": None if self.rb is None else self.rb.copy()}

    def restore(self, s):
        self.ev_t, self.ev_r, self.tap = list(s["ev_t"]), list(s["ev_r"]), list(s["tap"])
        self.otap = list(s["otap"])
        self.hdtap = list(s["hdtap"])
        if self.pop is not None:
            self.pop = s["pop"].copy()
        if self.rb is not None:
            self.rb = s["rb"].copy()

    def finish(self, eng):
        out = {}
        if self.ev_map is not None:
            t = np.concatenate(self.ev_t) if self.ev_t else np.empty(0, np.int32)
            r = np.concatenate(self.ev_r) if self.ev_r else np.empty(0, np.int32)
            o = np.lexsort((r, t))
            out["event_tick"], out["event_row"] = t[o], r[o]
        if self.deliv_src is not None:
            L = self.L
            c = np.zeros(2, dtype=np.uint32)
            if L.cns_engine_deliv_counts(eng.h, _ptr(c)):
                raise RuntimeError("delivery counts read failed")
            nd, ne = int(c[0]), int(c[1])
            if nd > self.cap_d or ne > self.cap_e:
                raise RuntimeError(f"delivery log overflow: {nd}/{self.cap_d} entries, "
                                   f"{ne}/{self.cap_e} events")
            d = {"dir_tick": np.empty(nd, np.int32), "dir_edge": np.empty(nd, np.int64),
                 "dir_w": np.empty(nd, np.float32), "ev_tick": np.empty(ne, np.int32),
                 "ev_src": np.empty(ne, np.int32), "ev_pos": np.empty(ne, np.float64),
                 "ev_neg": np.empty(ne, np.float64), "ev_nother": np.empty(ne, np.uint32),
                 "ev_ndir": np.empty(ne, np.uint32)}
            st = L.cns_engine_deliv_read(
                eng.h, _ptr(d["dir_tick"]), _ptr(d["dir_edge"]), _ptr(d["dir_w"]), nd,
                _ptr(d["ev_tick"]), _ptr(d["ev_src"]), _ptr(d["ev_pos"]), _ptr(d["ev_neg"]),
                _ptr(d["ev_nother"]), _ptr(d["ev_ndir"]), ne)
            if st:
                raise RuntimeError(f"delivery log read {st}: "
                                   + eng._L.cns_engine_last_error(eng.h).decode())
            o = np.lexsort((d["dir_edge"], d["dir_tick"]))
            for k in ("dir_tick", "dir_edge", "dir_w"):
                out["deliv_" + k] = d[k][o]
            o = np.lexsort((d["ev_src"], d["ev_tick"]))
            for k in ("ev_tick", "ev_src", "ev_pos", "ev_neg", "ev_nother", "ev_ndir"):
                out["deliv_" + k] = d[k][o]
        if self.ptap_rows is not None:
            n = sum(len(x) for _, x in self.tap)
            out["ptap"] = (np.concatenate([x for _, x in self.tap]) if self.tap
                           else np.empty((0, len(self.ptap_rows), len(PTAP_CHANNELS)), np.float32))
            out["ptap_tick"] = (np.concatenate([np.arange(a, a + len(x)) for a, x in self.tap])
                                if self.tap else np.empty(0, np.int64)).astype(np.int64)
            out["ptap_rows"] = self.ptap_rows.copy()
            assert len(out["ptap_tick"]) == n
        if self.hd_tap:
            n_hd = eng.holddown_n
            out["holddown_tap"] = (
                np.concatenate([x for _, x in self.hdtap]) if self.hdtap
                else np.empty((0, n_hd, len(HOLDDOWN_TAP_CHANNELS)), np.float32))
            out["holddown_tap_tick"] = (
                np.concatenate([np.arange(a, a + len(x)) for a, x in self.hdtap])
                if self.hdtap else np.empty(0, np.int64)).astype(np.int64)
            out["holddown_tap_rows"] = eng._keep["holddown_rows"].astype(np.int64)
        if self.pop is not None:
            out["population"] = self.pop
        if self.rb is not None:
            out["row_bins"], out["row_bins_rows"] = self.rb, self.rb_rows.copy()
        if self.occ_log:
            oc = eng.occ
            ca, cs = eng.read_buffer("occ_a_count"), eng.read_buffer("occ_s_count")
            if np.any(ca > oc["a_cap"]) or np.any(cs > oc["s_cap"]):
                raise RuntimeError("occupancy log overflow")
            logs = {k: eng.read_buffer("occ_" + k) for k in
                    ("a_tick", "a_edge", "a_n", "s_tick", "s_n", "s_val", "s_g")}
            ao = np.concatenate([[0], np.cumsum(oc["a_cap"].astype(np.int64))])
            so = np.concatenate([[0], np.cumsum(oc["s_cap"].astype(np.int64))])
            asel = np.concatenate([np.arange(ao[j], ao[j] + ca[j]) for j in range(len(ca))])
            ssel = np.concatenate([np.arange(so[j], so[j] + cs[j]) for j in range(len(cs))])
            atgt = np.repeat(np.arange(len(ca)), ca.astype(np.int64)).astype(np.int32)
            stgt = np.repeat(np.arange(len(cs)), cs.astype(np.int64)).astype(np.int32)
            at, ae, an = (logs[k][asel] for k in ("a_tick", "a_edge", "a_n"))
            o = np.lexsort((ae, atgt, at))
            out.update(occ_a_tick=at[o], occ_a_tgt=atgt[o], occ_a_edge=ae[o], occ_a_n=an[o])
            st, sn = logs["s_tick"][ssel], logs["s_n"][ssel]
            sv, sg = logs["s_val"].reshape(-1, 4)[ssel], logs["s_g"][ssel]
            o = np.lexsort((stgt, st))
            out.update(occ_s_tick=st[o], occ_s_tgt=stgt[o], occ_s_n=sn[o], occ_s_val=sv[o],
                       occ_s_g=sg[o])
        if self.occ_keep is not None:
            out["occ_tap"] = (np.concatenate([x for _, x in self.otap]) if self.otap
                              else np.empty((0, eng.occ["n_tgt"], len(OCC_CHANNELS))))
            out["occ_tap_tick"] = (np.concatenate([np.arange(a, a + len(x)) for a, x in self.otap])
                                   if self.otap else np.empty(0, np.int64)).astype(np.int64)
        if eng.occ is not None:
            out["occ_state_final"] = eng.read_buffer("occ_state").copy()
            out["occ_ring_final_sha256"] = hashlib.sha256(
                eng.read_buffer("occ_ring").tobytes()).hexdigest()
        if self.spec.get("final_state"):
            sec = eng.state_sections()
            buf = eng.state_read(-1)
            hdr, parts = state_canonical(buf, sec)
            out["final_state_digest"] = state_digest(buf, sec)
            out["final_state_sections"] = {
                name: hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest()
                for name, b in parts}
            out["final_state_header"] = [int(x) for x in hdr]
        return out


class BranchHandle:
    """A saved prefix of one run (design 02e57559): the engine that holds it in
    device slot 0 and the host state of every accumulator at the save tick.
    Continuations resume from it with the same arguments; close() frees it."""

    def __init__(self, eng, at_tick, n_steps, config, force_digest, host, record_host,
                 forced):
        self.eng, self.at_tick, self.n_steps = eng, int(at_tick), int(n_steps)
        self.config, self.force_digest = config, force_digest
        self.host, self.record_host, self.forced = host, record_host, forced
        sec = eng.state_sections()
        self.prefix_state_digest = state_digest(eng.state_read(0), sec)
        self.continuations = 0

    def close(self):
        if self.eng is not None:
            self.eng.close()
            self.eng = None


def _force_digest(force_ms, before_tick, dt):
    """sha256 of the forced (tick, row) pairs before a tick, sorted."""
    pairs = []
    for ms, ids in (force_ms or {}).items():
        t = int(round(ms / dt))
        if t < before_tick:
            pairs.extend((t, int(i)) for i in np.asarray(ids).ravel())
    a = np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)
    return hashlib.sha256(a.tobytes()).hexdigest()


def _branch_config(net, duration_ms, seed, record, record_trace_every,
                   record_voltage, spec):
    """Everything a continuation must share with its prefix, as one digest."""
    from cns import release_resource_arrays
    p = net.p
    rr = release_resource_arrays(p, net.N, net.graded)
    h = hashlib.sha256()
    for part in (repr((id(net), float(duration_ms), int(seed), record_trace_every,
                       float(p.dt))),
                 None if record is None else np.asarray(record, dtype=np.int64),
                 None if record_voltage is None else np.asarray(record_voltage, dtype=np.int64),
                 np.asarray(net.W.data), np.asarray(net.W.indices), np.asarray(net.W.indptr)):
        h.update(b"|" if part is None else
                 (part.encode() if isinstance(part, str) else np.ascontiguousarray(part).tobytes()))
    if rr is not None:
        for k in ("rows", "U", "tau_rec_ms"):
            h.update(np.ascontiguousarray(rr[k]).tobytes())
    tr = getattr(p, "release_resource_target_rows", None)
    h.update(b"T" if tr is None else np.ascontiguousarray(np.asarray(tr, np.int64)).tobytes())
    rest = {k: v for k, v in spec.items() if k != "branch"}
    for k in sorted(rest):
        v = rest[k]
        h.update(k.encode())
        if isinstance(v, dict):
            for kk in sorted(v):
                h.update(kk.encode() + np.ascontiguousarray(np.asarray(v[kk])).tobytes())
        else:
            h.update(np.ascontiguousarray(np.asarray(v)).tobytes())
    return h.hexdigest()


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


def _psi_out(psi, parts):
    """Design 302c1bfb, Build 8: cns.run()'s presynaptic_inhibition receipt
    and log from the engine's per-window factor logs, in its step-then-row
    order, the mean from its per-step float32 factor sums."""
    if parts:
        step, row = (np.concatenate([x[k] for x in parts]) for k in (0, 1))
        f, G, gi = (np.concatenate([x[k] for x in parts]) for k in (2, 3, 4))
        o = np.lexsort((row, step))
        step, row, f, G, gi = step[o], row[o], f[o], G[o], gi[o]
    else:
        step = row = np.empty(0, np.int64)
        f = G = gi = np.empty(0, np.float32)
    n = int(len(f))
    fsum = 0.0
    if n:
        cut = np.flatnonzero(np.diff(step)) + 1
        for a, b in zip(np.r_[0, cut], np.r_[cut, n]):
            fsum += float(f[a:b].sum())
    rec = {"rows": int(len(psi["rows"])), "source_spikes": n,
           "attenuated_spikes": int((f < 1.0).sum()),
           "release_factor_mean": None if not n else float(fsum / n),
           "release_factor_min": None if not n else float(min(1.0, float(f.min())))}
    if psi["gain"] is not None or psi["state"] == "source":
        rec.update(gain=psi["gain"], kg=float(psi["kg"]), state=psi["state"])
    if psi["state"] == "source":
        rec.update(source_rows=psi["source_rows"], pairs=psi["pairs"],
                   tau_ms=psi["tau_ms"],
                   r9=[float(x) for x in np.unique(psi["r9"])],
                   c9=[float(x) for x in np.unique(psi["c9"])])
    log = {"step": step.astype(np.int32), "row": row.astype(np.int32),
           "f": f.astype(np.float32), "G": G.astype(np.float32),
           "g_i": gi.astype(np.float32)}
    return rec, log


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


def _graded_gain_window_plan(net, eng, spec, dt, n_steps, window):
    """Design 336cacb4 Build 3: the commanded rows' gain vector and the gate.

    The commanded vector is the built one with the listed rows' entries
    multiplied by float32(factor) in float32, so factor 1.0 uploads the built
    bits and a power of two scales them exactly. Both edges must fall on
    window starts, and the float gate on w0 * dt must agree with the integer
    gate on every window, so chi(t) is exactly [start, stop) at every graded
    update."""
    _lib_336cacb4()
    if set(spec) != {"rows", "factor", "start_ms", "stop_ms"}:
        raise ValueError("graded_gain_window is {rows, factor, start_ms, stop_ms}")
    rows = np.asarray(spec["rows"], dtype=np.int64)
    gidx = np.asarray(getattr(net, "graded_idx", np.empty(0, np.int64)), dtype=np.int64)
    if (rows.ndim != 1 or not len(rows) or len(np.unique(rows)) != len(rows)):
        raise ValueError("graded_gain_window rows must be unique")
    pos = np.searchsorted(gidx, rows)
    if np.any(pos >= len(gidx)) or not np.array_equal(gidx[np.minimum(pos, len(gidx) - 1)], rows):
        raise ValueError("graded_gain_window rows must be graded rows")
    factor = float(spec["factor"])
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("graded_gain_window factor must be finite and > 0")
    start_ms, stop_ms = float(spec["start_ms"]), float(spec["stop_ms"])
    if not (np.isfinite(start_ms) and np.isfinite(stop_ms) and 0.0 <= start_ms < stop_ms):
        raise ValueError("graded_gain_window needs 0 <= start_ms < stop_ms")
    edges = []
    for ms in (start_ms, stop_ms):
        tk = int(round(ms / dt))
        if abs(tk * dt - ms) > 1e-9 * max(1.0, ms) or tk % window:
            raise ValueError(
                f"graded_gain_window edge {ms} ms is not on a window start")
        edges.append(tk)
    w0s = np.arange(0, n_steps, window)
    if not np.array_equal((start_ms <= w0s * dt) & (w0s * dt < stop_ms),
                          (edges[0] <= w0s) & (w0s < edges[1])):
        raise ValueError("graded_gain_window gate disagrees with its tick edges")
    base = np.ascontiguousarray(eng.graded_gain_scaled, dtype=np.float32).copy()
    commanded = base.copy()
    commanded[pos] = base[pos] * np.float32(factor)
    return {"rows": rows, "positions": pos.astype(np.int64), "factor": factor,
            "start_ms": start_ms, "stop_ms": stop_ms, "start_tick": edges[0],
            "stop_tick": edges[1], "base": base, "commanded": commanded,
            "on": False, "transition_ticks": [], "windows_commanded": 0}


def _graded_release_plan(net, spec, dt, engine_reltap_pos, g_every):
    """Design 336cacb4 Build 4: the recorder's windows, columns and weights.

    w_i is the sum of |weight| over row i's outgoing edges in the built net.W
    (CSC on the presynaptic axis), the entries the engine's graded gather
    delivers through."""
    if set(spec) != {"windows_ms", "onset_ms"}:
        raise ValueError("graded_release_record is {windows_ms, onset_ms}")
    gidx = np.asarray(net.graded_idx, dtype=np.int64)
    ng = len(gidx)
    cols = np.searchsorted(engine_reltap_pos, np.arange(ng))
    if not np.array_equal(engine_reltap_pos[cols], np.arange(ng)):
        raise RuntimeError("graded release recorder: a graded row is not tapped")

    def _tick(ms):
        ms = float(ms)
        tk = int(round(ms / dt))
        if not np.isfinite(ms) or abs(tk * dt - ms) > 1e-9 * max(1.0, abs(ms)):
            raise ValueError(f"graded_release_record time {ms} ms is not a tick")
        return tk
    windows_ms = [[float(a), float(b)] for a, b in spec["windows_ms"]]
    if not windows_ms or any(not 0.0 <= a < b for a, b in windows_ms):
        raise ValueError("graded_release_record windows need 0 <= a < b")
    W = net.W
    w_abs = np.array([float(np.abs(W.data[W.indptr[i]:W.indptr[i + 1]].astype(np.float64)).sum())
                      for i in gidx], dtype=np.float64)
    ids = [str(x) for x in np.asarray(net.ids)[gidx]]
    return {"cols": cols, "windows_ms": windows_ms,
            "windows_tick": [[_tick(a), _tick(b)] for a, b in windows_ms],
            "onset_ms": float(spec["onset_ms"]), "onset_tick": _tick(spec["onset_ms"]),
            "sum": np.zeros((len(windows_ms), ng), dtype=np.float64),
            "n": np.zeros(len(windows_ms), dtype=np.int64),
            "first": None, "gidx": gidx, "ids": ids, "w_abs": w_abs,
            "g_every": int(g_every)}


def _graded_release_consume(grr, w0, nt, release_tap, gain_live, g_every):
    """One window of the Build 4 recorder. The tap holds k_graded's pre-gain
    release f(v) on graded ticks; the product with the gains in force is taken
    in float32, which is k_graded's own FM, so each value is the gscale the
    engine delivered through W_g."""
    gk = np.flatnonzero((w0 + np.arange(nt)) % g_every == 0)
    if not len(gk):
        return
    f = release_tap[gk][:, grr["cols"]]
    post = gain_live[None, :] * f
    tk = w0 + gk
    for i, (a, b) in enumerate(grr["windows_tick"]):
        sel = (tk >= a) & (tk < b)
        if sel.any():
            grr["sum"][i] += post[sel].astype(np.float64).sum(axis=0)
            grr["n"][i] += int(sel.sum())
    if grr["first"] is None:
        hit = np.flatnonzero(tk >= grr["onset_tick"])
        if len(hit):
            j = int(hit[0])
            grr["first"] = {"tick": int(tk[j]), "gscale_f32": post[j].copy(),
                            "gain_scaled_f32": gain_live.copy(), "f_f32": f[j].copy()}


def _graded_release_receipt(grr, dt):
    """The Build 4 record: mean post-gain release rel_i = mean(gscale) /
    graded_every per window, so rel_i = G_i [1 + (m - 1) chi c_i] f(v_i)."""
    ge = float(grr["g_every"])
    first = grr["first"]
    return {
        "graded_rows": grr["gidx"].tolist(),
        "graded_ids": grr["ids"],
        "graded_row_order_sha256": hashlib.sha256(
            "\n".join(grr["ids"]).encode()).hexdigest(),
        "graded_every": int(grr["g_every"]),
        "windows_ms": grr["windows_ms"],
        "n_updates": grr["n"].tolist(),
        "mean_release": [(grr["sum"][i] / grr["n"][i] / ge).tolist()
                         if grr["n"][i] else None for i in range(len(grr["n"]))],
        "w_abs_out": grr["w_abs"].tolist(),
        "onset_ms": grr["onset_ms"],
        "first_update_tick": None if first is None else first["tick"],
        "first_update_ms": None if first is None else first["tick"] * dt,
        "first_update_gscale_f32": None if first is None else first["gscale_f32"].tolist(),
        "first_update_gain_scaled_f32": None if first is None else first["gain_scaled_f32"].tolist(),
        "first_update_f_f32": None if first is None else first["f_f32"].tolist(),
    }


def _run_metal_unlocked(net, duration_ms, drive=None, seed=0, record=None,
                        record_trace_every=None, record_voltage=None,
                        drive_window=None, drive_fn=None, feedback_rows=None,
                        drive_every_ms=1.0, force_spikes=None,
                        first_spike=False, force_spikes_fn=None,
                        feedback_graded_release_rows=None,
                        graded_drive_max_rate_hz=None,
                        graded_drive_signed=False, membrane_drive_rows=None,
                        membrane_drive_mv=0.0,
                        membrane_drive_fraction=None, record_input=None,
                        light_input=None, record_voltage_every_ms=None,
                        record_voltage_native=False, record_lamina=None,
                        native_record=None, graded_gain_window=None,
                        graded_release_record=None, record_opsin=False,
                        spike_tap_rows=None, spike_tap=None, row_current=None,
                        inject=None, silence_window=None, rest_drive=None,
                        graded_trace_record=None, record_holddown=False,
                        record_voltage_native_state=False):
    """CNS.run() on the Metal engine. Same arguments, same return shape.

    `record_voltage_native_state` (design 1e8f1a64, recorder only) needs
    record_voltage_native and returns out["voltage_native_state"], the whole
    native state tap of the record_voltage rows, (n_steps, rows, 4) float32,
    fields in out["voltage_native_state_fields"]: the post-integration
    pre-reset voltage (voltage_native's field), the voltage entering the
    membrane equation, and the post-decay/arrival excitatory and inhibitory
    accumulators. It reads the tap voltage_native already reads.

    `graded_trace_record` (design 6f704792 Build 1, readout only) is
    {"rows", "bin_ms"}: for each listed graded row, the charge it delivered
    per unit outgoing weight in each bin, in spike units. That is the sum over
    the bin's graded updates of the release k_graded computed (the release
    tap) times the gain in force (graded_gain_scaled, the built gain times
    graded_every), in float32 as k_graded multiplies them; a spike delivers 1.

    `native_record` (design 02e57559, CUDA only) is _NativeRecord's spec. With
    {"branch": {"save_at_tick": T}} the call runs [0, T) and returns only a
    "branch_handle"; with {"branch": {"resume": handle}} it continues that
    handle's saved state to the end, every host accumulator restored.

    `graded_gain_window` (design 336cacb4, CUDA only) is {"rows", "factor",
    "start_ms", "stop_ms"}: the listed graded rows' release gain is multiplied
    by factor for the windows whose start t = w0 * dt satisfies
    start_ms <= t < stop_ms, the gate walk_search's drive clamp uses, with
    both edges on window starts. `graded_release_record` (readout only) is
    {"windows_ms": [[a, b], ...], "onset_ms": t}: per graded row, the mean
    post-gain release over the graded updates in each [a, b), the first
    update at or after onset_ms, and the row's summed |outgoing weight|.

    `spike_tap_rows` and `spike_tap` (design fa5adfc8) go together: after each
    window's eng.run the (step, row) pairs of those rows' spikes in that
    window are appended to the caller's list, sorted by step then row, where
    step is the spike's CNS step w0 + tick. The tap reads the spikes the
    engine returned and writes nothing the run reads, so a run with the tap
    is the run without it; absent, nothing is appended and nothing changes."""
    p = net.p
    N = net.N
    tap_member = None
    if spike_tap_rows is not None or spike_tap is not None:
        if spike_tap_rows is None or type(spike_tap) is not list:
            raise ValueError("spike_tap_rows needs spike_tap, a list the caller owns")
        if native_record is not None:
            # a branch saves and resumes windows, which would tap a window twice
            raise ValueError("spike_tap does not compose with native_record")
        tap_rows = np.asarray(spike_tap_rows, dtype=np.int64)
        if (tap_rows.ndim != 1 or not len(tap_rows) or (tap_rows < 0).any()
                or (tap_rows >= N).any() or len(np.unique(tap_rows)) != len(tap_rows)):
            raise ValueError("spike_tap_rows must be distinct engine rows")
        tap_member = np.zeros(N, dtype=bool)
        tap_member[tap_rows] = True
    if (getattr(p, "tau_syn_exc", None) is not None
            or getattr(p, "tau_syn_inh", None) is not None):
        raise NotImplementedError(
            "Metal does not yet implement split excitatory/inhibitory "
            "synaptic decay; use the NumPy engine")
    requested_vrec = (None if record_voltage is None
                      else np.asarray(record_voltage, dtype=np.int64))
    if type(record_voltage_native_state) is not bool:
        raise ValueError("record_voltage_native_state must be Boolean")
    if record_voltage_native_state and not (record_voltage_native and requested_vrec is not None):
        raise ValueError("record_voltage_native_state needs record_voltage with record_voltage_native")
    if record_voltage_native_state and native_record is not None and native_record.get("branch"):
        raise ValueError("record_voltage_native_state is refused with a native_record branch")
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
    # design 87a3ad35, B2: the opsin channel, validated exactly as run() does
    from cns import opsin_arrays, opsin_out, OPSIN_TAP_FIELDS
    _op = opsin_arrays(p, N, duration_ms, np.asarray(p.e_inh, dtype=np.float32),
                       net.n_edges, graded)
    if record_opsin and _op is None:
        raise ValueError("record_opsin needs an opsin or threshold-window set")
    # design 2d12a3b3, B2: the held current, validated before anything is built
    _rc, _rc_receipt = ((None, None) if row_current is None
                        else row_current_arrays(row_current, N, n_steps))
    _rc_rows = None if _rc is None else set(_rc["rows"].tolist())
    # design 2163ba7d, Build 2: the slow inhibitory pool, validated as run() does
    from cns import slow_inh_arrays, slow_inh_out
    _sl = slow_inh_arrays(p, N, net._indptr, net._indices, net._data, n_steps, graded)
    if _sl is not None:
        _lib_2163ba7d_b2()
    # design d27f82df, B2 and B3: the exact-update mask and the hold-down
    _xm = getattr(p, "row_exact_mask", None)
    _hd = getattr(p, "holddown_slots", None)
    if _xm is not None or _hd is not None:
        _lib_d27f82df()
    if record_holddown and _hd is None:
        raise ValueError("record_holddown needs the hold-down (holddown_by_id)")
    # design b1746728, Build 1: the serotonin state, validated as run() does;
    # the held current would also set extra, and the state is in no
    # recorder's or branch's saved state. Design c380b266 Build 2 lifts the
    # injection: k_step adds the injection section to extra after the
    # serotonin and gap terms, set_inject grows clamp_val alone while the
    # state lives in args, and configure_serotonin below runs before the
    # first window's set_inject, so the engine's has_inj refusal in
    # cns_engine_serotonin_config is never reached and the library is unchanged
    _se = _serotonin_plan(p, N, dt, n_steps)
    if _se is not None and (_rc is not None or native_record is not None):
        raise NotImplementedError(
            "the serotonin state does not compose with row_current or native_record")
    # design 8b6e0441, Build 1: the octopamine state, refused where the
    # serotonin state is, and on a library without it (Metal)
    from cns import octopamine_state_plan
    _os = octopamine_state_plan(p, N, dt, n_steps, net.W)
    if _os is not None:
        _lib_8b6e0441_b1()
        if _rc is not None or native_record is not None:   # a branch is a native_record
            raise NotImplementedError(
                "the octopamine state does not compose with row_current or native_record")
    # design afc8fd03, B3: team-henri's injection, the segments cns.py adds
    if inject is not None:
        if force_spikes_fn is not None:
            raise ValueError("inject is refused with force_spikes_fn")
        if native_record is not None or graded_gain_window is not None:
            raise NotImplementedError(
                "inject does not compose with native_record or graded_gain_window")
        if p.gap_law is not None:
            raise NotImplementedError(
                "inject with gap_law has no NumPy reference (cns.py refuses gap_law)")
        _lib_afc8fd03()
    from cns import _inject_segments, octopamine_offset_plan
    inj_segs = _inject_segments(inject, dt, n_steps, N)
    # design 8b6e0441, Build 2: the offset; on Metal it holds the injection section
    _oa = octopamine_offset_plan(p, N)
    if _oa is not None and inject is not None:
        raise NotImplementedError("the octopamine offset is refused beside inject")
    if p.gap_current_tap and requested_vrec is None:
        raise ValueError("gap_current_tap requires record_voltage")
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
    if force_spikes_fn is not None:
        if drive_fn is not None:
            raise ValueError("drive_fn and force_spikes_fn are mutually exclusive")
        if feedback_rows is None:
            raise ValueError("force_spikes_fn requires feedback_rows")
        cadence = Fraction(str(drive_every_ms)) / Fraction(str(p.dt))
        if cadence <= 0 or cadence.denominator != 1:
            raise ValueError("force_spikes_fn cadence must be a positive integral number of CNS steps")
        fb_every = int(cadence)
    window = (fb_every if drive_fn is not None or force_spikes_fn is not None
              else min(n_steps, 256))
    # design be61e238 B0: inject beside drive_fn when every edge of the injected
    # current falls on a drive tick. The windows are then range(0, n_steps,
    # fb_every), as without inject, so drive_fn lands on every window's first
    # tick; an edge between drive ticks would start windows off the cadence.
    if inj_segs is not None and drive_fn is not None:
        cadence = Fraction(str(drive_every_ms)) / Fraction(str(p.dt))
        if cadence <= 0 or cadence.denominator != 1:
            raise ValueError("inject beside drive_fn needs a whole number of CNS steps per drive tick")
        off_cadence = [a for a, _, _ in inj_segs if a % fb_every]
        if off_cadence:
            raise ValueError(f"inject beside drive_fn: edges at ticks {off_cadence[:4]} fall "
                             f"between drive ticks (every {fb_every} steps)")
    force_ms = ({round(float(k), 3): np.asarray(v, dtype=np.int64)
                 for k, v in force_spikes.items()} if force_spikes else None)
    nr = resume = branch = config = None
    if native_record is not None:
        spec = dict(native_record)
        branch = spec.pop("branch", None)
        if spec.get("window_ticks") is not None:
            if drive_fn is not None or force_spikes_fn is not None:
                raise ValueError("native_record window_ticks cannot replace a callback cadence")
            window = int(spec["window_ticks"])
        if branch is not None:
            refused = [name for name, used in (
                ("drive", bool(drive)), ("drive_fn", drive_fn is not None),
                ("force_spikes_fn", force_spikes_fn is not None),
                ("feedback_rows", feedback_rows is not None),
                ("record_input", record_input is not None),
                ("light_input", light_input is not None),
                ("record_lamina", record_lamina is not None),
                ("first_spike", bool(first_spike)),
                ("membrane_drive_rows",
                 membrane_drive_rows is not None and len(membrane_drive_rows) > 0),
                ("rest_drive", rest_drive is not None),
                ("gap_current_tap", bool(p.gap_current_tap)),
                ("graded_gain_window", graded_gain_window is not None),
                ("graded_release_record", graded_release_record is not None),
                ("graded_trace_record", graded_trace_record is not None),
                ("the opsin channel", _op is not None),
                ("the held current", _rc is not None),
                ("the slow inhibitory pool", _sl is not None),
                ("the exact-update mask", _xm is not None),
                ("the hold-down", _hd is not None),
                ("presynaptic inhibition",
                 getattr(p, "presynaptic_inhibition_rows", None) is not None),
                ("silence_window", silence_window is not None)) if used]
            if refused:
                raise NotImplementedError(f"a state branch does not carry {refused}")
            if set(branch) not in ({"save_at_tick"}, {"resume"}):
                raise ValueError("branch is {'save_at_tick': T} or {'resume': handle}")
            config = _branch_config(net, duration_ms, seed, record, record_trace_every,
                                    record_voltage, spec)
            if "save_at_tick" in branch:
                T = int(branch["save_at_tick"])
                if not 0 < T < n_steps or T % window:
                    raise ValueError("save_at_tick must be a positive multiple of the "
                                     "window inside the run")
            else:
                resume = branch["resume"]
                if not isinstance(resume, BranchHandle) or resume.eng is None \
                        or resume.eng.h is None:
                    raise RuntimeError("branch handle's engine is closed")
                if config != resume.config:
                    raise ValueError("continuation arguments differ from its prefix")
                if _force_digest(force_ms, resume.at_tick, dt) != resume.force_digest:
                    raise ValueError("continuation's prescribed events before the "
                                     "branch tick differ from its prefix")
                if resume.n_steps != n_steps or resume.eng.window_ticks != max(window, 1):
                    raise ValueError("continuation length or window differs from its prefix")
        nr = _NativeRecord(net, spec, n_steps, force_ms, dt)
        if resume is not None and np.any(nr.forced > resume.forced):
            raise ValueError("continuation prescribes more events than its prefix sized for")
    # every run starts from rest, like run(); the engine holds state, so a
    # fresh one is built per run (graph upload is ~0.1 s against the run)
    # Every engine this process still holds, not only the one on this net:
    # walk_search's net_cache keeps one CNS per config and each keeps its last
    # engine, so on a multi-arm sweep the others are pure GPU memory. Closing
    # them here rather than in __del__ is what bounds the process, because the
    # cache holds a strong reference to every net it has built.
    for _other in list(_LIVE_ENGINES):
        if resume is None or _other is not resume.eng:
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
    if graded_release_record is not None:
        # design 336cacb4 Build 4: tap every graded row. The tap copies the
        # pre-gain release k_graded already computed and writes no state.
        if not len(graded_idx):
            raise ValueError("graded_release_record needs graded rows")
        engine_reltap_pos = np.unique(np.concatenate(
            [engine_reltap_pos, np.arange(len(graded_idx), dtype=np.int64)])
        ).astype(np.int64)
    gtr_pos = None
    if graded_trace_record is not None:
        # design 6f704792 Build 1: tap the named graded rows, as above.
        if set(graded_trace_record) != {"rows", "bin_ms"}:
            raise ValueError("graded_trace_record is {rows, bin_ms}")
        _gtr_rows = np.asarray(graded_trace_record["rows"], dtype=np.int64)
        _gpos = {int(r): i for i, r in enumerate(np.asarray(graded_idx))}
        if (_gtr_rows.ndim != 1 or len(np.unique(_gtr_rows)) != len(_gtr_rows)
                or any(int(r) not in _gpos for r in _gtr_rows)):
            raise ValueError("graded_trace_record rows must be distinct graded rows")
        gtr_pos = np.asarray([_gpos[int(r)] for r in _gtr_rows], dtype=np.int64)
        if len(gtr_pos):
            engine_reltap_pos = np.unique(np.concatenate(
                [engine_reltap_pos, gtr_pos])).astype(np.int64)
    if resume is None:
        eng = MetalEngine(
            net, window_ticks=max(window, 1), vrec=engine_vrec,
            reltap_pos=(engine_reltap_pos if len(engine_reltap_pos) else None),
            kmax=1 if (_se is not None or _os is not None) else 0)
    else:
        eng = resume.eng
    net._metal_engine = eng
    net._holddown_numpy = None   # walk_search reads this engine's hold-down
    sw_mask = None
    sw_spikes = 0
    if silence_window is not None:
        from cns import _silence_window_steps
        sw_rows, sw_on, sw_off = _silence_window_steps(silence_window, N, dt, drive_every_ms)
        sw_mask = np.zeros(N, dtype=bool)
        sw_mask[sw_rows] = True
        sw_set = {sw_on: np.full(len(sw_rows), 1e9, dtype=np.float32),
                  sw_off: eng.spike_at[sw_rows].astype(np.float32)}
    light_levels = None
    if resume is None:
        from gpu.optic.lamina_record import configure as configure_lamina
        light_levels = configure_lamina(eng, light_input, record_lamina)
    op_tap = None
    if _rc is not None:
        eng.configure_row_current(_rc, n_steps)
    if _sl is not None:
        eng.configure_slow(_sl)
    if _xm is not None:
        eng.configure_row_exact(_xm)
    hd_tap = None
    if _hd is not None:
        eng.configure_holddown(_hd, dt, tap=(nr is not None and nr.hd_tap) or record_holddown)
        if record_holddown:
            hd_tap = np.zeros((n_steps, eng.holddown_n, len(HOLDDOWN_TAP_CHANNELS)),
                              dtype=np.float32)
    if _se is not None:
        eng.configure_serotonin(_se)
    if _os is not None:   # design 8b6e0441, Build 1
        eng.configure_octopamine_state(_os)
    oa_route = None
    if _oa is not None:   # after the serotonin state, which Metal refuses beside an injection
        oa_route = eng.configure_octopamine_offset(_oa)
    if _op is not None:
        eng.configure_opsin(_op, record_opsin)
        if record_opsin:
            op_tap = {k: np.zeros((n_steps, len(_op["rows"])), dtype=np.float32)
                      for k in OPSIN_TAP_FIELDS}
    if getattr(p, "adapt_w_init_rows", None) is not None:
        if resume is not None:
            raise ValueError("adapt_w_init_rows sets w at t = 0 and cannot start a resumed run")
        eng.configure_adapt_w()
    lamina_recorder = None
    if record_lamina is not None:
        if input_recorder is not None:
            raise ValueError("lamina and route recorders cannot share a run")
        from gpu.optic.lamina_record import Recorder
        lamina_recorder = Recorder(eng, n_steps, dt, record_lamina)
    has_resource = (getattr(p, "release_resource_rows", None) is not None
                    or getattr(p, "release_resource_target_rows", None) is not None)
    if nr is None:
        if has_resource:
            eng.configure_resource(n_steps)
    elif resume is None:
        if has_resource:
            eng.configure_resource(nr.capacity, extra=nr.forced[nr.rr_rows])
        nr.configure(eng)
    else:
        nr.attach(eng)
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
    ggw = grr = gtr = None
    if (graded_gain_window is not None or graded_release_record is not None
            or graded_trace_record is not None):
        # design 336cacb4: the gains in force, the device's own on CUDA
        gain_live = eng.graded_gain_scaled.copy()
        if graded_gain_window is not None or not _MAC:
            gain0 = eng.graded_gain()
            if not np.array_equal(gain0.view(np.uint32),
                                  gain_live.view(np.uint32)):
                raise RuntimeError("device graded gain differs from the built gain")
    if graded_gain_window is not None:
        ggw = _graded_gain_window_plan(net, eng, graded_gain_window, dt,
                                       n_steps, max(window, 1))
    if graded_release_record is not None:
        grr = _graded_release_plan(net, graded_release_record, dt,
                                   engine_reltap_pos, g_every)
    if graded_trace_record is not None:
        _gb = float(graded_trace_record["bin_ms"])
        _gspb = int(round(_gb / dt))
        if _gspb < 1 or abs(_gspb * dt - _gb) > 1e-9 * max(1.0, _gb):
            raise ValueError("graded_trace_record bin_ms must be a whole number of ticks")
        gtr = {"rows": _gtr_rows, "cols": np.searchsorted(engine_reltap_pos, gtr_pos),
               "gain": gain_live[gtr_pos], "steps_per_bin": _gspb, "bin_ms": _gb,
               "acc": np.zeros((int(np.ceil(duration_ms / _gb)), len(gtr_pos)),
                               dtype=np.float64)}

    use_reg = float(getattr(p, "spike_reg", 1.0)) > 1.0
    if use_reg:
        _kreg = int(round(float(p.spike_reg)))
        reg_acc = None
        reg_thr = None
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
    voltage_every = 1
    if record_voltage_every_ms is not None:
        cadence = Fraction(str(record_voltage_every_ms)) / Fraction(str(p.dt))
        if cadence <= 0 or cadence.denominator != 1:
            raise ValueError("voltage cadence must be a positive integral number of steps")
        voltage_every = int(cadence)
    vtrace = (None if requested_vrec is None
              else np.full((n_steps // voltage_every, len(requested_vrec)), np.nan,
                           dtype=np.float32))
    native_vtrace = (np.empty((n_steps, len(requested_vrec)), np.float32)
                     if record_voltage_native and requested_vrec is not None else None)
    native_state = (np.empty((n_steps, len(requested_vrec), 4), np.float32)
                    if record_voltage_native_state else None)
    voltage_columns = (
        np.empty(0, dtype=np.int64) if requested_vrec is None
        else np.arange(len(requested_vrec), dtype=np.int64)
        if input_recorder is None
        else np.searchsorted(engine_vrec, requested_vrec))

    gap_tap = None
    if eng.gap_tap_enabled:
        gap_tap = {"gap_operands": np.empty((n_steps, eng.gap_pair_count, 4), np.float32),
                   "gap_current": np.empty((n_steps, eng.gap_pair_count, 2), np.float32),
                   "gap_extra_eff": np.empty((n_steps, eng.n_gap), np.float32),
                   "gap_rows": eng.gap_rows.copy(),
                   "membrane_input_mv": np.empty((n_steps, len(requested_vrec)), np.float32)}
    words = eng.words
    psi_log = []
    cand = np.zeros((window, words), dtype=np.uint32)
    force = (np.zeros((window, words), dtype=np.uint32)
             if force_ms or force_spikes_fn is not None else None)

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
    # Design 1e20a949, Build C1: the rest rows take the membrane-drive mask and
    # kick vector, which CNS.run keeps free by refusing membrane_drive_rows
    # beside them; their events join each window's candidates below.
    from cns import rest_drive_plan, rest_drive_refuse_overlap
    rest_plan = (None if rest_drive is None else rest_drive_plan(
        rest_drive, N, eng.spike_at, eng.v_rest, graded, seed, dt))
    rest_drive_refuse_overlap(rest_plan, drive_idx)
    if rest_plan is not None:
        rest_bits = np.zeros(words, dtype=np.uint32)
        _set_bits(rest_bits[None, :], np.zeros(len(rest_plan["rows"]), dtype=np.int64),
                  rest_plan["rows"])
        rest_kicks = np.zeros(N, dtype=np.float32)
        rest_kicks[rest_plan["rows"]] = rest_plan["kick"]
        eng.set_membrane_drive(rest_bits, rest_kicks)

    gd_mask = None
    if graded_drive_max_rate_hz is not None and len(drive_idx) and drive_fn is None:
        raise NotImplementedError(
            "metal backend: graded_drive_max_rate_hz without drive_fn")
    start, stop = 0, n_steps
    g0 = eng.gpu_seconds()
    if branch is not None and resume is None:
        stop = int(branch["save_at_tick"])
    elif resume is not None:
        start = resume.at_tick
        eng.state_load(0)
        hs = resume.host
        counts, total_spikes = hs["counts"].copy(), int(hs["total_spikes"])
        if trace is not None:
            trace = hs["trace"].copy()
        if vtrace is not None:
            vtrace = hs["vtrace"].copy()
        if native_vtrace is not None:
            native_vtrace = hs["native_vtrace"].copy()
        rng.bit_generator.state = copy.deepcopy(hs["rng"])
        if use_reg:
            reg_acc = None if hs["reg_acc"] is None else hs["reg_acc"].copy()
            reg_thr = None if hs["reg_thr"] is None else hs["reg_thr"].copy()
        nr.restore(resume.record_host)
        resume.continuations += 1
    # Design f431563f B3: an exception that leaves the window loop carries the
    # part of the run that finished, the windows before w0, so a caller can
    # record a censored run. A run that completes never enters the handler.
    w0 = start
    inj_seg_i = -1
    try:
        for w0, nt in _run_windows(start, stop, window, inj_segs,
                                   None if light_levels is None else light_levels.every):
            if inj_segs is not None:
                while inj_seg_i + 1 < len(inj_segs) and inj_segs[inj_seg_i + 1][0] <= w0:
                    inj_seg_i += 1
                    eng.set_inject(inj_segs[inj_seg_i][2])
            if sw_mask is not None:
                if any(w0 < edge < w0 + nt for edge in sw_set):
                    raise ValueError("silence_window edges must fall on engine windows")
                if w0 in sw_set:
                    eng.set_spike_at(sw_rows, sw_set[w0])
            if light_levels is not None and w0 % light_levels.every == 0:
                # design e2f5f784 P4-3: the per-row light level, renewed before drive_fn
                light_levels.upload(w0)
            cand[:nt] = 0
            cand_k, cand_id = [], []
            any_force = False
            if force is not None:
                force[:nt] = 0
            if force_spikes_fn is not None:
                # Each window begins at a callback tick. The callback sees only
                # prior ticks; forced rows enter this tick, not the whole window.
                # A bytes-backed snapshot is read-only and cannot alter feedback.
                callback_counts = np.frombuffer(
                    fb_accum.astype(np.int64, copy=False).tobytes(order="C"),
                    dtype=np.int64)
                returned = force_spikes_fn(w0 * dt, callback_counts)
                if (type(returned) is not np.ndarray or returned.ndim != 1
                        or returned.dtype != np.dtype(np.int64)):
                    raise ValueError("force_spikes_fn must return an exact 1-D np.int64 array")
                if len(returned) and ((returned < 0).any() or (returned >= N).any()):
                    raise IndexError("force_spikes_fn returned row out of bounds")
                if len(returned) != len(np.unique(returned)):
                    raise ValueError("force_spikes_fn returned duplicate rows")
                if len(returned):
                    _set_bits(force, np.zeros(len(returned), dtype=np.int64), returned)
                    any_force = True
                fb_accum[:] = 0
            if drive_fn is not None and w0 % fb_every == 0:
                # windows are fb_every ticks, so the callback lands on tick w0
                # and the rates hold for the whole window, as in run()
                idx_new, rate_new = drive_fn(w0 * dt, fb_accum)
                drive_idx = np.asarray(idx_new, dtype=np.int64)
                drive_rate = np.asarray(rate_new, dtype=np.float64)
                if drive_idx.shape != drive_rate.shape:
                    raise ValueError(
                        "drive_fn indices and rates must have matching shapes")
                rest_drive_refuse_overlap(rest_plan, drive_idx)
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
                    if _rc_rows is not None and _rc_rows.intersection(driven.tolist()):
                        raise ValueError("row_current names a row that set_clamp holds")
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
            if rest_plan is not None:
                # one (ticks, rows) block per window, filled in the order
                # run()'s per-step draws consume the rest stream
                u = rest_plan["gen"].random((nt, len(rest_plan["rows"])))
                rest_plan["draws"] += int(u.size)
                kk, ee = np.nonzero(u < rest_plan["prob"])
                if len(kk):
                    rest_plan["events"] += int(len(kk))
                    cand_k.append(kk)
                    cand_id.append(rest_plan["rows"][ee])
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
            if ggw is not None:
                want = bool(ggw["start_ms"] <= w0 * dt < ggw["stop_ms"])
                if want != ggw["on"]:
                    target = ggw["commanded"] if want else ggw["base"]
                    eng.set_graded_gain(target)
                    gain_live = eng.graded_gain()
                    if not np.array_equal(gain_live.view(np.uint32),
                                          target.view(np.uint32)):
                        raise RuntimeError(
                            "graded gain readback differs from the upload")
                    ggw["on"] = want
                    ggw["transition_ticks"].append(int(w0))
                if ggw["on"]:
                    ggw["windows_commanded"] += 1
            ticks, ids = eng.run(w0, nt, cand if any_cand else None,
                                 force if any_force else None)
            if sw_mask is not None and sw_on <= w0 < sw_off and len(ids):
                sw_spikes += int(sw_mask[ids].sum())
            if tap_member is not None and len(ids):
                tapped = tap_member[ids]
                if tapped.any():
                    spike_tap.extend(sorted(zip(
                        (w0 + ticks[tapped].astype(np.int64)).tolist(),
                        ids[tapped].astype(np.int64).tolist())))
            if nr is not None:
                nr.consume(w0, nt, ticks, ids)
            if eng.psi is not None:
                _pk, _pj, _pf, _pG, _pgi = eng.psi_log(nt)
                if len(_pk):
                    psi_log.append((w0 + _pk.astype(np.int64), eng.psi["rows"][_pj],
                                    _pf, _pG, _pgi))
            if op_tap is not None:
                _otap = eng.opsin_tap(nt)
                for _c, _k in enumerate(OPSIN_TAP_FIELDS):
                    op_tap[_k][w0:w0 + nt] = _otap[:, :, _c]
            if hd_tap is not None:
                hd_tap[w0:w0 + nt] = eng.holddown_tap(nt)
            state_tap = (None if engine_vrec is None
                         else eng.state_tap(nt))
            if vtrace is not None:
                chosen = np.flatnonzero((np.arange(w0, w0 + nt) + 1) % voltage_every == 0)
                dest = (w0 + chosen + 1) // voltage_every - 1
                vtrace[dest] = state_tap[chosen][:, voltage_columns, 0]
            if native_vtrace is not None:
                native_vtrace[w0:w0 + nt] = state_tap[:, voltage_columns, 0]
            if native_state is not None:
                native_state[w0:w0 + nt] = state_tap[:, voltage_columns, :]
            if lamina_recorder is not None:
                lamina_recorder.consume(w0, state_tap[:, voltage_columns, 0])
            if gap_tap is not None:
                sample = eng.gap_tap(nt)
                for key in ("gap_operands", "gap_current", "gap_extra_eff"):
                    gap_tap[key][w0:w0+nt] = sample[key]
                gap_tap["membrane_input_mv"][w0:w0+nt] = state_tap[:, voltage_columns, 1]
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
            if grr is not None:
                _graded_release_consume(grr, w0, nt, release_tap, gain_live,
                                        g_every)
            if gtr is not None and len(gtr["cols"]):
                _gk = np.flatnonzero((w0 + np.arange(nt)) % g_every == 0)
                if len(_gk):
                    _post = gtr["gain"][None, :] * release_tap[_gk][:, gtr["cols"]]
                    np.add.at(gtr["acc"], (w0 + _gk) // gtr["steps_per_bin"],
                              _post.astype(np.float64))
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
    except BaseException as exc:
        exc.partial = {"completed_ticks": int(w0),
                       "trace": None if trace is None else trace[: w0 // steps_per_bin],
                       "rec": rec}
        raise

    if branch is not None and resume is None:
        eng.state_save(0)
        host = {"counts": counts.copy(), "total_spikes": total_spikes,
                "trace": None if trace is None else trace.copy(),
                "vtrace": None if vtrace is None else vtrace.copy(),
                "native_vtrace": None if native_vtrace is None else native_vtrace.copy(),
                "rng": copy.deepcopy(rng.bit_generator.state),
                "reg_acc": (None if not use_reg or reg_acc is None else reg_acc.copy()),
                "reg_thr": (None if not use_reg or reg_thr is None else reg_thr.copy())}
        handle = BranchHandle(eng, stop, n_steps, config, _force_digest(force_ms, stop, dt),
                              host, nr.snapshot(), nr.forced.copy())
        return {"branch_handle": handle, "completed_ticks": stop,
                "prefix_state_digest": handle.prefix_state_digest,
                "n_spikes": total_spikes, "duration_ms": duration_ms,
                "backend": "metal", "gpu_seconds": eng.gpu_seconds() - g0}

    out = {
        "counts": counts[rec],
        **({"t_first_ms": np.where(t_first >= 0, t_first * dt, np.nan)}
           if t_first is not None else {}),
        "rates_hz": counts[rec] / (duration_ms / 1000.0),
        "n_spikes": total_spikes,
        "duration_ms": duration_ms,
        "recorded": rec,
        "backend": "metal",
        "gpu_seconds": eng.gpu_seconds() - g0,
        "fold_overflows": eng.fold_overflows(),
        "bin_max": eng.bin_max(),
        "touched_per_tick": eng.touched_per_tick(),
        "bin_hist": eng.bin_hist(),
    }
    if ggw is not None:
        out["graded_gain_window"] = {
            "rows": ggw["rows"].tolist(), "positions": ggw["positions"].tolist(),
            "factor": ggw["factor"], "factor_f32": float(np.float32(ggw["factor"])),
            "start_ms": ggw["start_ms"], "stop_ms": ggw["stop_ms"],
            "start_tick": ggw["start_tick"], "stop_tick": ggw["stop_tick"],
            "windows_commanded": ggw["windows_commanded"],
            "transition_ticks": ggw["transition_ticks"],
            "base_gain_scaled_f32": ggw["base"][ggw["positions"]].tolist(),
            "commanded_gain_scaled_f32": ggw["commanded"][ggw["positions"]].tolist()}
    if grr is not None:
        out["graded_release_record"] = _graded_release_receipt(grr, dt)
    if membrane_drive_receipt is not None:
        membrane_drive_receipt["drawn_unique_row_tick_events"] = int(
            membrane_drive_events)
        out["membrane_drive"] = membrane_drive_receipt
    if rest_plan is not None:
        out["rest_drive"] = dict(
            rest_plan["receipt"], steps=int(n_steps),
            draws_total=int(rest_plan["draws"]),
            membrane_drive_events=int(rest_plan["events"]), backend="metal")
    if getattr(p, "release_resource_rows", None) is not None:
        out["release_log"], out["release_log_sha256"] = eng.resource_log()
    if eng.psi is not None:
        out["presynaptic_inhibition"], out["presynaptic_inhibition_log"] = _psi_out(
            eng.psi, psi_log)
    if sw_mask is not None:
        from cns import silence_window_receipt
        out["silence_window"] = silence_window_receipt(sw_mask, sw_on, sw_off, dt, sw_spikes)
    if eng.coin2 is not None:
        out["coincidence_triples"] = eng.coin2_receipt()
    elif eng.n_coin:
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
    if gtr is not None:
        out["graded_trace"] = {
            "rows": gtr["rows"], "trace": gtr["acc"].astype(np.float32),
            "bin_ms": gtr["bin_ms"], "graded_every": int(g_every),
            "readback": ("k_graded release tap (MetalEngine.reltap) times "
                         "graded_gain_scaled, float32, summed per bin over the "
                         "graded updates")}
    if vtrace is not None:
        out["voltage"] = vtrace
        out["voltage_idx"] = requested_vrec
    if native_vtrace is not None:
        out["voltage_native"] = native_vtrace
    if native_state is not None:
        out["voltage_native_state"] = native_state
        out["voltage_native_state_fields"] = (
            "v_post_integration", "v_in", "g_exc_accumulator", "g_inh_accumulator")
    if lamina_recorder is not None:
        out["lamina_recording"] = lamina_recorder.finish()
    if light_levels is not None:
        out["light_levels"] = light_levels.receipt()
    if gap_tap is not None:
        out["gap_tap"] = gap_tap
    if input_recorder is not None:
        out["input_recording"] = input_recorder.finish()
    if nr is not None:
        out["native"] = nr.finish(eng)
    if _op is not None:
        out["opsin"] = opsin_out(_op, op_tap)
    if hd_tap is not None:
        from cns import holddown_out
        out["holddown"] = holddown_out(eng._keep["holddown_rows"], eng.holddown_state(),
                                       hd_tap)
    if _rc_receipt is not None:
        out["row_current"] = _rc_receipt
    if _sl is not None:
        # k_fold adds a spike's increments at emission into its arrival slot,
        # so the accumulators hold every emitted spike and the ring holds the
        # increments that had not arrived by the last tick
        sg = eng.read_slow()
        emitted = int(counts[_sl["receipt"]["rows"]].sum())
        out["slow_inh"] = slow_inh_out(
            _sl, sg["acc_fast"], sg["acc_slow"], sg["intg"], sg["g"],
            sg["ring"].sum(axis=0, dtype=np.float64), sg["rec_acc"], emitted,
            "emitted", emitted, eng.gscale)
    if _se is not None:
        sg = eng.read_serotonin()
        out["serotonin"] = {
            "source_rows": int(len(_se["source"])),
            "target_rows": int(len(_se["target"])),
            "source_spikes": sg["spikes"],
            "tau_ms": float(_se["tau"]),
            "gain_mv_per_hz": float(_se["gain"]),
            "state_final_hz": float(sg["state"]),
            "state_max_hz": float(sg["max"]),
            "state_mean_applied_hz": float(sg["applied"] / max(1, n_steps)),
            "init_hz": float(_se["init"]),
            "bin_ms": float(_se["spb"] * dt),
            "state_bin_end_hz": sg["bins"].tolist(),
        }
    if _os is not None:   # design 8b6e0441, Build 1
        from cns import octopamine_state_receipt
        sg = eng.read_octopamine_state()
        out["octopamine_state"] = octopamine_state_receipt(
            _os, sg["state"], sg["max"], sg["applied"], sg["spikes"], sg["bins"],
            n_steps, dt)
        out["octopamine_state"]["route"] = "engine (cns_engine_octopamine_state_config)"
    if _oa is not None:
        out["octopamine_offset"] = {"offset_mv": float(_oa[1]), "target_rows": int(len(_oa[0])),
                                    "route": oa_route}
    if resume is not None:
        out["branch"] = {"at_tick": resume.at_tick,
                         "prefix_state_digest": resume.prefix_state_digest,
                         "continuation": resume.continuations}
    return out
