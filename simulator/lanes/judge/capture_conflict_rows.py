#!/usr/bin/env python3
"""Readers for registry rows already present in a title-card capture.

The title card stores summaries, not the capture itself.  These readers run on
that one in-memory capture before it is discarded, so adding a row costs no
new simulation and cannot change the body.  The abdomen window and transform
match lane D's committed same-window instrument.  The knee read matches lane
A's corrected CD2-AF convention at the walking posture: positive FTi is
flexion, hence interior-angle change is the negative of FTi change.
"""
from __future__ import annotations

from typing import Any

import numpy as np

import motormap as MM
from flyscore import layout as L
from lanes.D import abd_real_lateral_output as ABD


ABDOMEN_START_MS = 500.0
ABDOMEN_SAMPLE_MS = 12.5
ABDOMEN_FRAMES = 234
ABDOMEN_BAND_DEG = (4.653, 11.629)
CONDUCTION_BAND_MS = (2.8, 3.3)
KNEE_TRANSIENT_MS = 500.0
KNEE_MIN_EPISODE_MS = 20.0
KNEE_EXCURSION_BAND_DEG = (4.05, 12.60)
TIP_ORDER = ("lf", "lh", "lm", "rf", "rh", "rm")
EXPECTED_KNEE_SIGN = {
    "lf": -1,
    "lm": +1,
    "lh": +1,
    "rf": -1,
    "rm": +1,
    "rh": +1,
}


def _exact_ticks(ms: float, tick_ms: float, name: str) -> int:
    value = float(ms) / float(tick_ms)
    rounded = int(round(value))
    if not np.isclose(value, rounded, rtol=0.0, atol=1e-10):
        raise ValueError(f"{name} {ms} ms is not aligned to the {tick_ms} ms tick")
    return rounded


def _episodes(mask: np.ndarray, min_ticks: int) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False])).astype(int)
    delta = np.diff(padded)
    starts = np.flatnonzero(delta == 1)
    stops = np.flatnonzero(delta == -1)
    return [(int(a), int(b)) for a, b in zip(starts, stops) if b - a >= min_ticks]


def _egocentric_abdomen(sampled: list[tuple[Any, ...]]) -> np.ndarray:
    root = np.asarray([row[0] for row in sampled], dtype=np.float64)
    quat = np.asarray([row[1] for row in sampled], dtype=np.float64)
    abdomen = np.asarray([row[18] for row in sampled], dtype=np.float64)
    if root.shape != (ABDOMEN_FRAMES, 3):
        raise ValueError(f"root shape {root.shape} != {(ABDOMEN_FRAMES, 3)}")
    if quat.shape != (ABDOMEN_FRAMES, 4):
        raise ValueError(f"quaternion shape {quat.shape} != {(ABDOMEN_FRAMES, 4)}")
    if abdomen.shape != (ABDOMEN_FRAMES, 3):
        raise ValueError(f"abdomen shape {abdomen.shape} != {(ABDOMEN_FRAMES, 3)}")
    if not (np.isfinite(root).all() and np.isfinite(quat).all()
            and np.isfinite(abdomen).all()):
        raise ValueError("abdomen reader received non-finite capture values")

    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    forward = np.stack(
        [1 - 2 * (y * y + z * z), 2 * (x * y + w * z)], axis=1)
    norm = np.linalg.norm(forward, axis=1, keepdims=True)
    if np.any(norm <= 0.0):
        raise ValueError("abdomen reader received a zero-length heading")
    # Match card_regrade_qh1.egocentric_tips exactly.  The epsilon is part of
    # that established transform, even though a valid quaternion leaves the
    # norm far from zero.
    forward = forward / (norm + 1e-9)
    right = np.stack([forward[:, 1], -forward[:, 0]], axis=1)
    rel = abdomen[:, :2] - root[:, :2]
    points = np.full((ABDOMEN_FRAMES, L.N_PARTS, 2), np.nan, dtype=float)
    points[:, L.THORAX, :] = 0.0
    points[:, L.ABDOMEN, :] = np.stack(
        [np.sum(rel * right, axis=1), -np.sum(rel * forward, axis=1)], axis=1)
    return points


def abdomen_lateral_excursion_deg(
        capture: list[tuple[Any, ...]], tick_ms: float) -> float:
    """P5-to-p95 thorax-to-abdomen angle in the physical-data window."""
    start = _exact_ticks(ABDOMEN_START_MS, tick_ms, "abdomen window start")
    step = _exact_ticks(ABDOMEN_SAMPLE_MS, tick_ms, "abdomen sample interval")
    indices = start + np.arange(ABDOMEN_FRAMES, dtype=np.int64) * step
    if int(indices[-1]) >= len(capture):
        raise ValueError("capture ends before the abdomen comparison window")
    sampled = [capture[int(i)] for i in indices]
    if not all(len(row) > 18 and len(row[18]) == 3 for row in sampled):
        raise ValueError("capture lacks c_abdomen6 at element 18")
    result, error = ABD.sequence_lateral_metrics(_egocentric_abdomen(sampled))
    if error is not None or result is None or result.get("excursion") is None:
        raise ValueError(f"abdomen reference reader rejected the capture: {error}")
    return float(np.degrees(result["excursion"]))


def knee_stance_rows(
        capture: list[tuple[Any, ...]], tick_ms: float) -> dict[str, dict[str, float | int | None]]:
    """Per-leg median interior-angle change over stance episodes.

    Stance is the CD2-AF model-side read: tip clearance no more than 0.05 mm
    above that leg's fifth percentile, after 500 ms, for at least 20 ms.
    """
    transient = _exact_ticks(KNEE_TRANSIENT_MS, tick_ms, "knee transient")
    min_ticks = _exact_ticks(KNEE_MIN_EPISODE_MS, tick_ms, "knee episode floor")
    if len(capture) <= transient:
        raise ValueError("capture ends inside the knee transient")
    if not all(len(row) > 12 for row in capture[transient:]):
        raise ValueError("capture lacks achieved joints at element 12")
    q = np.degrees(np.asarray([row[12] for row in capture], dtype=float)[transient:])
    tipz = np.asarray([row[5] for row in capture], dtype=float)[transient:]
    if q.ndim != 2 or q.shape[1] != len(MM.DOF_NAMES):
        raise ValueError(f"achieved-joint shape {q.shape} is not (*, {len(MM.DOF_NAMES)})")
    if tipz.ndim != 2 or tipz.shape[1] != len(TIP_ORDER):
        raise ValueError(f"tip-height shape {tipz.shape} is not (*, {len(TIP_ORDER)})")
    if not (np.isfinite(q).all() and np.isfinite(tipz).all()):
        raise ValueError("knee reader received non-finite capture values")

    clear = tipz - np.percentile(tipz, 5, axis=0, keepdims=True)
    out: dict[str, dict[str, float | int | None]] = {}
    for leg in MM.LEGS:
        tip_col = TIP_ORDER.index(leg)
        eps = _episodes(clear[:, tip_col] <= 0.05, min_ticks)
        angle = q[:, MM.DOF_INDEX[f"{leg}_FTi_pitch"]]
        changes = [float(angle[b - 1] - angle[a]) for a, b in eps]
        # At the occupied posture +FTi closes the anatomical knee.  The
        # registry uses interior angle, whose sign is therefore opposite.
        interior = None if not changes else -float(np.median(changes))
        out[leg] = {
            "n_stance_episodes": len(eps),
            "interior_angle_change_deg": interior,
            "expected_sign": EXPECTED_KNEE_SIGN[leg],
        }
    return out


def capture_rows(capture: list[tuple[Any, ...]], tick_ms: float) -> dict:
    return {
        "abdomen_lateral_excursion_deg": abdomen_lateral_excursion_deg(
            capture, tick_ms),
        "knee_stance": knee_stance_rows(capture, tick_ms),
    }


def built_delay_mean_ms(net_cache: dict) -> float:
    """Mean delivered presynaptic delay on the network the run actually used."""
    if len(net_cache) != 1:
        raise ValueError(f"card expected one built network, found {len(net_cache)}")
    net = next(iter(net_cache.values()))
    vector = getattr(net.p, "delay_per_neuron", None)
    value = float(net.p.delay) if vector is None else float(np.mean(vector))
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"built delay mean is invalid: {value!r}")
    return value
