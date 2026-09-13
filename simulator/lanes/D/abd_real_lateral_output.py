"""Audit lateral output of a tracked body-axis proxy in a real-fly dataset.

This is the seed-free ``ABD-REAL-LATERAL-OUTPUT`` instrument. It uses the same
thorax and abdomen landmarks and the same image-coordinate convention as
``flyscore.metrics.posture_variability``.  The 3,000 sequences used to build
the scorer's reference distributions are reconstructed and excluded, leaving
25,059 reference-disjoint sequences. The landmark-layout inference used a
separate 300-row systematic sample, so this is not independent of that step.

The resulting angle is the top-view orientation of the tracked thorax-to-
abdomen vector. It constrains that tracked axis's lateral output only. It is not a
per-joint range, does not observe dorsoventral pitch, and cannot allocate the
motion among the model's five abdomen joints.  The landmark layout is inferred
rather than certified; see ``flyscore/layout.py``.

The source identity is checked against the public Zenodo artifact before HDF5
is opened.  There is no command-line bypass for that check.

    PYTHONPATH=. .venv/bin/python lanes/D/abd_real_lateral_output.py \
        --h5 data/Fly_DLC_behavior_tracking.h5 \
        --out lanes/D/2026-09-05-abd-real-lateral-output.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import BinaryIO

import h5py
import numpy as np
import scipy
from scipy.stats import rankdata

from flyscore import layout as L


EXPERIMENT = "ABD-REAL-LATERAL-OUTPUT"
ZENODO_RECORD = "10.5281/zenodo.11002776"
EXPECTED_FILE_SIZE = 3_361_694_720
EXPECTED_MD5 = "0bb2ecb57d504fe13a1fe8b8b7090511"
EXPECTED_DATASET_SHAPE = (28_059, 234, 64)
EXPECTED_DATASET_DTYPE = np.dtype("float64")
REFERENCE_SEED = 20_260_805
REFERENCE_COUNT = 3_000
ANALYSIS_COUNT = 25_059
REFERENCE_INDEX_SHA256 = (
    "93f524fb59b12e6dd02b1c597ffae5134b6a6828115efbb5b447e25e6069fa3a"
)
ANALYSIS_INDEX_SHA256 = (
    "a6936826ccb23ccf47b413f458a60cc10cb821a1a999e77ec2c05213d2d0ed7d"
)
METRIC_NAMES = (
    "median",
    "p05",
    "p95",
    "excursion",
    "max_central90_side_excursion",
    "angle_std",
    "excursion_to_std_ratio",
)
SPLIT_LABEL = b"ABD-REAL-LATERAL-OUTPUT-SPLIT-v1\x00"
SPLIT_PASS_ABS_D = 0.1
SHAPE_RESPONSE_MIN_MEDIAN_ABS_DELTA = 0.25


class SourceIdentityError(RuntimeError):
    """The input bytes do not match the declared public source."""


class SourceFormatError(RuntimeError):
    """The authenticated input does not have the expected data layout."""


class ExecutionIdentityError(RuntimeError):
    """The reader is not executing from a committed file identity."""


def _require(condition: bool, message: str, error=SourceFormatError) -> None:
    if not condition:
        raise error(message)


def _int64_sha256(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype=np.dtype("<i8"))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _instrument_receipt() -> dict:
    """Bind a formal read to one committed copy of this lane instrument."""
    runner = Path(__file__).resolve()
    root = runner.parents[2]
    relative = runner.relative_to(root).as_posix()
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    _require(
        tracked.returncode == 0,
        f"instrument is not tracked: {relative}",
        ExecutionIdentityError,
    )
    clean = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", relative],
        cwd=root,
        check=False,
    )
    _require(
        clean.returncode == 0,
        f"instrument differs from HEAD: {relative}",
        ExecutionIdentityError,
    )

    def rev_parse(revision: str) -> str:
        completed = subprocess.run(
            ["git", "rev-parse", revision],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        _require(
            completed.returncode == 0,
            f"git cannot resolve {revision}",
            ExecutionIdentityError,
        )
        return completed.stdout.strip()

    return {
        "runner_repo_path": relative,
        "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
        "execution_commit": rev_parse("HEAD"),
        "origin_main_at_execution": rev_parse("origin/main"),
        "runner_matches_execution_commit": True,
    }


def reference_and_analysis_indices(
    n_total: int = EXPECTED_DATASET_SHAPE[0],
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the scorer's fixed 3,000 rows and their complement."""
    _require(
        type(n_total) is int and n_total == EXPECTED_DATASET_SHAPE[0],
        f"dataset sequence count {n_total!r} != {EXPECTED_DATASET_SHAPE[0]}",
    )
    rng = np.random.default_rng(REFERENCE_SEED)
    reference = np.sort(
        rng.choice(n_total, size=REFERENCE_COUNT, replace=False).astype(
            np.int64, copy=False
        )
    )
    keep = np.ones(n_total, dtype=bool)
    keep[reference] = False
    analysis = np.flatnonzero(keep).astype(np.int64, copy=False)

    _require(reference.size == REFERENCE_COUNT, "reference index count drift")
    _require(analysis.size == ANALYSIS_COUNT, "analysis index count drift")
    _require(
        _int64_sha256(reference) == REFERENCE_INDEX_SHA256,
        "reference index identity drift",
    )
    _require(
        _int64_sha256(analysis) == ANALYSIS_INDEX_SHA256,
        "analysis index identity drift",
    )
    return reference, analysis


def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def _verify_open_source(
    source: BinaryIO,
    *,
    expected_size: int = EXPECTED_FILE_SIZE,
    expected_md5: str = EXPECTED_MD5,
) -> tuple[dict, tuple[int, int, int, int, int]]:
    """Hash one already-open inode and reject mutation during the read."""
    before = os.fstat(source.fileno())
    _require(
        stat.S_ISREG(before.st_mode),
        "source is not a regular file",
        SourceIdentityError,
    )
    _require(
        before.st_size == expected_size,
        f"source byte size {before.st_size} != {expected_size}",
        SourceIdentityError,
    )

    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    byte_count = 0
    source.seek(0)
    while True:
        chunk = source.read(8 * 1024 * 1024)
        if not chunk:
            break
        md5.update(chunk)
        sha256.update(chunk)
        byte_count += len(chunk)
    after = os.fstat(source.fileno())
    identity = _stat_identity(before)
    _require(
        _stat_identity(after) == identity,
        "source identity changed while hashing",
        SourceIdentityError,
    )
    _require(
        byte_count == expected_size,
        f"hashed byte count {byte_count} != {expected_size}",
        SourceIdentityError,
    )
    observed_md5 = md5.hexdigest()
    _require(
        observed_md5 == expected_md5,
        f"source MD5 {observed_md5} != {expected_md5}",
        SourceIdentityError,
    )
    return (
        {
            "expected_bytes": int(expected_size),
            "observed_bytes": int(byte_count),
            "expected_md5": expected_md5,
            "observed_md5": observed_md5,
            "observed_sha256": sha256.hexdigest(),
        },
        identity,
    )


def verify_source(
    path: Path | str,
    *,
    expected_size: int = EXPECTED_FILE_SIZE,
    expected_md5: str = EXPECTED_MD5,
) -> dict:
    """Verify a source path; override arguments exist for small test fixtures."""
    with Path(path).open("rb") as source:
        receipt, _ = _verify_open_source(
            source, expected_size=expected_size, expected_md5=expected_md5
        )
    return receipt


def inspect_dataset(h5: h5py.File) -> tuple[h5py.Dataset, dict]:
    """Require and report the exact published dataset shape and dtype."""
    _require("data" in h5, "HDF5 dataset /data is absent")
    dataset = h5["data"]
    _require(isinstance(dataset, h5py.Dataset), "HDF5 /data is not a dataset")
    _require(
        tuple(dataset.shape) == EXPECTED_DATASET_SHAPE,
        f"HDF5 /data shape {tuple(dataset.shape)} != {EXPECTED_DATASET_SHAPE}",
    )
    _require(
        dataset.dtype == EXPECTED_DATASET_DTYPE,
        f"HDF5 /data dtype {dataset.dtype} != {EXPECTED_DATASET_DTYPE}",
    )
    return dataset, {
        "path": "/data",
        "shape": list(dataset.shape),
        "dtype": str(dataset.dtype),
        "chunks": None if dataset.chunks is None else list(dataset.chunks),
        "compression": dataset.compression,
        "root_keys": sorted(h5.keys()),
        "root_attribute_keys": sorted(h5.attrs.keys()),
        "dataset_attribute_keys": sorted(dataset.attrs.keys()),
        "animal_arena_trial_identifiers_available": False,
    }


def _metric_matrix_from_angles(angles: np.ndarray) -> np.ndarray:
    """Compute the fixed summaries for one or more unwrapped angle traces."""
    values = np.asarray(angles, dtype=float)
    _require(
        values.ndim == 2 and values.shape[1] == L.N_FRAMES,
        f"angle matrix shape {values.shape} is invalid",
    )
    _require(np.isfinite(values).all(), "angle matrix contains non-finite values")
    percentiles = np.percentile(
        values, [5.0, 50.0, 95.0], axis=1, method="linear"
    ).T
    p05 = percentiles[:, 0]
    median = percentiles[:, 1]
    p95 = percentiles[:, 2]
    excursion = p95 - p05
    angle_std = np.std(values, axis=1)
    ratio = np.full(angle_std.shape, np.nan, dtype=float)
    ratio_available = np.isfinite(angle_std) & (angle_std > 0.0)
    np.divide(excursion, angle_std, out=ratio, where=ratio_available)
    matrix = np.column_stack(
        (
            median,
            p05,
            p95,
            excursion,
            np.maximum(median - p05, p95 - median),
            angle_std,
            ratio,
        )
    )
    _require(
        np.isfinite(matrix[:, :-1]).all(),
        "finite angles produced a non-finite angular summary",
    )
    return matrix


def sequence_lateral_metrics(
    sequence: np.ndarray,
) -> tuple[dict[str, float | None] | None, str | None]:
    """Return one sequence's unwrapped thorax-to-abdomen angle summaries."""
    values = np.asarray(sequence)
    if values.shape == (L.N_FRAMES, L.N_PARTS * 2):
        points = values.reshape(L.N_FRAMES, L.N_PARTS, 2)
    elif values.shape == (L.N_FRAMES, L.N_PARTS, 2):
        points = values
    else:
        raise SourceFormatError(
            f"sequence shape {values.shape} is not "
            f"{(L.N_FRAMES, L.N_PARTS * 2)} or "
            f"{(L.N_FRAMES, L.N_PARTS, 2)}"
        )

    vector = points[:, L.ABDOMEN, :] - points[:, L.THORAX, :]
    if not np.isfinite(vector).all():
        return None, "nonfinite_thorax_or_abdomen"
    if np.any(np.sum(vector * vector, axis=1) <= 0.0):
        return None, "zero_length_thorax_abdomen_vector"

    # x is lateral; low-to-high y is anterior-to-posterior.  This is the
    # orientation convention in flyscore.metrics.posture_variability.
    angle = np.unwrap(np.arctan2(vector[:, 0], vector[:, 1]))
    row = _metric_matrix_from_angles(angle[None, :])[0]
    result = {
        name: (None if not np.isfinite(row[column]) else float(row[column]))
        for column, name in enumerate(METRIC_NAMES)
    }
    return result, None


def _summarise_matrix(matrix: np.ndarray, scale: float = 1.0) -> dict:
    _require(
        matrix.ndim == 2 and matrix.shape[1] == len(METRIC_NAMES),
        f"metric matrix shape {matrix.shape} is invalid",
    )
    _require(matrix.shape[0] > 0, "no valid sequences remain")
    out = {}
    for column, name in enumerate(METRIC_NAMES):
        column_scale = (
            1.0 if name == "excursion_to_std_ratio" else float(scale)
        )
        values = matrix[:, column] * column_scale
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            out[name] = {
                "n": 0,
                "unavailable": int(matrix.shape[0]),
                "p5": None,
                "median": None,
                "p95": None,
            }
            continue
        out[name] = {
            "n": int(finite.size),
            "unavailable": int(matrix.shape[0] - finite.size),
            "p5": float(np.percentile(finite, 5.0, method="linear")),
            "median": float(np.median(finite)),
            "p95": float(np.percentile(finite, 95.0, method="linear")),
        }
    return out


def _descriptive_correlations(x: np.ndarray, y: np.ndarray) -> dict:
    """Pearson and average-rank Spearman coefficients on finite pairs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    _require(x.shape == y.shape and x.ndim == 1, "correlation shape mismatch")
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2:
        return {
            "n": int(x.size),
            "pearson_r": None,
            "spearman_rho": None,
            "unavailable_reason": "fewer than two finite pairs",
        }
    if np.ptp(x) <= 0.0 or np.ptp(y) <= 0.0:
        return {
            "n": int(x.size),
            "pearson_r": None,
            "spearman_rho": None,
            "unavailable_reason": "at least one input is constant",
        }
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(
        np.corrcoef(rankdata(x, method="average"),
                    rankdata(y, method="average"))[0, 1]
    )
    _require(
        np.isfinite((pearson, spearman)).all(),
        "correlation produced a non-finite coefficient",
    )
    return {
        "n": int(x.size),
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "unavailable_reason": None,
    }


def _quality_diagnostics(
    valid_indices: np.ndarray,
    matrix: np.ndarray,
) -> dict:
    """Report extreme unwrap tails without changing the primary population."""
    excursion_column = METRIC_NAMES.index("excursion")
    std_column = METRIC_NAMES.index("angle_std")
    excursion = matrix[:, excursion_column]
    excursion_deg = np.rad2deg(excursion)
    thresholds = {}
    for threshold in (180.0, 360.0, 720.0):
        flagged = excursion_deg > threshold
        thresholds[str(int(threshold))] = {
            "count": int(flagged.sum()),
            "fraction_of_valid_rows": float(np.mean(flagged)),
            "dataset_indices": valid_indices[flagged].tolist(),
        }

    cutoff = float(np.percentile(excursion, 99.0, method="linear"))
    keep = excursion <= cutoff
    _require(keep.any() and (~keep).any(), "top-one-percent diagnostic split failed")
    return {
        "primary_population_excludes_flagged_rows": False,
        "extreme_excursion_degrees_strictly_above": thresholds,
        "interpretation": (
            "These tails flag possible tracking or angle-unwrapping pathologies. "
            "They do not establish anatomical motion."
        ),
        "top_one_percent_excursion_trim_descriptive_only": {
            "status": (
                "added after a BUILD-phase exploratory scan; not a protected "
                "selection rule"
            ),
            "cutoff_radians": cutoff,
            "cutoff_degrees": float(np.rad2deg(cutoff)),
            "kept_rows": int(keep.sum()),
            "excluded_rows": int((~keep).sum()),
            "excursion_vs_angle_std": _descriptive_correlations(
                excursion[keep], matrix[keep, std_column]
            ),
        },
    }


def _sample_summary(values: np.ndarray) -> dict:
    """Summarise finite values for a split-half comparison."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"n": 0, "mean": None, "median": None, "sd": None}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "sd": float(np.std(finite, ddof=1)) if finite.size > 1 else None,
    }


def _stable_split_indices(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split indices by a fixed SHA-256 ordering, without drawing a seed."""
    values = np.asarray(indices, dtype=np.int64)
    _require(values.ndim == 1 and values.size >= 2, "split indices are invalid")
    _require(
        np.unique(values).size == values.size,
        "split indices contain duplicates",
    )
    keys = [
        hashlib.sha256(
            SPLIT_LABEL + int(index).to_bytes(8, "little", signed=False)
        ).digest()
        for index in values
    ]
    order = sorted(range(values.size), key=lambda pos: (keys[pos], int(values[pos])))
    cut = values.size // 2
    half_a = np.sort(values[np.asarray(order[:cut], dtype=np.int64)])
    half_b = np.sort(values[np.asarray(order[cut:], dtype=np.int64)])
    _require(
        np.intersect1d(half_a, half_b, assume_unique=True).size == 0,
        "split halves overlap",
    )
    _require(
        np.array_equal(np.sort(np.concatenate((half_a, half_b))), np.sort(values)),
        "split halves do not reconstruct the input",
    )
    return half_a, half_b


def _split_half_stability(
    analysis_indices: np.ndarray,
    valid_indices: np.ndarray,
    matrix: np.ndarray,
) -> dict:
    """Compare deterministic disjoint halves using the scorer's Cohen-d gate."""
    _require(
        valid_indices.shape == (matrix.shape[0],),
        "valid index and metric row counts differ",
    )
    half_a, half_b = _stable_split_indices(analysis_indices)
    in_a = np.isin(valid_indices, half_a, assume_unique=True)
    in_b = np.isin(valid_indices, half_b, assume_unique=True)
    _require(np.all(in_a ^ in_b), "a valid sequence lacks one split assignment")

    metrics = {}
    new_metric_names = {
        "excursion",
        "max_central90_side_excursion",
        "excursion_to_std_ratio",
    }
    new_abs_d: list[float] = []
    for column, name in enumerate(METRIC_NAMES):
        summary_a = _sample_summary(matrix[in_a, column])
        summary_b = _sample_summary(matrix[in_b, column])
        pooled_sd = None
        d = None
        if (
            summary_a["sd"] is not None
            and summary_b["sd"] is not None
        ):
            pooled_sd = float(
                np.sqrt((summary_a["sd"] ** 2 + summary_b["sd"] ** 2) / 2.0)
            )
            if pooled_sd > 0.0:
                d = float((summary_a["mean"] - summary_b["mean"]) / pooled_sd)
        passed = d is not None and abs(d) < SPLIT_PASS_ABS_D
        metrics[name] = {
            "half_a": summary_a,
            "half_b": summary_b,
            "pooled_sd": pooled_sd,
            "cohens_d_a_minus_b": d,
            "abs_d_pass_below": SPLIT_PASS_ABS_D,
            "outcome": "PASS" if passed else "FAIL",
        }
        if name in new_metric_names and d is not None:
            new_abs_d.append(abs(d))

    all_new_available = len(new_abs_d) == len(new_metric_names)
    worst_new = max(new_abs_d) if new_abs_d else None
    new_pass = (
        all_new_available
        and worst_new is not None
        and worst_new < SPLIT_PASS_ABS_D
    )
    return {
        "method": (
            "sort all reference-disjoint indices by SHA-256 of the fixed split "
            "label plus "
            "the unsigned little-endian int64 index, then divide once at the middle"
        ),
        "fixed_label_ascii": SPLIT_LABEL.rstrip(b"\x00").decode("ascii"),
        "random_seed_drawn": False,
        "sequence_independence_is_not_asserted": True,
        "half_a": {
            "assigned_rows": int(half_a.size),
            "valid_rows": int(in_a.sum()),
            "sorted_int64_le_sha256": _int64_sha256(half_a),
        },
        "half_b": {
            "assigned_rows": int(half_b.size),
            "valid_rows": int(in_b.sum()),
            "sorted_int64_le_sha256": _int64_sha256(half_b),
        },
        "metrics": metrics,
        "new_metrics": sorted(new_metric_names),
        "worst_abs_cohens_d_new_metrics": worst_new,
        "outcome_new_metrics": "PASS" if new_pass else "FAIL",
    }


def _paired_ratio_summary(
    original: np.ndarray,
    transformed: np.ndarray,
    *,
    expected: float,
) -> dict:
    finite = (
        np.isfinite(original)
        & np.isfinite(transformed)
        & (np.asarray(original) != 0.0)
    )
    ratios = np.asarray(transformed)[finite] / np.asarray(original)[finite]
    _require(ratios.size > 0, "no finite nonzero pairs for response ratio")
    return {
        "n": int(ratios.size),
        "expected_transformed_over_original": float(expected),
        "median_transformed_over_original": float(np.median(ratios)),
        "max_abs_error_from_expected": float(np.max(np.abs(ratios - expected))),
    }


def _manipulation_response(
    matrix: np.ndarray,
    half_amplitude_matrix: np.ndarray,
    two_level_ratios: np.ndarray,
) -> dict:
    """Check width response and normalized-shape response on the same traces."""
    _require(matrix.shape == half_amplitude_matrix.shape, "response shape mismatch")
    _require(two_level_ratios.shape == (matrix.shape[0],), "shape ratio mismatch")
    columns = {name: METRIC_NAMES.index(name) for name in METRIC_NAMES}
    amplitude = {}
    for name in ("excursion", "max_central90_side_excursion", "angle_std"):
        amplitude[name] = _paired_ratio_summary(
            matrix[:, columns[name]],
            half_amplitude_matrix[:, columns[name]],
            expected=0.5,
        )
    amplitude_ratio = matrix[:, columns["excursion_to_std_ratio"]]
    half_ratio = half_amplitude_matrix[:, columns["excursion_to_std_ratio"]]
    finite_amplitude_ratio = np.isfinite(amplitude_ratio) & np.isfinite(half_ratio)
    amplitude_ratio_delta = half_ratio[finite_amplitude_ratio] - amplitude_ratio[
        finite_amplitude_ratio
    ]
    _require(amplitude_ratio_delta.size > 0, "no finite amplitude-ratio pairs")
    amplitude["excursion_to_std_ratio"] = {
        "n": int(amplitude_ratio_delta.size),
        "expected_transformed_minus_original": 0.0,
        "median_transformed_minus_original": float(
            np.median(amplitude_ratio_delta)
        ),
        "max_abs_error_from_expected": float(
            np.max(np.abs(amplitude_ratio_delta))
        ),
    }
    amplitude_pass = all(
        item["max_abs_error_from_expected"] <= 1e-12
        for item in amplitude.values()
    )

    finite_shape = np.isfinite(amplitude_ratio) & np.isfinite(two_level_ratios)
    original_ratio = amplitude_ratio[finite_shape]
    transformed_ratio = two_level_ratios[finite_shape]
    delta = transformed_ratio - original_ratio
    absolute_delta = np.abs(delta)
    _require(delta.size > 0, "no finite two-level ratio pairs")
    median_abs_delta = float(np.median(absolute_delta))
    shape_pass = median_abs_delta >= SHAPE_RESPONSE_MIN_MEDIAN_ABS_DELTA

    return {
        "half_amplitude_about_each_sequence_median": {
            "definition": (
                "angle' = sequence median + 0.5 * (angle - sequence median)"
            ),
            "purpose": (
                "width metrics must halve; excursion divided by angle SD must "
                "remain invariant because it measures waveform shape"
            ),
            "metrics": amplitude,
            "numerical_tolerance": 1e-12,
            "outcome": "PASS" if amplitude_pass else "FAIL",
        },
        "two_level_shape": {
            "definition": (
                "each angle below its sequence median becomes -1; every other "
                "angle becomes +1"
            ),
            "purpose": (
                "the dimensionless excursion-to-SD ratio must respond when a "
                "continuous trace is reduced to a two-level waveform"
            ),
            "n": int(delta.size),
            "original_median_ratio": float(np.median(original_ratio)),
            "transformed_median_ratio": float(np.median(transformed_ratio)),
            "median_paired_delta": float(np.median(delta)),
            "median_absolute_paired_delta": median_abs_delta,
            "fraction_abs_delta_at_least_threshold": float(
                np.mean(absolute_delta >= SHAPE_RESPONSE_MIN_MEDIAN_ABS_DELTA)
            ),
            "pass_threshold_median_absolute_delta_at_least": (
                SHAPE_RESPONSE_MIN_MEDIAN_ABS_DELTA
            ),
            "outcome": "PASS" if shape_pass else "FAIL",
        },
        "outcome_new_metrics": (
            "PASS" if amplitude_pass and shape_pass else "FAIL"
        ),
    }


def analyse_dataset(
    dataset: h5py.Dataset,
    analysis_indices: np.ndarray,
    *,
    block_rows: int = 256,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, list[int]],
]:
    """Read the disjoint rows in bounded sequential blocks."""
    _require(
        tuple(dataset.shape) == EXPECTED_DATASET_SHAPE,
        "analysis received a dataset with the wrong shape",
    )
    _require(
        type(block_rows) is int and block_rows > 0,
        "block_rows must be a positive integer",
    )
    requested = np.zeros(EXPECTED_DATASET_SHAPE[0], dtype=bool)
    requested[analysis_indices] = True
    _require(
        int(requested.sum()) == ANALYSIS_COUNT,
        f"analysis index selection has {int(requested.sum())} rows",
    )

    metric_blocks: list[np.ndarray] = []
    half_amplitude_blocks: list[np.ndarray] = []
    two_level_ratio_blocks: list[np.ndarray] = []
    valid_index_blocks: list[np.ndarray] = []
    invalid = {
        "nonfinite_thorax_or_abdomen": [],
        "zero_length_thorax_abdomen_vector": [],
    }

    total = EXPECTED_DATASET_SHAPE[0]
    for start in range(0, total, block_rows):
        stop = min(start + block_rows, total)
        wanted = requested[start:stop]
        if not wanted.any():
            continue
        block = np.asarray(dataset[start:stop])
        expected_block_shape = (stop - start, L.N_FRAMES, L.N_PARTS * 2)
        _require(
            block.shape == expected_block_shape,
            f"HDF5 block shape {block.shape} != {expected_block_shape}",
        )
        points = block.reshape(stop - start, L.N_FRAMES, L.N_PARTS, 2)
        vector = points[:, :, L.ABDOMEN, :] - points[:, :, L.THORAX, :]
        finite = np.isfinite(vector).all(axis=(1, 2))
        squared_length = np.sum(vector * vector, axis=2)
        zero_length = finite & np.any(squared_length <= 0.0, axis=1)

        global_indices = np.arange(start, stop, dtype=np.int64)
        nonfinite_wanted = wanted & ~finite
        zero_wanted = wanted & zero_length
        invalid["nonfinite_thorax_or_abdomen"].extend(
            global_indices[nonfinite_wanted].tolist()
        )
        invalid["zero_length_thorax_abdomen_vector"].extend(
            global_indices[zero_wanted].tolist()
        )

        valid = wanted & finite & ~zero_length
        if valid.any():
            selected = vector[valid]
            angles = np.unwrap(
                np.arctan2(selected[:, :, 0], selected[:, :, 1]), axis=1
            )
            metrics = _metric_matrix_from_angles(angles)
            metric_blocks.append(metrics)

            center = metrics[:, METRIC_NAMES.index("median")][:, None]
            half_angles = center + 0.5 * (angles - center)
            half_amplitude_blocks.append(
                _metric_matrix_from_angles(half_angles)
            )

            two_level_angles = np.where(angles < center, -1.0, 1.0)
            two_level_ratio_blocks.append(
                _metric_matrix_from_angles(two_level_angles)[
                    :, METRIC_NAMES.index("excursion_to_std_ratio")
                ]
            )
            valid_index_blocks.append(global_indices[valid])
        block_number = start // block_rows + 1
        if stop == total or block_number % 8 == 0:
            print(f"read {stop}/{total} source sequences", flush=True)

    _require(metric_blocks, "no valid sequences remain")
    matrix = np.concatenate(metric_blocks, axis=0)
    half_amplitude_matrix = np.concatenate(half_amplitude_blocks, axis=0)
    two_level_ratios = np.concatenate(two_level_ratio_blocks)
    valid_indices = np.concatenate(valid_index_blocks)
    invalid_count = sum(len(indices) for indices in invalid.values())
    _require(
        matrix.shape == (ANALYSIS_COUNT - invalid_count, len(METRIC_NAMES)),
        "valid plus invalid sequence accounting drift",
    )
    _require(
        valid_indices.size + invalid_count == ANALYSIS_COUNT,
        "analysis sequence accounting drift",
    )
    _require(
        np.isfinite(matrix[:, :-1]).all(),
        "an angular metric contains a non-finite value",
    )
    _require(
        half_amplitude_matrix.shape == matrix.shape,
        "half-amplitude metric shape drift",
    )
    _require(
        two_level_ratios.shape == (matrix.shape[0],),
        "two-level ratio shape drift",
    )
    return (
        valid_indices,
        matrix,
        half_amplitude_matrix,
        two_level_ratios,
        invalid,
    )


def _metric_matrix_sha256(indices: np.ndarray, matrix: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(indices, dtype=np.dtype("<i8")).tobytes())
    digest.update(np.ascontiguousarray(matrix, dtype=np.dtype("<f8")).tobytes())
    return digest.hexdigest()


def analyse_hdf5(path: Path | str) -> dict:
    """Authenticate, inspect, and analyse the public HDF5 artifact."""
    instrument_receipt = _instrument_receipt()
    source_path = Path(path)
    with source_path.open("rb") as source:
        source_receipt, source_identity = _verify_open_source(source)
        source.seek(0)
        try:
            with h5py.File(source, "r") as h5:
                dataset, dataset_receipt = inspect_dataset(h5)
                reference, analysis = reference_and_analysis_indices(
                    int(dataset.shape[0])
                )
                (
                    valid_indices,
                    matrix,
                    half_amplitude_matrix,
                    two_level_ratios,
                    invalid,
                ) = analyse_dataset(dataset, analysis)
        except OSError as exc:
            raise SourceFormatError(f"HDF5 open/read failed: {exc}") from exc
        _require(
            _stat_identity(os.fstat(source.fileno())) == source_identity,
            "source identity changed during HDF5 analysis",
            SourceIdentityError,
        )

    invalid_count = sum(len(indices) for indices in invalid.values())
    ratio_unavailable = valid_indices[~np.isfinite(matrix[:, -1])]
    radians = _summarise_matrix(matrix)
    degrees = _summarise_matrix(matrix, 180.0 / np.pi)
    excursion_column = METRIC_NAMES.index("excursion")
    side_excursion_column = METRIC_NAMES.index("max_central90_side_excursion")
    std_column = METRIC_NAMES.index("angle_std")
    ratio_column = METRIC_NAMES.index("excursion_to_std_ratio")
    split_half = _split_half_stability(analysis, valid_indices, matrix)
    manipulation = _manipulation_response(
        matrix, half_amplitude_matrix, two_level_ratios
    )
    metric_entitled = (
        split_half["outcome_new_metrics"] == "PASS"
        and manipulation["outcome_new_metrics"] == "PASS"
    )
    return {
        "experiment": EXPERIMENT,
        "status": "COMPLETE",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "instrument": instrument_receipt,
        "source": {
            "citation": (
                "Schulz et al. 2024, Walking behavior of flies "
                f"(Drosophila melanogaster), Zenodo {ZENODO_RECORD}, CC-BY-4.0"
            ),
            "filename": source_path.name,
            **source_receipt,
            "default_identity_check_enforced": True,
            "record_metadata": {
                "animal": "Drosophila melanogaster",
                "cohort": (
                    "female genetic-screen flies, not wild-type; the deposit "
                    "reports no visible morphological or behavioural difference "
                    "from wild type during capture"
                ),
                "capture": (
                    "three flies simultaneously in one acrylic 2D arena, filmed "
                    "from below for approximately 3 seconds at 80 Hz; repeated "
                    "for different sets of flies"
                ),
                "preprocessing": (
                    "Tracktor centroid tracking, per-fly frame crops, alignment "
                    "to one direction, DeepLabCut 32-part tracking, then "
                    "Savitzky-Golay smoothing"
                ),
                "metadata_url": f"https://zenodo.org/records/{ZENODO_RECORD.split('.')[-1]}",
            },
        },
        "dataset": dataset_receipt,
        "software": {
            "numpy": np.__version__,
            "h5py": h5py.__version__,
            "scipy": scipy.__version__,
            "percentile_method": "numpy.percentile method='linear'",
        },
        "selection": {
            "reference_exclusion": {
                "reason": (
                    "these rows built flyscore/reference_real_flies.json and "
                    "are excluded to keep this output audit disjoint"
                ),
                "numpy_generator": "default_rng",
                "seed": REFERENCE_SEED,
                "count": int(reference.size),
                "sorted_int64_le_sha256": _int64_sha256(reference),
            },
            "analysis": {
                "status": "disjoint from the 3,000-row scorer reference only",
                "layout_inference_independence": (
                    "not disjoint: flyscore/layout.py was inferred from a "
                    "separate systematic 300-row sample"
                ),
                "count_before_invalid_exclusion": int(analysis.size),
                "sorted_int64_le_sha256": _int64_sha256(analysis),
                "first_index": int(analysis[0]),
                "last_index": int(analysis[-1]),
            },
            "simulation_seeds_drawn": 0,
        },
        "readout": {
            "landmarks": {
                "thorax": int(L.THORAX),
                "abdomen": int(L.ABDOMEN),
                "layout_status": "inferred, not certified",
            },
            "coordinate_convention": (
                "x is lateral; low-to-high y is anterior-to-posterior"
            ),
            "angle_formula": (
                "unwrap(atan2(abdomen_x - thorax_x, "
                "abdomen_y - thorax_y))"
            ),
            "unwrap_axis": "time within each sequence",
            "absolute_angle_use": (
                "diagnostic only: per-frame alignment, inferred landmark "
                "identities, and the 2-pi branch prevent an absolute rest or "
                "sign target"
            ),
            "sequence_frames": int(L.N_FRAMES),
            "sampling_hz": float(L.FPS),
            "per_sequence_percentiles": [5.0, 50.0, 95.0],
            "excursion_formula": "p95 - p05",
            "max_central90_side_excursion_formula": (
                "max(median - p05, p95 - median)"
            ),
            "angle_std_formula": (
                "numpy.std(unwrapped angle, ddof=0), identical to the "
                "existing flyscore posture_variability statistic"
            ),
            "excursion_to_std_ratio_formula": "excursion / angle_std",
        },
        "invalid": {
            "count": int(invalid_count),
            "fraction_of_analysis": float(invalid_count / analysis.size),
            "by_reason": {
                reason: {"count": len(indices), "indices": indices}
                for reason, indices in invalid.items()
            },
            "metric_unavailable": {
                "excursion_to_std_ratio": {
                    "reason": "angle_std is zero or non-finite",
                    "count": int(ratio_unavailable.size),
                    "indices": ratio_unavailable.tolist(),
                }
            },
        },
        "population": {
            "percentiles_across_sequences": [5.0, 50.0, 95.0],
            "radians": {
                name: radians[name]
                for name in METRIC_NAMES
                if name != "excursion_to_std_ratio"
            },
            "degrees": {
                name: degrees[name]
                for name in METRIC_NAMES
                if name != "excursion_to_std_ratio"
            },
            "dimensionless": {
                "excursion_to_std_ratio": radians[
                    "excursion_to_std_ratio"
                ]
            },
        },
        "quality_diagnostics": _quality_diagnostics(valid_indices, matrix),
        "novelty_against_existing_posture_variability": {
            "existing_statistic": (
                "angle_std is exactly flyscore.metrics.posture_variability: "
                "population standard deviation of the unwrapped angle "
                "within one sequence"
            ),
            "correlations_are_descriptive": True,
            "sequence_independence_is_not_asserted": True,
            "rank_correlation_is_the_primary_redundancy_read": (
                "rare extreme unwrap tails inflate raw Pearson coefficients"
            ),
            "excursion_vs_angle_std": _descriptive_correlations(
                matrix[:, excursion_column], matrix[:, std_column]
            ),
            "max_central90_side_excursion_vs_angle_std": _descriptive_correlations(
                matrix[:, side_excursion_column], matrix[:, std_column]
            ),
            "excursion_vs_max_central90_side_excursion": _descriptive_correlations(
                matrix[:, excursion_column], matrix[:, side_excursion_column]
            ),
            "excursion_to_std_ratio_vs_excursion": _descriptive_correlations(
                matrix[:, ratio_column], matrix[:, excursion_column]
            ),
            "excursion_to_std_ratio_vs_max_central90_side_excursion": (
                _descriptive_correlations(
                    matrix[:, ratio_column], matrix[:, side_excursion_column]
                )
            ),
            "excursion_to_std_ratio_vs_angle_std": _descriptive_correlations(
                matrix[:, ratio_column], matrix[:, std_column]
            ),
            "ratio_temporal_order_sensitive": False,
        },
        "split_half_stability": split_half,
        "manipulation_response": manipulation,
        "new_metric_entitlement": {
            "rule": (
                "descriptive adoption requires both the deterministic row-split "
                "stability gate and the declared synthetic response gate to pass"
            ),
            "outcome": "PASS" if metric_entitled else "FAIL",
            "failure_consequence": (
                "retain the descriptive output but do not adopt a new benchmark"
            ),
        },
        "metric_matrix_receipt": {
            "columns": list(METRIC_NAMES),
            "row_order": "ascending valid dataset index",
            "valid_rows": int(valid_indices.size),
            "bytes": (
                "little-endian int64 indices followed by row-major "
                "little-endian float64 metric values"
            ),
            "sha256": _metric_matrix_sha256(valid_indices, matrix),
        },
        "claim_boundary": (
            "This physical-recording read constrains top-view orientation "
            "output of the tracked thorax-to-abdomen landmark "
            "vector over 234-frame sequences sampled at 80 Hz. It does not "
            "constrain a single abdomen joint, allocate "
            "motion among five joints, measure dorsoventral pitch, certify "
            "the inferred landmark layout, establish anatomical range of "
            "motion, establish that every retained sequence is walking, or "
            "validate the abdominal motor map. The normalized ratio reads the "
            "distribution of angle values but discards their temporal order. "
            "The HDF5 artifact supplies no animal, arena, or trial identifiers, "
            "so rows are not treated as independent flies."
        ),
    }


def write_result(path: Path | str, result: dict) -> None:
    """Publish once so a prior result cannot be silently replaced."""
    output = Path(path)
    _require(output.parent.is_dir(), f"output directory absent: {output.parent}")
    payload = json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit reference-disjoint real-fly lateral thorax-abdomen axis output."
        )
    )
    parser.add_argument("--h5", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    _require(
        args.h5.resolve() != args.out.resolve(),
        "input and output paths must differ",
    )
    result = analyse_hdf5(args.h5)
    write_result(args.out, result)
    print(json.dumps(result["population"], indent=2, sort_keys=True))
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
