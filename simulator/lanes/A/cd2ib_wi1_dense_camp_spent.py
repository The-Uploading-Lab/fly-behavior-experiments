#!/usr/bin/env python3
"""CD2-IB: test the dense per-segment campaniform correction on wi1.

Simple per-leg gain equalisation was already null.  The distinct correction
that survived earlier work adds 26 deterministic unclassified sensory rows per
leg, splits each leg's load over tarsus, tibia and claw, and rescales the global
campaniform gain so total delivered population weight is unchanged.  On we1 it
produced six standing rescues and no losses over 48 paired seeds.  This BUILD
screen asks whether exact champion wi1 can carry the same default-off
correction.

The frozen CD2-IA wi1 arm supplies a bit-identical reference on spent seed
203760, so only the candidate CNS arm runs.  A retained, clear behavioural move
is a spent-seed chase signal only.  It does not license fresh seeds, a clip, a
model-gain claim, or an identity claim for the transferred sensory rows.

Run from the committed execution source in its own tmux session:

    FLY_WBE_DATA_ROOT=/Users/robin/Projects/fly-wbe-A/data \
    FLY_WBE_DECISION_GRADE=1 FLY_WBE_METAL_LOCK_MODE=shared \
    LANE_A_EXECUTION_CONTEXT=tmux_session_lane-a-cd2ib PYTHONPATH=. \
    /Users/robin/Projects/fly-wbe-A/.venv/bin/python -u \
      lanes/A/cd2ib_wi1_dense_camp_spent.py
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cns_metal as CM  # noqa: E402
import motormap as MM  # noqa: E402
from load_loop import camp_by_leg, camp_extra_by_leg  # noqa: E402
from lanes.A import cd2hc_wi1_front_claw_mirror as HC  # noqa: E402
from lanes.A import cd2ia_wi1_front_claw_mirror_spent as IA  # noqa: E402
from lanes.A import cd2ia_read_wi1_front_claw_mirror_spent as IA_READER  # noqa: E402
from lanes.A.seedguard import all_used  # noqa: E402


EXPERIMENT = "CD2-IB"
SEED = 203760
SEALED_FRESH_SEEDS = (203762, 203763)
CURRENT_BLOB = "8eae189f36067085fcffd899d7f4a3448b34feaa"
GOAL_BLOB = "428ec01093d73801c08fef962e7feb8fe19d39b2"
ORIGIN_MAIN_AT_SELECTION = "44f87c2220441ad78b3751a7d0830df7da7812a7"
EXPECTED_THETA_SHA256 = (
    "ba18676b65b3db0747532d344cd2a2847ac8e38335bb77d3c34b29e9b7802fe7")
EXPECTED_BASELINE_RESULT_SHA256 = (
    "6a0a278a91fec3c00eb6416faf76c8f1a3877eed05ef52a5bc662eabf55e6c9c")
EXPECTED_METAL_DYLIB_SHA256 = (
    "c40c5e1ab0f16eb13ebd1bfa6a6e69363542c313d353dfafa26e42cc0fc55ce9")
BASELINE_RESULT = ROOT / "lanes/A/2026-09-09-cd2ia-wi1-front-claw-mirror-spent.json"
RESULT = ROOT / "lanes/A/2026-09-09-cd2ib-wi1-dense-camp-spent.json"
DATA_ROOT = Path(os.environ.get("FLY_WBE_DATA_ROOT", str(ROOT / "data")))
META_PATH = DATA_ROOT / "banc_888_meta.feather"
EDGES_PATH = DATA_ROOT / "banc_888_edgelist_simple_v3.feather"
CANDIDATE_PATCH = {
    "camp_load": "per_segment",
    "camp_extra": 26,
    "camp_match": 1,
}
EXPECTED_BASE_COUNTS = {
    "lf": 28, "lm": 2, "lh": 2, "rf": 25, "rm": 2, "rh": 2,
}
EXPECTED_EXTRA_COUNTS = {leg: 26 for leg in MM.LEGS}
FOOTFALL_BAND = (1.02, 1.11)
POSTURE_VARIABILITY_BAND = (0.0256, 0.0859)
BODY_HEIGHT_BAND = (0.80, 0.89)
HEAVE_BAND = (0.0, 0.10)
STRAIGHTNESS_BAND = (0.50, 0.98)
MIN_SPEED_RATIO = 0.80
MIN_TARGET_GAP_REDUCTION = 0.05
MIN_PHASE_MOVE_DEG = 20.0
TRACKED_SOURCES = tuple(dict.fromkeys((
    "lanes/A/cd2ib_wi1_dense_camp_spent.py",
    "lanes/A/test_cd2ib_wi1_dense_camp_spent.py",
    "lanes/A/cd2ia_read_wi1_front_claw_mirror_spent.py",
    "lanes/A/2026-09-09-cd2ia-wi1-front-claw-mirror-spent.json",
    *IA.TRACKED_SOURCES,
)))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True).strip()


def finite(record: dict[str, Any], key: str) -> float | None:
    value = record.get(key)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value))):
        return None
    return float(value)


def in_band(value: float | None, band: tuple[float, float]) -> bool:
    return value is not None and band[0] <= value <= band[1]


def target_gap_reduction(reference: float | None, candidate: float | None,
                         band: tuple[float, float]) -> float | None:
    """Fraction of the reference's distance to a measured interval removed."""
    if reference is None or candidate is None:
        return None
    if reference < band[0]:
        old_gap = band[0] - reference
        new_gap = max(0.0, band[0] - candidate)
    elif reference > band[1]:
        old_gap = reference - band[1]
        new_gap = max(0.0, candidate - band[1])
    else:
        return 0.0
    return (old_gap - new_gap) / old_gap if old_gap > 0.0 else 0.0


def theta_arms() -> tuple[dict[str, Any], dict[str, Any]]:
    reference = copy.deepcopy(HC.theta_arms()[HC.ARMS[0]])
    require(not set(CANDIDATE_PATCH).intersection(reference),
            "wi1 already carries part of the campaniform correction")
    candidate = copy.deepcopy(reference)
    candidate.update(copy.deepcopy(CANDIDATE_PATCH))
    expected = {
        key: {"reference": None, "candidate": value}
        for key, value in sorted(CANDIDATE_PATCH.items())
    }
    require(HC.theta_difference(reference, candidate) == expected,
            "candidate changes more than the dense campaniform correction")
    return reference, candidate


def correction_audit(meta: pd.DataFrame) -> dict[str, Any]:
    base = camp_by_leg(meta, None)
    extra = camp_extra_by_leg(meta, int(CANDIDATE_PATCH["camp_extra"]))
    base_counts = {leg: len(base[leg]) for leg in MM.LEGS}
    extra_counts = {leg: len(extra[leg]) for leg in MM.LEGS}
    overlaps = {
        leg: len(set(base[leg]).intersection(extra[leg])) for leg in MM.LEGS
    }
    dense_counts = {
        leg: len(set(base[leg]).union(extra[leg])) for leg in MM.LEGS
    }
    require(base_counts == EXPECTED_BASE_COUNTS,
            f"base campaniform counts changed: {base_counts}")
    require(extra_counts == EXPECTED_EXTRA_COUNTS,
            f"transferred row counts changed: {extra_counts}")
    require(not any(overlaps.values()),
            f"transferred rows overlap known campaniform rows: {overlaps}")
    n0 = sum(base_counts.values())
    n1 = sum(dense_counts.values())
    match_factor = n0 / n1
    matched_total = sum(count * match_factor
                        for count in dense_counts.values())
    require(math.isclose(matched_total, float(n0), rel_tol=0.0,
                         abs_tol=1e-12),
            "camp_match does not conserve total population weight")
    return {
        "known_rows_by_leg": base_counts,
        "transferred_rows_by_leg": extra_counts,
        "dense_rows_by_leg": dense_counts,
        "known_total": n0,
        "dense_total": n1,
        "camp_match_factor": match_factor,
        "matched_population_weight": matched_total,
        "segment_signal": "measured MuJoCo contact load by tarsus/tibia/claw",
        "row_identity": (
            "TRANSFERRED GUESS: deterministic unclassified leg sensory rows; "
            "not identified as campaniform in BANC"),
    }


def compare(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    speed_before = finite(reference, "speed")
    speed_after = finite(candidate, "speed")
    speed_ratio = (None if speed_before is None or speed_after is None
                   or speed_before <= 0.0 else speed_after / speed_before)
    footfall = finite(candidate, "footfall_sep_bl")
    posture_variability = finite(candidate, "fs_posture_variability_rad")
    min_lifts = min(candidate.get("lifts_h10") or [-1])
    retention = {
        "standing_v32": candidate.get("standing_v32") is True,
        "endogenous_all_leg_cycle": bool(
            candidate.get("endogenous_timing")
            and finite(candidate, "freq_tip_h10") is not None
            and 5.0 <= float(candidate["freq_tip_h10"]) <= 16.0
            and min_lifts >= 5),
        "footfall_in_band": in_band(footfall, FOOTFALL_BAND),
        "posture_variability_in_band": in_band(
            posture_variability, POSTURE_VARIABILITY_BAND),
        "speed_retained_ge_0p80": bool(
            speed_ratio is not None and speed_ratio >= MIN_SPEED_RATIO),
    }
    gaps = {
        "heave": target_gap_reduction(
            finite(reference, "heave_frac"), finite(candidate, "heave_frac"),
            HEAVE_BAND),
        "body_height": target_gap_reduction(
            finite(reference, "body_height_mm"),
            finite(candidate, "body_height_mm"), BODY_HEIGHT_BAND),
        "straightness": target_gap_reduction(
            finite(reference, "straightness_5p5s"),
            finite(candidate, "straightness_5p5s"), STRAIGHTNESS_BAND),
    }
    phase_moves: dict[str, Any] = {}
    for pair in ("lm_rm", "lh_rh"):
        body_before = finite(reference, f"phase_{pair}_deg")
        body_after = finite(candidate, f"phase_{pair}_deg")
        fixed_before = finite(reference, f"phase_{pair}_fixed_deg")
        fixed_after = finite(candidate, f"phase_{pair}_fixed_deg")
        body_move = (None if body_before is None or body_after is None
                     else body_after - body_before)
        fixed_move = (None if fixed_before is None or fixed_after is None
                      else fixed_after - fixed_before)
        phase_moves[pair] = {
            "body_before_deg": body_before,
            "body_after_deg": body_after,
            "body_move_toward_180_deg": body_move,
            "fixed_before_deg": fixed_before,
            "fixed_after_deg": fixed_after,
            "fixed_move_toward_180_deg": fixed_move,
            "both_reads_move_ge_20_deg": bool(
                reference.get(f"phase_{pair}_readable")
                and candidate.get(f"phase_{pair}_readable")
                and body_move is not None and body_move >= MIN_PHASE_MOVE_DEG
                and fixed_move is not None and fixed_move >= MIN_PHASE_MOVE_DEG),
        }
    movers = {
        "speed_gain_ge_5pct": bool(
            speed_ratio is not None and speed_ratio >= 1.05),
        "heave_target_gap_reduction_ge_5pct": bool(
            gaps["heave"] is not None
            and gaps["heave"] >= MIN_TARGET_GAP_REDUCTION),
        "body_height_target_gap_reduction_ge_5pct": bool(
            gaps["body_height"] is not None
            and gaps["body_height"] >= MIN_TARGET_GAP_REDUCTION),
        "straightness_target_gap_reduction_ge_5pct": bool(
            gaps["straightness"] is not None
            and gaps["straightness"] >= MIN_TARGET_GAP_REDUCTION),
        "middle_phase_both_reads_ge_20deg": phase_moves["lm_rm"][
            "both_reads_move_ge_20_deg"],
        "hind_phase_both_reads_ge_20deg": phase_moves["lh_rh"][
            "both_reads_move_ge_20_deg"],
    }
    retention_pass = all(retention.values())
    clear_mover = any(movers.values())
    return HC.clean({
        "retention": retention,
        "retention_pass": retention_pass,
        "speed": {
            "wi1_mm_s": speed_before,
            "candidate_mm_s": speed_after,
            "candidate_over_wi1": speed_ratio,
        },
        "candidate_footfall_sep_bl": footfall,
        "candidate_min_lifts_h10": min_lifts,
        "target_gap_reduction_fraction": gaps,
        "phase_moves": phase_moves,
        "movers": movers,
        "clear_behavioural_mover": clear_mover,
        "chase_gate": retention_pass and clear_mover,
    })


def decision(comparison: dict[str, Any]) -> dict[str, Any]:
    chase = comparison.get("chase_gate") is True
    retained = comparison.get("retention_pass") is True
    return {
        "one_spent_seed_chase_signal": chase,
        "model_gain_claimed": False,
        "fresh_seed_allowed": False,
        "clip_allowed": False,
        "outcome": (
            "CHASE_DENSE_CAMP_ON_SEPARATE_SPENT_SEED" if chase else
            "DROP_DIRECT_DENSE_CAMP_ON_WI1_RETENTION" if not retained else
            "DROP_DIRECT_DENSE_CAMP_ON_WI1_NO_SIGNAL"),
        "next": (
            "test unchanged correction on one separate spent Lane A seed"
            if chase else
            "do not tune this correction on wi1; choose a distinct mechanism"),
    }


def source_gate() -> tuple[str, dict[str, str], str]:
    require(os.environ.get("FLY_WBE_DECISION_GRADE") == "1",
            "set FLY_WBE_DECISION_GRADE=1")
    require(os.environ.get("FLY_WBE_METAL_LOCK_MODE") == "shared",
            "CD2-IB requires the deterministic shared Metal lock")
    require(os.environ.get("LANE_A_EXECUTION_CONTEXT")
            == "tmux_session_lane-a-cd2ib",
            "CD2-IB requires its own tmux session")
    require(git("branch", "--show-current") == "lane-a",
            "CD2-IB runs only on branch lane-a")
    head = git("rev-parse", "HEAD")
    require(head == git("rev-parse", "origin/lane-a"),
            "HEAD differs from origin/lane-a")
    require(git("rev-parse", "origin/main") == ORIGIN_MAIN_AT_SELECTION,
            "origin/main moved after route selection; refresh authority")
    require(git("rev-parse", "origin/main:exchange/feedback/CURRENT.md")
            == CURRENT_BLOB, "CURRENT changed; read it before running")
    require(git("rev-parse", "origin/main:exchange/goals/lane-a.md")
            == GOAL_BLOB, "goal changed; read it before running")
    require(not git("status", "--porcelain=v1", "--untracked-files=all"),
            "worktree is not clean before execution")
    require(not RESULT.exists(), f"refusing to overwrite {RESULT}")
    require(sha256(ROOT / "results-wi1-theta.json")
            == EXPECTED_THETA_SHA256, "wi1 theta bytes changed")
    require(sha256(BASELINE_RESULT) == EXPECTED_BASELINE_RESULT_SHA256,
            "frozen CD2-IA baseline bytes changed")
    IA_READER.validate()
    baseline = json.loads(BASELINE_RESULT.read_text())
    for relative in HC.TRACKED_SOURCES:
        expected = baseline["provenance"]["source_sha256"].get(relative)
        require(expected is not None,
                f"baseline omitted execution source: {relative}")
        require(sha256(ROOT / relative) == expected,
                f"shared execution source changed since baseline: {relative}")
    block = json.loads((ROOT / "seed_blocks.json").read_text())["blocks"][
        "lane-a"]
    require(block[0] <= SEED <= block[1], "selection seed is outside Lane A")
    require(SEED in all_used(), "CD2-IB accepts only an already-spent seed")
    require(SEED not in SEALED_FRESH_SEEDS, "sealed fresh seed selected")
    require(META_PATH.is_file() and EDGES_PATH.is_file(),
            f"BANC files absent under {DATA_ROOT}")
    CM._lib()
    require(sha256(CM._LIB_PATH) == EXPECTED_METAL_DYLIB_SHA256,
            "Metal dylib identity changed")
    source_hashes: dict[str, str] = {}
    for relative in TRACKED_SOURCES:
        payload = (ROOT / relative).read_bytes()
        committed = subprocess.check_output(
            ["git", "show", f"HEAD:{relative}"], cwd=ROOT)
        require(payload == committed,
                f"execution source differs from HEAD: {relative}")
        source_hashes[relative] = hashlib.sha256(payload).hexdigest()
    return head, source_hashes, git("rev-parse", "origin/main")


def main() -> int:
    head, source_hashes, origin_main = source_gate()
    reference_theta, candidate_theta = theta_arms()
    expected_difference = {
        key: {"reference": None, "candidate": value}
        for key, value in sorted(CANDIDATE_PATCH.items())
    }
    require(HC.theta_difference(reference_theta, candidate_theta)
            == expected_difference, "candidate theta difference changed")

    baseline_result = json.loads(BASELINE_RESULT.read_text())
    reference = copy.deepcopy(baseline_result["runs"]["wi1"])
    require(reference.get("seed") == SEED
            and reference.get("engine_receipt", {}).get("backend") == "metal",
            "frozen reference is not the spent-seed Metal wi1 arm")

    meta = pd.read_feather(META_PATH)
    edges = pd.read_feather(EDGES_PATH)
    audit = correction_audit(meta)
    started = time.time()
    candidate = IA.run_with_engine_receipt(
        candidate_theta, {}, meta, edges, SEED)
    require("error" not in candidate,
            f"candidate failed: {candidate.get('error')}")
    comparison = compare(reference, candidate)
    verdict = decision(comparison)
    print(
        f"[wi1_dense_camp] seed={SEED} "
        f"stand={candidate.get('standing_v32')} "
        f"speed={candidate.get('speed')} "
        f"sep={candidate.get('footfall_sep_bl')} "
        f"lifts={min(candidate.get('lifts_h10') or [-1])} "
        f"heave={candidate.get('heave_frac')} "
        f"height={candidate.get('body_height_mm')} "
        f"mid={candidate.get('phase_lm_rm_deg')}/"
        f"{candidate.get('phase_lm_rm_fixed_deg')} "
        f"hind={candidate.get('phase_lh_rh_deg')}/"
        f"{candidate.get('phase_lh_rh_fixed_deg')}", flush=True)

    result = {
        "id": EXPERIMENT,
        "lane": "A",
        "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "phase": "BUILD",
        "question": (
            "can exact champion wi1 carry the dense per-segment "
            "campaniform correction that previously rescued standing on we1"),
        "engine": "metal",
        "engine_exception": None,
        "substrate": "whole BANC graph, unfiltered title path",
        "selection_seed_reused": SEED,
        "new_seeds_spent": 0,
        "sealed_fresh_seeds_preserved": list(SEALED_FRESH_SEEDS),
        "theta": "results-wi1-theta.json",
        "theta_sha256": EXPECTED_THETA_SHA256,
        "arms": {
            "wi1": {"source": str(BASELINE_RESULT.relative_to(ROOT))},
            "wi1_dense_camp": CANDIDATE_PATCH,
        },
        "candidate_control_difference": expected_difference,
        "correction": audit,
        "prior_evidence": {
            "simple_per_leg_gain": (
                "null on CD2-CZ and CD2-FS; not repeated here"),
            "dense_per_segment_on_we1": (
                "six standing flips for and zero against over 48 paired seeds; "
                "speed cost unresolved"),
        },
        "registered_read": {
            "retention": {
                "standing_v32": True,
                "endogenous_cadence_hz": [5.0, 16.0],
                "minimum_per_leg_lifts_h10": 5,
                "footfall_band_bl": list(FOOTFALL_BAND),
                "posture_variability_band_rad": list(
                    POSTURE_VARIABILITY_BAND),
                "minimum_speed_ratio": MIN_SPEED_RATIO,
            },
            "clear_mover": (
                "at least 5% of a measured heave, body-height, or "
                "straightness target gap removed; or speed +5%; or middle/hind "
                "phase moves at least 20 degrees toward 180 in body and "
                "fixed-heading reads"),
            "stopping_rule": (
                "stop after this one candidate arm; chase only if retention "
                "and one clear-mover clause both pass"),
        },
        "graph": {
            "neurons": len(meta),
            "edges": len(edges),
            "expanded_synapses": int(edges["count"].sum()),
        },
        "runs": {
            "wi1_reused": reference,
            "wi1_dense_camp": candidate,
        },
        "comparison": comparison,
        "decision": verdict,
        "authority": {
            "current_blob": CURRENT_BLOB,
            "goal_blob": GOAL_BLOB,
            "origin_main": origin_main,
        },
        "provenance": {
            "execution_commit": head,
            "source_sha256": source_hashes,
            "baseline_result_sha256": EXPECTED_BASELINE_RESULT_SHA256,
            "meta_path": str(META_PATH),
            "meta_sha256": sha256(META_PATH),
            "edges_path": str(EDGES_PATH),
            "edges_sha256": sha256(EDGES_PATH),
            "metal_dylib_sha256": sha256(CM._LIB_PATH),
            "metal_source_sha256": CM._source_sha256(),
            "machine": platform.machine(),
            "python": sys.version,
            "execution_context": os.environ["LANE_A_EXECUTION_CONTEXT"],
            "exact_command": (
                "FLY_WBE_DATA_ROOT=/Users/robin/Projects/fly-wbe-A/data "
                "FLY_WBE_DECISION_GRADE=1 FLY_WBE_METAL_LOCK_MODE=shared "
                "LANE_A_EXECUTION_CONTEXT=tmux_session_lane-a-cd2ib "
                "PYTHONPATH=. /Users/robin/Projects/fly-wbe-A/.venv/bin/"
                "python -u lanes/A/cd2ib_wi1_dense_camp_spent.py"),
        },
        "wall_seconds": round(time.time() - started, 3),
    }
    RESULT.write_text(json.dumps(HC.clean(result), indent=1) + "\n")
    print(json.dumps(verdict, indent=1), flush=True)
    print(f"receipt -> {RESULT.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
