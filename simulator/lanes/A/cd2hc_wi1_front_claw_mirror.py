#!/usr/bin/env python3
"""CD2-HC: can exact champion wi1 absorb the front-claw typing correction?

BANC's right-front SNpp50/SNpp51 split is the sole mirror reversal in eleven
leg-side comparisons across BANC and MANC (Fisher p = 1.6e-6).  The transferred
25-cell correction in ``feco_front_claw_mirror.json`` moved front left-right
phase 46 to 63 degrees toward the animal on wb1m, but that body lost speed and
qualifying runs.  Four later champions changed the interaction, and wi1's only
new mechanism is selective claw high-pass gain, so the correction gets one
fresh paired BUILD screen on the current executable fly.

The candidate changes exactly ``feco_reflex_map``.  Both arms run the full BANC
graph for 6,000 ms on Metal.  Seed 203761 runs only if seed 203760 preserves
standing, at least 80% of control speed, the accepted footfall band, and all-leg
cycling, while moving the front pair at least 30 degrees toward 180 in BOTH the
body-frame and fixed-heading reads.  The fixed-heading pair must be readable in
both arms.  Two seeds clearing that gate are a chase signal, not closure; they
license the required live clip and a larger judge comparison.

Robin's live 2026-09-08 10:50 EEST instruction superseded the expired CURRENT
hold without waiting for the goal file to be rewritten.  The authority blobs
below preserve the stale files that were visible when this route was chosen.

Reproduce from the committed execution source:

    FLY_WBE_DATA_ROOT=/Users/robin/Projects/fly-wbe-A/data \
    FLY_WBE_DECISION_GRADE=1 FLY_WBE_METAL_LOCK_MODE=shared \
    PYTHONPATH=.:lanes/A /Users/robin/Projects/fly-wbe-A/.venv/bin/python -u \
      lanes/A/cd2hc_wi1_front_claw_mirror.py \
      --out lanes/A/2026-09-08-cd2hc-wi1-front-claw-mirror.json
"""
from __future__ import annotations

import argparse
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

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "lanes/A") not in sys.path:
    sys.path.insert(0, str(ROOT / "lanes/A"))

import body as B  # noqa: E402
import cns_metal as CM  # noqa: E402
import motormap as MM  # noqa: E402
import regression_walk as RW  # noqa: E402
import walk_search as WS  # noqa: E402
from lanes.A.seedguard import all_used  # noqa: E402
from lanes.judge import card_regrade_qh1 as CARD  # noqa: E402
from walk_search import _resolve_body_joint_drive_scale  # noqa: E402


EXPERIMENT = "CD2-HC"
FIRST_SEED = 203760
EXTENSION_SEED = 203761
SEEDS = (FIRST_SEED, EXTENSION_SEED)
ARMS = ("wi1", "wi1_front_claw_mirror")
CURRENT_BLOB = "2f1ca47ba8121b414240aabb86a4d76e257b4168"
GOAL_BLOB = "4d2a861ce4044f4dba2dbbfa224f45d1aaca808e"
MAIN_AT_SELECTION = "a7f97d2f8f13a066a7159b4811621efa8b365226"
MIRROR_PATCH = {
    "feco_reflex_map": {
        "path": "lanes/A/feco_front_claw_mirror.json",
        "kinds": ["claw"],
    }
}
FRONT_MOVE_DEG = 30.0
MIN_SPEED_RATIO = 0.80
FOOTFALL_BAND = (1.02, 1.11)
EXPECTED_MAP_ROWS = 25
EXPECTED_MAP_SPLIT = {"flex": 20, "ext": 5}
RESULT_DEFAULT = "lanes/A/2026-09-08-cd2hc-wi1-front-claw-mirror.json"
DATA_ROOT = Path(os.environ.get("FLY_WBE_DATA_ROOT", str(ROOT / "data")))
META_PATH = DATA_ROOT / "banc_888_meta.feather"
EDGES_PATH = DATA_ROOT / "banc_888_edgelist_simple_v3.feather"
TRACKED_SOURCES = (
    "lanes/A/cd2hc_wi1_front_claw_mirror.py",
    "lanes/A/test_cd2hc_wi1_front_claw_mirror.py",
    "lanes/A/feco_front_claw_mirror.json",
    "lanes/A/seedguard.py",
    "lanes/A/phasemeasures.py",
    "lanes/judge/card_regrade_nc1.py",
    "lanes/judge/card_regrade_qh1.py",
    "lanes/judge/phase_fixed_heading.py",
    "results-wi1-theta.json",
    "walk_search.py",
    "regression_walk.py",
    "footfall1.py",
    "body.py",
    "motormap.py",
    "cns.py",
    "cns_metal.py",
    "gpu/cns_engine.m",
    "gpu/cns_engine.h",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True).strip()


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def theta_difference(reference: dict, candidate: dict) -> dict:
    return {
        key: {"reference": reference.get(key), "candidate": candidate.get(key)}
        for key in sorted(set(reference) | set(candidate))
        if reference.get(key) != candidate.get(key)
    }


def theta_arms() -> dict[str, dict]:
    document = json.loads((ROOT / "results-wi1-theta.json").read_text())
    reference = copy.deepcopy(document.get("winner", document))
    require(reference.get("backend") == "metal", "wi1 backend drifted")
    require(reference.get("joint_int") == "exact", "wi1 joint integrator drifted")
    require(float(reference.get("spike_reg", 0.0)) == 44.0,
            "wi1 spike_reg drifted")
    require(reference.get("feco_modality_mode") == "exact_type",
            "wi1 exact FeCO modality correction drifted")
    require(reference.get("feco_polarity_mode") == "banc_type",
            "wi1 BANC-type polarity base drifted")
    require(reference.get("phasic") == 1.25,
            "wi1 selective claw high-pass gain drifted")
    require("feco_reflex_map" not in reference,
            "wi1 already carries a reflex correction map")
    candidate = copy.deepcopy(reference)
    candidate.update(copy.deepcopy(MIRROR_PATCH))
    expected = {
        "feco_reflex_map": {
            "reference": None,
            "candidate": MIRROR_PATCH["feco_reflex_map"],
        }
    }
    require(theta_difference(reference, candidate) == expected,
            "candidate changes more than feco_reflex_map")
    return {ARMS[0]: reference, ARMS[1]: candidate}


def map_audit(meta: pd.DataFrame) -> dict:
    document = json.loads(
        (ROOT / MIRROR_PATCH["feco_reflex_map"]["path"]).read_text())
    mapping = document["map"]
    require(document.get("n") == EXPECTED_MAP_ROWS == len(mapping),
            "front-claw correction row count drifted")
    split = {
        polarity: sum(v["polarity"] == polarity for v in mapping.values())
        for polarity in ("flex", "ext")
    }
    require(split == EXPECTED_MAP_SPLIT,
            f"front-claw correction split drifted: {split}")
    ids = set(meta["banc_888_id"].astype(str))
    missing = sorted(set(mapping) - ids)
    require(not missing, f"front-claw correction IDs absent from BANC: {missing}")
    return {
        "rows": len(mapping),
        "corrected_split": split,
        "missing_from_banc": missing,
        "status": "transferred correction; not a BANC measurement",
        "source_note": document.get("note"),
    }


def source_gate(out_path: Path) -> tuple[str, dict[str, str]]:
    require(os.environ.get("FLY_WBE_DECISION_GRADE") == "1",
            "set FLY_WBE_DECISION_GRADE=1")
    require(os.environ.get("FLY_WBE_METAL_LOCK_MODE") == "shared",
            "CD2-HC requires the deterministic shared Metal lock")
    require(git("branch", "--show-current") == "lane-a",
            "CD2-HC runs only on branch lane-a")
    head = git("rev-parse", "HEAD")
    require(head == git("rev-parse", "origin/lane-a"),
            "HEAD differs from origin/lane-a")
    require(subprocess.run(
        ["git", "merge-base", "--is-ancestor", "origin/main", "HEAD"],
        cwd=ROOT).returncode == 0, "current origin/main is not in HEAD")
    require(git("rev-parse", "origin/main") == MAIN_AT_SELECTION,
            "origin/main moved after route selection; refresh authority")
    require(git("rev-parse", "origin/main:exchange/feedback/CURRENT.md")
            == CURRENT_BLOB, "CURRENT changed; read it before running")
    require(git("rev-parse", "origin/main:exchange/goals/lane-a.md")
            == GOAL_BLOB, "goal file changed; read it before running")
    require(not git("status", "--porcelain=v1", "--untracked-files=all"),
            "worktree is not clean before execution")
    require(not out_path.exists(), f"refusing to overwrite {out_path}")
    hashes: dict[str, str] = {}
    for rel in TRACKED_SOURCES:
        local = (ROOT / rel).read_bytes()
        committed = subprocess.check_output(
            ["git", "show", f"HEAD:{rel}"], cwd=ROOT)
        require(local == committed,
                f"execution source differs from HEAD: {rel}")
        hashes[rel] = hashlib.sha256(local).hexdigest()
    block = json.loads((ROOT / "seed_blocks.json").read_text())["blocks"][
        "lane-a"]
    require(all(block[0] <= seed <= block[1] for seed in SEEDS),
            "CD2-HC seeds are outside Lane A's block")
    overlap = sorted(set(SEEDS).intersection(all_used()))
    require(not overlap, f"refusing used seeds: {overlap}")
    require(META_PATH.is_file() and EDGES_PATH.is_file(),
            f"BANC files absent under {DATA_ROOT}")
    CM._lib()  # Validates the compiled-in source stamp before any run.
    return head, hashes


def run_arm(theta: dict, cache: dict, meta: pd.DataFrame,
            edges: pd.DataFrame, seed: int) -> dict:
    RW._G.clear()
    RW._G.update({"th": theta, "meta": meta, "edges": edges, "cache": cache})
    held: dict[str, Any] = {}
    original_evaluate = WS.evaluate

    def capture_evaluate(*args, **kwargs):
        held["capture"] = kwargs.get("capture")
        result = original_evaluate(*args, **kwargs)
        held["extras"] = copy.deepcopy(WS.LAST_EXTRAS)
        return result

    WS.evaluate = capture_evaluate
    started = time.time()
    try:
        record = RW._run(seed)
    finally:
        WS.evaluate = original_evaluate
    record["wall_s"] = round(time.time() - started, 3)
    if "error" in record:
        return clean(record)
    capture = held.get("capture")
    require(capture, "regression reader did not retain the live capture")
    tick_ms = float(theta.get("tick_ms", 2.5))
    record.update(CARD.footfall(capture, tick_ms))
    record.update(CARD.flyscore_rows(capture, tick_ms))
    record.update(CARD.phase_rows(capture, tick_ms))
    record.update(CARD.phase_fixed_rows(
        capture, tick_ms, record.get("phase_line_hz")))
    extras = held.get("extras") or {}
    record["neural_physics"] = extras.get("neural_physics")
    record["prop_rate_mean"] = extras.get("prop_rate_mean")
    record["feco_type_polarity"] = extras.get("feco_type_polarity")
    record["proprioceptor_phasic"] = extras.get("proprioceptor_phasic")
    record["feco_reflex_map"] = (
        (theta.get("feco_reflex_map") or {}).get("path"))
    lifts = record.get("lifts_h10")
    record["qualifies_minimum"] = bool(
        record.get("standing_v32") and lifts and min(lifts) >= 5
        and record.get("freq_tip_h10") is not None
        and 5.0 <= float(record["freq_tip_h10"]) <= 16.0
        and record.get("endogenous_timing"))
    return clean(record)


def finite(record: dict, key: str) -> float | None:
    value = record.get(key)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value))):
        return None
    return float(value)


def compare_seed(reference: dict, candidate: dict) -> dict:
    body_before = finite(reference, "phase_lf_rf_deg")
    body_after = finite(candidate, "phase_lf_rf_deg")
    fixed_before = finite(reference, "phase_lf_rf_fixed_deg")
    fixed_after = finite(candidate, "phase_lf_rf_fixed_deg")
    speed_before = finite(reference, "speed")
    speed_after = finite(candidate, "speed")
    footfall = finite(candidate, "footfall_sep_bl")
    body_move = (None if body_before is None or body_after is None
                 else body_after - body_before)
    fixed_move = (None if fixed_before is None or fixed_after is None
                  else fixed_after - fixed_before)
    speed_ratio = (None if speed_before is None or speed_after is None
                   or speed_before <= 0.0 else speed_after / speed_before)
    fixed_readable = bool(
        reference.get("phase_lf_rf_readable")
        and candidate.get("phase_lf_rf_readable"))
    footfall_in_band = bool(
        footfall is not None and FOOTFALL_BAND[0] <= footfall <= FOOTFALL_BAND[1])
    minimum_lifts = min(candidate.get("lifts_h10") or [-1])
    extension_gate = bool(
        reference.get("standing_v32") and candidate.get("standing_v32")
        and fixed_readable
        and body_move is not None and body_move >= FRONT_MOVE_DEG
        and fixed_move is not None and fixed_move >= FRONT_MOVE_DEG
        and speed_ratio is not None and speed_ratio >= MIN_SPEED_RATIO
        and footfall_in_band and minimum_lifts >= 5)
    return clean({
        "both_standing": bool(
            reference.get("standing_v32") and candidate.get("standing_v32")),
        "front_body_frame": {
            "wi1_deg": body_before,
            "candidate_deg": body_after,
            "toward_180_deg": body_move,
        },
        "front_fixed_heading": {
            "wi1_deg": fixed_before,
            "candidate_deg": fixed_after,
            "toward_180_deg": fixed_move,
            "both_readable": fixed_readable,
        },
        "speed": {
            "wi1_mm_s": speed_before,
            "candidate_mm_s": speed_after,
            "candidate_over_wi1": speed_ratio,
            "retained_ge_0p80": bool(
                speed_ratio is not None and speed_ratio >= MIN_SPEED_RATIO),
        },
        "candidate_footfall_sep_bl": footfall,
        "candidate_footfall_in_band": footfall_in_band,
        "candidate_min_lifts_h10": minimum_lifts,
        "candidate_heave_frac": finite(candidate, "heave_frac"),
        "wi1_heave_frac": finite(reference, "heave_frac"),
        "candidate_body_height_mm": finite(candidate, "body_height_mm"),
        "wi1_body_height_mm": finite(reference, "body_height_mm"),
        "candidate_straightness_5p5s": finite(
            candidate, "straightness_5p5s"),
        "wi1_straightness_5p5s": finite(
            reference, "straightness_5p5s"),
        "extension_gate": extension_gate,
    })


def write_partial(path: Path, head: str, runs: dict,
                  comparisons: dict) -> None:
    executed = sorted({
        int(seed) for arm in runs.values() for seed in arm
    })
    path.write_text(json.dumps(clean({
        "partial": True,
        "id": EXPERIMENT,
        "execution_commit": head,
        "seeds": executed,
        "runs": runs,
        "comparisons": comparisons,
    }), indent=1) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=RESULT_DEFAULT)
    args = parser.parse_args()
    out_path = ROOT / args.out
    head, source_hashes = source_gate(out_path)
    arms = theta_arms()
    meta = pd.read_feather(META_PATH)
    edges = pd.read_feather(EDGES_PATH)
    correction = map_audit(meta)
    caches = {arm: {} for arm in ARMS}
    runs: dict[str, dict[str, dict]] = {arm: {} for arm in ARMS}
    comparisons: dict[str, dict] = {}
    execution_seeds = [FIRST_SEED]
    started = time.time()

    for seed_index, seed in enumerate(execution_seeds):
        order = ARMS if seed_index % 2 == 0 else tuple(reversed(ARMS))
        for arm in order:
            row = run_arm(arms[arm], caches[arm], meta, edges, seed)
            runs[arm][str(seed)] = row
            print(
                f"[{arm}] {seed} stand={row.get('standing_v32')} "
                f"speed={row.get('speed')} sep={row.get('footfall_sep_bl')} "
                f"front body/fixed={row.get('phase_lf_rf_deg')}/"
                f"{row.get('phase_lf_rf_fixed_deg')} "
                f"read={row.get('phase_lf_rf_readable')} "
                f"lifts={min(row.get('lifts_h10') or [-1])} "
                f"heave={row.get('heave_frac')} ({row.get('wall_s')}s)",
                flush=True)
            write_partial(out_path, head, runs, comparisons)
        comparisons[str(seed)] = compare_seed(
            runs[ARMS[0]][str(seed)], runs[ARMS[1]][str(seed)])
        write_partial(out_path, head, runs, comparisons)
        print(json.dumps({str(seed): comparisons[str(seed)]}, indent=1),
              flush=True)
        if seed == FIRST_SEED and comparisons[str(seed)]["extension_gate"]:
            execution_seeds.append(EXTENSION_SEED)

    repeated = bool(
        len(execution_seeds) == 2
        and all(comparisons[str(seed)]["extension_gate"]
                for seed in execution_seeds))
    if repeated:
        decision = "CHASE: two fresh seeds license a live clip and larger paired block"
    elif len(execution_seeds) == 1:
        decision = "DROP: correction missed the shortest fair wi1 screen"
    else:
        decision = "DROP: first-seed signal did not repeat"
    result = clean({
        "id": EXPERIMENT,
        "lane": "A",
        "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "question": (
            "can exact champion wi1 absorb the biology-backed right-front "
            "claw typing correction and improve true front left-right phase"),
        "phase": "BUILD",
        "engine": "metal",
        "engine_exception": None,
        "lock": os.environ.get("FLY_WBE_METAL_LOCK_MODE"),
        "decision_grade": os.environ.get("FLY_WBE_DECISION_GRADE"),
        "substrate": "whole BANC graph, unfiltered title path",
        "theta": "results-wi1-theta.json",
        "theta_sha256": sha256(ROOT / "results-wi1-theta.json"),
        "arms": {ARMS[0]: {}, ARMS[1]: MIRROR_PATCH},
        "candidate_control_difference": theta_difference(
            arms[ARMS[0]], arms[ARMS[1]]),
        "correction": correction,
        "authority": {
            "robin_live_instruction": "2026-09-08 10:50 EEST",
            "current_blob_at_selection": CURRENT_BLOB,
            "goal_blob_at_selection": GOAL_BLOB,
            "origin_main_at_selection": MAIN_AT_SELECTION,
            "note": "live instruction superseded the expired hold",
        },
        "registered_read": {
            "first_seed": FIRST_SEED,
            "conditional_extension_seed": EXTENSION_SEED,
            "standing": "both arms clear standing_v32",
            "front_body_frame_move_toward_180_deg": FRONT_MOVE_DEG,
            "front_fixed_heading_move_toward_180_deg": FRONT_MOVE_DEG,
            "fixed_heading": "front pair readable in both arms",
            "minimum_speed_ratio": MIN_SPEED_RATIO,
            "candidate_footfall_band_bl": list(FOOTFALL_BAND),
            "candidate_minimum_per_leg_lifts_h10": 5,
            "extension": (
                "203761 runs only if 203760 clears every clause above"),
            "gain_claim": (
                "two seeds plus a live clip; this screen alone closes no row"),
        },
        "graph": {
            "neurons": len(meta),
            "edges": len(edges),
            "expanded_synapses": int(edges["count"].sum()),
        },
        "provenance": {
            "execution_commit": head,
            "source_sha256": source_hashes,
            "meta_path": str(META_PATH),
            "meta_sha256": sha256(META_PATH),
            "edges_path": str(EDGES_PATH),
            "edges_sha256": sha256(EDGES_PATH),
            "engine_dylib": str(CM._LIB_PATH),
            "engine_dylib_sha256": sha256(CM._LIB_PATH),
            "engine_source_sha256": CM._source_sha256(),
            "engine_source_sha16_loaded": CM._ENGINE_SHA,
            "machine": platform.machine(),
            "python": sys.version,
            "exact_command": (
                "FLY_WBE_DATA_ROOT=/Users/robin/Projects/fly-wbe-A/data "
                "FLY_WBE_DECISION_GRADE=1 FLY_WBE_METAL_LOCK_MODE=shared "
                "PYTHONPATH=.:lanes/A "
                "/Users/robin/Projects/fly-wbe-A/.venv/bin/python -u "
                "lanes/A/cd2hc_wi1_front_claw_mirror.py "
                "--out lanes/A/2026-09-08-cd2hc-wi1-front-claw-mirror.json"),
        },
        "seeds": execution_seeds,
        "new_seeds_spent": len(execution_seeds),
        "runs": runs,
        "comparisons": comparisons,
        "verdict": {
            "first_seed_extension_gate": comparisons[str(FIRST_SEED)][
                "extension_gate"],
            "extension_seed_run": EXTENSION_SEED in execution_seeds,
            "repeated_two_seed_signal": repeated,
            "decision": decision,
            "claim_limit": "BUILD route selection; no phase row closes",
        },
        "wall_seconds": round(time.time() - started, 3),
    })
    out_path.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["verdict"], indent=1), flush=True)
    print(f"receipt -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
