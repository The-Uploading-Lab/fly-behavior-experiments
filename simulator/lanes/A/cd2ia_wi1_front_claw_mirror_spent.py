#!/usr/bin/env python3
"""CD2-IA: run the frozen wi1 front-claw mirror read on one spent seed.

The biology-backed right-front claw correction was measured on wb1m, but the
same comparison frozen for wi1 in CD2-HC never ran.  Its proposed fresh seeds
were later consumed by CD2-HD and the next Lane A seeds are sealed.  This
BUILD route therefore reuses spent selection seed 203760 and applies CD2-HC's
unchanged one-seed gate.  A pass licenses a mirror-in dial search on spent
seeds; it is not a model gain, confirmation, closure, or clip license.

Run from the committed execution source in its own tmux session:

    FLY_WBE_DATA_ROOT=/Users/robin/Projects/fly-wbe-A/data \
    FLY_WBE_DECISION_GRADE=1 FLY_WBE_METAL_LOCK_MODE=shared \
    LANE_A_EXECUTION_CONTEXT=tmux_session_lane-a-cd2ia PYTHONPATH=. \
    /Users/robin/Projects/fly-wbe-A/.venv/bin/python -u \
      lanes/A/cd2ia_wi1_front_claw_mirror_spent.py
"""

from __future__ import annotations

import copy
import hashlib
import json
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
from lanes.A import cd2hc_wi1_front_claw_mirror as HC  # noqa: E402
from lanes.A.seedguard import all_used  # noqa: E402


EXPERIMENT = "CD2-IA"
SEED = 203760
SEALED_FRESH_SEEDS = (203762, 203763)
CURRENT_BLOB = "50e5ffe129726f826a761b5b5eaf20fe6e72b437"
GOAL_BLOB = "428ec01093d73801c08fef962e7feb8fe19d39b2"
EXPECTED_THETA_SHA256 = (
    "ba18676b65b3db0747532d344cd2a2847ac8e38335bb77d3c34b29e9b7802fe7")
EXPECTED_MIRROR_SHA256 = (
    "cd6a48bd87c68ba1b492ef015e658c4573a31cbf46a76e0ef4661e3e3dc78948")
EXPECTED_METAL_DYLIB_SHA256 = (
    "c40c5e1ab0f16eb13ebd1bfa6a6e69363542c313d353dfafa26e42cc0fc55ce9")
RESULT = ROOT / "lanes/A/2026-09-09-cd2ia-wi1-front-claw-mirror-spent.json"
DATA_ROOT = Path(os.environ.get("FLY_WBE_DATA_ROOT", str(ROOT / "data")))
META_PATH = DATA_ROOT / "banc_888_meta.feather"
EDGES_PATH = DATA_ROOT / "banc_888_edgelist_simple_v3.feather"
TRACKED_SOURCES = tuple(dict.fromkeys((
    "lanes/A/cd2ia_wi1_front_claw_mirror_spent.py",
    "lanes/A/test_cd2ia_wi1_front_claw_mirror_spent.py",
    *HC.TRACKED_SOURCES,
)))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True).strip()


def decision(comparison: dict[str, Any]) -> dict[str, Any]:
    passed = comparison.get("extension_gate") is True
    return {
        "one_spent_seed_chase_signal": passed,
        "model_gain_claimed": False,
        "fresh_seed_allowed": False,
        "clip_allowed": False,
        "outcome": (
            "CHASE_MIRROR_IN_DIAL_SEARCH_ON_SPENT_SEEDS"
            if passed else "DROP_DIRECT_MIRROR_TRANSFER_ON_WI1"
        ),
        "next": (
            "run one bounded mirror-in dial search on spent Lane A seeds"
            if passed else
            "do not tune this direct transfer; choose a different mechanism"
        ),
    }


def engine_run_receipt(requested_backend: str,
                       run_out: dict[str, Any]) -> dict[str, Any]:
    """Record the backend returned by CNS.run rather than inferring it."""
    reported = run_out.get("backend")
    actual = (str(reported) if reported is not None
              else "numpy" if requested_backend == "numpy" else None)
    return {
        "backend": actual,
        "requested_backend": str(requested_backend),
        "run_output_reported_backend": reported is not None,
        "engine_source_sha256_16": CM._ENGINE_SHA,
    }


def run_with_engine_receipt(theta: dict[str, Any], cache: dict,
                            meta: pd.DataFrame, edges: pd.DataFrame,
                            seed: int) -> dict[str, Any]:
    receipts: list[dict[str, Any]] = []
    original_run = HC.WS.cns.CNS.run

    def capture_run(net: Any, *args: Any, **kwargs: Any) -> Any:
        run_out = original_run(net, *args, **kwargs)
        receipts.append(engine_run_receipt(
            str(getattr(net.p, "backend", "numpy")), run_out))
        return run_out

    HC.WS.cns.CNS.run = capture_run
    try:
        row = HC.run_arm(theta, cache, meta, edges, seed)
    finally:
        HC.WS.cns.CNS.run = original_run
    require(len(receipts) == 1,
            f"captured {len(receipts)} CNS engine receipts")
    row["engine_receipt"] = receipts[0]
    require(row["engine_receipt"] == {
        "backend": "metal",
        "requested_backend": "metal",
        "run_output_reported_backend": True,
        "engine_source_sha256_16": CM._source_sha256()[:16],
    }, f"run did not receipt Metal: {row['engine_receipt']}")
    return row


def source_gate() -> tuple[str, dict[str, str], str]:
    require(os.environ.get("FLY_WBE_DECISION_GRADE") == "1",
            "set FLY_WBE_DECISION_GRADE=1")
    require(os.environ.get("FLY_WBE_METAL_LOCK_MODE") == "shared",
            "CD2-IA requires the deterministic shared Metal lock")
    require(os.environ.get("LANE_A_EXECUTION_CONTEXT")
            == "tmux_session_lane-a-cd2ia",
            "CD2-IA requires its own tmux session")
    require(git("branch", "--show-current") == "lane-a",
            "CD2-IA runs only on branch lane-a")
    head = git("rev-parse", "HEAD")
    require(head == git("rev-parse", "origin/lane-a"),
            "HEAD differs from origin/lane-a")
    require(git("rev-parse", "origin/main:exchange/feedback/CURRENT.md")
            == CURRENT_BLOB, "CURRENT changed; read it before running")
    require(git("rev-parse", "origin/main:exchange/goals/lane-a.md")
            == GOAL_BLOB, "goal changed; read it before running")
    require(not git("status", "--porcelain=v1", "--untracked-files=all"),
            "worktree is not clean before execution")
    require(not RESULT.exists(), f"refusing to overwrite {RESULT}")
    require(sha256(ROOT / "results-wi1-theta.json")
            == EXPECTED_THETA_SHA256, "wi1 theta bytes changed")
    require(sha256(ROOT / "lanes/A/feco_front_claw_mirror.json")
            == EXPECTED_MIRROR_SHA256, "front-claw mirror bytes changed")
    block = json.loads((ROOT / "seed_blocks.json").read_text())["blocks"][
        "lane-a"]
    require(block[0] <= SEED <= block[1], "selection seed is outside Lane A")
    used = all_used()
    require(SEED in used, "CD2-IA accepts only an already-spent seed")
    require(not set(SEALED_FRESH_SEEDS).intersection({SEED}),
            "sealed fresh seed selected")
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
    arms = HC.theta_arms()
    expected_difference = {
        "feco_reflex_map": {
            "reference": None,
            "candidate": copy.deepcopy(HC.MIRROR_PATCH["feco_reflex_map"]),
        }
    }
    require(HC.theta_difference(arms[HC.ARMS[0]], arms[HC.ARMS[1]])
            == expected_difference, "arms differ outside the mirror map")

    meta = pd.read_feather(META_PATH)
    edges = pd.read_feather(EDGES_PATH)
    correction = HC.map_audit(meta)
    runs: dict[str, dict[str, Any]] = {}
    started = time.time()
    for arm in HC.ARMS:
        row = run_with_engine_receipt(arms[arm], {}, meta, edges, SEED)
        require("error" not in row, f"{arm} failed: {row.get('error')}")
        runs[arm] = row
        print(
            f"[{arm}] seed={SEED} stand={row.get('standing_v32')} "
            f"speed={row.get('speed')} sep={row.get('footfall_sep_bl')} "
            f"front_body={row.get('phase_lf_rf_deg')} "
            f"front_fixed={row.get('phase_lf_rf_fixed_deg')} "
            f"readable={row.get('phase_lf_rf_readable')} "
            f"min_lifts={min(row.get('lifts_h10') or [-1])}", flush=True)

    comparison = HC.compare_seed(runs[HC.ARMS[0]], runs[HC.ARMS[1]])
    verdict = decision(comparison)
    result = {
        "id": EXPERIMENT,
        "lane": "A",
        "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "phase": "BUILD",
        "question": (
            "does the unrun biology-backed front-claw mirror transfer clear "
            "CD2-HC's frozen one-seed gate on exact champion wi1"),
        "engine": "metal",
        "engine_exception": None,
        "substrate": "whole BANC graph, unfiltered title path",
        "selection_seed_reused": SEED,
        "new_seeds_spent": 0,
        "sealed_fresh_seeds_preserved": list(SEALED_FRESH_SEEDS),
        "theta": "results-wi1-theta.json",
        "theta_sha256": EXPECTED_THETA_SHA256,
        "arms": {HC.ARMS[0]: {}, HC.ARMS[1]: HC.MIRROR_PATCH},
        "candidate_control_difference": expected_difference,
        "correction": correction,
        "registered_read": {
            "source": "CD2-HC's committed unchanged compare_seed gate",
            "standing": "both arms clear standing_v32",
            "front_body_frame_move_toward_180_deg": HC.FRONT_MOVE_DEG,
            "front_fixed_heading_move_toward_180_deg": HC.FRONT_MOVE_DEG,
            "fixed_heading": "front pair readable in both arms",
            "minimum_speed_ratio": HC.MIN_SPEED_RATIO,
            "candidate_footfall_band_bl": list(HC.FOOTFALL_BAND),
            "candidate_minimum_per_leg_lifts_h10": 5,
        },
        "graph": {
            "neurons": len(meta),
            "edges": len(edges),
            "expanded_synapses": int(edges["count"].sum()),
        },
        "runs": runs,
        "comparison": comparison,
        "decision": verdict,
        "authority": {
            "current_blob": CURRENT_BLOB,
            "goal_blob": GOAL_BLOB,
            "origin_main": origin_main,
        },
        "startup_catches": [{
            "failure": (
                "the first launch wrote its log inside the worktree before "
                "the source gate, so the clean-tree check refused execution"),
            "disposition": "execution logging moved to /private/tmp",
            "cns_steps": 0,
            "seeds_spent": 0,
            "failure_log_sha256": (
                "10218b0d4ff584cd3f4c1014e646a0a37ff1592b2468ea31563184582b27faa5"),
        }, {
            "failure": (
                "the second launch completed the control CNS run, then "
                "expected engine_receipt inside regression_walk's row even "
                "though that wrapper does not emit the field"),
            "disposition": (
                "capture CNS.run's reported backend around each arm, using "
                "the established Lane A receipt schema"),
            "completed_arms": ["wi1"],
            "candidate_arms": 0,
            "selection_seed": SEED,
            "new_seeds_spent": 0,
            "failure_log_sha256": (
                "ed81df53591e7ae61b46be14188d60811e2df93b9f653a0f34a2ea677620a66a"),
        }],
        "provenance": {
            "execution_commit": head,
            "source_sha256": source_hashes,
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
                "LANE_A_EXECUTION_CONTEXT=tmux_session_lane-a-cd2ia "
                "PYTHONPATH=. /Users/robin/Projects/fly-wbe-A/.venv/bin/"
                "python -u lanes/A/cd2ia_wi1_front_claw_mirror_spent.py"),
        },
        "wall_seconds": round(time.time() - started, 3),
    }
    RESULT.write_text(json.dumps(HC.clean(result), indent=1) + "\n")
    print(json.dumps(verdict, indent=1), flush=True)
    print(f"receipt -> {RESULT.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
