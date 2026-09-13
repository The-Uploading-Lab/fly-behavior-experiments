#!/usr/bin/env python3
"""Strict committed reader for the CD2-IA spent-seed Metal receipt."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "lanes/A/2026-09-09-cd2ia-wi1-front-claw-mirror-spent.json"
EXPECTED_RESULT_SHA256 = (
    "6a0a278a91fec3c00eb6416faf76c8f1a3877eed05ef52a5bc662eabf55e6c9c")
EXPECTED_EXECUTION_COMMIT = "201047b25919bf14b8c115af2b4b22aaaea7084c"
EXPECTED_THETA_SHA256 = (
    "ba18676b65b3db0747532d344cd2a2847ac8e38335bb77d3c34b29e9b7802fe7")
EXPECTED_DYLIB_SHA256 = (
    "c40c5e1ab0f16eb13ebd1bfa6a6e69363542c313d353dfafa26e42cc0fc55ce9")
EXPECTED_ENGINE_SOURCE_SHA256 = (
    "48ebcded0df397512b099af10634050261b28beba900c1697fe440aad40c30ef")
EXPECTED_COMMAND = (
    "FLY_WBE_DATA_ROOT=/Users/robin/Projects/fly-wbe-A/data "
    "FLY_WBE_DECISION_GRADE=1 FLY_WBE_METAL_LOCK_MODE=shared "
    "LANE_A_EXECUTION_CONTEXT=tmux_session_lane-a-cd2ia PYTHONPATH=. "
    "/Users/robin/Projects/fly-wbe-A/.venv/bin/python -u "
    "lanes/A/cd2ia_wi1_front_claw_mirror_spent.py")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def close(actual: Any, expected: float) -> bool:
    return (isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isclose(float(actual), expected,
                             rel_tol=0.0, abs_tol=1e-12))


def git(*args: str, binary: bool = False) -> Any:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=not binary)


def validate() -> dict[str, Any]:
    require(sha256_bytes(RESULT.read_bytes()) == EXPECTED_RESULT_SHA256,
            "CD2-IA result bytes differ from the frozen receipt")
    result = json.loads(RESULT.read_text())
    require(result.get("id") == "CD2-IA" and result.get("phase") == "BUILD",
            "experiment identity changed")
    require(result.get("engine") == "metal"
            and result.get("engine_exception") is None,
            "execution engine changed")
    require(result.get("substrate")
            == "whole BANC graph, unfiltered title path",
            "execution substrate changed")
    require(result.get("selection_seed_reused") == 203760
            and result.get("new_seeds_spent") == 0
            and result.get("sealed_fresh_seeds_preserved") == [203762, 203763],
            "seed classification changed")
    require(result.get("theta_sha256") == EXPECTED_THETA_SHA256,
            "wi1 theta identity changed")
    require(result.get("candidate_control_difference") == {
        "feco_reflex_map": {
            "reference": None,
            "candidate": {
                "path": "lanes/A/feco_front_claw_mirror.json",
                "kinds": ["claw"],
            },
        },
    }, "candidate differs from wi1 outside the mirror map")

    correction = result["correction"]
    require(correction.get("rows") == 25
            and correction.get("corrected_split") == {"flex": 20, "ext": 5}
            and correction.get("missing_from_banc") == [],
            "transferred correction audit changed")
    require(result.get("graph") == {
        "neurons": 188508,
        "edges": 13620865,
        "expanded_synapses": 42309621,
    }, "whole-graph receipt changed")

    provenance = result["provenance"]
    require(provenance.get("execution_commit") == EXPECTED_EXECUTION_COMMIT,
            "execution commit changed")
    require(subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_EXECUTION_COMMIT,
         "HEAD"], cwd=ROOT).returncode == 0,
        "execution commit is not ancestral to HEAD")
    for relative, expected in provenance["source_sha256"].items():
        payload = git("show", f"{EXPECTED_EXECUTION_COMMIT}:{relative}",
                      binary=True)
        require(sha256_bytes(payload) == expected,
                f"execution source differs: {relative}")
    require(provenance.get("metal_dylib_sha256") == EXPECTED_DYLIB_SHA256
            and provenance.get("metal_source_sha256")
            == EXPECTED_ENGINE_SOURCE_SHA256,
            "Metal engine identity changed")
    require(provenance.get("execution_context")
            == "tmux_session_lane-a-cd2ia",
            "execution context changed")
    require(provenance.get("exact_command") == EXPECTED_COMMAND,
            "exact command changed")

    runs = result["runs"]
    require(set(runs) == {"wi1", "wi1_front_claw_mirror"},
            "run arm set changed")
    expected_receipt = {
        "backend": "metal",
        "requested_backend": "metal",
        "run_output_reported_backend": True,
        "engine_source_sha256_16": EXPECTED_ENGINE_SOURCE_SHA256[:16],
    }
    for arm, run in runs.items():
        require(run.get("seed") == 203760
                and run.get("standing_v32") is True,
                f"{arm}: seed or standing result changed")
        require(run.get("engine_receipt") == expected_receipt,
                f"{arm}: Metal receipt changed")
        require((run.get("neural_physics") or {}).get("neuron_rows") == 188508,
                f"{arm}: neural row count changed")
    require(runs["wi1"].get("feco_reflex_map") is None
            and runs["wi1_front_claw_mirror"].get("feco_reflex_map")
            == "lanes/A/feco_front_claw_mirror.json",
            "applied mirror receipt changed")

    comparison = result["comparison"]
    require(comparison.get("both_standing") is True
            and comparison.get("extension_gate") is False,
            "gate result changed")
    body = comparison["front_body_frame"]
    fixed = comparison["front_fixed_heading"]
    speed = comparison["speed"]
    require(close(body.get("wi1_deg"), 140.0)
            and close(body.get("candidate_deg"), 54.0)
            and close(body.get("toward_180_deg"), -86.0),
            "body-frame phase result changed")
    require(close(fixed.get("wi1_deg"), 27.0)
            and close(fixed.get("candidate_deg"), 8.0)
            and close(fixed.get("toward_180_deg"), -19.0)
            and fixed.get("both_readable") is True,
            "fixed-heading phase result changed")
    require(close(speed.get("wi1_mm_s"), 1.8492900569098412)
            and close(speed.get("candidate_mm_s"), 1.715926764690589)
            and close(speed.get("candidate_over_wi1"), 0.9278840592253538)
            and speed.get("retained_ge_0p80") is True,
            "speed result changed")
    require(close(comparison.get("candidate_footfall_sep_bl"),
                  1.0639628479471606)
            and comparison.get("candidate_footfall_in_band") is True
            and comparison.get("candidate_min_lifts_h10") == 5,
            "candidate retention result changed")

    decision = result["decision"]
    require(decision == {
        "one_spent_seed_chase_signal": False,
        "model_gain_claimed": False,
        "fresh_seed_allowed": False,
        "clip_allowed": False,
        "outcome": "DROP_DIRECT_MIRROR_TRANSFER_ON_WI1",
        "next": "do not tune this direct transfer; choose a different mechanism",
    }, "decision changed")
    catches = result.get("startup_catches") or []
    require(len(catches) == 2
            and catches[0].get("cns_steps") == 0
            and catches[1].get("completed_arms") == ["wi1"]
            and catches[1].get("candidate_arms") == 0,
            "startup catch record changed")
    return {
        "status": "VALIDATED_CD2IA_DIRECT_MIRROR_DROP",
        "result_sha256": EXPECTED_RESULT_SHA256,
        "execution_commit": EXPECTED_EXECUTION_COMMIT,
        "engine": "metal",
        "engine_source_sha256": EXPECTED_ENGINE_SOURCE_SHA256,
        "seed": 203760,
        "new_seeds_spent": 0,
        "both_standing": True,
        "body_phase_move_toward_180_deg": -86.0,
        "fixed_phase_move_toward_180_deg": -19.0,
        "speed_ratio": 0.9278840592253538,
        "decision": decision["outcome"],
        "exact_command": EXPECTED_COMMAND,
    }


def main() -> int:
    print(json.dumps(validate(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
