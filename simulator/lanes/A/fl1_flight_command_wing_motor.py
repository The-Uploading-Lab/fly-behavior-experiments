#!/usr/bin/env python3
"""FL1: does the whole CNS engage the flight power motor under a flight command?

Robin, 2026-09-09 (live instruction): flight is the Lane A goal. This is the
first bounded step. It holds wi1's exact theta and changes ONE key, the
descending command population, from the walking pair DNg100 to a flight
population, then reads the 64 wing motor neurons through evaluate()'s read-only
record_rows (default off, bit-identical; the record set never feeds back). The
body loop runs unchanged, so this is the whole executable model driven by a
flight command instead of a walking command. Spent Lane A seed 203760,
deterministic Metal, no shared code changed, no new seed spent.

Arms (same seed; theta differs from wi1 only in `cmd`):
  wi1           cmd DNg100, the walking champion (control).
  flight_dng02  cmd = the 38 DNg02 cells (7 BANC subtypes). Namiki et al. 2022
                (Curr Biol 32:1189): >=15 DNg02 pairs regulate wingbeat
                amplitude by a population code and can elicit maximum
                flight-motor power. In BANC the family carries the largest
                two-hop normalised weight onto the 24 DLM/DVM motor neurons.
  flight_dnp31  cmd = the DNp31 pair, BANC's strongest DIRECT descending input
                to the DLM/DVM motor neurons (1,422 synapses). No published
                function; a connectome-derived hypothesis, labelled as such.

Biological reference for the readout: DLM and DVM motor neurons are silent at
rest and fire tonically at roughly 5-20 Hz in flight (asynchronous power
muscle; Harcombe & Wyman 1977 J Comp Physiol; Gordon & Dickinson 2006 J Exp
Biol). The b1 steering motor neuron fires once per wingbeat (~200 Hz), driven
by haltere afferents; this body has no haltere mechanics, so steering, haltere
and neck motor rates are reported, not gated.

Gate: an arm ENGAGES the flight motor when at least 12 of the 24 power motor
neurons fire tonically (mean 2-40 Hz over 1-6 s, active in >=4 of the 5
one-second bins) while the wi1 control's power pool stays below 2 Hz mean. A
power-pool mean above 60 Hz is flagged unphysiological. No behavioural gain is
claimed either way; this decides the design of the next flight step.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
for entry in (str(ROOT), str(ROOT / "lanes/A")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import cns  # noqa: E402
import cns_metal as CM  # noqa: E402
import walk_search as WS  # noqa: E402
from lanes.A import cd2hc_wi1_front_claw_mirror as HC  # noqa: E402
from lanes.A import cd2ib_wi1_dense_camp_spent as IB  # noqa: E402
from lanes.A.seedguard import all_used  # noqa: E402

require, git, sha, clean = HC.require, HC.git, HC.sha256, HC.clean
theta_difference = HC.theta_difference

SEED = 203760
SEALED_FRESH_SEEDS = (203762, 203763)
DURATION_MS = 6000.0
BIN_MS = 5.0          # evaluate() records its read-only trace at record_trace_every=5.0
SETTLE_MS = 1000.0
POWER_MUSCLES = ("dorsal_longitudinal_muscle", "dorsoventral_muscle")
LEG_EFFECTORS = ("front_leg", "middle_leg", "hind_leg")
DNG02_TYPES = ("DNg02_a", "DNg02_b", "DNg02_c", "DNg02_d", "DNg02_f", "DNg02_g",
               "DNg02_h")
ARMS = {"wi1": "DNg100", "flight_dng02": list(DNG02_TYPES), "flight_dnp31": "DNp31"}
FLIGHT_ARMS = ("flight_dng02", "flight_dnp31")
TONIC_LO_HZ, TONIC_HI_HZ, UNPHYSIOLOGICAL_HZ = 2.0, 40.0, 60.0
TONIC_MIN_ACTIVE_SECONDS, ENGAGED_MIN_CELLS, CONTROL_QUIET_HZ = 4, 12, 2.0
EXPECTED_COUNTS = {"motor_all": 805, "wing": 64, "power": 24, "steering": 40,
                   "haltere": 27, "neck": 49, "leg": 391}
PRIOR_RESULT = ROOT / "lanes/A/2026-09-09-cd2ig-wi1-split-decay-spent.json"
RESULT = ROOT / "lanes/A/2026-09-09-fl1-flight-command-wing-motor.json"
EXECUTION_CONTEXT = "tmux_session_lane-a-fl1"
COMMAND = (
    "FLY_WBE_DATA_ROOT=/Users/robin/Projects/fly-wbe-A/data "
    "FLY_WBE_DECISION_GRADE=1 FLY_WBE_METAL_LOCK_MODE=shared "
    f"LANE_A_EXECUTION_CONTEXT={EXECUTION_CONTEXT} PYTHONPATH=. "
    "/Users/robin/Projects/fly-wbe-A/.venv/bin/python -u "
    "lanes/A/fl1_flight_command_wing_motor.py")
SOURCES = (
    "cns.py", "cns_metal.py", "walk_search.py", "body.py", "motormap.py",
    "regression_walk.py", "abdomen.py", "closed_loop.py", "adapt_and_score.py",
    "results-wi1-theta.json", "lanes/A/cd2hc_wi1_front_claw_mirror.py",
    "lanes/A/cd2ib_wi1_dense_camp_spent.py", "lanes/A/seedguard.py",
    "lanes/judge/card_regrade_qh1.py", "lanes/A/fl1_flight_command_wing_motor.py",
    "lanes/A/fl1_read_flight_command.py", "lanes/A/test_fl1_flight_command.py")
ROBIN_INSTRUCTION = (
    "2026-09-09, live: make flight the Lane A goal now; work on the highest-value "
    "bounded step toward a biologically constrained flying fly, using the whole "
    "executable model where possible; persist the flight assignment in NEXT and "
    "keep choosing another flight route rather than returning to walking.")


# ----------------------------------------------------------------- selection
def motor_rows(meta: pd.DataFrame) -> dict[str, np.ndarray]:
    motor = (meta["super_class"] == "motor").to_numpy()
    effector = meta["body_part_effector"].fillna("").to_numpy()
    target = meta["peripheral_target_type"].fillna("").to_numpy()
    wing = motor & (effector == "wing")
    power = wing & np.isin(target, POWER_MUSCLES)
    return {
        "motor_all": np.flatnonzero(motor),
        "wing": np.flatnonzero(wing),
        "power": np.flatnonzero(power),
        "steering": np.flatnonzero(wing & ~power),
        "haltere": np.flatnonzero(motor & (effector == "haltere")),
        "neck": np.flatnonzero(motor & (effector == "neck")),
        "leg": np.flatnonzero(motor & np.isin(effector, LEG_EFFECTORS)),
    }


def command_rows(meta: pd.DataFrame, cmd) -> np.ndarray:
    types = [cmd] if isinstance(cmd, str) else list(cmd)
    return np.flatnonzero(meta["cell_type"].isin(types).to_numpy())


# ------------------------------------------------------------------- readout
def rates(rec: np.ndarray, trace: np.ndarray, rows: np.ndarray,
          bin_ms: float = BIN_MS, settle_ms: float = SETTLE_MS):
    """Per-cell mean rate (Hz) after the settle window and per-second rates."""
    rec = np.asarray(rec)
    cols = np.searchsorted(rec, rows)
    require(cols.max(initial=-1) < len(rec) and np.array_equal(rec[cols], rows),
            "recorded rows do not contain the requested rows")
    start = int(round(settle_ms / bin_ms))
    counts = np.asarray(trace)[start:, cols].astype(np.int64)
    per_sec_bins = int(round(1000.0 / bin_ms))
    require(counts.shape[0] % per_sec_bins == 0, "window is not whole seconds")
    seconds = counts.shape[0] / per_sec_bins
    per_cell = counts.sum(axis=0) / seconds
    per_second = counts.reshape(-1, per_sec_bins, counts.shape[1]).sum(axis=1)
    return per_cell, per_second.astype(float)


def tonic_mask(per_cell: np.ndarray, per_second: np.ndarray,
               lo: float = TONIC_LO_HZ, hi: float = TONIC_HI_HZ,
               min_active: int = TONIC_MIN_ACTIVE_SECONDS) -> np.ndarray:
    active_seconds = (per_second > 0).sum(axis=0)
    return (per_cell >= lo) & (per_cell <= hi) & (active_seconds >= min_active)


def pool(rec, trace, rows):
    per_cell, per_second = rates(rec, trace, rows)
    return {
        "n": int(len(rows)),
        "mean_hz": round(float(per_cell.mean()), 4),
        "median_hz": round(float(np.median(per_cell)), 4),
        "max_hz": round(float(per_cell.max()), 4),
        "active_ge_2hz": int((per_cell >= TONIC_LO_HZ).sum()),
        "tonic_2_40hz": int(tonic_mask(per_cell, per_second).sum()),
        "per_second_pool_mean_hz": [round(float(x), 3)
                                    for x in per_second.mean(axis=1)],
    }


def summarize(meta, rec, trace, rows, cmd_rows):
    out = {name: pool(rec, trace, r) for name, r in rows.items()}
    per_cell, _ = rates(rec, trace, rows["wing"])
    wing = meta.iloc[rows["wing"]]
    cells = [{"row": int(r), "cell_type": str(ct), "side": str(sd), "muscle": str(m),
              "hz": round(float(h), 3)}
             for r, ct, sd, m, h in zip(rows["wing"], wing["cell_type"], wing["side"],
                                        wing["peripheral_target_type"], per_cell)]
    out["wing_cells"] = cells
    by_type: dict[str, list[float]] = {}
    for c in cells:
        by_type.setdefault(c["cell_type"], []).append(c["hz"])
    out["wing_by_type_hz"] = {
        k: {"n": len(v), "mean": round(float(np.mean(v)), 3),
            "min": round(min(v), 3), "max": round(max(v), 3)}
        for k, v in sorted(by_type.items())}
    power_cells = [c for c in cells if c["muscle"] in POWER_MUSCLES]
    out["power_by_side_hz"] = {
        side: round(float(np.mean([c["hz"] for c in power_cells if c["side"] == side])), 4)
        for side in ("left", "right")}
    cmd_cell, _ = rates(rec, trace, cmd_rows)
    out["command_rows"] = {"n": int(len(cmd_rows)), "mean_hz": round(float(cmd_cell.mean()), 3),
                           "min_hz": round(float(cmd_cell.min()), 3),
                           "max_hz": round(float(cmd_cell.max()), 3)}
    return out


def decide(summary: dict) -> dict:
    control_power = summary["wi1"]["power"]["mean_hz"]
    control_quiet = control_power < CONTROL_QUIET_HZ
    engaged = {arm: summary[arm]["power"]["tonic_2_40hz"] >= ENGAGED_MIN_CELLS
               for arm in FLIGHT_ARMS}
    unphysiological = {arm: summary[arm]["power"]["mean_hz"] > UNPHYSIOLOGICAL_HZ
                       for arm in FLIGHT_ARMS}
    winners = [arm for arm in FLIGHT_ARMS if engaged[arm] and not unphysiological[arm]]
    if not control_quiet:
        outcome = "CONTROL_POWER_POOL_NOT_QUIET"
    elif winners:
        outcome = "FLIGHT_MOTOR_ENGAGED:" + ",".join(winners)
    elif any(unphysiological.values()):
        outcome = "FLIGHT_MOTOR_SATURATED"
    else:
        outcome = "FLIGHT_MOTOR_NOT_ENGAGED"
    return {"outcome": outcome, "control_power_mean_hz": control_power,
            "control_quiet": control_quiet, "engaged": engaged,
            "unphysiological": unphysiological, "engaged_arms": winners,
            "model_gain_claimed": False, "fresh_seed_allowed": False,
            "next_step_if_engaged": (
                "FL2: couple the engaged arm's DLM/DVM spike trains to the flybody "
                "wing oscillator as asynchronous-muscle activation and measure lift"),
            "next_step_if_not": (
                "FL2': drive the wing premotor layer or the GF-PSI takeoff path; "
                "do not return to walking")}


# ------------------------------------------------------------------ running
def run_arm(name, theta, meta, edges, record_rows):
    traces, receipts = [], []
    original_evaluate, original_run = WS.evaluate, cns.CNS.run

    def recording_evaluate(*args, **kwargs):
        require("record_rows" not in kwargs and "trace_out" not in kwargs,
                "evaluate already records extra rows")
        return original_evaluate(*args, record_rows=record_rows, trace_out=traces,
                                 **kwargs)

    def observe(net, *args, **kwargs):
        p = net.p
        require(net.N == 188508 and p.backend == "metal", "wrong graph or engine")
        require(float(args[0]) == DURATION_MS and kwargs.get("seed") == SEED,
                "wrong duration or seed")
        require(float(kwargs.get("record_trace_every")) == BIN_MS, "trace bin drifted")
        recorded = np.asarray(kwargs.get("record"))
        require(np.isin(record_rows, recorded).all(), "record set lacks the motor rows")
        out = original_run(net, *args, **kwargs)
        receipts.append({
            "backend": p.backend, "engine_sha256": sha(ROOT / "cns.py"),
            "metal_source_sha256": CM._source_sha256(),
            "metal_dylib_sha256": sha(CM._LIB_PATH),
            "duration_ms": float(args[0]), "seed": kwargs.get("seed"),
            "neuron_rows": int(net.N), "dt_ms": float(p.dt),
            "recorded_rows": int(len(recorded)), "trace_bin_ms": BIN_MS,
            "backend_evidence": "instrumented cns.CNS.run dispatch; Metal branch"})
        return out

    WS.evaluate = recording_evaluate
    try:
        with patch.object(cns.CNS, "run", observe):
            row = HC.run_arm(theta, {}, meta, edges, SEED)
    finally:
        WS.evaluate = original_evaluate
    require("error" not in row,
            f"{name}: run failed: {row.get('error')}: {row.get('error_message')}")
    require(len(receipts) == 1 and len(traces) == 1,
            "expected one whole-CNS run with one trace")
    rec, trace = traces[0]
    require(trace is not None and trace.shape[0] == int(DURATION_MS / BIN_MS),
            "trace shape drifted")
    row["engine_receipt"] = receipts[0]
    return row, np.asarray(rec), np.asarray(trace)


def source_hashes(head):
    out = {}
    for path in SOURCES:
        payload = (ROOT / path).read_bytes()
        require(payload == subprocess.check_output(
            ["git", "show", f"{head}:{path}"], cwd=ROOT), f"uncommitted {path}")
        out[path] = hashlib.sha256(payload).hexdigest()
    return out


def write(document):
    RESULT.write_text(json.dumps(clean(document), indent=1, allow_nan=False) + "\n")


def main():
    started = time.monotonic()
    require(git("branch", "--show-current") == "lane-a", "wrong branch")
    head = git("rev-parse", "HEAD")
    require(head == git("rev-parse", "origin/lane-a") and not git("status", "--porcelain"),
            "commit and push before execution")
    require(not RESULT.exists(), "refusing to overwrite a receipt")
    require(os.environ.get("FLY_WBE_DECISION_GRADE") == "1"
            and os.environ.get("FLY_WBE_METAL_LOCK_MODE") == "shared"
            and os.environ.get("LANE_A_EXECUTION_CONTEXT") == EXECUTION_CONTEXT,
            "use the dedicated execution command")
    sources = source_hashes(head)
    require(sha(ROOT / "results-wi1-theta.json") == IB.EXPECTED_THETA_SHA256,
            "wi1 theta changed")
    block = json.loads((ROOT / "seed_blocks.json").read_text())["blocks"]["lane-a"]
    require(block[0] <= SEED <= block[1] and SEED in all_used()
            and SEED not in SEALED_FRESH_SEEDS, "seed must be owned and spent")
    prior = json.loads(PRIOR_RESULT.read_text())
    data_hashes = {"meta": sha(IB.META_PATH), "edges": sha(IB.EDGES_PATH)}
    require(data_hashes == prior["provenance"]["data_sha256"], "pinned data changed")
    subprocess.run(["git", "fetch", "-q", "origin", "main"], cwd=ROOT, check=True)
    authority = {
        "robin_live_instruction": ROBIN_INSTRUCTION,
        "current_blob": git("rev-parse", "origin/main:exchange/feedback/CURRENT.md"),
        "goal_blob": git("rev-parse", "origin/main:exchange/goals/lane-a.md"),
        "origin_main": git("rev-parse", "origin/main")}
    wi1 = HC.theta_arms()[HC.ARMS[0]]
    require(wi1["cmd"] == "DNg100" and wi1["backend"] == "metal", "wi1 command drifted")
    thetas = {name: dict(wi1, cmd=cmd) for name, cmd in ARMS.items()}
    for name in FLIGHT_ARMS:
        require(set(theta_difference(wi1, thetas[name])) == {"cmd"},
                f"{name} changes more than cmd")
    load_start = time.monotonic()
    meta, edges = pd.read_feather(IB.META_PATH), pd.read_feather(IB.EDGES_PATH)
    load_seconds = time.monotonic() - load_start
    require(len(meta) == 188508, "graph drifted")
    rows = motor_rows(meta)
    require({k: len(v) for k, v in rows.items()} == EXPECTED_COUNTS, "motor census drifted")
    cmd_rows = {name: command_rows(meta, cmd) for name, cmd in ARMS.items()}
    require(len(cmd_rows["wi1"]) == 2 and len(cmd_rows["flight_dng02"]) == 38
            and len(cmd_rows["flight_dnp31"]) == 2, "command census drifted")
    record_rows = np.unique(np.concatenate([rows["motor_all"]] + list(cmd_rows.values())))
    runs, summary, walls = {}, {}, {}
    for name in ARMS:
        print(f"START {name} seed={SEED} backend=metal cmd={ARMS[name]}", flush=True)
        arm_start = time.monotonic()
        row, rec, trace = run_arm(name, thetas[name], meta, edges, record_rows)
        walls[name] = round(time.monotonic() - arm_start, 2)
        runs[name] = row
        summary[name] = summarize(meta, rec, trace, rows, cmd_rows[name])
        print(json.dumps({"arm": name, "wall_s": walls[name],
                          "power": summary[name]["power"],
                          "command": summary[name]["command_rows"],
                          "standing": row.get("standing")}), flush=True)
        write({"partial": True, "id": "FL1", "execution_commit": head, "seed": SEED,
               "runs": runs, "summary": summary, "exact_command": COMMAND})
    decision = decide(summary)
    document = {
        "id": "FL1", "written": subprocess.check_output(
            ["date", "+%Y-%m-%dT%H:%M:%S%z"], text=True).strip(),
        "phase": "BUILD", "engine": "metal", "engine_sha256": sha(ROOT / "cns.py"),
        "whole_cns": True, "body_loop": True,
        "graph": {"neurons": len(meta), "edges": len(edges),
                  "expanded_synapses": int(edges["count"].sum())},
        "selection_seed_reused": SEED, "new_seeds_spent": 0,
        "sealed_fresh_seeds_preserved": list(SEALED_FRESH_SEEDS),
        "question": ("Does the whole CNS engage the DLM/DVM flight power motor when the "
                     "descending command is a flight population instead of DNg100?"),
        "arms": {name: {"cmd": ARMS[name], "command_rows": int(len(cmd_rows[name])),
                        "theta_difference_from_wi1": theta_difference(wi1, thetas[name])}
                 for name in ARMS},
        "readout": {"record_rows": int(len(record_rows)), "bin_ms": BIN_MS,
                    "window_ms": [SETTLE_MS, DURATION_MS],
                    "pools": {k: int(len(v)) for k, v in rows.items()},
                    "gate": {"tonic_hz": [TONIC_LO_HZ, TONIC_HI_HZ],
                             "tonic_min_active_seconds": TONIC_MIN_ACTIVE_SECONDS,
                             "engaged_min_power_cells": ENGAGED_MIN_CELLS,
                             "control_quiet_hz": CONTROL_QUIET_HZ,
                             "unphysiological_hz": UNPHYSIOLOGICAL_HZ},
                    "mechanism": "walk_search.evaluate record_rows/trace_out, read-only"},
        "biological_reference": {
            "power_motor_neurons_flight_hz": "5-20 tonic, silent at rest",
            "sources": ["Harcombe & Wyman 1977 J Comp Physiol 123:271",
                        "Gordon & Dickinson 2006 J Exp Biol 209:4183",
                        "Namiki et al. 2022 Curr Biol 32:1189 (DNg02)"],
            "not_gated": "steering (b1 ~wingbeat-locked via halteres), haltere, neck"},
        "theta_wi1": wi1, "runs": runs, "summary": summary, "decision": decision,
        "authority": authority,
        "provenance": {
            "execution_commit": head, "source_sha256": sources, "data_sha256": data_hashes,
            "prior_receipt_for_data_pin": str(PRIOR_RESULT.relative_to(ROOT)),
            "metal_source_sha256": CM._source_sha256(),
            "metal_dylib_sha256": sha(CM._LIB_PATH), "exact_command": COMMAND,
            "python": sys.version, "machine": platform.machine(),
            "numpy_version": np.__version__, "pandas_version": pd.__version__},
        "timing": {"load_seconds": round(load_seconds, 2), "arm_wall_seconds": walls,
                   "total_wall_seconds": round(time.monotonic() - started, 2)},
    }
    write(document)
    print(json.dumps({"decision": decision}), flush=True)
    print(f"WROTE {RESULT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
