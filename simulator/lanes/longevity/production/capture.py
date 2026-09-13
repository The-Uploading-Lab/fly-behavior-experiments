"""Capture the existing demo's body poses and neural activity once.

Simulation uses the lane's unchanged run/evaluate path. Presentation can then
change cameras and layout by replaying poses, without rerunning or retuning it.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ESC = ROOT / "lanes/longevity/escape"
sys.path[:0] = [str(ROOT), str(ESC)]
import jumpfly_arms as JA
import walk_arms as WA


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def network_rows(meta, subsample=16):
    """Same display-only position filter as canonical netview.NetworkView."""
    pos = meta["position"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    ok = ~np.isnan(pos).any(axis=1)
    pos = np.where(np.isnan(pos), np.nanmedian(pos, axis=0), pos)
    lo, hi = np.percentile(pos, (.1, 99.9), axis=0)
    return np.flatnonzero(np.all((pos >= lo) & (pos <= hi), axis=1) & ok)[::subsample]


def source_files():
    paths = set()
    for module in list(sys.modules.values()):
        f = getattr(module, "__file__", None)
        if f:
            p = Path(f).resolve()
            if p.suffix == ".py" and p.is_relative_to(ROOT) and ".venv" not in p.parts:
                paths.add(p)
    paths.update(ESC / f for f in ("theta-wj1-jump-film.json", "jumpfly-stance-v2-best.json"))
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def run(args):
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"run from {ROOT}")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)
    start = time.perf_counter()
    meta = pd.read_feather(ROOT / "data/banc_888_meta.feather")
    rows = network_rows(meta)
    bodies, traces = [], []
    original_body, original_evaluate = JA.ColourBody, JA.W.evaluate

    class CaptureBody(original_body):
        def __init__(self, *a, **k):
            k["render_path"] = None
            super().__init__(*a, **k)
            self.pose_time, self.pose_qpos, self.pose_qvel, self.pose_ctrl = [], [], [], []
            self.pose_act, self.pose_xyz = [], []
            self.next_pose = 0.0
            bodies.append(self)

        def _capture(self):
            d = self.h["data"]
            if d.time + 1e-9 < self.next_pose:
                return
            self.pose_time.append(float(d.time))
            self.pose_qpos.append(d.qpos.copy())
            self.pose_qvel.append(d.qvel.copy())
            self.pose_ctrl.append(d.ctrl.copy())
            self.pose_act.append(d.act.copy())
            self.pose_xyz.append(d.xpos[self.h["thorax"]].copy())
            dense = (args.behavior == "escape" and
                     (args.onset_ms - 10) * .001 <= d.time <= (args.onset_ms + 160) * .001)
            self.next_pose = d.time + (.0005 if dense else 1/120)

    def record_evaluate(*a, **k):
        k["record_rows"] = rows
        k["trace_out"] = traces
        return original_evaluate(*a, **k)

    before = source_files()
    data_identity = {}
    for name in ("banc_888_meta.feather", "banc_888_edgelist_simple_v3.feather"):
        p = ROOT / "data" / name
        data_identity[name] = {"sha256": sha256(p), "bytes": p.stat().st_size,
                               "mtime_ns": p.stat().st_mtime_ns}
    preparation_s = time.perf_counter() - start
    JA.ColourBody, JA.W.evaluate = CaptureBody, record_evaluate
    sim_start = time.perf_counter()
    try:
        if args.behavior == "walking":
            receipt = WA.run(args.arm, args.seed, dur_ms=args.duration_ms)
        else:
            receipt = JA.run(args.arm, args.seed, dur_ms=args.duration_ms,
                             flash_ms=args.onset_ms, land_at_s=args.land_s, cmd_hz=104)
    finally:
        JA.ColourBody, JA.W.evaluate = original_body, original_evaluate
    simulation_s = time.perf_counter() - sim_start
    after = source_files()
    changed = [p for p, h in before.items() if after.get(p) != h]
    for name, info in data_identity.items():
        p = ROOT / "data" / name
        if p.stat().st_size != info["bytes"] or p.stat().st_mtime_ns != info["mtime_ns"]:
            changed.append("data/" + name)
    if changed:
        raise RuntimeError(f"inputs changed during capture: {changed}")
    if len(bodies) != 1 or len(traces) != 1:
        raise RuntimeError(f"expected one body and trace, got {len(bodies)}, {len(traces)}")
    b = bodies[0]
    rec_rows, activity = traces[0]
    cols = np.searchsorted(rec_rows, rows)
    if not np.array_equal(np.asarray(rec_rows)[cols], rows):
        raise RuntimeError("recorded neuron identity mismatch")
    activity = np.asarray(activity)[:, cols]
    if not np.isfinite(activity).all() or (activity < 0).any():
        raise RuntimeError("invalid neural activity")
    archive = args.out / "capture.npz"
    model_path = args.out / "body.mjb"
    JA.mj.mj_saveModel(b.h["model"], str(model_path), None)
    np.savez_compressed(archive, time_s=b.pose_time, qpos=b.pose_qpos, qvel=b.pose_qvel,
                        ctrl=b.pose_ctrl, act=b.pose_act, thorax_xyz=b.pose_xyz,
                        body_state=np.asarray(b.rows), network_rows=rows, activity=activity)
    raw = args.out / "receipt.json"
    raw.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    body_state = np.asarray(b.rows)
    manifest = {
        "schema": "longevity-production-capture-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "behavior": args.behavior, "arm": args.arm, "seed": args.seed,
        "seed_scope": "existing_demo_replay; not_new_walking_or_title_evidence",
        "duration_ms": args.duration_ms, "onset_ms": args.onset_ms if args.behavior == "escape" else None,
        "land_s": args.land_s if args.behavior == "escape" else None,
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "current_blob": subprocess.check_output(["git", "rev-parse", "origin/main:exchange/feedback/CURRENT.md"], text=True).strip(),
        "command": [sys.executable, *sys.argv], "cwd": str(ROOT),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy", "mujoco", "flygym")},
        "engine": "numpy", "engine_reason": "pinned cns_metal.py refuses gap junctions and per-neuron spike regularity",
        "source_sha256": after, "data": data_identity, "capture_sha256": sha256(archive),
        "body_model_sha256": sha256(model_path),
        "receipt_sha256": sha256(raw), "network_display_subsample": 16, "network_bin_ms": 5.0,
        "pose_frames": len(b.pose_time), "network_bins": len(activity), "displayed_neurons": len(rows),
        "posture": {"samples": len(body_state), "min_upright": float(body_state[:, 2].min()),
                    "fraction_upright_ge_half": float((body_state[:, 2] >= .5).mean()),
                    "final_upright": float(body_state[-1, 2])},
        "limits": ["walking joint targets transferred from separate sensory body",
                   "walking rest schedule imposed for treatment arms",
                   "flight power, attitude and landing programmed"],
        "timing_s": {"preparation": preparation_s, "simulation": simulation_s,
                     "total": time.perf_counter() - start},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": str(args.out), "timing_s": manifest["timing_s"],
                      "posture": manifest["posture"], "pose_frames": len(b.pose_time)}, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("behavior", choices=("escape", "walking"))
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--duration-ms", type=float, required=True)
    ap.add_argument("--onset-ms", type=float, default=1000)
    ap.add_argument("--land-s", type=float, default=2.1)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.arm not in (WA.WALK_FRACTION if a.behavior == "walking" else JA.ARMS):
        ap.error("arm is not defined by the pinned model")
    run(a)
