"""Run one fixed condition and save its numerical states for analysis/video."""
from contextlib import ExitStack
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np

from portable_integrity import ROOT, sha, verify_sources, verify_data


def simulate(out, condition="control", protocol="walk", age=None):
    from lanes.longevity.ageing import integrated_probe as IP
    from lanes.longevity.ageing import escape_membrane as EM
    from lanes.longevity.ageing import global_ageing as GA
    from karolina_experiment import apply_intervention, ADAPT
    from successor_body import dense_readout

    out = Path(out)
    if out.exists():
        raise FileExistsError(f"Refusing to replace a run: {out}")
    verify_sources()
    verify_data()
    out.mkdir(parents=True)
    started = time.perf_counter()
    candidate = json.loads((ROOT / "simulator/lanes/longevity/ageing/candidate-fixed-native-walk-visual125-lplc2-quarter-complete-ttm.json").read_text())
    if condition != "control":
        candidate.update(id=candidate["id"] + "-" + condition, karolina_intervention=condition)
    duration = 6000. if protocol == "walk" else 1100. if protocol == "sham" else 3000.
    original_body = IP.JA.ColourBody
    original_theta = IP.candidate_theta
    original_neural = IP.JA.cns.CNS.run
    bodies, live, changes = [], [], []

    class CaptureBody(original_body):
        def __init__(self, *a, **kw):
            kw["render_path"] = None
            super().__init__(*a, **kw)
            self.pose_time, self.pose_qpos, self.pose_xyz = [], [], []
            self.next_pose = 0.
            bodies.append(self)

        def _capture(self):
            d = self.h["data"]
            if d.time + 1e-9 < self.next_pose:
                return
            self.pose_time.append(float(d.time))
            self.pose_qpos.append(d.qpos.copy())
            self.pose_xyz.append(d.xpos[self.h["thorax"]].copy())
            if age is None:
                self.next_pose = len(self.pose_time) / 240.
            else:
                # Original aging capture clock, including its denser input window.
                dense = not (age == 0 and protocol == "walk") and .590 <= d.time <= .760
                self.next_pose = d.time + (.0005 if dense else 1 / 120.)

    def theta_wrapper(theta, config, meta=None):
        base = original_theta(theta, config, meta)
        changed = apply_intervention(base, condition, meta)
        difference = {k: {"before": base.get(k), "after": changed.get(k)}
                      for k in set(base) | set(changed) if base.get(k) != changed.get(k)}
        expected = set(ADAPT) if condition == "ibuprofen" else {"row_pair_seams"} if condition == "deprivation_b_half" else set()
        if set(difference) != expected:
            raise ValueError("Unexpected treatment parameter delta")
        changes.append(difference)
        return changed

    def neural_wrapper(net, *a, **kw):
        actual = {k: np.broadcast_to(np.asarray(getattr(net.p, k)), (net.N,)) for k in ADAPT}
        if condition == "ibuprofen" and any(not np.all(actual[k] == v) for k, v in ADAPT.items()):
            raise ValueError("Ibuprofen adaptation did not reach every neuron")
        live.append({"neurons": net.N, "adaptation": {
            k: {"min": float(v.min()), "max": float(v.max())} for k, v in actual.items()}})
        return original_neural(net, *a, **kw)

    try:
        with ExitStack() as stack:
            stack.enter_context(IP.patched(IP.JA, "ColourBody", CaptureBody))
            stack.enter_context(IP.patched(IP, "candidate_theta", theta_wrapper))
            stack.enter_context(IP.patched(IP.JA.cns.CNS, "run", neural_wrapper))
            result = (EM.run(candidate, protocol=protocol, duration_ms=duration) if age is None else
                      GA.run(candidate, age, protocol, duration))
        if len(bodies) != 1 or len(live) != 1 or len(changes) != 1 or result["seed"] != 0:
            raise ValueError("Expected exactly one seed-zero simulation")
        body = bodies[0]
        state = np.asarray(body.rows)
        readout = dense_readout(result, state, protocol)
        if result["review_required"]:
            raise ValueError("Source changed during execution")
        np.savez_compressed(out / "poses.npz", time_s=body.pose_time, qpos=body.pose_qpos,
                            thorax_xyz=body.pose_xyz, body_state=state)
        IP.JA.mj.mj_saveModel(body.h["model"], str(out / "body.mjb"), None)
        for name, value in (("integrated-receipt.json", result), ("readout.json", readout)):
            (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        manifest = {
            "schema": "portable-fly-experiment-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "condition": condition, "protocol": protocol, "age_days": age, "seed": 0,
            "duration_ms": duration, "source_manifest_sha256": verify_sources(),
            "files": {p.name: sha(p) for p in out.iterdir()}, "theta_changes": changes[0],
            "live_neural_parameters": live[0], "aging": result.get("global_ageing"),
            "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "mujoco", "flygym", "pandas")},
            "python": platform.python_version(), "platform": platform.platform(),
            "timing_s": {**result["timing_s"], "worker_total": time.perf_counter() - started},
            "status": "complete", "parameter_time_model": "fixed snapshot; no treatment exposure kinetics",
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"run": out.name, "seconds": manifest["timing_s"]["worker_total"],
                          "retention_pass": readout["successor_retention_pass"]}), flush=True)
    except Exception as exc:
        (out / "failure.json").write_text(json.dumps({"error": str(exc), "seconds": time.perf_counter() - started}) + "\n")
        raise
