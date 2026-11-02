"""Run paired muscle, delay and inhibitory-scaling controls on the frozen walker.

This is a new controlled experiment on the September build, not a replay of
the confounded August comparison or the later October inhibitory study.
"""
from pathlib import Path
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SOURCE_COMMIT = "88fc005eecd0e181f44f7e828750605d36ef8745"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean(value):
    """JSON has no NaN: keep missing numerical results explicit."""
    import math
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def stiffness_schedule(reference, mode, assignment_seed):
    """Keep joint identity; alter only the time dependence of baseline stiffness."""
    import numpy as np
    reference = np.asarray(reference)
    if reference.ndim != 2 or len(reference) < 4 or not np.isfinite(reference).all():
        raise ValueError("A finite time-by-joint reference with at least four ticks is required")
    if mode == "stiffness-replay":
        return reference.copy(), {"operation": "identity"}
    if mode == "stiffness-mean":
        return np.broadcast_to(reference.mean(axis=0), reference.shape).copy(), {
            "operation": "per-joint temporal mean"}
    if mode == "stiffness-shift-after-startup":
        # Frozen 2.5 ms body tick: preserve the first 500 ms exactly.
        startup = 200
        if len(reference) < startup + 4:
            raise ValueError("Startup-preserving shift needs at least 204 ticks")
        tail = reference[startup:]
        offset = int(np.random.default_rng(assignment_seed).integers(
            len(tail) // 4, 3 * len(tail) // 4 + 1))
        shifted = reference.copy()
        shifted[startup:] = np.roll(tail, offset, axis=0)
        np.testing.assert_array_equal(shifted[:startup], reference[:startup])
        np.testing.assert_array_equal(np.sort(shifted[startup:], axis=0), np.sort(tail, axis=0))
        return shifted, {"operation": "common circular shift after preserved startup",
                         "startup_ticks": startup, "offset_ticks": offset}
    if mode == "stiffness-shift":
        offset = int(np.random.default_rng(assignment_seed).integers(
            len(reference) // 4, 3 * len(reference) // 4 + 1))
        shifted = np.roll(reference, offset, axis=0)
        np.testing.assert_array_equal(np.sort(shifted, axis=0), np.sort(reference, axis=0))
        return shifted, {"operation": "common circular time shift", "offset_ticks": offset}
    raise ValueError(mode)


def delay_control(reference, mode, assignment_seed):
    import numpy as np
    reference = np.asarray(reference)
    if mode == "delay-uniform":
        return np.full_like(reference, reference.mean())
    if mode == "delay-narrow-clock-mean":
        # The September engine rounds to 0.1 ms slots with a one-slot floor.
        # Two adjacent slot values are the narrowest integer distribution
        # with exactly the original integer slot sum.
        if reference.ndim != 1 or not len(reference) or not np.isfinite(reference).all():
            raise ValueError("A nonempty finite delay vector is required")
        slots = np.maximum(1, np.rint(reference / .1)).astype(np.int64)
        lower, high_count = divmod(int(slots.sum()), len(slots))
        narrow = np.full(len(slots), lower, dtype=np.int64)
        selected = np.random.default_rng(assignment_seed).permutation(len(slots))[:high_count]
        narrow[selected] += 1
        result = narrow.astype(float) * .1
        np.testing.assert_array_equal(np.maximum(1, np.rint(result / .1)), narrow)
        assert int(narrow.sum()) == int(slots.sum())
        return result
    if mode == "delay-shuffle":
        result = reference[np.random.default_rng(assignment_seed).permutation(len(reference))]
        np.testing.assert_array_equal(np.sort(result), np.sort(reference))
        return result
    return reference.copy()


def mechanical_metrics(arrays, dt_s):
    """Descriptive readouts; no animal-validation or energy-efficiency score."""
    import numpy as np
    body, quat = arrays["body"], arrays["quat"]
    trans = int(round(.5 / dt_s))
    norm = np.sum(quat * quat, axis=1)
    tilt = np.degrees(np.arccos(np.clip(
        1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2) / norm, -1, 1)))
    # Post-transient sustained posture loss: at least 50 ms continuously.
    bad = (tilt >= 51.68) | (body[:, 2] <= 1.0)
    width = int(round(.05 / dt_s))
    hits = np.flatnonzero(np.convolve(bad[trans:].astype(int), np.ones(width, int), "valid") == width)
    fall = int(trans + hits[0]) if len(hits) else None
    end = fall if fall is not None else len(body) - 1
    delta = body[end, :2] - body[trans, :2]
    force, achieved = arrays["actuator_force"], arrays["achieved_joint_angle"]
    # Trapezoidal torque-angle sum in FlyGym model units, sampled each control tick.
    work = .5 * (force[1:] + force[:-1]) * np.diff(achieved, axis=0)
    limits = arrays["force_limits"]
    saturated = ((force <= limits[:, 0] + 1e-8) | (force >= limits[:, 1] - 1e-8))
    contact = arrays["loads"] > 1.0  # same threshold as cruse1.CONTACT_N
    slip = np.linalg.norm(np.diff(arrays["tips"][:, :, :2], axis=0), axis=2) / dt_s
    mask = contact[1:] & contact[:-1]
    return {"sustained_posture_loss_s": None if fall is None else (fall + 1) * dt_s,
            "displacement_before_posture_loss_mm": float(np.linalg.norm(delta)),
            "net_displacement_mm": float(np.linalg.norm(body[-1, :2] - body[trans, :2])),
            "joint_tracking_rmse_rad": float(np.sqrt(np.mean((arrays["target_joint_angle"][trans:] - achieved[trans:]) ** 2))),
            "actuator_force_limit_fraction": float(saturated[trans:].mean()),
            "positive_actuator_work_model_units": float(np.maximum(work[trans:], 0).sum()),
            "negative_actuator_work_model_units": float(np.minimum(work[trans:], 0).sum()),
            "contact_tip_speed_mean_mm_s": float(slip[trans:][mask[trans:]].mean()) if mask[trans:].any() else None}


def main(followup=False):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=([2026110301] if followup else [201303]), help="Paired neural noise seeds")
    ap.add_argument("--null-seeds", type=int, nargs="+", default=([2026110311, 2026110312, 2026110313, 2026110314] if followup else [9001]), help="Independent assignment/shift seeds")
    ap.add_argument("--out", type=Path, default=REPO / ("outputs/publication/followup" if followup else "outputs/publication/controls"))
    args = ap.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.null_seeds)) != len(args.null_seeds):
        ap.error("Duplicate seeds would duplicate deterministic trials")
    if any(s < 0 for s in args.seeds + args.null_seeds):
        ap.error("Seeds must be nonnegative")
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"Preserve existing results and choose a fresh --out: {output}")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "osmesa")
    from analyse import verify
    verify()
    sys.path.insert(0, str(REPO))
    from run import download
    download()
    output.mkdir(parents=True)
    work = output / "model"
    shutil.copytree(HERE / "delay-model", work)
    for name in json.loads((REPO / "experiments/data.json").read_text()):
        (work / "data" / name).symlink_to(REPO / "simulator/data" / name)
    protocol_path = HERE / ("FOLLOWUP_PROTOCOL.md" if followup else "PROTOCOL.md")
    source_files = [HERE / "reproduce.py", protocol_path, HERE / "manifest.json", REPO / "uv.lock"]
    if followup:
        source_files.extend([HERE / "followup.py", HERE / "analyse_followup.py", HERE / "analyse_controls.py"])
    fixed_modes = ["base", "stiffness-replay", "stiffness-mean", "activation", "delay-uniform", "inhibition-unscaled", "motor-zero"]
    null_modes = ["stiffness-shift", "delay-shuffle"]
    if followup:
        null_modes += ["stiffness-shift-after-startup", "delay-narrow-clock-mean"]
    provenance = {"historical_source_commit": SOURCE_COMMIT, "backend": "numpy",
                  "release_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                  "source_hashes": {str(p.relative_to(REPO)): sha(p) for p in source_files},
                  "python": sys.version, "platform": platform.platform(),
                  "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "pandas", "mujoco", "flygym")},
                  "noise_seeds": args.seeds, "null_seeds": args.null_seeds, "duration_ms": 6000,
                  "protocol_version": 2 if followup else 1,
                  "execution_host": platform.node(), "machine": platform.machine(),
                  "numerical_environment": {k: os.environ.get(k) for k in ("OPENBLAS_CORETYPE", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NPY_DISABLE_CPU_FEATURES")},
                  "slurm": {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK")},
                  "status": "running", "rows": []}
    write_json(output / "summary.json", provenance)
    os.chdir(work)
    sys.path.insert(0, str(work))
    import numpy as np
    import cd1sc_delay_dispersion as runner
    import walk_search as W
    import muscle_split as MS
    import abdomen as AB
    import cruse1
    if cruse1.CONTACT_N != 1.0:
        raise ValueError("Contact threshold changed; update the explicit metrics definition")
    original_delay = W.length_delay_vector
    init_start = time.perf_counter()
    with (output / "initialization.log").open("w") as log, contextlib.redirect_stdout(log):
        runner._init("results-qh1-theta.json", "numpy")
    provenance["initialization_seconds"] = time.perf_counter() - init_start
    original_controller = MS.StiffnessController
    original_abdomen_step = AB.AbdomenAdapter.step
    active = {}

    def observed_abdomen_step(self, counts, cscale, tick_ms):
        if active["mode"] == "motor-zero":
            counts = np.zeros_like(counts)
        drive = original_abdomen_step(self, counts, cscale, tick_ms)
        active["abdomen_drive"].append(np.asarray(drive).copy())
        return drive

    def controlled_delay(meta, mean_ms, cap_ms, groups):
        reference = original_delay(meta, mean_ms, cap_ms, groups)
        result = delay_control(reference, active["mode"], active["null_seed"])
        active["delay_ms"] = result.copy()
        runner._BASE["delay_vec_stats"] = {"base": {"mean": float(result.mean()), "min": float(result.min()), "max": float(result.max())}}
        return result

    class ObservedController(original_controller):
        def __init__(self, sim, fly, *a, **kw):
            super().__init__(sim, fly, *a, **kw)
            from flygym.compose.fly.base_fly import ActuatorType
            ids = np.asarray(sim._intern_actuatorids_by_type_by_fly[ActuatorType.POSITION][fly.name])
            # The historical capture records 42 leg actuators, excluding abdomen.
            active["force_limits"] = self.m.actuator_forcerange[ids[:42]].copy()
            active["modulated_actuator_ids"] = self.ids.copy()
            active["modulated_dof_indices"] = self.mask.copy()

        def _write(self, k):
            index = len(active["stiffness"])
            if active["schedule"] is not None:
                if index >= len(active["schedule"]):
                    raise ValueError("Stiffness replay exhausted")
                k = active["schedule"][index]
            super()._write(k)
            np.testing.assert_array_equal(self.m.actuator_gainprm[self.ids, 0], -self.m.actuator_biasprm[self.ids, 1])
            active["stiffness"].append(np.asarray(k).copy())

    W.length_delay_vector = controlled_delay
    MS.StiffnessController = ObservedController
    AB.AbdomenAdapter.step = observed_abdomen_step
    try:
        for seed in args.seeds:
            baseline = None
            baseline_neural_hash = None
            cases = [(m, None) for m in fixed_modes]
            cases += [(m, ns) for ns in args.null_seeds for m in null_modes]
            for mode, null_seed in cases:
                label = f"{mode}-{seed}" + (f"-null{null_seed}" if null_seed is not None else "")
                case = output / label
                case.mkdir()
                active.clear()
                active.update(mode=mode, null_seed=null_seed, stiffness=[], abdomen_drive=[], schedule=None)
                intervention = {}
                if mode.startswith("stiffness-"):
                    active["schedule"], intervention = stiffness_schedule(baseline["stiffness"], mode, null_seed)
                overrides = {"joint_int": "exact"}
                if mode == "activation":
                    overrides["k_mode"] = "cocon"
                if mode == "inhibition-unscaled":
                    overrides["bal_target"] = 0.0
                if mode == "motor-zero":
                    overrides["neural_motor_scale"] = 0.0
                runner.ARMS[label] = overrides
                runner.TRACE_DIR = str(case)
                full_theta = dict(runner._BASE["th"], **overrides)
                write_json(case / "parameters.json", full_theta)
                start = time.perf_counter()
                print(f"Running {label}", flush=True)
                with (case / "run.log").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    record = runner._job((label, seed))
                cap = runner._BASE["cap"]
                arrays = {name: np.array([c[index] for c in cap]) for name, index in {
                    "body": 0, "quat": 1, "target_joint_angle": 3, "loads": 4, "tips": 10,
                    "actuator_force": 11, "achieved_joint_angle": 12, "motor_drive": 17}.items()}
                arrays.update({k: np.asarray(active[k]) for k in ("stiffness", "abdomen_drive", "delay_ms", "force_limits", "modulated_actuator_ids", "modulated_dof_indices")})
                if len(arrays["stiffness"]) != 2400 or len(cap) != 2400:
                    raise ValueError("Expected 2400 control ticks, including the first 500 ms")
                if not all(np.isfinite(v).all() for v in arrays.values()):
                    raise ValueError("Nonfinite simulation output")
                if baseline is not None:
                    np.testing.assert_array_equal(arrays["force_limits"], baseline["force_limits"])
                    np.testing.assert_array_equal(arrays["modulated_dof_indices"], baseline["modulated_dof_indices"])
                if mode == "base":
                    baseline = {k: v.copy() for k, v in arrays.items()}
                neural_rows, neural_trace = runner._BASE["last_trace"]
                neural_hash = hashlib.sha256(np.ascontiguousarray(neural_trace).tobytes()).hexdigest()
                if mode == "base":
                    baseline_neural_hash = neural_hash
                if mode == "stiffness-replay":
                    for k, v in baseline.items():
                        np.testing.assert_array_equal(arrays[k], v, err_msg=f"Identity replay changed {k}")
                    intervention["all_mechanical_arrays_identical"] = True
                    if neural_hash != baseline_neural_hash:
                        raise ValueError("Identity replay changed neural activity")
                    intervention["neural_activity_identical"] = True
                if mode == "motor-zero":
                    np.testing.assert_array_equal(arrays["motor_drive"], np.zeros_like(arrays["motor_drive"]))
                    np.testing.assert_array_equal(arrays["abdomen_drive"], np.zeros_like(arrays["abdomen_drive"]))
                net = list(runner.RW._G["cache"].values())[-1]
                scale = net.balance_scale
                if mode == "inhibition-unscaled" and scale is not None:
                    raise ValueError("Inhibitory scaling was not disabled")
                arrays["balance_scale"] = np.ones(net.N) if scale is None else np.asarray(scale)
                np.savez_compressed(case / "mechanics.npz", **arrays)
                row = {"case": label, "mode": mode, "seed": seed, "null_seed": null_seed,
                       "parameter_overrides": overrides, "intervention": intervention,
                       "neural_counts_sha256": neural_hash,
                       "wall_seconds": time.perf_counter() - start, "historical_metrics": record,
                       "mechanical_metrics": mechanical_metrics(arrays, full_theta["tick_ms"] / 1000.),
                       "files": {p.name: sha(p) for p in sorted(case.iterdir()) if p.is_file()}}
                write_json(case / "result.json", row)
                provenance["rows"].append(row)
                write_json(output / "summary.json", provenance)
                print(f"Finished {label}: standing={record['standing_v32']}, speed={record.get('speed')}, {row['wall_seconds']:.1f}s", flush=True)
        provenance["status"] = "complete"
        write_json(output / "summary.json", provenance)
    except BaseException as exc:
        provenance["status"] = "failed"
        provenance["error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json(output / "summary.json", provenance)
        raise
    finally:
        W.length_delay_vector = original_delay
        MS.StiffnessController = original_controller
        AB.AbdomenAdapter.step = original_abdomen_step


if __name__ == "__main__":
    main()
