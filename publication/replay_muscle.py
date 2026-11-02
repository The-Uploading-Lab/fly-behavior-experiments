"""Recompute paired August muscle-configuration trials from their original source."""
from pathlib import Path
import argparse
import contextlib
import hashlib
import json
import os
import platform
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="46", help="One seed or an inclusive range, e.g. 46-75")
    ap.add_argument("--out", type=Path, default=REPO / "outputs/publication/muscle-history")
    args = ap.parse_args()
    try:
        parts = [int(x) for x in args.seeds.split("-")]
        lo, hi = (parts * 2) if len(parts) == 1 else parts
        if not 46 <= lo <= hi <= 75:
            raise ValueError
    except ValueError:
        ap.error("Use one archived seed or an inclusive range within 46-75")
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(f"Preserve existing results and choose a fresh --out: {output}")
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "osmesa")
    from analyse import verify
    verify()
    manifest = json.loads((HERE / "muscle-model/manifest.json").read_text())
    for row in manifest["files"]:
        p = HERE / "muscle-model" / row["path"]
        if hashlib.sha256(p.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Changed historical source: {p}")
    sys.path.insert(0, str(REPO))
    from run import download
    download()
    output.mkdir(parents=True)
    model = output / "model"
    shutil.copytree(HERE / "muscle-model", model)
    for name in json.loads((REPO / "experiments/data.json").read_text()):
        (model / "data" / name).symlink_to(REPO / "simulator/data" / name)
    os.chdir(model)
    sys.path.insert(0, str(model))
    import title_match as runner
    archived = json.loads((HERE / "data/muscle-configuration-rows.json").read_text())
    if runner.CHAMP != archived["champ"] or runner.CAND5 != archived["cand5"]:
        raise ValueError("Historical configuration mismatch")
    by = {(r["who"], r["seed"]): r for r in archived["results"]}
    result = {"source_commit": manifest["source_commit"], "results_commit": manifest["results_commit"],
              "source_manifest_sha256": hashlib.sha256((HERE / "muscle-model/manifest.json").read_bytes()).hexdigest(),
              "runner_sha256": hashlib.sha256((HERE / "replay_muscle.py").read_bytes()).hexdigest(),
              "environment_lock_sha256": hashlib.sha256((REPO / "uv.lock").read_bytes()).hexdigest(),
              "platform": platform.platform(), "python": sys.version,
              "duration_ms": runner.DUR, "seeds": list(range(lo, hi + 1)),
              "configurations": {"champ": runner.CHAMP, "cand5": runner.CAND5},
              "status": "running", "rows": [],
              "limit": "This is a comparison of two complete configurations. Command and other parameters changed with the muscle interface. The pinned modern environment is not the original August environment."}

    def save():
        (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    try:
        runner._init()
        import walk_search
        for seed in range(lo, hi + 1):
            for who, theta in (("champ", runner.CHAMP), ("cand5", runner.CAND5)):
                start = time.perf_counter()
                print(f"Running {who}-{seed}", flush=True)
                runner._G["cache"].clear()
                with (output / f"{who}-{seed}.log").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    # Call the same evaluator directly so exceptions retain their
                    # traceback instead of becoming zero-valued result records.
                    score, raw, up, z = walk_search.evaluate(
                        runner._G["cache"], runner._G["meta"], runner._G["edges"],
                        theta, seed=seed, dur_ms=runner.DUR)
                record = {"who": who, "seed": seed, "score": score, "raw": raw,
                          "upright": up, "z_end": z}
                reference = by[who, seed]
                differences = {k: record[k] - reference[k] for k in ("score", "raw", "upright", "z_end")}
                row = {"record": record, "archived": reference, "new_minus_archived": differences,
                       "all_four_metrics_exact": all(x == 0. for x in differences.values()),
                       "wall_seconds": time.perf_counter() - start}
                result["rows"].append(row)
                save()
                print(json.dumps(row), flush=True)
        result["status"] = "complete"
        save()
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        save()
        raise


if __name__ == "__main__":
    main()
