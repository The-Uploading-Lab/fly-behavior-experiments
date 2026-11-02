"""Run one six-second historical delay trial on the portable NumPy backend."""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["base", "flat6", "flat15", "noshort", "nolong", "perm"], default="base")
    parser.add_argument("--seed", type=int, default=201303)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "osmesa")
    from analyse import verify
    verify()
    # Download through the repository's existing, checksum-verified public route.
    sys.path.insert(0, str(REPO))
    from run import download
    download()
    output = REPO / "outputs/publication/delay" / f"{args.arm}-{args.seed}"
    if output.exists():
        raise FileExistsError(f"Preserve the existing trial before rerunning: {output}")
    output.mkdir(parents=True)
    work = output / "model"
    shutil.copytree(ROOT / "delay-model", work)
    for name in json.loads((REPO / "experiments/data.json").read_text()):
        (work / "data" / name).symlink_to(REPO / "simulator/data" / name)
    traces = output / "traces"
    traces.mkdir()
    os.chdir(work)
    # Historical source must precede the hackathon simulator imports added by run.py.
    sys.path.insert(0, str(work))
    started = time.perf_counter()
    import cd1sc_delay_dispersion as runner
    runner.TRACE_DIR = str(traces)
    runner._init("results-qh1-theta.json", "numpy")
    initialized = time.perf_counter()
    record = runner._job((args.arm, args.seed))
    finished = time.perf_counter()
    receipt = {"historical_commit": "88fc005eecd0e181f44f7e828750605d36ef8745",
               "backend": "numpy", "arm": args.arm, "seed": args.seed,
               "timing_seconds": {"import_and_initialization": initialized-started,
                                  "simulation_and_trace_export": finished-initialized,
                                  "total": finished-started},
               "record": record,
               "comparison_limit": "Historical archived replay used Metal. Backend RNG and numerical differences require comparison before numerical equivalence is claimed."}
    (output / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
