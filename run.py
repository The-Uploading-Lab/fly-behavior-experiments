"""One entry point for the two hackathon experiments."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT / "simulator"),
                str(ROOT / "simulator/lanes/longevity/production"),
                str(ROOT / "simulator/lanes/longevity/escape")]
from portable_integrity import sha, verify_sources


def download():
    for name, info in json.loads((ROOT / "experiments/data.json").read_text()).items():
        path = ROOT / "simulator/data" / name
        if path.exists() and sha(path) == info["sha256"]:
            continue
        if path.exists():
            raise ValueError(f"Input hash mismatch: {path}. Preserve it elsewhere before downloading the pinned input.")
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(path.suffix + ".part")
        print(f"Downloading {name} ({info['bytes'] / 1e6:.1f} MB)", flush=True)
        with urllib.request.urlopen(info["url"], timeout=90) as source, part.open("wb") as dest:
            while block := source.read(4 * 1024 * 1024):
                dest.write(block)
        if part.stat().st_size != info["bytes"] or sha(part) != info["sha256"]:
            raise ValueError(f"Download checksum failed: {part}")
        part.replace(path)
    print("Pinned BANC inputs verified.", flush=True)


def verified_run(path):
    manifest = path / "manifest.json"
    if not manifest.exists():
        return False
    m = json.loads(manifest.read_text())
    if m["source_manifest_sha256"] != verify_sources():
        raise ValueError(f"Run belongs to different source: {path}; choose a fresh --out directory")
    for name, expected in m["files"].items():
        if sha(path / name) != expected:
            raise ValueError(f"Changed run output: {path / name}")
    return m["status"] == "complete"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment", choices=("download", "wetlab", "aging", "_case"))
    ap.add_argument("--out", type=Path, help="Output directory; completed identical runs are verified and reused")
    ap.add_argument("--video", action="store_true", help="Also render MP4s from the computed poses")
    ap.add_argument("--days", type=int, nargs="+", help="Aging subset from 0,10,...,90; default is all ten ages")
    ap.add_argument("--condition", choices=("control", "ibuprofen", "deprivation_b_half"), default="control", help=argparse.SUPPRESS)
    ap.add_argument("--protocol", choices=("walk", "puff", "sham"), default="walk", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.experiment == "download":
        download()
        return
    if args.experiment == "_case":
        from simulate import simulate
        os.chdir(ROOT / "simulator")
        simulate(args.out, args.condition, args.protocol, args.days[0] if args.days else None)
        return
    if args.days and args.experiment != "aging":
        ap.error("--days applies only to aging")
    days = sorted(set(args.days)) if args.days else list(range(0, 91, 10))
    if any(d not in range(0, 91, 10) for d in days):
        ap.error("Ages must be 0,10,...,90")
    verify_sources()
    download()
    out = (args.out or ROOT / "outputs" / args.experiment).resolve()
    out.mkdir(parents=True, exist_ok=True)
    jobs = ([(f"{c}-{p}", c, p, None) for c in ("control", "ibuprofen", "deprivation_b_half") for p in ("walk", "puff")]
            if args.experiment == "wetlab" else
            [(f"day-{d:02d}-{p}", "control", p, d) for d in days for p in ("walk", "puff", "sham")])
    for i, (name, condition, protocol, day) in enumerate(jobs, 1):
        folder = out / "runs" / name
        if verified_run(folder):
            print(f"[{i}/{len(jobs)}] Verified completed run: {name}", flush=True)
            continue
        print(f"[{i}/{len(jobs)}] Simulating {name}", flush=True)
        command = [sys.executable, str(ROOT / "run.py"), "_case", "--out", str(folder),
                   "--condition", condition, "--protocol", protocol]
        if day is not None:
            command += ["--days", str(day)]
        subprocess.run(command, cwd=ROOT, check=True)
    os.chdir(ROOT / "simulator")
    from statistics_output import wetlab, aging
    if args.experiment == "wetlab":
        wetlab(out)
    else:
        aging(out, days)
    if args.video:
        from video import render
        for name, _, _, _ in jobs:
            render(out / "runs" / name, out / "videos" / f"{name}.mp4")
    print(f"Results: {out}", flush=True)


if __name__ == "__main__":
    main()
