"""Verify the portable source and input snapshot without research-fleet refs."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_sources():
    manifest = ROOT / "provenance/source-sha256.json"
    for name, expected in json.loads(manifest.read_text()).items():
        path = ROOT / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Source differs from the released snapshot: {name}")
    return sha(manifest)


def verify_data():
    for name, info in json.loads((ROOT / "experiments/data.json").read_text()).items():
        path = ROOT / "simulator/data" / name
        if not path.is_file() or sha(path) != info["sha256"]:
            raise ValueError(f"Missing or changed input {name}; run: uv run python run.py download")
