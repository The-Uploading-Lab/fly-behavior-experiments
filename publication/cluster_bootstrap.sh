#!/usr/bin/env bash
# Submit this as a one-task Slurm array on the normal partition, never execute
# a full model on the login node. All installation stays in this checkout.
set -euo pipefail
: "${SLURM_JOB_ID:?Run through Slurm}"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_CORETYPE=Haswell MUJOCO_GL=glfw
export UV_CACHE_DIR="$PWD/outputs/uv-cache"
export MPLCONFIGDIR="$PWD/outputs/matplotlib-$SLURM_JOB_ID"
mkdir -p outputs/publication "$MPLCONFIGDIR"
python3 -m venv outputs/bootstrap-env
outputs/bootstrap-env/bin/python -m pip install --disable-pip-version-check 'uv==0.12.0'
outputs/bootstrap-env/bin/uv sync --locked --python python3
# Reuse only the two exact public inputs. The normal launcher independently
# verifies these hashes again before every experiment.
python3 - <<'PY'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path('experiments/data.json').read_text())
for name, info in manifest.items():
    source = Path.home() / 'fly-wbe-data' / name
    target = Path('simulator/data') / name
    if source.exists() and not target.exists():
        if hashlib.sha256(source.read_bytes()).hexdigest() != info['sha256']:
            raise ValueError(f'Existing cluster input differs: {name}')
        target.symlink_to(source)
PY
.venv/bin/python publication/test_reproduce.py
.venv/bin/python publication/timing_fixture.py > outputs/publication/cluster-timing-fixture.json
.venv/bin/python publication/replay_delay.py --arm base --seed 201303
.venv/bin/python - <<'PY'
import hashlib, importlib.metadata as m, json, os, platform, subprocess, sys
from pathlib import Path
receipt = {'status': 'complete', 'commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
           'host': platform.node(), 'python': sys.version,
           'packages': {p:m.version(p) for p in ('numpy','scipy','pandas','mujoco','flygym')},
           'environment': {k:os.environ.get(k) for k in ('OPENBLAS_CORETYPE','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','MUJOCO_GL')},
           'job_id': os.environ['SLURM_JOB_ID'],
           'baseline': json.loads(Path('outputs/publication/delay/base-201303/result.json').read_text()),
           'lock_sha256': hashlib.sha256(Path('uv.lock').read_bytes()).hexdigest()}
Path('outputs/publication/cluster-bootstrap-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
PY
