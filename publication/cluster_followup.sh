#!/usr/bin/env bash
# sbatch --partition=normal --cpus-per-task=1 --mem=6G --array=0-5%3
# Set wall time from the completed bootstrap probe before submission.
set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?Run as the six-task Slurm array}"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_CORETYPE=Haswell MUJOCO_GL=glfw
export MPLCONFIGDIR="$PWD/outputs/matplotlib-$SLURM_JOB_ID"
mkdir -p "$MPLCONFIGDIR"
.venv/bin/python - <<'PY'
import json, subprocess
from pathlib import Path
receipt=json.loads(Path('outputs/publication/cluster-bootstrap-receipt.json').read_text())
assert receipt['status']=='complete'
assert receipt['commit']==subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
PY
case "$SLURM_ARRAY_TASK_ID" in
  0|1|2|3|4|5) ;;
  *) echo 'Task index must be 0 through 5' >&2; exit 2;;
esac
noise_seed=$((2026110301 + SLURM_ARRAY_TASK_ID))
.venv/bin/python -u publication/followup.py --seeds "$noise_seed" \
  --null-seeds 2026110311 2026110312 2026110313 2026110314 \
  --out "outputs/publication/followup-$noise_seed"
