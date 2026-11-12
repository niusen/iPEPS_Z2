#!/bin/sh
# POSIX sh; works with either `sh run_D4_D10.sh` or `bash run_D4_D10.sh`.
# Same conda selection as kagome_su_server/run_kagome_D4_seed_once.sh.
set -eu
task_foreground=0
case "${1:-}" in
    '') ;;
    --foreground) task_foreground=1; shift ;;
    *) printf 'Usage: sh run_D4_D10.sh [--foreground]\n' >&2; exit 2 ;;
esac
if [ "$#" -ne 0 ]; then
    printf 'Usage: sh run_D4_D10.sh [--foreground]\n' >&2
    exit 2
fi
case "$0" in
    */*) task_parent=${0%/*} ;;
    *) task_parent=. ;;
esac
task_dir=$(CDPATH= cd -- "$task_parent" && pwd)
cd "$task_dir/../.."
if [ "${KAGOME_SKIP_CONDA:-0}" != 1 ]; then
    task_conda_sh=${CONDA_SH:-/home/sniu/miniconda3/etc/profile.d/conda.sh}
    task_conda_env=${CONDA_ENV:-pytorch}
    if [ ! -r "$task_conda_sh" ]; then
        printf 'Cannot read conda initialization file: %s\nSet CONDA_SH, or activate your environment and set KAGOME_SKIP_CONDA=1.\n' "$task_conda_sh" >&2
        exit 2
    fi
    # conda's shell initialization may access unset variables.
    set +u
    . "$task_conda_sh"
    conda activate "$task_conda_env"
    set -u
fi
task_python=${PYTHON_BIN:-python}
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
# Preflight exits before backgrounding on missing dependencies, an unavailable
# GPU, different settings, or a currently active owner of the output folder.
# This probe does not evolve or measure a state.
"$task_python" -B - <<'PY_CHECK'
import json
import sys
print('Python:', sys.executable, flush=True)
import torch
import numpy
import opt_einsum
print('PyTorch:', torch.__version__, flush=True)
from kagome_hexagon_triangle.server.run_D4_D10 import OUT, settings
from kagome_hexagon_triangle.server.device_support import resolve_measurement_device, immutable_settings
from kagome_hexagon_triangle.server.workflow import writer_lock
config = settings()
device = resolve_measurement_device(config.get('measurement_device', 'cpu'))
OUT.mkdir(parents=True, exist_ok=True)
with writer_lock(OUT):
    saved = OUT/'settings.json'
    if saved.exists() and immutable_settings(json.loads(saved.read_text())) != immutable_settings(config):
        raise ValueError('Settings changed in this output folder. Choose a new OUT in run_D4_D10.py.')
print(f'Preflight OK: SU=cpu, CTMRG/observables={device}, output={OUT}', flush=True)
PY_CHECK

if [ "$task_foreground" = 1 ]; then
    exec "$task_python" -B -u -m kagome_hexagon_triangle.server.run_D4_D10
fi

# Default matches the user's older job scripts: return the terminal, keep
# computing after logout, and put stdout AND stderr into one named log.
# There is one long-running Python process, with no restart/watchdog loop.
mkdir -p "$task_dir/logs"
task_stamp=$(date +'%Y_%m_%d_%H_%M_%S')_$$
task_log="$task_dir/logs/D4_D10_${task_stamp}.log"
task_pid_file="$task_dir/logs/D4_D10_${task_stamp}.pid"
nohup "$task_python" -B -u -m kagome_hexagon_triangle.server.run_D4_D10 > "$task_log" 2>&1 < /dev/null &
task_pid=$!
printf '%s\n' "$task_pid" > "$task_pid_file"
printf 'Started PID=%s\nLog: %s\nPID file: %s\nWatch: tail -f "%s"\nStop: kill -TERM %s\nResume: rerun the same sh.\n' \
    "$task_pid" "$task_log" "$task_pid_file" "$task_log" "$task_pid"
