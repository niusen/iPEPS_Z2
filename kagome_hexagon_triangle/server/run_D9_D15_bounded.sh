#!/bin/sh
# One background process; no watchdog or automatic restart.
set -eu
task_foreground=0
case "${1:-}" in
    '') ;;
    --foreground) task_foreground=1; shift ;;
    *) printf 'Usage: sh run_D9_D15_bounded.sh [--foreground]\n' >&2; exit 2 ;;
esac
[ "$#" -eq 0 ] || exit 2
case "$0" in */*) task_parent=${0%/*} ;; *) task_parent=. ;; esac
task_dir=$(CDPATH= cd -- "$task_parent" && pwd)
cd "$task_dir/../.."
if [ "${KAGOME_SKIP_CONDA:-0}" != 1 ]; then
    task_conda_sh=${CONDA_SH:-/home/sniu/miniconda3/etc/profile.d/conda.sh}
    task_conda_env=${CONDA_ENV:-pytorch}
    if [ ! -r "$task_conda_sh" ]; then
        printf 'Cannot read conda initialization: %s\n' "$task_conda_sh" >&2
        exit 2
    fi
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
"$task_python" -B - <<'PY_CHECK'
import sys,torch,numpy,opt_einsum
from kagome_hexagon_triangle.server.run_D9_D15_bounded import settings,OUT
from kagome_hexagon_triangle.server.workflow_bounded_resume import preflight
from kagome_hexagon_triangle.server.device_support import resolve_measurement_device
cfg=settings();device=resolve_measurement_device(cfg['measurement_device'])
source=preflight(cfg,OUT)
print('Python:',sys.executable,'PyTorch:',torch.__version__,flush=True)
print('Saved source:',source['source_file'],'old steps:',source['inherited_steps'],flush=True)
print('D9: at most 2000 additional fine steps; D10..15: at most 9000 per D.',flush=True)
print('Every D inherits the preceding state; ONE final measurement per D.',flush=True)
print('Measurement:',device,'chi=',cfg['chi'],'CTMRG max=',cfg['ctm_max'],'OUT=',OUT,flush=True)
PY_CHECK
if [ "$task_foreground" = 1 ]; then
    exec "$task_python" -B -u -m kagome_hexagon_triangle.server.run_D9_D15_bounded
fi
mkdir -p "$task_dir/logs"
task_stamp=$(date +'%Y_%m_%d_%H_%M_%S')_$$
task_log="$task_dir/logs/D9_D15_bounded_${task_stamp}.log"
task_pid_file="$task_dir/logs/D9_D15_bounded_${task_stamp}.pid"
nohup "$task_python" -B -u -m kagome_hexagon_triangle.server.run_D9_D15_bounded > "$task_log" 2>&1 < /dev/null &
task_pid=$!
printf '%s\n' "$task_pid" > "$task_pid_file"
printf 'Started PID=%s\nLog: %s\nPID file: %s\nWatch: tail -f "%s"\nStop: kill -TERM %s\nResume: rerun the same sh.\n' \
    "$task_pid" "$task_log" "$task_pid_file" "$task_log" "$task_pid"
