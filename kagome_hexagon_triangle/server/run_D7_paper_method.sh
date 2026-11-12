#!/bin/sh
# POSIX sh; one background Python process, ordinary TERM stops it, no restart loop.
set -eu
task_foreground=0
case "${1:-}" in
    '') ;;
    --foreground) task_foreground=1; shift ;;
    *) printf 'Usage: sh run_D7_paper_method.sh [--foreground]\n' >&2; exit 2 ;;
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
import json,sys,torch,numpy,opt_einsum
from kagome_hexagon_triangle.server.run_D7_paper_method import settings,OUT
from kagome_hexagon_triangle.paper_method.workflow import validate,fingerprint
from kagome_hexagon_triangle.server.device_support import resolve_measurement_device,immutable_settings
from kagome_hexagon_triangle.server.workflow import writer_lock
cfg=settings();validate(cfg);device=resolve_measurement_device(cfg['measurement_device'])
OUT.mkdir(parents=True,exist_ok=True)
with writer_lock(OUT):
    marker=OUT/'settings.json'
    if marker.exists() and immutable_settings(json.loads(marker.read_text()))!=immutable_settings(cfg):
        raise ValueError('Settings changed; use a new OUT')
    marker=OUT/'provenance.json'
    if marker.exists() and json.loads(marker.read_text())['fingerprint']!=fingerprint(cfg):
        raise ValueError('Algorithm changed; use a new OUT')
print('Python:',sys.executable,'PyTorch:',torch.__version__,flush=True)
print('Physical-NN SU, six independent tensors initially; conditional C3 only.',flush=True)
print('Fixed full steps:',sum(round(s['tau']/s['dt']) for s in cfg['schedule']),
      'SVDs/step: 72 with six tensors, 24 after C3 reduction.',flush=True)
print('D=',cfg['D'],'seeds=',cfg['seeds'],'chis=',cfg['chis'],'measurement=',device,'OUT=',OUT,flush=True)
PY_CHECK
if [ "$task_foreground" = 1 ]; then
    exec "$task_python" -B -u -m kagome_hexagon_triangle.server.run_D7_paper_method
fi
mkdir -p "$task_dir/logs"
task_stamp=$(date +'%Y_%m_%d_%H_%M_%S')_$$
task_log="$task_dir/logs/D7_paper_method_${task_stamp}.log"
task_pid_file="$task_dir/logs/D7_paper_method_${task_stamp}.pid"
nohup "$task_python" -B -u -m kagome_hexagon_triangle.server.run_D7_paper_method > "$task_log" 2>&1 < /dev/null &
task_pid=$!
printf '%s\n' "$task_pid" > "$task_pid_file"
printf 'Started PID=%s\nLog: %s\nPID file: %s\nWatch: tail -f "%s"\nStop: kill -TERM %s\nResume: rerun the same sh.\n' \
    "$task_pid" "$task_log" "$task_pid_file" "$task_log" "$task_pid"
