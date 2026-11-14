#!/bin/sh
# POSIX sh, background job and log. All physics settings are in the Python file.
set -eu
task_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$task_dir/../../.."
if [ "${KAGOME_SKIP_CONDA:-0}" != 1 ]; then
    set +u
    . "${CONDA_SH:-/home/sniu/miniconda3/etc/profile.d/conda.sh}"
    conda activate "${CONDA_ENV:-pytorch}"
    set -u
fi
task_python=${PYTHON_BIN:-python}
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
"$task_python" -B -c 'import sys,torch; from kagome_hexagon_triangle.dmi.server.run_independent_random_server import SERVER_OPTIONS; from kagome_hexagon_triangle.server.device_support import resolve_measurement_device; print(sys.executable,torch.__version__,resolve_measurement_device(SERVER_OPTIONS["measurement_device"]))'
mkdir -p "$task_dir/../data/logs"
task_stamp=$(date +'%Y_%m_%d_%H_%M_%S')_$$
task_log="$task_dir/../data/logs/independent_random_${task_stamp}.log"
nohup "$task_python" -B -u -m kagome_hexagon_triangle.dmi.server.run_independent_random_server > "$task_log" 2>&1 < /dev/null &
task_pid=$!
printf '%s\n' "$task_pid" > "$task_dir/../data/logs/independent_random_${task_stamp}.pid"
printf 'PID=%s\nLog: %s\nStop: kill -TERM %s\nResume: rerun the same sh with unchanged settings.\n' "$task_pid" "$task_log" "$task_pid"
