#!/bin/sh
# Edit run.py options. POSIX sh; one background process, no restart loop.
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
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 PYTHONDONTWRITEBYTECODE=1
"$task_python" -B -c 'from kagome_hexagon_triangle.dmi.server.run import OPTIONS; from kagome_hexagon_triangle.server.device_support import resolve_measurement_device; print(resolve_measurement_device(OPTIONS["measurement_device"]))'
mkdir -p "$task_dir/../data/logs"
task_stamp=$(date +'%Y_%m_%d_%H_%M_%S')_$$
task_log="$task_dir/../data/logs/dmi_${task_stamp}.log"
nohup "$task_python" -B -u -m kagome_hexagon_triangle.dmi.server.run > "$task_log" 2>&1 < /dev/null &
task_pid=$!
printf '%s\n' "$task_pid" > "$task_dir/../data/logs/dmi_${task_stamp}.pid"
printf 'PID=%s\nLog: %s\nStop: kill -TERM %s\n' "$task_pid" "$task_log" "$task_pid"
