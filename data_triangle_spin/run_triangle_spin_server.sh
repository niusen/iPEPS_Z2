#!/bin/bash

. /home/sniu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

cd "$(dirname "$0")"
mkdir -p logs

now=$(date +'%Y_%m_%d_%H_%M_%S')
python -u optimize_triangle_spin_iPESS.py \
    >> "logs/${now}.out" \
    2>> "logs/${now}.err" &

pid=$!
echo "job started: pid=${pid}"
echo "$pid" > "logs/${now}.pid"
