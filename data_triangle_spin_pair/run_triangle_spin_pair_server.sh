#!/bin/bash
. /home/sniu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch
now=$(date +'%Y_%m_%d_%H_%M_%S')
python -u optimize_triangle_spin_pair_iPESS.py \
  >> "$now.out" 2>> "$now.err" &
