#!/bin/bash
. /home/sniu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch
now=$(date +'%Y_%m_%d_%H_%M_%S')
python -u compute_triangle_spin_entanglement_spectrum.py \
  >> "ES_$now.out" 2>> "ES_$now.err" &
