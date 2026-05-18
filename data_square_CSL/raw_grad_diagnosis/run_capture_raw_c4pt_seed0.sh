#!/bin/bash
. /home/sniu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch
now=$(date +'%Y_%m_%d_%H_%M_%S')
python -u capture_raw_c4pt_lbfgs.py \
  --instate-prefix ../ipepsz2_random_C4PT_init_D3_seed0 \
  --out-dir capture_470_seed0_$now \
  --D 3 \
  --chi 40 \
  --maxiter 4 \
  --ctm-iters 80 \
  --ctm-conv-tol 1.0e-8 \
  --seed 0 \
  --threads 4 \
  --device cpu \
  --lr 0.5 \
  --history-size 6 \
  --line-search backtracking >> capture_470_seed0_$now.out 2>> capture_470_seed0_$now.err &
