#!/bin/bash
. /home/sniu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch
now=$(date +'%Y_%m_%d_%H_%M_%S')
python -u optimize_square_C4PT_iPEPS_native_fast.py \
  --mode opt \
  --random-init \
  --D 3 \
  --seed 0 \
  --out-prefix native_fast_C4PT_D3_chi40_seed0_$now \
  --chi 40 \
  --ctm-max-iter 80 \
  --ctm-conv-tol 1.0e-8 \
  --opt-max-iter 200 \
  --optimizer lbfgs \
  --lr 0.5 \
  --history-size 6 \
  --line-search backtracking \
  --max-grad-norm 0.0 \
  --target-energy -0.98 \
  --check-every 1 \
  --threads 4 \
  --device cpu \
  --dtype complex128 >> native_fast_C4PT_chi40_seed0_$now.out 2>> native_fast_C4PT_chi40_seed0_$now.err &
