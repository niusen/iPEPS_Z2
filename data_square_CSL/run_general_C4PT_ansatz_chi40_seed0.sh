#!/bin/bash
. /home/sniu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch
now=$(date +'%Y_%m_%d_%H_%M_%S')
python -u smoke_random_opt_C4PT.py \
  --D 3 \
  --chi 40 \
  --maxiter 160 \
  --ad-ctm-iters 8 \
  --ls-ctm-iters 30 \
  --seed 0 \
  --threads 4 \
  --max-grad-norm 1.0 \
  --step0 0.2 \
  --ls-maxiter 8 \
  --history-size 8 \
  --line-search hager_zhang \
  --target-energy -0.98 \
  --out-prefix general_C4PT_ansatz_D3_chi40_seed0_$now >> general_C4PT_ansatz_chi40_seed0_$now.out 2>> general_C4PT_ansatz_chi40_seed0_$now.err &
