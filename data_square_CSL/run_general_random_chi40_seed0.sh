#!/bin/bash
. /home/sniu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch
now=$(date +'%Y_%m_%d_%H_%M_%S')
python -u smoke_random_opt_C4PT.py \
  --D 3 \
  --chi 40 \
  --maxiter 80 \
  --ad-ctm-iters 4 \
  --ls-ctm-iters 12 \
  --seed 0 \
  --threads 4 \
  --max-grad-norm 1.0 \
  --step0 0.25 \
  --ls-maxiter 4 \
  --history-size 4 \
  --target-energy -0.98 \
  --out-prefix general_random_C4PT_ansatz_D3_chi40_seed0_$now >> general_random_C4PT_ansatz_chi40_seed0_$now.out 2>> general_random_C4PT_ansatz_chi40_seed0_$now.err &
