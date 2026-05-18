#!/bin/bash
. /home/sniu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch
now=$(date +'%Y_%m_%d_%H_%M_%S')
python -u smoke_random_opt_noC4PT.py \
  --D 3 \
  --chi 40 \
  --Lx 1 \
  --Ly 1 \
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
  --out-prefix random_noC4PT_D3_chi40_Lx1_Ly1_seed0_$now >> general_random_noC4PT_chi40_seed0_$now.out 2>> general_random_noC4PT_chi40_seed0_$now.err &
