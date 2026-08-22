#!/bin/bash
. /home/sniu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch
now=$(date +'%Y_%m_%d_%H_%M_%S')
python -u smoke_random_opt_noC4PT.py \
  --D 3 \
  --chi 40 \
  --Lx 2 \
  --Ly 2 \
  --maxiter 200 \
  --ad-ctm-iters 80 \
  --ls-ctm-iters 80 \
  --seed 0 \
  --threads 4 \
  --max-grad-norm 0.0 \
  --lr 0.5 \
  --ls-maxiter 8 \
  --history-size 6 \
  --line-search backtracking \
  --optimizer lbfgsmod \
  --target-energy -0.98 \
  --out-prefix random_noC4PT_lbfgsmod_D3_chi40_Lx2_Ly2_seed0_$now >> general_random_noC4PT_lbfgsmod_Lx2_Ly2_chi40_seed0_$now.out 2>> general_random_noC4PT_lbfgsmod_Lx2_Ly2_chi40_seed0_$now.err &
