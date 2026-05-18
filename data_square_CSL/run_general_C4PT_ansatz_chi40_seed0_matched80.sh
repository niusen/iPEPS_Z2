#!/bin/bash
. /home/sniu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch
now=$(date +'%Y_%m_%d_%H_%M_%S')
python -u smoke_random_opt_C4PT_lbfgsmod.py \
  --D 3 \
  --chi 40 \
  --maxiter 200 \
  --ctm-iters 80 \
  --ctm-conv-tol 1.0e-8 \
  --seed 0 \
  --threads 4 \
  --max-grad-norm 0.0 \
  --lr 0.5 \
  --history-size 6 \
  --line-search backtracking \
  --target-energy -0.98 \
  --check-every 1 \
  --out-prefix general_C4PT_ansatz_matched80_lbfgsmod_D3_chi40_seed0_$now >> general_C4PT_ansatz_matched80_lbfgsmod_chi40_seed0_$now.out 2>> general_C4PT_ansatz_matched80_lbfgsmod_chi40_seed0_$now.err &
