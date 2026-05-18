#!/bin/bash
. /home/sniu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch
now=$(date +'%Y_%m_%d_%H_%M_%S')
python -u probe_repeated_step2_grad.py \
  --instate-prefix ../ipepsz2_random_C4PT_init_D3_seed0 \
  --chi 40 \
  --ctm-iters 80 \
  --ctm-conv-tol 1.0e-8 \
  --seed 0 \
  --threads 4 \
  --device cpu \
  --lr 0.5 \
  --history-size 6 >> probe_repeated_step2_grad_seed0_$now.out 2>> probe_repeated_step2_grad_seed0_$now.err &
