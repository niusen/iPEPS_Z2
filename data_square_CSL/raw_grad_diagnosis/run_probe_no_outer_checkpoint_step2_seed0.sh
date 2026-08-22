#!/bin/bash
. /home/sniu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch
now=$(date +'%Y_%m_%d_%H_%M_%S')
python -u probe_no_outer_checkpoint_step2.py --chi 40 --ctm-iters 80 --ctm-conv-tol 1e-8 --seed 0 --threads 4 --lr 0.5 --history-size 6 >> probe_no_outer_checkpoint_step2_seed0_$now.out 2>> probe_no_outer_checkpoint_step2_seed0_$now.err &
