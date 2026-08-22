#!/bin/bash
. /home/sniu/miniconda3/etc/profile.d/conda.sh
conda activate pytorch
now=$(date +'%Y_%m_%d_%H_%M_%S')
python -u save_random_C4PT_init.py \
  --D 3 \
  --seed 0 \
  --device cpu \
  --out-prefix ipepsz2_random_C4PT_init_D3_seed0 >> ipepsz2_random_C4PT_init_D3_seed0_$now.out 2>> ipepsz2_random_C4PT_init_D3_seed0_$now.err &
