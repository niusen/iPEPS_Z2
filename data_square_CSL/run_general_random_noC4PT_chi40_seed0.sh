#!/usr/bin/env bash
set -euo pipefail

# Run from the iPEPS_Z2 repository root:
#   bash data_square_CSL/run_general_random_noC4PT_chi40_seed0.sh
#
# This uses the general square iPEPS ansatz, IPEPS_SQUARE, without
# imposing C4 or PT symmetry. It also does not use the native C4/PT
# fast optimizer.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO_ROOT}/data_square_CSL/logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/general_random_noC4PT_D3_chi40_Lx1_Ly1_seed0_${STAMP}.log"
OUT_PREFIX="data_square_CSL/random_noC4PT_D3_chi40_Lx1_Ly1_seed0_${STAMP}"

echo "python: ${PYTHON_BIN}"
echo "repo: ${REPO_ROOT}"
echo "log: ${LOG_FILE}"
echo "out_prefix: ${OUT_PREFIX}"

"${PYTHON_BIN}" data_square_CSL/smoke_random_opt_noC4PT.py \
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
  --out-prefix "${OUT_PREFIX}" \
  2>&1 | tee "${LOG_FILE}"
