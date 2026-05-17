#!/usr/bin/env bash
set -euo pipefail

# Run from the iPEPS_Z2 repository root:
#   bash data_square_CSL/run_general_random_chi40_seed0.sh
#
# This is the general iPEPS_Z2 optimization path through
# data_square_CSL/smoke_random_opt_C4PT.py. It does not use the native
# C4/PT fast optimizer.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO_ROOT}/data_square_CSL/logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/general_random_D3_chi40_seed0_${STAMP}.log"

echo "python: ${PYTHON_BIN}"
echo "repo: ${REPO_ROOT}"
echo "log: ${LOG_FILE}"

"${PYTHON_BIN}" data_square_CSL/smoke_random_opt_C4PT.py \
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
  2>&1 | tee "${LOG_FILE}"
