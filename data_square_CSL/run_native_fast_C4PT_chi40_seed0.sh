#!/usr/bin/env bash
set -euo pipefail

# Run from the iPEPS_Z2 repository root:
#   bash data_square_CSL/run_native_fast_C4PT_chi40_seed0.sh
#
# This is the fast C4/PT-specific iPEPS_Z2 optimizer. It imposes the
# one-tensor C4/PT ansatz and uses the native fast code in this repo,
# not Juraj's package.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO_ROOT}/data_square_CSL/logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/native_fast_C4PT_D3_chi40_seed0_${STAMP}.log"
OUT_PREFIX="data_square_CSL/native_fast_C4PT_D3_chi40_seed0_${STAMP}"

echo "python: ${PYTHON_BIN}"
echo "repo: ${REPO_ROOT}"
echo "log: ${LOG_FILE}"
echo "out_prefix: ${OUT_PREFIX}"

"${PYTHON_BIN}" data_square_CSL/optimize_square_C4PT_iPEPS_native_fast.py \
  --mode opt \
  --random-init \
  --D 3 \
  --seed 0 \
  --out-prefix "${OUT_PREFIX}" \
  --chi 40 \
  --ctm-max-iter 80 \
  --ctm-conv-tol 1.0e-8 \
  --opt-max-iter 200 \
  --optimizer lbfgs \
  --lr 0.5 \
  --history-size 6 \
  --line-search backtracking \
  --max-grad-norm 0.0 \
  --target-energy -0.98 \
  --check-every 1 \
  --threads 4 \
  --device cpu \
  --dtype complex128 \
  2>&1 | tee "${LOG_FILE}"
