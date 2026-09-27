#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo "ER-X Ultimate: Full Universe (10.32M) Model Training"
echo "=========================================================="

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python -u -m src.erx_ultimate.train "$@"
