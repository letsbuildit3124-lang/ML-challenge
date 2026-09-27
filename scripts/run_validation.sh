#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo "ER-X Ultimate: 5-Fold Cross-Validation Evaluation"
echo "=========================================================="

python -u -m src.erx_ultimate.validate --eval-fold 0
