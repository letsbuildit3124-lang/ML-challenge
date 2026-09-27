#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo "ER-X Ultimate: Running Deliverable Audit"
echo "=========================================================="

python -u -m src.erx_ultimate.output_audit \
    --results outputs/erx_ultimate/matching_results.tsv \
    --expected-rows 1732544
