#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo "ER-X Ultimate: Dual-Process Final Production Inference"
echo "=========================================================="

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Launch S2 and S3 processes concurrently
python -u -m src.erx_ultimate.final_inference --source S2 > logs_inference_s2.log 2>&1 &
PID_S2=$!

python -u -m src.erx_ultimate.final_inference --source S3 > logs_inference_s3.log 2>&1 &
PID_S3=$!

echo "Inference running -> Process A (S2): PID $PID_S2 | Process B (S3): PID $PID_S3"
wait $PID_S2
wait $PID_S3

# Consolidate final results
python -u -m src.erx_ultimate.final_inference

# Run deliverable audit
python -u -m src.erx_ultimate.output_audit --results outputs/erx_ultimate/matching_results.tsv
