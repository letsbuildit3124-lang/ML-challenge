#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo "ER-X Ultimate: Ingestion, Normalization & CSR Index Build"
echo "=========================================================="

python -u -m src.erx_ultimate.cache_manager
