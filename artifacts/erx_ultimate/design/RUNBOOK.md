# ER-X Ultimate: Production Runbook & Operational Guide

## 1. Quick Start / Commands Summary

All commands are run from the project root directory.

### Step 1: Ingestion & Cache Preparation
```bash
python -u -m src.erx_ultimate.cache_manager --action prepare --clean
```
- Ingests raw TSV datasets into `cache/erx_ultimate/entity_resolution.duckdb`.
- Performs SIMD normalization and builds normalized Parquet files.
- Builds CSR inverted indexes for S1.

### Step 2: 5-Fold Validation & Evaluation
```bash
python -u -m src.erx_ultimate.validate --folds 5 --eval-fold 0
```
- Runs end-to-end retrieval, feature extraction, LightGBM training, calibration, and Expected-F0.5 decoding on held-out 20% fold.
- Reports exact Macro F0.5 score.

### Step 3: Full-Universe Model Training (10.32M Universe)
```bash
python -u -m src.erx_ultimate.train --shards 12 --num-threads 8
```
- Generates 40.56M candidate training pairs in 12 disk-backed Parquet shards.
- Consolidates shards directly into memory-mapped arrays (`consolidated_X.mmap`, `consolidated_y.mmap`).
- Trains final production LightGBM model and fits Isotonic probability calibrators.
- Saves model artifacts to `artifacts/erx_ultimate/models/`.

### Step 4: High-Throughput Production Inference (9.97M Test Universe)
```bash
# S2 and S3 split across two processes for maximum throughput
python -u -m src.erx_ultimate.final_inference --dual-process --threads 4
```
- Spawns Process A for S2 (5.03M targets, 4 threads) and Process B for S3 (4.93M targets, 4 threads).
- Uses zero-copy memory-mapped CSR inverted indexes.
- Decodes target matches with Expected-F0.5 thresholding.
- Generates `outputs/erx_ultimate/matching_results.tsv` and `outputs/erx_ultimate/candidate_pairs.tsv`.

### Step 5: Automated 10-Point Output Audit
```bash
python -u -m src.erx_ultimate.output_audit --results outputs/erx_ultimate/matching_results.tsv
```
- Validates row counts, S1 ordering, S2/S3 exclusivity, singletons, and formats.
- Generates `outputs/erx_ultimate/submission_manifest.json`.
