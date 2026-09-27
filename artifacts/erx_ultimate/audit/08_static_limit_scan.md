# ER-X Ultimate: Static Limit & Truncation Scan Report

## 1. Executive Summary
A comprehensive static code scan of all files under `src/erx_ultimate/` was performed to identify any occurrences of hardcoded limits, top-N shortcuts, sample reductions, or unintended data truncations.

---

## 2. Scan Results Matrix

| File Path | Function / Block | Pattern Found | Code Context | Classification | Impact / Risk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `src/erx_ultimate/learned_rules.py` | `LearnedRulesEngine.fit_from_ground_truth` (Line 44) | `LIMIT 500000;` | `SELECT ... FROM train_ground_truth ... LIMIT 500000;` | **PRODUCTION LIMIT** | Truncates ground truth scan to 500k pairs out of 7.64M. Limits vocabulary of mined typos. |
| `src/erx_ultimate/train.py` | `run_training_pipeline` (Lines 141-145) | `dummy_rows = 50000 if smoke_test else 100000` | Fallback shard generation when no shards exist on disk | **PRODUCTION LIMIT** | Uses synthetic 50k/100k random features if shards were not pre-generated, skipping full 10.32M target universe candidate mining. |
| `src/erx_ultimate/train.py` | `run_training_pipeline` (Line 157) | `min(50000, total_rows)` | `val_sample = np.memmap(mmap_x, ... shape=(min(50000, total_rows), ...))` | **PRODUCTION LIMIT** | In-sample 50,000 slice used for Isotonic calibrator fitting instead of full OOF dataset. |
| `src/erx_ultimate/validate.py` | `run_validation` (Lines 94-104) | Synthetic mock clusters | `mock_clusters = [...]` | **NOT IMPLEMENTED** | Validation runner executes hardcoded synthetic 3-entity mock instead of streaming full held-out 20% validation partition. |
| `src/erx_ultimate/cache_manager.py` | `build_normalized_parquet` (Line 143) | `batch_size = 50000` | `rows = cursor.fetchmany(batch_size)` | **SAFE CHUNKING** | Memory safety cursor fetching; continues until `cursor.fetchmany()` returns empty. Streams full dataset. |
| `src/erx_ultimate/final_inference.py` | `stream_target_source_inference` (Line 73) | `batch_size: int = 50000` | `for i in range(0, total_targets, batch_size):` | **SAFE CHUNKING** | Streams full target dataset (5.03M S2, 4.93M S3) to `total_targets` with zero early termination. |
| `src/erx_ultimate/retrieval.py` | `retrieve_candidates_for_record` (Lines 215, 224, 235, 246, 260) | `[:100]`, `[:50]`, `[:top_k]` | Candidate channel truncation | **SAFE BENCHMARK** | Intended retrieval candidate bounds per channel before RRF fusion. |

---

## 3. Required Code Adjustments
1. **Remove `LIMIT 500000` in `learned_rules.py`**: Stream all 7.64M ground truth positive links through DuckDB.
2. **Implement Real Full-Universe Candidate & Negative Mining in `train.py`**: Iterate through all 10.32M targets (S2 + S3), perform blind CSR retrieval, identify positives from DuckDB GT, mine 4 hard negatives per positive, and write out chunked Parquet shards.
3. **Implement Full-Dataset 5-Fold Evaluator in `validate.py`**: Load disjoint validation entity fold, stream targets, compute full-universe Macro F0.5.
