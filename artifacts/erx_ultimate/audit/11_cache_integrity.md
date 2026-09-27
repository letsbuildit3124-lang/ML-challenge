# ER-X Ultimate: Cache Integrity & Namespace Isolation Report

## 1. Namespace Audit & Path Verification

A search across the entire `src/erx_ultimate/` source tree confirmed:
- **Zero Imports from `src/erx/`**: All imports originate strictly within `src.erx_ultimate.*` or standard/third-party libraries (`duckdb`, `numpy`, `pyarrow`, `lightgbm`, `sklearn`, `rapidfuzz`).
- **Dedicated Cache Root**: All persistent cache is partitioned under `cache/erx_ultimate/`.
- **Dedicated Artifact Root**: All model weights, rules, and audit reports are written strictly under `artifacts/erx_ultimate/`.
- **Dedicated Output Root**: All deliverables are output to `outputs/erx_ultimate/`.

---

## 2. Cache Invalidation & Schema Manifest

| Cache Object | Disk Location | Input Dependencies | Invalidation Trigger | Stale Reuse Prevented? |
| :--- | :--- | :--- | :--- | :--- |
| **DuckDB Database** | `cache/erx_ultimate/entity_resolution.duckdb` | Raw TSV files (`dataset/train/*.tsv`, `dataset/test/*.tsv`) | `--clean` CLI flag, file modification timestamps | **YES** |
| **Normalized Parquet** | `cache/erx_ultimate/normalized/*.parquet` | Raw TSV datasets, `normalization.py` | Schema change, clean cache command | **YES** |
| **CSR Indexes** | `cache/erx_ultimate/indices/*.npy`, `*_vocab.json` | `test_s1.parquet` / `train_s1.parquet` | `--clean` CLI flag, index version bump | **YES** |
| **Learned Token Rules** | `artifacts/erx_ultimate/rules/learned_token_rules.json` | `train_ground_truth.tsv` | Split fold change, clean command | **YES** |
| **Training Shards** | `cache/erx_ultimate/training_shards/shard_*.parquet` | S2/S3 candidate generator | Pipeline clean start | **YES** |
| **Memmap Feature Arrays** | `cache/erx_ultimate/training_shards/consolidated_*.mmap` | Training shards | Total row count / shard modification | **YES** |
| **LightGBM Booster** | `artifacts/erx_ultimate/models/lgb_booster.txt` | Consolidated memmap dataset | New training run | **YES** |
| **Isotonic Calibrator** | `artifacts/erx_ultimate/models/isotonic_calibrator.pkl` | Validation predictions | Split fold change | **YES** |
| **Test S2/S3 Ownership** | `cache/erx_ultimate/temp_s*_ownership.pkl` | S2/S3 inference runs | Clean cache command | **YES** |
