# ER-X Ultimate: Architecture Specification

## 1. System Scope & Namespace Isolation
The `erx_ultimate` system is a complete clean-room implementation of the business entity resolution engine for the Amazon ML Challenge 2026.

### Directory Structure & Namespace Rules
```
src/erx_ultimate/
├── __init__.py
├── types.py             # Strongly-typed dataclasses & compact array definitions
├── config.py            # Centralized typed configuration dataclass
├── resource_monitor.py  # Background memory/CPU telemetry and backpressure controls
├── normalization.py     # SIMD/precompiled text & address normalization pipelines
├── cache_manager.py     # Fresh DuckDB instance and Parquet cache management
├── learned_rules.py     # In-fold typo/alias/OCR rule induction engine
├── retrieval.py         # 6-channel CSR uint32 inverted indexes + RRF fusion
├── features.py          # Vectorized 73-feature extractor using RapidFuzz C-API
├── model.py            # LightGBM GBDT with memmap streaming & Isotonic calibration
├── postprocessing.py    # Target ownership resolver & Expected-F0.5 cluster decoder
├── train.py             # 10.32M universe training with disk-backed memmap consolidation
├── validate.py          # 5-fold disjoint entity cross-validation & F0.5 metrics
├── final_inference.py   # Dual-process (S2/S3) streaming inference engine
└── output_audit.py      # Automated 10-point deliverable validation engine

artifacts/erx_ultimate/
├── audit/               # Audit documentation (01 through 07)
├── design/              # Architecture, dataflow, and performance specifications
├── config/              # YAML config files
├── rules/               # Extracted typo and alias dictionaries
└── models/              # LightGBM model binary and Isotonic calibrator dumps

cache/erx_ultimate/
├── entity_resolution.duckdb  # Dedicated DuckDB database file
├── normalized/               # Normalized Parquet partitions for S1, S2, S3
├── indices/                  # CSR uint32 inverted index arrays (.npy)
└── training_shards/          # Intermediate training feature shards (.parquet)

outputs/erx_ultimate/
├── matching_results.tsv      # Final entity clusters keyed by S1
├── candidate_pairs.tsv       # Top candidate retrieval pairs
└── submission_manifest.json  # Checksums, record counts, and run telemetry
```

## 2. Invariant: Absolute Namespace Segregation
- No imports from `src/erx` or legacy modules.
- No reading from or writing to `cache/erx` or `artifacts/erx`.
- Every artifact and cache file is created fresh under `erx_ultimate`.
