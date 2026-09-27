# ER-X Ultimate: Cache Management & Storage Architecture

## 1. Storage Layout & Path Isolation

All persistent cache is organized under `cache/erx_ultimate/`:
```
cache/erx_ultimate/
├── entity_resolution.duckdb          # Primary analytical DuckDB engine
├── normalized/
│   ├── train_s1.parquet
│   ├── train_s2.parquet
│   ├── train_s3.parquet
│   ├── test_s1.parquet
│   ├── test_s2.parquet
│   └── test_s3.parquet
├── indices/
│   ├── s1_exact_indptr.npy
│   ├── s1_exact_indices.npy
│   ├── s1_tfidf_indptr.npy
│   ├── s1_tfidf_indices.npy
│   ├── s1_tfidf_weights.npy
│   ├── s1_token_indptr.npy
│   ├── s1_token_indices.npy
│   ├── s1_phonetic_indptr.npy
│   ├── s1_phonetic_indices.npy
│   ├── s1_address_indptr.npy
│   └── s1_address_indices.npy
└── training_shards/
    ├── shard_00.parquet
    ├── shard_01.parquet
    └── ...
```

## 2. Invalidation & Freshness Strategy
- A unique pipeline hash is computed over configuration parameters and code versions.
- If raw dataset timestamps or configurations change, the cache manager marks tables as stale and rebuilds them.
- Clean-start flag `--clean` purges `cache/erx_ultimate/` completely to ensure deterministic fresh generation.
