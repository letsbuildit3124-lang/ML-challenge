# ER-X Ultimate: Performance & Concurrency Plan

## 1. Concurrency Model: Dual-Process Architecture

```
[System: 8 vCPU / 32 GB RAM Linux VM]
│
├── Process A: S2 Inference Engine (4 Cores)
│   ├── Target Universe: 5,034,616 records
│   ├── Private Heap: ~4.8 GB RSS
│   ├── Direct Memory-Mapped Index: S1 CSR (Shared Read-Only Page Cache)
│   └── Output Shard: cache/erx_ultimate/temp_s2_matches.parquet
│
└── Process B: S3 Inference Engine (4 Cores)
    ├── Target Universe: 4,934,973 records
    ├── Private Heap: ~4.8 GB RSS
    ├── Direct Memory-Mapped Index: S1 CSR (Shared Read-Only Page Cache)
    └── Output Shard: cache/erx_ultimate/temp_s3_matches.parquet
```

## 2. Low-Level Performance Optimizations
1. **Zero Thread Contention**:
   - Environment variables explicitly set to `1` thread for OpenMP, MKL, BLAS, and NumExpr.
   - Eliminates thread scheduler context switches.
2. **C-Level RapidFuzz Batching**:
   - `rapidfuzz.process.cdist` and `rapidfuzz.distance.Levenshtein.distance` executed over C arrays.
3. **DuckDB High-Performance Ingestion**:
   - Uses zero-copy Arrow table scanning with streaming cursors.
   - Disk spilling enabled to `/tmp/duckdb_spill` if memory reaches 4GB limit.
4. **Fast TSV Streaming Writer**:
   - Line-by-line buffered I/O with 16MB write buffer.
   - Avoids serializing entire output dataframes into RAM.
