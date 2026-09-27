# 02 — Reuse vs. Replace Architectural Decision Matrix

## Decision Philosophy
We adopt an uncompromising performance and correctness principle: **Reuse proven mathematical algorithms and low-level C-accelerated kernels; replace memory-inefficient object graphs, uncalibrated heuristics, and fragile data pathways with high-throughput native integer structures and strict validation gates.**

---

## Detailed Component Decision Matrix

| Subsystem | Component | Legacy Approach | Ultimate ER-X Decision | Technical Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Data Ingestion** | Normalization Cache | Pre-normalized Parquet via DuckDB | **REUSE & HARDEN** | Columnar Parquet with Snappy/ZSTD is optimal. Add schema hash and record count integrity validation. |
| **Data Types** | S1 In-Memory Representation | `CompactS1Record` dataclass | **REPLACE -> NumPy CSR Arrays** | Replace Python dataclass instances with contiguous 1D arrays (`uint32` IDs, precomputed token counts, integer hashes) to achieve $<350\text{ MB}$ memory footprint for 2.2M entities. |
| **Retrieval** | Inverted Postings | `Dict[str, List[int]]` | **REPLACE -> CSR uint32 Index** | Contiguous flat integer arrays (`offsets.npy` and `candidate_ids.npy`) support $O(1)$ zero-copy slice indexing and zero-RAM cross-process memory mapping (`mmap_mode="r"`). |
| **Retrieval** | Score Combination | Ad-hoc heuristic float addition | **REPLACE -> Reciprocal Rank Fusion (RRF)** | $RRF(d) = \sum_{c \in C} \frac{1}{60 + r_c(d)}$ provides rank-invariant, calibrated multi-channel fusion across disparate score distributions. |
| **Feature Extraction** | String Distances | Scalar RapidFuzz calls | **REPLACE -> Batched C-API (`process.cpdist`)** | Pure C++ SIMD multi-threaded execution (`workers=-1`) releases the Python GIL and computes 50k string comparisons in milliseconds. |
| **Feature Extraction** | Numeric & Context Features | Scalar Python arithmetic | **REPLACE -> Fully Vectorized NumPy** | All 73 features computed in-place in pre-allocated `(N, 73)` `float32` contiguous memory buffers. |
| **Model** | GBDT Classifier | LightGBM Booster | **REUSE & OPTIMIZE** | LightGBM provides best-in-class CPU throughput and split finding. Tune tree depth (`max_depth=7`, `num_leaves=63`), feature fraction, and single-threaded BLAS to eliminate CPU thrashing. |
| **Calibration** | Probability Calibration | Isotonic Regression / Platt | **REUSE & HARDEN** | Retain Isotonic regression fitted strictly on Out-Of-Fold (OOF) cross-validation predictions. |
| **Decoding** | Match Selection | Greedy argmax threshold | **REPLACE -> Target Ownership & Expected-F0.5 Decoder** | Resolve target competition before S1 aggregation; evaluate $k \in \{0, 1, \dots, N\}$ candidate subsets to maximize Macro F0.5 with strict singleton protection. |
| **Storage & Resume** | Checkpointing | Sharded Parquet with JSON metadata | **REUSE & EXPAND** | Granular shard-level MD5 checksums, auto-resume, and corrupted shard recovery. |
| **Output Pipeline** | Submission Formatting | Python dictionary accumulator | **REPLACE -> DuckDB Streaming Aggregation** | Zero-RAM streaming union of S2 and S3 shard outputs directly into `matching_results.tsv` and `candidate_pairs.tsv`. |
