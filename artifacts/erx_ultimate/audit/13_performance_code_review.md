# ER-X Ultimate: CPU Performance & Hot-Path Code Review

## 1. Top 10 Runtime Bottleneck Analysis

| Rank | Operation / Stage | Code Location | Mechanism | Execution Model | Bottleneck Severity | Optimization Opportunity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Pairwise String Distance Calculation | `features.py` (`extract_pair_features`, lines 100-127) | RapidFuzz C-API vs Python loop | Per-pair loop over candidate matches | **HIGH** | Batch candidate string arrays and call `rapidfuzz.process.cdist` in C++ batch mode. |
| **2** | Micro-Batch Candidate Retrieval Loop | `retrieval.py` (`retrieve_candidates_for_record`, lines 195-281) | CSR inverted index array slicing + RRF sum | Python loop per query record | **MEDIUM-HIGH** | Vectorize posting lookup using NumPy integer array indices where possible. |
| **3** | Feature Matrix Assembly | `features.py` (`extract_batch_features`, lines 172-187) | Pre-allocated NumPy array (`zeros(N, 73)`) | Row-by-row feature population in Python | **MEDIUM** | Group identical string comparisons across pairs. |
| **4** | DuckDB Cursor Batch Ingestion | `cache_manager.py` (`build_normalized_parquet`, lines 145-195) | `cursor.fetchmany(50000)` | Batched PyArrow table serialization | **LOW** | Already optimized via Snappy Parquet streaming. |
| **5** | LightGBM Batch Prediction | `model.py` (`predict_batch`, lines 95-106) | C++ OpenMP LightGBM Booster | Native batch evaluation | **LOW** | Highly parallelized and fast. |
| **6** | Target Ownership Resolution | `postprocessing.py` (`resolve_target_ownership`, lines 25-59) | Python dict grouping & sorting | Linear scan over scored pairs | **LOW** | Fast ($O(N \log K)$ with $K \le 30$). |
| **7** | TSV Deliverable Export | `final_inference.py` (lines 244-250) | 16MB buffered file I/O | Line-by-line stream | **LOW** | Minimal I/O overhead. |

---

## 2. Multi-Threading & Thread Oversubscription Audit
- **Environment Flags**: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` set at top of `final_inference.py` and `train.py`.
- **Process Allocation**: Process A (S2) and Process B (S3) run concurrently with 4 dedicated threads each, totaling 8 threads across the 8 physical cores without scheduler contention.
