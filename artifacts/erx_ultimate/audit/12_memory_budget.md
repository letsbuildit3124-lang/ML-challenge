# ER-X Ultimate: Memory Budget & OOM Vulnerability Proof

## 1. Static Allocation & Footprint Analysis

Hardware: 8 vCPU, 32 GB RAM, 0 GPU.

### System Allocation Limits:
- **Soft RSS Limit**: 12.0 GB
- **Warning Threshold**: 16.0 GB
- **Hard Safety Ceiling**: 20.0 GB

---

## 2. Stage-by-Stage Peak RSS Estimate

| Stage | Main In-Memory Structures | Estimated RSS | Safe Under 20 GB? | Risk Factor / Notes |
| :--- | :--- | :--- | :--- | :--- |
| **DuckDB Ingestion** | DuckDB buffer pool, chunk cursor | $\approx 2.5\text{ GB}$ | **YES** | `SET max_memory = '4GB'` enforced. |
| **CSR Index Build** | Python postings dictionaries (1.73M S1s) | $\approx 3.8\text{ GB}$ | **YES** | Converted directly to uint32 1D NumPy arrays on disk; Python dicts garbage-collected. |
| **Memmap Consolidation** | 1 Parquet shard at a time (500k rows) + disk memmap | $\approx 1.2\text{ GB}$ | **YES** | Shards streamed individually; previous shards cleared via `del` and `gc.collect()`. |
| **LightGBM Training** | Disk memmap reader + LightGBM histogram bins | $\approx 11.5\text{ GB}$ | **YES** | `free_raw_data=True`, `max_bin=255`, 40.56M rows $\times 73$ features in uint8/float32 bins. |
| **S2 Inference (Process A)** | S1 records lookup dict (1.73M) + micro-batch buffer (50k) | $\approx 4.8\text{ GB}$ | **YES** | 1.73M `EntityRecord` objects $\approx 2.6\text{ GB}$ + CSR memory maps $\approx 1.8\text{ GB}$. |
| **S3 Inference (Process B)** | S1 records lookup dict (1.73M) + micro-batch buffer (50k) | $\approx 4.8\text{ GB}$ | **YES** | Shared read-only CSR index pages in OS page cache; total physical RAM $\approx 9.6\text{ GB}$ combined. |
| **Post-Processing & Output** | Scored pairs ownership dict + TSV write buffer (16MB) | $\approx 1.8\text{ GB}$ | **YES** | Streaming TSV serialization directly to disk. |

---

## 3. High-Memory Risk Vectors Identified
1. **`s1_records_map` Python Dict in `final_inference.py`**:
   - `s1_records_map: Dict[int, EntityRecord]` creates 1,732,544 Python dataclass objects in heap.
   - At $\approx 1.5\text{ KB}$ per Python object overhead, this consumes $\approx 2.6\text{ GB}$ RAM in each worker process.
   - *Recommendation for future optimization*: Store S1 string columns in flat contiguous byte arrays or query columnar PyArrow tables directly by uint32 indices to reduce heap allocation from 2.6 GB down to $\approx 300\text{ MB}$.
