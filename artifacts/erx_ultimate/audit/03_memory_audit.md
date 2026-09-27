# ER-X Ultimate: Memory & OOM Elimination Audit

## 1. Executive Summary & Root Cause Analysis
During previous pipeline runs on the 8-vCPU / 32 GB RAM VM, process terminations (OOM killer signal 9) occurred during candidate consolidation and feature training. The root causes were:

1. **In-Memory Arrow/Numpy Merging**:
   - 40.56M candidate rows across 12 training shards loaded into memory simultaneously.
   - 73 float32 features per row = $40,563,360 \times 73 \times 4\text{ bytes} \approx 11.84\text{ GB}$.
   - Adding Arrow table overhead ($\approx 15.3\text{ GB}$), label vectors ($\approx 0.16\text{ GB}$), query groups ($\approx 0.08\text{ GB}$), and intermediate buffers led to peak RSS exceeding 27.2 GB, exceeding the 32 GB system budget when combined with OS caches and background buffers.

2. **DuckDB Query Materialization Spikes**:
   - Unconstrained `duckdb.query().to_arrow_table()` or `df()` executes synchronous vector operations allocating large transient arenas before garbage collection.

3. **Subprocess Thread Multiplication**:
   - Python `ProcessPoolExecutor` with default BLAS/OpenMP thread allocation caused each worker process to spawn 8 OpenMP threads ($8 \text{ workers} \times 8 \text{ threads} = 64 \text{ threads}$), thrashing CPU caches and causing memory fragmentation in `malloc` arena caches.

---

## 2. Hard Limits and Memory Budget Matrix

| Component | Max Safe Allocation | Invariant / Enforcement Mechanism |
| :--- | :--- | :--- |
| **DuckDB Buffer Pool** | $\le 4.0\text{ GB}$ | `SET max_memory = '4GB'; SET threads = 4;` |
| **S1 CSR Inverted Indexes** | $\le 2.5\text{ GB}$ | Zero-copy disk `mmap_mode='r'` for uint32 indptr & indices |
| **Inference Streaming Buffer** | $\le 1.5\text{ GB}$ | Micro-batches of 50,000 targets; garbage collected per chunk |
| **Feature Extraction Buffers** | $\le 1.0\text{ GB}$ | Pre-allocated reusable NumPy float32 arrays per worker |
| **Training Consolidation** | $\le 0.5\text{ GB}$ RAM | Disk-backed `np.memmap` (`consolidated_X.mmap`, `consolidated_y.mmap`) |
| **LightGBM Native Dataset** | $\le 14.0\text{ GB}$ | `free_raw_data=True`, `max_bin=255`, 2-pass histogram construction |
| **OS & Python Runtime** | $\le 4.5\text{ GB}$ | Kernel buffers, page tables, Python GC headroom |
| **TOTAL PEAK CEILING** | **$\le 28.0\text{ GB}$** | **Hard abort guard at 20.0 GB soft threshold** |

---

## 3. Zero-Copy Architecture & Memory Mapping Specifications

### 3.1 Training Consolidation via Disk-Backed Memmap
Rather than loading 40.56M rows into memory:
1. Shards write directly to individual `.parquet` files on disk during candidate generation.
2. The consolidation stage scans Parquet metadata to determine total row count $N$.
3. Pre-allocates disk files:
   - `consolidated_X.mmap`: Shape $(N, 73)$, dtype `float32` on NVMe.
   - `consolidated_y.mmap`: Shape $(N,)$, dtype `uint8` on NVMe.
   - `consolidated_groups.npy`: Shape $(M,)$, dtype `uint32`.
4. Streams each shard into the memory-mapped slices in chunks of 500,000 rows without keeping previous shards in RAM.
5. LightGBM constructs binary dataset directly from memory-mapped arrays with `free_raw_data=True`.

### 3.2 Dual-Process Inference Memory Isolation
- S2 (5.03M targets) and S3 (4.93M targets) execute in separate OS processes.
- Read-only S1 CSR inverted indexes are mapped with `np.load(..., mmap_mode="r")`.
- The Linux page cache shares these identical physical RAM pages across both S2 and S3 processes without double-allocating RAM ($2.2\text{ GB}$ total physical RAM shared).
- Memory footprints:
  - Process A (S2): $\approx 4.8\text{ GB}$ RSS (including private working heap).
  - Process B (S3): $\approx 4.8\text{ GB}$ RSS.
  - Combined system footprint: $< 12.0\text{ GB}$ physical RSS.

---

## 4. OOM Guard & Real-Time Telemetry
`ResourceMonitor` thread monitors memory at 100ms intervals:
- **Normal Zone ($< 12\text{ GB}$)**: Maximum throughput, batch size 50,000.
- **Warning Zone ($12\text{ GB} - 16\text{ GB}$)**: Explicit `gc.collect()`, batch size throttled to 25,000.
- **Critical Zone ($> 16\text{ GB}$)**: Flush buffers to disk, force DuckDB checkpointing.
- **Hard Safety Ceiling ($20\text{ GB}$)**: Fail-safe exit with detailed crash telemetry before OS kernel OOM killer fires.
