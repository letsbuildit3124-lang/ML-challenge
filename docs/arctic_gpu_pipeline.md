# Antigravity V4.1 — Arctic GPU Orchestration Pipeline Guide

This guide explains how to execute automated GPU-accelerated Arctic dense embedding generation (`themelder/arctic-embed-xs-entity-resolution`) using Kaggle as a remote batch worker.

---

## 1. System Architecture

```
+-----------------------------------------------------------------------------------+
|                                 EC2 MASTER NODE                                   |
|                                                                                   |
|  [Persistent DuckDB Cache]                                                        |
|  targets (10.32M rows)                                                            |
|          |                                                                        |
|          v                                                                        |
|  [1. Export Chunks]                                                               |
|  cache/arctic_gpu/input/chunk_000000.parquet ... chunk_000103.parquet             |
|  cache/arctic_gpu/manifest.json (Checkpoints & SHA256)                            |
|          |                                                                        |
|          v (kaggle kernels push)                 (kaggle kernels output)          |
|    +-------------+                             +--------------------+             |
|    | Staging Job |                             | Verified Retrieval |             |
|    +-------------+                             +--------------------+             |
|           |                                               ^                       |
+-----------|-----------------------------------------------|-----------------------+
            |                                               |
            v                                               |
+-----------------------------------------------------------------------------------+
|                            KAGGLE REMOTE GPU WORKER                               |
|                                                                                   |
|  - Receives chunk_XXXXXX.parquet                                                  |
|  - Encodes using Arctic ER on Nvidia L4 / T4 (FP16 Batched Inference)            |
|  - Generates L2-Normalized float32 embeddings + Positional Target IDs             |
|  - Computes Output SHA256 & writes metadata                                       |
+-----------------------------------------------------------------------------------+
            |
            v (On EC2)
+-----------------------------------------------------------------------------------+
|                         LOCAL ASSEMBLY & BENCHMARKING                             |
|                                                                                   |
|  [4. Positional Verification] (assert target_ids[i] == DuckDB[i])                 |
|          |                                                                        |
|          v                                                                        |
|  [5. Persistent Storage] cache/embeddings/arctic/target_embeddings.npy (Memmap)   |
|          |                                                                        |
|          v                                                                        |
|  [6. FAISS Index Builder] cache/ann/arctic/target.index (IndexIVFPQ ~495MB RAM)   |
|          |                                                                        |
|          v                                                                        |
|  [7. Hybrid Benchmark] V4 Sparse + Arctic Dense Recall @ K (F0.5 Optimized)      |
|          |                                                                        |
|          v                                                                        |
|  [8. Detailed Report] reports/arctic_gpu_benchmark.md                             |
+-----------------------------------------------------------------------------------+
```

---

## 2. One-Time Setup & Authentication

### Step 1: Install Kaggle CLI (if not already installed)
```bash
pip install kaggle pyyaml
```

### Step 2: Configure Kaggle API Token
1. Go to [https://www.kaggle.com/settings](https://www.kaggle.com/settings) $\to$ **API** $\to$ Click **Create New Token**.
2. Download `kaggle.json` and place it on your EC2 instance:
   ```bash
   mkdir -p ~/.kaggle
   mv /path/to/downloaded/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```
3. Test authentication:
   ```bash
   kaggle --version
   ```

### Step 3: Configure `config/arctic_gpu.yaml`
Open `config/arctic_gpu.yaml` and set your Kaggle username:
```yaml
kaggle:
  kernel: "YOUR_KAGGLE_USERNAME/arctic-entity-resolution-worker"
  accelerator: "NvidiaL4" # or NvidiaTeslaT4
```

---

## 3. Operational Workflow Commands

### Step A: Verify Execution Plan (Dry-Run)
Verify that paths, chunk sizes, and storage limits are valid without making remote calls:
```bash
PYTHONPATH=. python3 -m src.arctic_pipeline --gpu --dry-run
```

### Step B: Run 1-Chunk Validation Test (100k targets)
Run an isolated single-chunk end-to-end smoke test:
```bash
PYTHONPATH=. python3 -m src.arctic_pipeline --gpu --limit-chunks 1
```

### Step C: Execute Full Production Pipeline (10.32M targets)
Run the automated end-to-end orchestration for the entire target universe:
```bash
PYTHONPATH=. python3 -m src.arctic_pipeline --gpu
```

---

## 4. Recovery & Resumability

The pipeline is completely fault-tolerant and saves checkpoint status to `cache/arctic_gpu/manifest.json`.

- **If EC2 or Kaggle disconnects**: Simply re-run `python3 -m src.arctic_pipeline --gpu`. It automatically skips already completed & verified chunks.
- **To skip export and continue Kaggle execution**:
  ```bash
  PYTHONPATH=. python3 -m src.arctic_pipeline --gpu --skip-export
  ```
- **To rebuild only the FAISS index from downloaded embeddings**:
  ```bash
  PYTHONPATH=. python3 -m src.arctic_pipeline --gpu --skip-export --skip-kaggle
  ```
- **To run only the benchmark against the finished index**:
  ```bash
  PYTHONPATH=. python3 -m src.arctic_pipeline --gpu --skip-export --skip-kaggle --skip-faiss
  ```

---

## 5. Storage Requirements

| Storage Component | Approximate Size | Location / Notes |
| :--- | :--- | :--- |
| **Input Parquet Chunks** | ~1.2 GB | `cache/arctic_gpu/input/` (Compressed Zstandard) |
| **Chunk Output Embeddings** | ~15.8 GB | `cache/arctic_gpu/output/` (104 chunk `.npy` files) |
| **Merged Target Memmap** | ~15.8 GB | `cache/embeddings/arctic/target_embeddings.npy` (Disk-backed) |
| **Merged Target IDs** | ~180 MB | `cache/embeddings/arctic/target_ids.json` |
| **FAISS IVF-PQ Index** | ~495 MB | `cache/ann/arctic/target.index` (Bounded RAM < 1 GB) |
| **Total Free Disk Buffer Required** | **~35 GB** | Verified before job staging |

---

## 6. Model License & Competition Compliance

- **Model Used**: `themelder/arctic-embed-xs-entity-resolution` ($384$-dimensional dense representation, Apache-2.0 License).
- **Rule Compliance**:
  - Max parameters: $22\text{M} \le 8\text{B}$ (Compliant).
  - External lookup: **ZERO external data lookups** (The worker encodes only the challenge text columns).
  - License: Apache-2.0 open license.

---

## 7. Disabling Arctic & Returning to Pure V4 Pipeline

Arctic dense retrieval is an independent, additive branch. The base V4 sparse + deterministic pipeline remains fully functional without it.

To keep Arctic disabled in candidate generation:
1. Ensure `use_arctic: false` in `config/arctic_gpu.yaml`.
2. Run standard V4 retrieval:
   ```bash
   PYTHONPATH=. python3 -m src.v4_recall_benchmark --s1-count 1000 --budget 250
   ```
