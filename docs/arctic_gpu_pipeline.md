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
|  - Kernel Slug: rajeshshitap/arctic-entity-resolution-worker                     |
|  - Deterministic Single-File Discovery (chunk_XXXXXX.parquet)                     |
|  - Encodes using Arctic ER on Nvidia L4 / T4 (Batched FP32 Inference)             |
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

### Step 1: Install Kaggle CLI and PyYAML (if needed)
```bash
pip install kaggle pyyaml
```

### Step 2: Kaggle Authentication via OAuth
Authenticate using the standard Kaggle CLI OAuth flow:
```bash
kaggle auth login
```
*(Alternatively, place your `kaggle.json` API token in `~/.kaggle/kaggle.json` with permissions `chmod 600 ~/.kaggle/kaggle.json`)*.

Test that the CLI connection is live:
```bash
kaggle kernels list --page-size 1
```

### Step 3: Verified Configuration (`config/arctic_gpu.yaml`)
`config/arctic_gpu.yaml` is pre-configured with your kernel slug:
```yaml
kaggle:
  kernel: "rajeshshitap/arctic-entity-resolution-worker"
  accelerator: "NvidiaL4" # or NvidiaTeslaT4
  timeout_seconds: 3600

embedding:
  model_name: "themelder/arctic-embed-xs-entity-resolution"
  dimension: 384
  batch_size: 256
  fp16: false # Standard precision for initial validation

pipeline:
  chunk_size: 100000
  chunks_per_job: 1
  max_retries: 3
  dense_top_k: 50
  resume: true
```

---

## 3. Operational Workflow Commands

### Step A: Verify Execution Plan (Dry-Run)
Verify configuration, paths, target counts, and disk space without executing any remote jobs or altering files:
```bash
PYTHONPATH=. python3 -m src.arctic_pipeline --gpu --dry-run
```

### Step B: Run 1-Chunk Validation Test (100k targets)
Run an isolated single-chunk end-to-end test to verify GPU execution on Kaggle, output retrieval, and strict positional verification:
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

The pipeline is completely fault-tolerant and checks `cache/arctic_gpu/manifest.json`. Completed chunks are automatically skipped.

- **Standard Resume**: Re-run the main command (it automatically resumes from the first pending/failed chunk):
  ```bash
  PYTHONPATH=. python3 -m src.arctic_pipeline --gpu
  ```
- **Skip Export Stage** (if Parquet chunks are already exported):
  ```bash
  PYTHONPATH=. python3 -m src.arctic_pipeline --gpu --skip-export
  ```
- **Rebuild FAISS Only** (after all chunks are downloaded):
  ```bash
  PYTHONPATH=. python3 -m src.arctic_pipeline --gpu --skip-export --skip-kaggle
  ```
- **Run Benchmark Only** (against existing FAISS index):
  ```bash
  PYTHONPATH=. python3 -m src.arctic_pipeline --gpu --skip-export --skip-kaggle --skip-faiss
  ```

---

## 5. Storage Requirements

| Storage Component | Approximate Size | Location / Notes |
| :--- | :--- | :--- |
| **Input Parquet Chunks** ($104$ chunks $\times 100\text{k}$) | $\approx 1.2\text{ GB}$ | Disk (`cache/arctic_gpu/input/`) |
| **Downloaded Chunk Embeddings** | $\approx 15.8\text{ GB}$ | Disk (`cache/arctic_gpu/output/`) |
| **Merged Persistent Target Memmap** | $\approx 15.8\text{ GB}$ | Disk (`cache/embeddings/arctic/`) |
| **Target IDs JSON Index** | $\approx 180\text{ MB}$ | Disk (`cache/embeddings/arctic/`) |
| **FAISS IVF-PQ Index ($M=48$)** | $\approx 495\text{ MB}$ | Disk & RAM ($< 1\text{ GB}$ resident RAM) |
| **Total Free Disk Buffer Recommended** | **$\approx 35\text{ GB}$** | Checked automatically before staging |

---

## 6. Model License & Competition Compliance

- **Model**: `themelder/arctic-embed-xs-entity-resolution` ($22.6\text{M}$ parameters, Apache-2.0 License).
- **Rule Compliance**:
  - Model Size: $22.6\text{M} \le 8\text{B}$ parameter cap (**Compliant**).
  - License: Apache-2.0 open license (**Compliant**).
  - External Lookups: **Strictly ZERO external lookups** (The worker encodes solely the competition-provided text fields).

---

## 7. Disabling Arctic & Returning to Pure V4 Pipeline

Arctic is an independent candidate retrieval branch. To keep it disabled:
1. In `config/arctic_gpu.yaml`, keep `retrieval.use_arctic: false`.
2. Run standard V4 retrieval:
   ```bash
   PYTHONPATH=. python3 -m src.v4_recall_benchmark --s1-count 1000 --budget 250
   ```
