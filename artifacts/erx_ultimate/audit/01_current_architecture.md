# 01 — Current Architecture Forensic Audit

## Executive Summary
This document provides a comprehensive forensic analysis of the legacy ER-X pipeline, identifying structural capabilities, computational bottlenecks, memory allocation patterns, and failure modes across the data ingestion, retrieval, feature extraction, model training, and inference lifecycles.

---

## 1. System Inventory & Component Call Graph

```
Raw TSV Data (Source 1, 2, 3)
      │
      ▼
DuckDB Ingestion & Pre-Normalization (cache_manager.py)
      │  └── Writes: cache/erx/train_sX_normalized.parquet
      │
      ▼
S1 Multi-Channel Indexing (retrieval.py)
      │  ├── Channel A: Exact / Learned Canonical, Compact, Sorted Tokens
      │  ├── Channel B: Char 3-5 TF-IDF Sparse Cosine Similarity
      │  ├── Channel C: Rare Token Inverted Index with Sublinear IDF Weighting
      │  ├── Channel D: Address & House Number / Numeric Signature Index
      │  ├── Channel E: Phonetic Token Signature Index (Double Metaphone/Soundex)
      │  └── Channel F: Learned Typo / OCR Variant Keys
      │
      ▼
Target Streaming & Candidate Retrieval (retrieval.py)
      │  └── Candidate Union & Deduplication
      │
      ▼
Feature Extraction (features.py)
      │  └── 73-Feature Extraction (Name, Address, Numeric, Cross-field, Context, Provenance)
      │
      ▼
GBDT Model Training & Probability Calibration (model.py / final_train.py)
      │  ├── LightGBM Pairwise Classifier
      │  └── Isotonic Probability Calibrator
      │
      ▼
Dual-Source Inference & Output Merge (final_inference.py)
      ├── Process A: S2 Target Stream -> Intermediate Shards
      ├── Process B: S3 Target Stream -> Intermediate Shards
      └── DuckDB Final Aggregation -> output/matching_results.tsv & output/candidate_pairs.tsv
```

---

## 2. Forensic Breakdown by Lifecycle Stage

### 2.1 Ingestion & Normalization (`cache_manager.py` / `normalization.py`)
- **Strengths**: Pre-normalizing raw TSVs into Snappy/ZSTD Parquet tables avoids repeated text parsing.
- **Weaknesses**:
  - Legacy parsing instantiated large Python `MultiViewRecord` dataclasses in memory for millions of rows.
  - Legal suffix regexes were repeatedly evaluated if not strictly precompiled.
  - DuckDB connection parameters were not uniformly shared across worker threads.

### 2.2 Retrieval Engine (`retrieval.py`)
- **Strengths**: 6 complementary channels capture diverse lexical, structural, and phonetic match patterns.
- **Weaknesses**:
  - Legacy indexes used nested Python `defaultdict(list)` structures, incurring substantial pointer-chasing overhead and GIL contention during multi-threaded lookups.
  - Channel score combination lacked principled rank fusion (e.g. Reciprocal Rank Fusion), relying on heuristic float additions.

### 2.3 Feature Engineering (`features.py`)
- **Strengths**: 73 features cover fine-grained lexical similarities (Levenshtein, Jaro-Winkler, Token-Sort, Token-Set), n-gram Dice/Jaccard, address component alignments, house number conflicts, and candidate-competition context.
- **Weaknesses**:
  - Per-candidate Python scalar dispatch loops caused excessive GIL locking and string copy operations.
  - Vectorization was incomplete across string-token sets.

### 2.4 Training & Shard Consolidation (`final_train.py`)
- **Strengths**: Chunked streaming over all 104 shards ensured full 10.32M target coverage without GT injection.
- **Weaknesses**:
  - When loading all 40.56M candidate rows at the final step, loading an entire consolidated Arrow table into RAM caused memory spikes ($>27\text{ GB}$) that exceeded VM memory without disk-backed memory-mapped staging.

### 2.5 Inference & Merging (`final_inference.py`)
- **Strengths**: Multi-process split (S2 on 4 cores, S3 on 4 cores) achieves true parallelism without Python GIL bottlenecks.
- **Weaknesses**:
  - Requires automated checksum verification and strict validation gates to guarantee 100% S1 row coverage ($1,732,544$ rows) and strict target exclusivity.
