# ER-X Ultimate: End-to-End Dataflow Specification

## 1. Pipeline Phases

```mermaid
sequenceDiagram
    autonumber
    actor CLI as Orchestration Script
    participant DM as CacheManager (DuckDB)
    participant NORM as Normalization Engine
    participant RET as CSR Retrieval Engine
    participant FEAT as Feature Extractor
    participant ML as LightGBM / Calibrator
    participant POST as Post-Processing Engine
    participant OUT as Output Deliverables

    CLI->>DM: Ingest raw TSVs (Train / Test)
    DM->>NORM: Normalize names, addresses, phones, URLs
    NORM->>DM: Save normalized columnar Parquet partitions
    
    rect rgb(240, 248, 255)
    Note over DM,RET: Retrieval & Candidate Generation
    DM->>RET: Build 6-channel CSR Inverted Indexes on S1
    RET-->>RET: Persist indptr, indices, weights to disk (.npy)
    RET->>RET: Stream S2/S3 queries, compute RRF scores, extract Top-K (K=30)
    end

    rect rgb(255, 245, 238)
    Note over RET,ML: Feature Extraction & Model Execution
    RET->>FEAT: Pass Candidate Pairs (Query ID, S1 Candidate IDs, RRF ranks)
    FEAT->>FEAT: Vectorized SIMD 73-Feature Extraction via RapidFuzz C-API
    FEAT->>ML: Predict pairwise match probabilities via LightGBM
    ML->>ML: Apply Isotonic Probability Calibration
    end

    rect rgb(245, 255, 245)
    Note over ML,OUT: Decision, Aggregation & Audit
    ML->>POST: Scored Candidate Graph
    POST->>POST: Resolve Target Competition (Max 1 S1 per Target)
    POST->>POST: Apply Expected-F0.5 Decision Thresholds
    POST->>OUT: Aggregate S2/S3 matches by S1 primary key
    OUT->>OUT: Export matching_results.tsv & candidate_pairs.tsv
    OUT->>CLI: Automated 10-Point Invariant Audit Pass
    end
```

## 2. In-Memory Streaming & Micro-Batching
1. Raw records are loaded in chunks of 50,000 via DuckDB cursor fetches.
2. Retrieval queries the memory-mapped CSR indexes in C arrays without converting to Python objects.
3. RapidFuzz features are computed in C-level array batches.
4. Intermediate pairs are streamed directly into output TSV file handles.
5. RSS is continuously kept $< 12.0\text{ GB}$.
