# ER-X Ultimate: End-to-End Dataflow Proof

## 1. Single-Target Dataflow Proof (Training Mode)

Tracing a target record from raw input to training feature row:

```
[Raw Target TSV Row]
    id: 504910, source: 2
    name: "Amazon Web Services Inc.", address: "410 Terry Ave N", city: "Seattle", phone: "+1 206-266-1000"
      │
      ▼ CacheManager.build_normalized_parquet / normalization.py
[Normalized EntityRecord]
    id: 504910, source: SOURCE2
    name_norm: "amazon web", address_norm: "410 terry avenue north", city_norm: "seattle", phone_norm: "2062661000"
      │
      ▼ ERXRetrievalEngine.retrieve_candidates_for_record (ZERO GT KNOWLEDGE)
[CSR Inverted Query on S1 Indexes]
    Channel 1 (Exact): Hits on name "amazon web" -> [S1_doc: 10, S1_doc: 42]
    Channel 2 (N-gram): Hits on n-grams -> [S1_doc: 10, S1_doc: 95]
    Channel 3 (Token): Hits on "amazon", "web" -> [S1_doc: 10, S1_doc: 110]
    Channel 4 (Phonetic): Hits on Soundex -> [S1_doc: 10]
    Channel 5 (Address): Hits on "410" -> [S1_doc: 10, S1_doc: 500]
      │
      ▼ Reciprocal Rank Fusion: RRF = sum(1 / (60 + rank + 1))
[Ranked Candidate Matches]
    Rank 1: s1_id = 1010 (S1_doc: 10), rrf_score = 0.081, mask = 31 (all 5 channels)
    Rank 2: s1_id = 2042 (S1_doc: 42), rrf_score = 0.016, mask = 1 (exact only)
    Rank 3: s1_id = 5095 (S1_doc: 95), rrf_score = 0.015, mask = 2 (ngram only)
      │
      ▼ features.extract_pair_features (73 Vectorized SIMD Features)
[Candidate Feature Vectors]
    Pair (504910, 1010) -> [f_0=1.0, f_1=0.0, ..., f_72=2.0] (73 float32 elements)
    Pair (504910, 2042) -> [f_0=1.0, f_1=0.0, ..., f_72=2.0] (73 float32 elements)
      │
      ▼ Ground Truth Lookup in train_ground_truth table (POST-RETRIEVAL LABEL ATTACHMENT)
[Label Assignment]
    Is (1010, 504910, 2) in train_ground_truth? YES -> label = 1 (Positive)
    Is (2042, 504910, 2) in train_ground_truth? NO  -> label = 0 (Hard Negative)
      │
      ▼ Shard Writer
[Disk Shard] -> cache/erx_ultimate/training_shards/shard_XX.parquet
```

---

## 2. Single-Target Dataflow Proof (Inference Mode)

Tracing a target record from raw test set to final TSV cluster:

```
[Test Target TSV: test_source2.tsv]
    id: 910234, source: 2
      │
      ▼ Normalization -> EntityRecord
      ▼ CSR Blind Retrieval -> Top 30 Candidates
      ▼ 73-Feature SIMD Extraction
      ▼ LightGBM Booster -> Raw Logit / Probability (p_raw = 0.942)
      ▼ Isotonic Calibrator -> Calibrated Probability (p_cal = 0.918)
      │
      ▼ PostProcessingEngine.resolve_target_ownership
[Target Ownership Check]
    Top S1: 1084 (p_cal = 0.918 >= 0.78 threshold)
    Runner-up S1: 4402 (p_cal = 0.310 -> Margin = 0.608 >= 0.05 min_margin)
    Decision: Assign Target (910234, Source2) to S1 ID 1084.
      │
      ▼ PostProcessingEngine.aggregate_clusters
[S1 Cluster Record]
    source1_id = 1084
    source2_ids = [910234, ...]
    source3_ids = [...]
      │
      ▼ TSV Streaming Serializer
[matching_results.tsv]
    1084\t910234\t...
```
