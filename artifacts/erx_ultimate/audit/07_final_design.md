# ER-X Ultimate: Final Architecture & Production Design Synthesis

## 1. System High-Level Topology

```mermaid
flowchart TD
    subgraph Data Layer [High-Speed Data Backbone]
        TSV[Raw TSV Datasets] --> DuckDB[(DuckDB Cache & Inverted Index DB)]
        DuckDB --> Parquet[Parquet Columnar Shards]
        Parquet --> MMAP[Memory-Mapped Disk Arrays]
    end

    subgraph Retrieval Layer [Multi-Channel CSR Retrieval]
        DuckDB --> Exact[Exact / Alias CSR Index]
        DuckDB --> TFIDF[Char 3-5 TF-IDF CSR Index]
        DuckDB --> Rare[Rare Token IDF CSR Index]
        DuckDB --> Phone[Phonetic / Metaphone CSR Index]
        DuckDB --> Num[Address & Numeric CSR Index]
        DuckDB --> Typo[Learned Typo / OCR Index]
        
        Exact & TFIDF & Rare & Phone & Num & Typo --> RRF[Reciprocal Rank Fusion Engine]
        RRF --> TopK[Top-K Candidates K=30]
    end

    subgraph Modeling Layer [Vectorized LightGBM & Calibration]
        TopK --> Feat[73 Vectorized SIMD Features]
        Feat --> LGBM[LightGBM GBDT Engine]
        LGBM --> Calib[OOF Isotonic Probability Calibrator]
    end

    subgraph Decision Layer [Exclusivity & Target Ownership]
        Calib --> EDec[Expected-F0.5 Threshold Decoder]
        EDec --> TargetRes[Target Ownership & Margin Resolver]
        TargetRes --> Agg[S1 Cluster Aggregator]
        Agg --> TSVOut[matching_results.tsv & candidate_pairs.tsv]
        TSVOut --> Audit[Automated 10-Point Audit Engine]
    end
```

## 2. Key Component Specifications

### 2.1 CSR Inverted Index Representation
- Inverted index for terms $\to$ document IDs stored in flattened contiguous 1D uint32 NumPy arrays (`indptr`, `indices`, `weights`).
- Saved directly to disk (`.npy`) and accessed via zero-copy `mmap_mode="r"`.

### 2.2 Vectorized 73-Feature Extraction Matrix
- Exact Matches (12 features): Exact string match, stripped match, alphanumeric match, casefold, token set equality across name, street, city, postal code, phone, website.
- N-gram Overlaps (16 features): Char 2-gram, 3-gram, 4-gram Jaccard and Dice similarities.
- Fuzzy Distances (20 features): Levenshtein distance, Token Sort Ratio, Token Set Ratio, Partial Ratio, WRatio via batch RapidFuzz C API.
- Token Statistics & IDF (12 features): Common token count, Jaccard token overlap, max/sum token IDF weight.
- Address & Geographic Alignment (8 features): Numeric house number match, postal code prefix match, street suffix alignment, phonetic Soundex/Metaphone equality.
- Retrieval & Structural Metadata (5 features): RRF fusion rank score, channel activation bitmask, target-to-candidate length ratio, token count delta, target source provenance.

### 2.3 Expected-F0.5 Optimal Decoder
- For binary classification evaluation under Macro F0.5 ($F_{0.5} = \frac{1.25 \times P \times R}{0.25 \times P + R}$), standard threshold $p > 0.50$ is suboptimal because precision is weighted $4\times$ higher than recall ($\beta = 0.5$).
- The Bayes-optimal decision threshold for $F_\beta$ satisfies:
  $$\tau^* = \frac{1}{1 + \beta^2} = \frac{1}{1 + 0.25} = 0.80$$
- Empirical calibration tunes $\tau \in [0.72, 0.85]$ to account for channel retrieval priors and singleton noise.
