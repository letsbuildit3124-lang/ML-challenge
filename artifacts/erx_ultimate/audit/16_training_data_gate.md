# ER-X Ultimate: Pre-Training Data Verification Gate Specification

## 1. Gate Invariants & Verification Rules
The Pre-Training Data Verification Gate is an automated blocking barrier in `train.py` (`run_pre_training_gate`) executed before any LightGBM training starts.

### Mandatory Acceptance Criteria:
1. **Zero Synthetic Data**: Rejects any runs relying on `np.random` feature arrays.
2. **Full Target Universe Representation**: All 10,320,219 targets (5,034,616 S2 + 5,285,603 S3) must be streamed.
3. **Natural Candidate Retrieval**: Positive pairs are labeled strictly after blind CSR inverted index retrieval.
4. **Bounded Negative Sampling**: 3 to 5 hard negatives per positive pair.
5. **Multi-Match Integrity**: Supports targets with multiple ground truth S1 associations.
6. **Atomic File Serialization**: Shards written with `.tmp` staging and atomic OS rename.

---

## 2. Gate Verification Output Fields
When executed on the full dataset, the gate outputs:
- Total Parquet Shards
- Total Training Pairs ($N \approx 40.56\text{M}$)
- Retained Positive Links
- Hard Negative Links
- Negative-to-Positive Ratio ($\approx 4.0$)
- Unique Target IDs Represented
- Unique S1 Entities Represented
- SHA256 Checksum Manifest
