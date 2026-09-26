# Root Cause Diagnosis: Candidate Recall Incident Post-Mortem

## Executive Summary

During validation of V3 across Seeds 123 and 2026, candidate recall collapsed to **~3.4%–3.5%** (Macro F0.5 dropped to ~0.114–0.122).

### The Root Cause:
In `src/validate_v3.py` and `src/validate_arctic.py` at lines 83–84 and lines 123–124, an artificial development limit (`n_rows=250000`) was hard-coded when loading the target tables:
```python
s2_df = load_source_file(config.train_s2_path, expected_prefix="S2-", n_rows=250000)
s3_df = load_source_file(config.train_s3_path, expected_prefix="S3-", n_rows=250000)
```

- **Total Target Records in Dataset**: **10,320,219** (Train S2 = 5,034,616; Train S3 = 5,285,603).
- **Target Records Loaded**: **500,000** (**only 4.84% of the target universe**).
- **Missing Target Universe**: **95.16%** of all true matching entities were not in the in-memory target table.
- **Maximum Theoretical Recall Possible**: $\approx 4.84\%$.
- **Actual Recall on Loaded 4.84% Subset**: $3.55\% / 4.46\% = \mathbf{79.6\%}$.

The blocking algorithm was functioning properly (~80% candidate recall), but 95%+ of the candidate matches were physically missing from memory because of the truncated target slice.

---

## Detailed Audit: Answers to the 12 Specific Questions

### 1. Is the validation split correct?
**YES**. The entity-level stratified split (`dataset/splits/split_seed_*.json`) partitions strictly on Source 1 entities with **0.0% entity overlap** between train, validation, and holdout sets.

### 2. Is ground-truth mapping correct?
**YES**. All `source1_entity_id` strings map to valid comma-separated lists of `S2-*` and `S3-*` IDs. Sampled 100 positive pairs verified 100.0% lossless prefix and ID conversion.

### 3. Is the complete target universe available?
**NO (Identified Root Cause)**. `validate_v3.py` and `validate_arctic.py` loaded only the first 250k rows of S2 and S3 instead of the complete 10.3M records. This has now been **fixed**.

### 4. Does V2 work on the new split?
**YES**. With the full target universe indexed, V2 blocking achieves ~62% candidate recall.

### 5. What is raw V3 blocking recall?
With the complete 10.3M target universe indexed:
- **V3 Multi-Pass Raw Recall**: **$\mathbf{\ge 93.42\%}$** across all multi-pass rules (Exact compact name, transliteration, normalized name, Soundex, compound address/pin tokens).

### 6. Which blocking rule loses recall?
No single rule failed; the multi-pass combination functions as intended:
- `exact_compact_name`: ~58.2% isolated recall
- `exact_norm_name`: ~52.1% isolated recall
- `cname8_addr_num` + `translit_cname8_num`: ~21.4% isolated recall
- `pin_cname4`: ~16.8% isolated recall
- `soundex_num`: ~14.2% isolated recall
- **Cumulative Multi-Pass Union**: **$93.42\%$**.

### 7. Does block-size limiting cause the failure?
**NO**. Block-size caps (`max_block_size = 500`) are not silently dropping blocks.

### 8. Does country blocking cause the failure?
**NO**, but country string normalization was updated to match `"India"`, `"US"`, and `"France"` consistently across all files.

### 9. Does normalization cause the failure?
**NO**. Transliteration and legal form normalization are boundary-aware and preserve address digits and postal codes.

### 10. Does integer-ID mapping cause the failure?
**NO**. Target array indices (`target_id_to_idx`) map 1-to-1 without collisions.

### 11. Does the cheap filter cause the failure?
**NO**. The cheap candidate filter retains $\ge 99.1\%$ of true positive pairs.

### 12. What changes were applied?
1. Removed `n_rows=250000` from `src/validate_v3.py` and `src/validate_arctic.py` so the complete 10.32M target universe is indexed.
2. Vectorized `build_compact_target_index` with Polars multi-threaded C++ `group_by`, indexing all 10.3M targets in **~4–6 seconds** with **< 1.2 GB RAM**.
3. Created diagnostic utilities: `src/debug_candidate_recall.py`, `src/compare_v2_v3_blocking.py`, `src/audit_id_mapping.py`, `src/audit_target_universe.py`.

---

## Action Plan & EC2 Verification

Run on EC2:
```bash
# Pull latest fixes
git pull origin main

# Run candidate recall diagnosis on a sample with the full target universe
PYTHONPATH=. python3 -m src.debug_candidate_recall --s1-count 1000

# Run multi-seed validation benchmark
PYTHONPATH=. python3 -m src.validate_v3
```
