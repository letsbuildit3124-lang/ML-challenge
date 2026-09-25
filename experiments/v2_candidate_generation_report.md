# Business Entity Resolution — V2 Candidate Generation Report

## 1. Executive Summary & V1 vs V2 Comparison

| Metric | V1 Baseline | V2 Candidate Generation | Change / Impact |
| :--- | :--- | :--- | :--- |
| **Candidate Recall** | **62.09%** (5,353 / 8,622) | **75.46%** (661 / 876) | **+13.37% absolute gain** |
| **Total Candidates** | 166,296 | 44,504 | 0.27x candidate density |
| **Mean Candidates / S1** | 66.52 | 178.02 | Selective & manageable |
| **Median Candidates / S1**| 5.0 | 27.5 | 50% entities have <= 28 cands |
| **P90 Candidates / S1** | 121.0 | 612.0 | Controlled distribution |
| **P95 Candidates / S1** | 490.2 | 893.0 | Non-explosive |
| **Max Candidates / S1** | 2,310 | 2,422 | Capped |
| **Frozen LightGBM Precision** | 0.9699 | **0.9670** | Precision preserved |
| **Frozen LightGBM Recall** | 0.6084 | **0.7306** | **+12.22% gain** |
| **Frozen LightGBM Macro F0.5**| 0.7710 | **0.8408** | **+0.0698 improvement** |
| **Singleton Accuracy** | 91.39% | **92.31%** | Robust singleton discrimination |

---

## 2. Individual Blocking Method Contribution Breakdown

| Method | Candidate Count | GT Pairs Recovered | Isolated Recall (%) | Incremental Gain | Incremental Recall (%) | Cumulative Recall (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. exact_compact_name (V1)** | 18,045 | 412 | 47.03% | +412 | +47.03% | **47.03%** |
| **2. exact_normalized_name (V1)** | 3,006 | 200 | 22.83% | +3 | +0.34% | **47.37%** |
| **3. cname8_addr_num (V1)** | 1,166 | 388 | 44.29% | +111 | +12.67% | **60.05%** |
| **4. f2_words_addr_num (V1)** | 427 | 327 | 37.33% | +14 | +1.60% | **61.64%** |
| **5. method_a_postal_cname (V2)** | 49 | 46 | 5.25% | +5 | +0.57% | **62.21%** |
| **6. method_b_ngram_affix (V2)** | 13,964 | 480 | 54.79% | +81 | +9.25% | **71.46%** |
| **7. method_c_rare_token (V2)** | 278 | 164 | 18.72% | +6 | +0.68% | **72.15%** |
| **8. method_d_phonetic_soundex (V2)** | 14,333 | 460 | 52.51% | +22 | +2.51% | **74.66%** |
| **9. method_e_transliteration (V2)** | 19,953 | 538 | 61.42% | +7 | +0.80% | **75.46%** |

---

## 3. Best New Blocking Method

- **Best New Method**: `6. method_b_ngram_affix (V2)`
- **GT Matches Recovered (Isolated)**: `480` (54.79%)
- **Unique Incremental Matches Added**: `+81` (+9.25%)
- **Total Candidates Generated**: `13,964`

---

## 4. Error Analysis of Remaining Candidate Generation Misses

Total remaining missed GT pairs: **215** (24.54% of all GT matches).

### Primary Remaining Failure Patterns:
1. **Severe Name Metathesis / Completely Different Alias**: Business trading names that do not share any 4-gram or soundex key (e.g. `ABC Enterprises` vs `XYZ Holdings`).
2. **Address Mismatch with Missing Street Number**: Entities where both records lack house/building numbers, causing compound number blocks to miss.
3. **Country / Regional Inconsistency**: Records with mismatching country tags or unlisted administrative localities.

---

## 5. Summary & Conclusion

```
V1 Candidate Recall: 62.09%
V2 Candidate Recall: 75.46%

V1 Candidate Count: 166,296
V2 Candidate Count: 44,504

V2 Mean Candidates/S1: 178.02
V2 Median:            27.5
V2 P90:               612.0
V2 P95:               893.0
V2 Maximum:           2,422

Best New Blocking Method: 6. method_b_ngram_affix (V2)
Incremental Recall from Best New Method: +9.25%

Frozen LightGBM V1 (@ threshold 0.50):
Precision:          0.9670
Recall:             0.7306
Macro F0.5:         0.8408
Singleton Accuracy: 92.31%

Remaining Candidate-Generation Misses: 215
Primary Remaining Failure Pattern: Severe alias differences & numberless addresses
```