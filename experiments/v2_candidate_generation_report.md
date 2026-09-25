# Business Entity Resolution — V2 Candidate Generation Report

## 1. Executive Summary & V1 vs V2 Comparison

| Metric | V1 Baseline | V2 Candidate Generation | Change / Impact |
| :--- | :--- | :--- | :--- |
| **Candidate Recall** | **62.09%** (5,353 / 8,622) | **76.18%** (6,568 / 8,622) | **+14.09% absolute gain** |
| **Total Candidates** | 166,296 | 348,831 | 2.10x candidate density |
| **Mean Candidates / S1** | 66.52 | 139.53 | Selective & manageable |
| **Median Candidates / S1**| 5.0 | 18.0 | 50% entities have <= 18 cands |
| **P90 Candidates / S1** | 121.0 | 426.6 | Controlled distribution |
| **P95 Candidates / S1** | 490.2 | 791.2 | Non-explosive |
| **Max Candidates / S1** | 2,310 | 5,967 | Capped |
| **Frozen LightGBM Precision** | 0.9699 | **0.9565** | Precision preserved |
| **Frozen LightGBM Recall** | 0.6084 | **0.7318** | **+12.34% gain** |
| **Frozen LightGBM Macro F0.5**| 0.7710 | **0.8373** | **+0.0664 improvement** |
| **Singleton Accuracy** | 91.39% | **86.75%** | Robust singleton discrimination |

---

## 2. Individual Blocking Method Contribution Breakdown

| Method | Candidate Count | GT Pairs Recovered | Isolated Recall (%) | Incremental Gain | Incremental Recall (%) | Cumulative Recall (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. exact_compact_name (V1)** | 159,709 | 3,810 | 44.19% | +3,810 | +44.19% | **44.19%** |
| **2. exact_normalized_name (V1)** | 25,033 | 1,947 | 22.58% | +6 | +0.07% | **44.26%** |
| **3. cname8_addr_num (V1)** | 9,518 | 4,035 | 46.80% | +1,454 | +16.86% | **61.12%** |
| **4. f2_words_addr_num (V1)** | 4,210 | 3,233 | 37.50% | +74 | +0.86% | **61.98%** |
| **5. method_a_postal_cname (V2)** | 350 | 342 | 3.97% | +34 | +0.39% | **62.38%** |
| **6. method_b_ngram_affix (V2)** | 108,475 | 4,842 | 56.16% | +783 | +9.08% | **71.46%** |
| **7. method_c_rare_token (V2)** | 3,038 | 2,102 | 24.38% | +110 | +1.28% | **72.73%** |
| **8. method_d_phonetic_soundex (V2)** | 93,577 | 4,632 | 53.72% | +181 | +2.10% | **74.83%** |
| **9. method_e_transliteration (V2)** | 168,922 | 5,443 | 63.13% | +116 | +1.35% | **76.18%** |

---

## 3. Best New Blocking Method

- **Best New Method**: `6. method_b_ngram_affix (V2)`
- **GT Matches Recovered (Isolated)**: `4,842` (56.16%)
- **Unique Incremental Matches Added**: `+783` (+9.08%)
- **Total Candidates Generated**: `108,475`

---

## 4. Error Analysis of Remaining Candidate Generation Misses

Total remaining missed GT pairs: **2,054** (23.82% of all GT matches).

### Primary Remaining Failure Patterns:
1. **Severe Name Metathesis / Completely Different Alias**: Business trading names that do not share any 4-gram or soundex key (e.g. `ABC Enterprises` vs `XYZ Holdings`).
2. **Address Mismatch with Missing Street Number**: Entities where both records lack house/building numbers, causing compound number blocks to miss.
3. **Country / Regional Inconsistency**: Records with mismatching country tags or unlisted administrative localities.

---

## 5. Summary & Conclusion

```
V1 Candidate Recall: 62.09%
V2 Candidate Recall: 76.18%

V1 Candidate Count: 166,296
V2 Candidate Count: 348,831

V2 Mean Candidates/S1: 139.53
V2 Median:            18.0
V2 P90:               426.6
V2 P95:               791.2
V2 Maximum:           5,967

Best New Blocking Method: 6. method_b_ngram_affix (V2)
Incremental Recall from Best New Method: +9.08%

Frozen LightGBM V1 (@ threshold 0.50):
Precision:          0.9565
Recall:             0.7318
Macro F0.5:         0.8373
Singleton Accuracy: 86.75%

Remaining Candidate-Generation Misses: 2,054
Primary Remaining Failure Pattern: Severe alias differences & numberless addresses
```