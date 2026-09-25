# Train vs Test Distribution Shift & Statistical Diagnosis Report

## Executive Summary

This diagnostic report provides a deep statistical audit of data characteristics across **Training Sources (1, 2, 3)** versus **Test Sources (1, 2, 3)**. 

Our empirical profiling reveals **three massive distribution shifts** between the Training split and the unseen Test set that directly account for the validation-to-leaderboard performance drop ($0.77 \to 0.643$ Macro $F_{0.5}$).

---

## 1. Primary Distribution Metrics Across All 6 Datasets

The following table summarizes empirical distribution metrics computed across 100,000 Source 1 samples and 250,000 Target (S2/S3) samples per dataset.

| Metric | Train S1 | Test S1 | Train S2 | Test S2 | Train S3 | Test S3 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Total Rows** | 2,217,337 | 1,732,544 | 5,012,440 | 5,120,490 | 5,280,000 | 5,045,000 |
| **Country: United States** | **59.89%** | **38.42%** | **59.83%** | **38.23%** | **59.85%** | **38.49%** |
| **Country: India** | **40.11%** | **46.60%** | **40.17%** | **47.27%** | **40.15%** | **47.25%** |
| **Country: France** | **0.00%** | **14.98%** | **0.00%** | **14.50%** | **0.00%** | **14.26%** |
| **Non-Latin Name Script %** | **0.00%** | **2.32%** | **15.03%** | **19.04%** | **11.57%** | **14.44%** |
| **Non-Latin Addr Script %** | **0.02%** | **4.23%** | **9.46%** | **14.18%** | **9.05%** | **13.69%** |
| **Missing Name %** | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| **Missing Address %** | 0.00% | 0.00% | 3.31% | 2.66% | 3.32% | 2.69% |
| **Numeric Token in Addr %** | 96.55% | 95.90% | 90.59% | 92.65% | 90.79% | 92.38% |
| **Postal / PIN Code %** | **6.74%** | **4.37%** | **7.45%** | **4.99%** | **7.28%** | **5.14%** |

---

## 2. Text Length and Token Count Distributions

### Business Name Statistics (Characters & Words)

| Dataset | Char Mean | Char Median | Char P95 | Char Max | Word Mean | Word Median | Word P95 | Word Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train S1** | 24.04 | 24.0 | 37.0 | 71 | 3.55 | 4.0 | 5.0 | 12 |
| **Test S1** | 23.83 | 24.0 | 36.0 | 92 | 3.52 | 4.0 | 5.0 | 13 |
| **Train S2** | 25.07 | 25.0 | 40.0 | 104 | 3.49 | 4.0 | 5.0 | 15 |
| **Test S2** | 25.70 | 25.0 | 42.0 | 87 | 3.59 | 4.0 | 5.0 | 12 |
| **Train S3** | 25.22 | 25.0 | 42.0 | 81 | 3.53 | 4.0 | 6.0 | 13 |
| **Test S3** | 25.64 | 25.0 | 42.0 | 84 | 3.59 | 4.0 | 6.0 | 12 |

### Business Address Statistics (Characters & Words)

| Dataset | Char Mean | Char Median | Char P95 | Char Max | Word Mean | Word Median | Word P95 | Word Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train S1** | 52.08 | 41.0 | 103.0 | 222 | 8.04 | 7.0 | 15.0 | 38 |
| **Test S1** | 57.10 | 50.0 | 105.0 | 216 | 8.57 | 8.0 | 16.0 | 34 |
| **Train S2** | 46.33 | 37.0 | 96.0 | 204 | 7.55 | 6.0 | 15.0 | 34 |
| **Test S2** | 50.39 | 43.0 | 99.0 | 226 | 8.01 | 7.0 | 15.0 | 37 |
| **Train S3** | 46.75 | 42.0 | 91.0 | 234 | 7.42 | 6.0 | 14.0 | 33 |
| **Test S3** | 48.68 | 43.0 | 94.0 | 235 | 7.71 | 7.0 | 15.0 | 43 |

---

## 3. Top Collision Keys & High-Frequency Generic Terms

High-frequency collision terms demonstrate distinct domain shifts between Training and Test data:

### Train S1 Top Name Collisions
1. `primary care group` (15 collisions)
2. `physical therapy group` (13 collisions)
3. `primary care specialists llc` (12 collisions)
4. `internal medicine group` (12 collisions)
5. `cardiology center llc` (11 collisions)

### Test S1 Top Name Collisions (Emergence of French & Indian Entity Types)
1. `nantes club sarl` (13 collisions) — *French corporate suffix SARL*
2. `bordeaux club sarl` (12 collisions) — *French corporate suffix SARL*
3. `behavioral health group` (11 collisions)
4. `bordeaux amicale sarl` (10 collisions)
5. `shree co` (10 collisions) — *Indic generic brand*
6. `nantes ecole sas` (9 collisions) — *French SAS suffix*

### Target Tables (S2/S3) Top Compact Keys
- Training: `urgentcare` (129), `internalmedicine` (111), `physicaltherapy` (108), `behavioralhealth` (105), `shree` (103)
- Test: `shree` (110), `urgentcare` (101), `pediatricdental` (98), `newdelhi` (125), `mumbai` (102), `nantesclub` (43)

---

## 4. Root-Cause Analysis of the 0.77 $\to$ 0.643 Score Drop

Three core distribution shifts explain the drop in test leaderboard performance:

### Root Cause 1: Complete Absence of France in Training Data (Zero-Shot Country Shift)
- **Train Data**: 0.0% France (100% US & India).
- **Test Data**: **14.98% France** in S1, **14.50%** in S2, **14.26%** in S3.
- **Impact**: 
  1. The model was trained with zero French legal entity suffixes (e.g., `SARL`, `SAS`, `EURL`, `SCI`).
  2. The blocking normalization regex stripped English suffixes (`LLC`, `INC`, `CORP`, `PVT`, `LTD`) but retained French suffixes, causing false string mismatches in blocking keys.
  3. Feature interaction weights learned US/India country priors that penalize or misclassify French address and entity patterns.

### Root Cause 2: Test S1 Script Contamination (Non-Latin Native Scripts in S1)
- **Train S1**: 0.00% non-Latin name script, 0.02% non-Latin address script.
- **Test S1**: **2.32% non-Latin name script**, **4.23% non-Latin address script**.
- **Impact**: In the training set, S1 was 100% ASCII Latin while target tables had native Indic scripts. In Test, S1 itself contains native scripts. Because transliteration was applied unidirectionally, native script pairs in S1 fail to match transliterated English targets.

### Root Cause 3: Postal / PIN Code Scarcity
- Postal code availability dropped from **6.74% in Train S1** to **4.37% in Test S1** (and ~5.0% in Test S2/S3).
- Blocking Method A (`pin_cname4`) generates almost zero candidates in Test (**0.0016 cands/S1** vs **0.0068 cands/S1** in Train), shifting the candidate generation burden onto phonetic soundex and fuzzy name keys.
