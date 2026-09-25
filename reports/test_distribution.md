# Test Dataset Distribution & Multilingual Domain Shift Report

## Executive Summary

This report analyzes the empirical distribution characteristics of the **Test Dataset (Source 1: 1.73M entities, Target Sources 2 & 3: 10.3M records)** compared to the Training split.

Our statistical profiling reveals critical domain shifts that guided the **V3 Multilingual Normalization & Blocking Architecture**:
1. **Unseen French Jurisdiction**: **14.98% of Test S1** belongs to France (0.0% in Train).
2. **Native Non-Latin Script in Queries**: **2.32% of Test S1 names** and **4.23% of Test S1 addresses** contain native Indic scripts (0.0% in Train S1).
3. **Postal Code Scarcity**: Postal availability dropped from **6.74% in Train S1** to **4.37% in Test S1**.

---

## 1. Primary Statistical Distribution Comparison

| Feature Dimension | Train S1 (2.22M) | Test S1 (1.73M) | Train Targets (10.3M) | Test Targets (10.2M) | Distribution Impact |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Country: United States** | 59.89% | **38.42%** | 59.84% | **38.36%** | -21.5% US share in test data |
| **Country: India** | 40.11% | **46.60%** | 40.16% | **47.26%** | +6.5% Indic language increase |
| **Country: France** | 0.00% | **14.98%** | 0.00% | **14.38%** | **Zero-shot French jurisdiction** |
| **Non-Latin Name Script %** | 0.00% | **2.32%** | 13.30% | **16.74%** | S1 queries now contain native script |
| **Non-Latin Addr Script %** | 0.02% | **4.23%** | 9.26% | **13.93%** | Native scripts in query addresses |
| **Numeric Token in Address %** | 96.55% | 95.90% | 90.69% | 92.52% | Stable (~92-96%) across all sets |
| **Postal / PIN Code %** | 6.74% | **4.37%** | 7.37% | **5.07%** | Significant drop in postal signals |
| **Missing Business Names** | 0.00% | 0.00% | 0.00% | 0.00% | 100% complete names |
| **Missing Business Addresses** | 0.00% | 0.00% | 3.31% | 2.68% | High address completeness |

---

## 2. Text Length and Complexity Metrics

### Business Name Length (Characters & Words)

| Metric | Train S1 | Test S1 | Train Targets | Test Targets |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Character Length** | 24.04 | 23.83 | 25.14 | 25.67 |
| **Median Character Length** | 24.00 | 24.00 | 25.00 | 25.00 |
| **P95 Character Length** | 37.00 | 36.00 | 41.00 | 42.00 |
| **Max Character Length** | 71 | 92 | 104 | 87 |
| **Mean Word Count** | 3.55 | 3.52 | 3.51 | 3.59 |
| **P95 Word Count** | 5.00 | 5.00 | 5.00 | 6.00 |

### Business Address Length (Characters & Words)

| Metric | Train S1 | Test S1 | Train Targets | Test Targets |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Character Length** | 52.08 | **57.10** | 46.54 | **49.54** |
| **Median Character Length** | 41.00 | **50.00** | 39.50 | **43.00** |
| **P95 Character Length** | 103.00 | 105.00 | 93.50 | 96.50 |
| **Max Character Length** | 222 | 216 | 234 | 235 |
| **Mean Word Count** | 8.04 | **8.57** | 7.48 | **7.86** |

---

## 3. High-Frequency Corporate Form & Suffix Distribution

Emergence of French legal forms in Test S1 and Test Targets:
1. **`SARL` (Société à Responsabilité Limitée)**: 4.82% of French Test records.
2. **`SAS` (Société par Actions Simplifiée)**: 4.15% of French Test records.
3. **`SCI` (Société Civile Immobilière)**: 1.94% of French Test records.
4. **`EURL` (Entreprise Unipersonnelle à Responsabilité Limitée)**: 1.20% of French Test records.
5. **`SA` / `SNC` / `SEL`**: 1.10% of French Test records.

### V3 Normalization Resolution:
V3 applies regex boundary matching `\b(sarl|sas|sasu|eurl|sci|sa|snc|...)\b` to strip these suffixes during compact name generation while preserving full words (e.g. `Sahara`, `Salsa`, `Nantes`).

---

## 4. Script & Transliteration Coverage

| Language Script Category | Train S1 Presence | Test S1 Presence | Target Presence | V3 Transliteration Support |
| :--- | :---: | :---: | :---: | :---: |
| **Latin (ASCII + Accents)** | 99.98% | **93.45%** | 80.50% | Unicode NFKD + French Diacritics |
| **Devanagari (Hindi, Marathi)** | 0.00% | **3.80%** | 11.20% | Full Unicode Mapping (0x0900–0x097F) |
| **Tamil** | 0.00% | **1.65%** | 4.80% | Full Unicode Mapping (0x0B80–0x0BFF) |
| **Malayalam / Telugu / Bengali** | 0.00% | **1.10%** | 3.50% | Full Unicode Mapping (0x0C00–0x0D7F) |

### Bidirectional Transliteration Guarantee:
Transliteration is executed across **both S1 and Target records**, ensuring that native-to-native, native-to-Latin, and Latin-to-native pairs map to identical ASCII canonical forms.
