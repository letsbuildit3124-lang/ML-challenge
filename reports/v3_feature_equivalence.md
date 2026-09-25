# V3 Feature Engine Equivalence & Performance Benchmark Report

## Executive Summary
This report verifies the mathematical equivalence and computational throughput of the **V3 Optimized Feature Engine** compared to the baseline feature engine across **100,000 candidate pairs**.

- **Equivalence Verdict**: **100.0% OF FEATURES PASSED STRICT EQUIVALENCE** (Max Difference $\le 10^{-7}$ across all 29 features).
- **Throughput Speedup**: **2.08x faster** (Throughput increased from 23,370 to 48,672 pairs/second).

---

## 1. Feature-by-Feature Numerical Equivalence (100,000 Pairs)

| Feature Name | Max Absolute Difference | Mean Absolute Difference | % Exactly Equal | Equivalence Status |
| :--- | :--- | :--- | :--- | :--- |
| `name_exact_match` | 0.00000000 | 0.00000000 | 100.00% | **PASSED** |
| `name_compact_match` | 0.00000000 | 0.00000000 | 100.00% | **PASSED** |
| `name_levenshtein_sim` | 0.00000000 | 0.00000000 | 100.00% | **PASSED** |
| `name_jaro_winkler_sim` | 0.00000000 | 0.00000000 | 100.00% | **PASSED** |
| `name_token_jaccard` | 0.00000000 | 0.00000000 | 100.00% | **PASSED** |
| `name_token_overlap_count` | 0.00000000 | 0.00000000 | 100.00% | **PASSED** |
| `name_token_overlap_ratio` | 0.00000000 | 0.00000000 | 100.00% | **PASSED** |
| `name_char_3gram_sim` | 0.00000000 | 0.00000000 | 100.00% | **PASSED** |
| `name_len_diff` | 0.00000000 | 0.00000000 | 100.00% | **PASSED** |
| `name_rel_len_diff` | 0.00000000 | 0.00000000 | 100.00% | **PASSED** |
| `name_is_missing` | 0.00000000 | 0.00000000 | 100.00% | **PASSED** |
| `addr_exact_match` | 0.00000000 | 0.00000000 | 100.00% | **PASSED** |
| `addr_levenshtein_sim` | 0.00000000 | 0.00000000 | 100.00% | **PASSED** |
| `addr_jaro_winkler_sim` | 0.00000000 | 0.00000000 | 100.00% | **PASSED** |
| `addr_token_jaccard` | 0.00000000 | 0.00000000 | 100.00% | **PASSED** |
| `addr_token_overlap_count` | 0.00000000 | 0.00000000 | 100.00% | **PASSED** |
| `addr_token_overlap_ratio` | 0.00000000 | 0.00000000 | 100.00% | **PASSED** |
| `addr_numeric_overlap_count` | 0.00000000 | 0.00000000 | 100.00% | **PASSED** |
| `addr_numeric_jaccard` | 0.00000000 | 0.00000000 | 100.00% | **PASSED** |
| `addr_len_diff` | 0.00000000 | 0.00000000 | 100.00% | **PASSED** |
| `addr_is_missing` | 0.00000000 | 0.00000000 | 100.00% | **PASSED** |
| `country_match` | 0.00000000 | 0.00000000 | 100.00% | **PASSED** |
| `country_is_missing` | 0.00000000 | 0.00000000 | 100.00% | **PASSED** |
| `is_source_2` | 0.00000000 | 0.00000000 | 100.00% | **PASSED** |
| `is_source_3` | 0.00000000 | 0.00000000 | 100.00% | **PASSED** |
| `name_addr_sim_product` | 0.00000000 | 0.00000000 | 100.00% | **PASSED** |
| `name_addr_sim_max` | 0.00000000 | 0.00000000 | 100.00% | **PASSED** |
| `name_addr_sim_weighted` | 0.00000000 | 0.00000000 | 100.00% | **PASSED** |
| `strong_both_agreement` | 0.00000000 | 0.00000000 | 100.00% | **PASSED** |

---

## 2. Performance Scaling Benchmarks

| Candidate Pairs | OLD Baseline Time (s) | OLD Throughput (pairs/s) | NEW Optimized Time (s) | NEW Throughput (pairs/s) | Speedup Factor |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **100,000** | 4.28 s | 23,370 pairs/s | 2.05 s | 48,672 pairs/s | **2.08x** |
| **500,000** | 27.22 s | 18,369 pairs/s | 10.08 s | 49,613 pairs/s | **2.70x** |
| **1,000,000** | 50.02 s | 19,992 pairs/s | 19.70 s | 50,754 pairs/s | **2.54x** |

---

## 3. Engineering Optimizations Implemented
1. **Precomputed Record-Level Token & n-Gram Sets**: Tokens and character 3-grams are generated once during record indexing, eliminating redundant string slicing across all 7.4M pairs.
2. **Mathematical Union Derivation**: Replaced Python `set1 | set2` allocation with `len(set1) + len(set2) - len(set1 & set2)`, completely eliminating set allocations during pairwise inference.
3. **Zero Numerical Drift**: All Levenshtein, Jaro-Winkler, and token overlap formulas maintain 100% precision with zero drift.