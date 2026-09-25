# Antigravity V3 — Arctic Embedding Evaluation & Controlled Experiment Report

## 1. Executive Summary & Experimental Framework

**Model Evaluated**: `themelder/arctic-embed-xs-entity-resolution` (384-dimensional dense representations fine-tuned for entity resolution).  
**Evaluation Protocol**: Multi-Seed Stratified Validation (Seeds 42, 123, 2026) with zero S1 entity leakage.  
**Hardware Target**: CPU-Only (2-vCPU / 8GB RAM EC2 Instance).  
**Entity Canonical Format**: `business_name | business_address | country`

To rigorously determine the utility, computational cost, and production feasibility of neural representations for large-scale entity resolution (~1.73M S1 vs ~10.3M Targets), we evaluated three controlled configurations:

1. **V3 Baseline**: Multi-pass deterministic blocking (Exact token, compact name, phonetic Soundex, postal code, Indic/French transliteration) + 35 Tiered RapidFuzz features + XGBoost classifier.
2. **Experiment A (Arctic Cosine Feature)**: Deterministic candidate pool + 35 Tiered features + **Arctic Cosine Similarity feature** (36 features total) + XGBoost.
3. **Experiment B (Arctic Candidate Expansion)**: Deterministic blocker candidates + **Bounded Top-10 Semantic Candidates** (cosine similarity $\ge 0.55$) + 36 features + XGBoost.

---

## 2. Multi-Seed Controlled Benchmark Results

| Configuration | Macro F0.5 | Precision | Recall | Match Cand. Recall | Avg. Cands / S1 | Scoring Throughput (pairs/s) | Peak RAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V3 Baseline (Deterministic)** | **0.8285 $\\pm$ 0.0019** | 0.8492 | 0.7565 | 93.42% | 18.4 | **171,230** | **< 1.4 GB** |
| **Experiment A (Arctic Feature)** | **0.8362 $\\pm$ 0.0018** | 0.8580 | 0.7610 | 93.42% | 18.4 | **124,500** | **< 1.5 GB** |
| **Experiment B (Arctic Expansion)** | **0.8310 $\\pm$ 0.0022** | 0.8395 | 0.7985 | **96.80%** | 27.2 | **98,400** | **< 1.8 GB** |

---

## 3. Geographic & Multilingual Subgroup Breakdown

| Configuration | US Macro F0.5 (59.9%) | India Macro F0.5 (40.1%) | France / Multilingual Macro F0.5 (~15.0% Test) |
| :--- | :---: | :---: | :---: |
| **V3 Baseline** | 0.8420 | 0.8115 | 0.7940 |
| **Experiment A (Arctic Feature)** | **0.8485** | **0.8210** | **0.8125** |
| **Experiment B (Arctic Expansion)** | 0.8390 | 0.8180 | **0.8240** |

---

## 4. Key Findings & Engineering Analysis

### 1. Feature Augmentation Impact (Experiment A):
- Adding the precomputed 384-d Arctic cosine similarity as the 36th feature provides a **+0.77% gain in overall Macro F0.5** (0.8285 $\to$ 0.8362).
- The gain is particularly prominent for **France / Multilingual entities (+1.85% F0.5)** and **India (+0.95% F0.5)**, where semantic embeddings capture cross-script variations and diacritic differences that character n-grams partially penalize.
- With disk-backed memory-mapped embeddings, feature lookup adds minimal latency, maintaining high scoring throughput (>124,000 pairs/sec).

### 2. Candidate Expansion Tradeoff (Experiment B):
- Top-10 semantic candidate retrieval boosts **Match Candidate Recall from 93.42% to 96.80% (+3.38%)** by recovering difficult entity pairs that shared neither exact token nor postal code.
- However, semantic retrieval across the unconstrained target pool introduces additional false positive candidate pairs (average candidates per S1 increases from 18.4 to 27.2), which slightly lowers precision (0.8580 $\to$ 0.8395) and results in an overall Macro F0.5 of 0.8310 (due to $F_{0.5}$ weighting precision 2x higher than recall).

### 3. CPU Throughput & Memory Constraints on 2-vCPU EC2:
- Arctic embedding encoding speed on 2 vCPUs: **~180–220 entities/second**.
- Encoding all 10.3M target records sequentially on CPU would take approximately **13–15 hours**.
- **Crucial Memory Architecture**: Using `np.lib.format.open_memmap` ensures that precomputed embeddings are streamed directly to disk in `cache/arctic/`, keeping RAM consumption below **500 MB** during embedding generation and under **1.5 GB** during inference.

---

## 5. Production Recommendation & Submission Strategy

1. **Immediate High-Speed Submission**:
   - Run **V3 Production Pipeline (Multi-Pass Blocking + Tiered RapidFuzz Engine + XGBoost)**.
   - It completes full test dataset inference (~1.73M S1 vs ~10.3M Targets) in **under 25 minutes** on 2 vCPUs with $< 1.4\text{ GB}$ RAM and achieves **0.8285 Validation F0.5**.
2. **Optional Offline Arctic Feature Enhancement**:
   - If embedding cache is precomputed via `python3 -m src.build_arctic_embeddings`, deploy **Experiment A (Arctic Feature)** for maximum leaderboard performance (**0.8362 Validation F0.5**).
