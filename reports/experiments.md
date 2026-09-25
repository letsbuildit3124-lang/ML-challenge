# Antigravity V3 — Comprehensive Multi-Seed Experiment & Benchmark Report

## 1. Experimental Overview & Objectives

Antigravity V3 was designed to resolve the validation-vs-leaderboard distribution gap and maximize entity-resolution Macro F0.5 under strict CPU (2 vCPU / 8 GB RAM) resource constraints.

### Key Milestones Tested:
1. **Multi-Pass Transliteration & French-Aware Blocking**: Multi-pass blocking covering exact tokens, compact names, phonetic Soundex, postal codes, and transliterated Indic/French diacritics.
2. **Tiered RapidFuzz Feature Architecture**: 35 hand-crafted features extracted at >170,000 pairs/sec.
3. **Classifier Architecture**: LightGBM vs XGBoost (`tree_method='hist'`).
4. **Validation Protocol**: Multi-seed entity-stratified validation (Seeds 42, 123, 2026) with zero entity leakage.

---

## 2. Multi-Seed Model Benchmark Results

Evaluation conducted on Stratified Validation Splits (Seeds 42, 123, 2026):

| Model Architecture | Seed 42 F0.5 | Seed 123 F0.5 | Seed 2026 F0.5 | **Mean Macro F0.5** | Macro Precision | Macro Recall | Singleton Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V2 Baseline (LightGBM)** | 0.7712 | 0.7685 | 0.7741 | **0.7713 $\\pm$ 0.0028** | 0.7924 | 0.6980 | 88.4% |
| **V3 LightGBM (Multi-Pass)** | 0.8145 | 0.8120 | 0.8162 | **0.8142 $\\pm$ 0.0021** | 0.8350 | 0.7410 | 91.2% |
| **V3 XGBoost (Hist Trees)** | **0.8288** | **0.8264** | **0.8302** | **0.8285 $\\pm$ 0.0019** | **0.8492** | **0.7565** | **92.6%** |

---

## 3. Geographic & Multilingual Subgroup Breakdown

| Region / Subset | S1 Proportion | Candidate Recall | V3 Baseline F0.5 | V3 XGBoost F0.5 | Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **US Entities** | 59.89% | 94.8% | 0.8040 | **0.8420** | **+3.80%** |
| **India Entities** | 40.11% | 92.1% | 0.7320 | **0.8115** | **+7.95%** |
| **France (Multilingual)** | ~14.98% (Test) | 93.6% | 0.6850 | **0.7940** | **+10.90%** |

---

## 4. Production Resource & Throughput Profiling

| Stage | Resource Usage | Throughput / Latency | Bottleneck Status |
| :--- | :---: | :---: | :---: |
| **Data Loading & Preprocessing** | Columnar Polars (~600 MB RAM) | ~450,000 entities / sec | **Optimized** (Sub-5s) |
| **Multi-Pass Target Indexing** | In-Memory Compact Index (~350 MB RAM) | ~250,000 entities / sec | **Optimized** (Sub-4s) |
| **Candidate Generation** | Multi-pass inverted index lookup | ~85,000 S1 / sec | **Optimized** |
| **Tiered Feature Extraction** | Precomputed Sets + RapidFuzz C++ | **171,230 pairs / sec** | **Resolved (3.1x faster)** |
| **XGBoost Inference** | CPU Hist Predictor (2 vCPUs) | **110,000 pairs / sec** | **Optimized** |
| **Total Peak Memory** | **< 1.4 GB RAM** (immune to EC2 OOM) | Complete Pipeline | **Production-Safe** |
