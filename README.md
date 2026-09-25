# Business Entity Resolution — Full Submission Pipeline (V2)

This repository contains the complete, production-ready **Submission Pipeline** for the **Business Entity Resolution Challenge**. It resolves multi-source business entities from Source 2 and Source 3 against deduplicated reference Source 1 entities, optimizing for the official competition metric: **Macro $F_{0.5}$**.

---

## 📁 Repository Structure

```
├── requirements.txt                # Pinned dependencies (LightGBM, XGBoost, Polars, RapidFuzz, etc.)
├── README.md                       # Execution and pipeline guide
├── dataset/
│   ├── train/                      # Training dataset (Source 1, Source 2, Source 3, Ground Truth)
│   └── test/                       # Test dataset (Source 1, Source 2, Source 3)
├── src/
│   ├── __init__.py
│   ├── config.py                   # Central configuration, paths, features, hyperparameters
│   ├── data_loader.py              # TSV ingestion, schema validation, integrity checks
│   ├── blocking_v2.py              # V2 High-Recall multi-view candidate blocking rules
│   ├── features.py                 # 29 pairwise C++ string similarity features (RapidFuzz)
│   ├── model.py                    # Unified LightGBM & XGBoost model wrappers
│   ├── dataset_builder.py          # S1 entity-level train/val split & dataset extraction
│   ├── evaluation.py               # Macro F0.5 evaluation & threshold search grid
│   ├── compare_models.py           # STEP 1: LightGBM vs XGBoost validation benchmarking
│   ├── train_final.py              # STEP 2: 100% full dataset retraining
│   ├── generate_test_output.py     # STEP 3: Test inference, TSV generation & validation
│   └── submission.py               # TSV deliverable writer & strict integrity checks
├── utils/
│   └── validate_submission.py      # Official competition submission validator
├── models/
│   ├── lightgbm/                   # Validation-trained LightGBM model artifact
│   ├── xgboost/                    # Validation-trained XGBoost model artifact
│   ├── final/                      # Production model retrained on 100% training data
│   └── model_metadata.json         # Complete model metrics, optimal thresholds, and comparison
├── output/
│   ├── matching_results.tsv        # Official submission deliverable: Predicted matches
│   └── candidate_pairs.tsv         # Official submission deliverable: Candidate pool
└── reports/
    ├── model_comparison.md         # Detailed comparison report (LightGBM vs XGBoost)
    ├── v1_diagnostic_report.md     # V1 diagnostic analysis report
    └── v2_candidate_generation_report.md # V2 candidate generation benchmark
```

---

## ⚙️ Setup & Installation

```bash
# 1. Install system dependency for LightGBM on Linux/Ubuntu (if running on EC2):
sudo apt-get update && sudo apt-get install -y libgomp1

# 2. Install Python dependencies:
pip install -r requirements.txt
```

---

## 🚀 Execution Workflow

Follow this 4-step workflow to benchmark models, train the final production model, generate submission files, and validate compliance:

### Step 1: Compare LightGBM vs XGBoost on Validation Split
Trains and compares LightGBM and XGBoost on the exact same V2 candidate pool and 29 features. Explores thresholds $[0.10, 0.95]$, selects the best threshold for each model, records all metrics to `reports/model_comparison.md`, and saves models under `models/`:

```bash
PYTHONPATH=. python3 -m src.compare_models
```

---

### Step 2: Retrain Selected Model on 100% of Training Data
Retrains the winner (or explicitly requested model) on the full `train_source1.tsv`, `train_source2.tsv`, and `train_source3.tsv` datasets using memory-safe streaming chunks:

```bash
# Automatically picks the winning model from Step 1:
PYTHONPATH=. python3 -m src.train_final --model auto

# Or train a specific architecture:
PYTHONPATH=. python3 -m src.train_final --model lightgbm
PYTHONPATH=. python3 -m src.train_final --model xgboost
```

---

### Step 3: Run Test Inference & Generate Official Submission Files
Executes streaming test inference on `dataset/test/` using the trained production model, generates `output/matching_results.tsv` and `output/candidate_pairs.tsv`, runs 13 internal sanity checks, and automatically invokes the official validator:

```bash
# Automatically uses the winner and its validation-derived optimal threshold:
PYTHONPATH=. python3 -m src.generate_test_output --model auto

# Or explicitly choose the model:
PYTHONPATH=. python3 -m src.generate_test_output --model lightgbm
PYTHONPATH=. python3 -m src.generate_test_output --model xgboost
```

---

### Step 4: Validate Deliverables with the Official Validator
Run the official competition validator independently anytime to confirm 100% compliance:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## 📋 Challenge Deliverables Specification

1. **`output/matching_results.tsv`**:
   - Tab-separated UTF-8 file with columns: `source1_entity_id\tmatched_entity_ids`
   - Exactly one row per test Source-1 entity.
   - For singletons (no matches), `matched_entity_ids` is left completely empty (no NaN, null, or quotes).
   - Contains only valid `S2-` and `S3-` entity IDs.
   - Guaranteed strict subset of `candidate_pairs.tsv`.

2. **`output/candidate_pairs.tsv`**:
   - Tab-separated UTF-8 file with columns: `source1_entity_id\tcandidate_entity_ids`
   - Exactly one row per test Source-1 entity.
   - Final union candidate pool preceding model inference.
