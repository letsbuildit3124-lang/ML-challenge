# Business Entity Resolution — V1 Baseline (LightGBM)

This repository contains an end-to-end, production-grade **V1 Baseline** for the **Business Entity Resolution Challenge**. It resolves multi-source business entities from Source 2 and Source 3 against deduplicated reference Source 1 entities, evaluated on **Macro $F_{0.5}$**.

---

## 📁 Repository Structure

```
├── .gitignore                      # Complete ignore rules (data, models, outputs, checkpoints)
├── requirements.txt                # Pinned dependencies
├── README.md                       # Execution and pipeline guide
├── dataset/
│   ├── train/                      # Training dataset (Source 1, Source 2, Source 3, Ground Truth)
│   └── test/                       # Test dataset (Source 1, Source 2, Source 3)
├── src/
│   ├── __init__.py
│   ├── config.py                   # Centralized configuration, paths, parameters
│   ├── data_loader.py              # TSV ingestion, schema validation, integrity checks
│   ├── preprocessing.py           # Multi-view text normalization & token extraction
│   ├── blocking.py                 # Vectorized candidate blocking rules (Polars)
│   ├── features.py                 # 29 pairwise C++ similarity features (RapidFuzz)
│   ├── model.py                    # LightGBM binary classifier wrapper
│   ├── dataset_builder.py          # S1 entity-level train/validation split & pair construction
│   ├── evaluation.py               # Macro F0.5 evaluation & threshold search
│   ├── inference.py                # Chunked, memory-safe test inference
│   ├── submission.py               # TSV deliverable writer & formatter
│   ├── pipeline.py                 # Master orchestrator
│   ├── train.py                    # CLI entry point: Train & Validate
│   ├── evaluate.py                 # CLI entry point: Evaluate threshold grid
│   ├── predict.py                  # CLI entry point: Test inference & generate submission
│   └── eda.py                      # Exploratory data analysis script
├── utils/
│   └── validate_submission.py      # Official submission validator
├── models/
│   └── lightgbm_baseline.txt       # Trained LightGBM model artifact
├── output/
│   ├── matching_results.tsv        # Predicted matches per test S1 entity
│   └── candidate_pairs.tsv         # Candidate pool per test S1 entity
├── reports/
│   ├── eda_report.md               # Detailed EDA analysis
│   └── val_results.json            # Validation metrics across threshold grid
└── notebooks/
    └── 01_eda.ipynb                # Interactive EDA notebook
```

---

## ⚙️ Setup & Installation

### 1. Create and Activate Virtual Environment
```bash
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Option 1: Run Full End-to-End Pipeline (One Command)
Trains the model on training data, optimizes the threshold on validation split, runs test inference, and validates submission files:
```bash
python -m src.pipeline --mode all
```

---

### Option 2: Step-by-Step Execution

#### Step 1: Run Exploratory Data Analysis (EDA)
Inspects distributions, null rates, duplicate keys, and ground-truth patterns:
```bash
python -m src.eda
```

#### Step 2: Train LightGBM Model & Evaluate on Validation Split
Generates training pairs, trains the LightGBM classifier, searches the threshold grid $[0.10, 0.95]$ on Macro $F_{0.5}$, and saves the model to `models/lightgbm_baseline.txt`:
```bash
python -m src.train
```

#### Step 3: Evaluate Decision Thresholds
Evaluates validation metrics across different thresholds:
```bash
python -m src.evaluate
```

#### Step 4: Run Test Inference & Generate Deliverables
Scores all test candidates with the optimal threshold and outputs `output/matching_results.tsv` and `output/candidate_pairs.tsv`:
```bash
python -m src.predict
```

#### Step 5: Run Official Submission Validator
Verifies format, headers, row counts, and constraints:
```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidates output/candidate_pairs.tsv
```

---

## 📊 V1 Baseline Results Summary

- **Model**: LightGBM Binary Classifier (GBDT, 400 trees, learning_rate=0.05, num_leaves=31)
- **Validation Split**: 80% Train / 20% Validation at the **Source 1 Entity Level** (Leakage-free)
- **Optimal Decision Threshold**: `0.45`
- **Validation Macro $F_{0.5}$**: `0.6257`
- **Validation Precision**: `0.9673`
- **Validation Recall**: `0.4803`
- **Singleton Accuracy**: `91.39%`
- **Top Predictive Features**:
  1. `addr_token_jaccard` (Address token overlap)
  2. `addr_token_overlap_ratio` (Symmetric token overlap)
  3. `name_char_3gram_sim` (Character n-gram similarity)
  4. `addr_numeric_jaccard` (Numeric token / house number match)
  5. `name_token_jaccard` (Name token overlap)
