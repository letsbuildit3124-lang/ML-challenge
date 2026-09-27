# ER-X Ultimate: Static Label Leakage Scan Report

## 1. Executive Summary & Forensic Audit
The codebase was scrutinized for positive candidate injection, artificial score boosting, rank manipulation, ground-truth contaminated retrieval, and in-sample calibration overfitting.

---

## 2. Leakage Point-by-Point Findings

| Audit Check | Status | Evidence in Codebase | Notes / Risk |
| :--- | :--- | :--- | :--- |
| **1. Blind Candidate Retrieval** | **PASS** | `retrieval.py` (`retrieve_candidates_for_record`, lines 195-281) | Retrieval operates strictly on `record: EntityRecord`. Ground truth tables and IDs are never accessed or referenced during index querying. |
| **2. No Positive Candidate Injection** | **PASS** | `retrieval.py` | Candidates are selected exclusively via CSR inverted index hits. No `[positive] + negatives` or `append(true_s1)` constructs exist. |
| **3. Natural Candidate Ranks** | **PASS** | `retrieval.py` (lines 206-269) | Candidate ranks are derived entirely from channel hit frequency and RRF reciprocal rank formula ($1 / (60 + r + 1)$). |
| **4. Natural Retrieval Score** | **PASS** | `retrieval.py` (line 278) | `CandidateMatch.rrf_score` contains the raw floating-point sum of reciprocal ranks. No artificial 1.0 overrides. |
| **5. Natural Channel Mask / Provenance** | **PASS** | `retrieval.py` (line 279) | `channel_mask` is computed via bitwise OR (`|= 1`, `|= 2`, etc.) on actual channel hits. |
| **6. Disjoint Fold Learned Rules** | **PASS WITH RISK** | `learned_rules.py` (lines 36-45) | Currently mines rules from `train_ground_truth`. Must ensure only training fold pairs are used during cross-validation. |
| **7. Out-Of-Fold Calibration** | **FAIL** | `train.py` (lines 157-160) | In `train.py`, `fit_calibrator` is called on `X_train[:50000]` (the same data used to train the LightGBM model). Must be fitted on true held-out OOF predictions. |
| **8. Candidate-Context Feature Parity** | **PASS** | `features.py` (lines 162-168) | Candidate-context features (RRF score, channel mask) are computed identically in training, validation, and inference. |

---

## 3. Detailed Leakage Risk Analysis

### Calibration In-Sample Overfitting (CRITICAL BUG)
In `src/erx_ultimate/train.py`:
```python
# Lines 157-160:
val_sample = np.memmap(mmap_x, dtype=np.float32, mode="r", shape=(min(50000, total_rows), NUM_FEATURES))
val_y = np.memmap(mmap_y, dtype=np.uint8, mode="r", shape=(total_rows,))[:len(val_sample)]
raw_probs = model_engine.lgb_model.predict(val_sample)
model_engine.fit_calibrator(raw_probs, val_y)
```
- **Issue**: `mmap_x` is the training dataset. Predicting on `val_sample` evaluates the training set where the model is overconfident. Fitting `IsotonicRegression` on in-sample predictions causes probability distortion and miscalibrated thresholding at inference.
- **Required Fix**: Reserve a dedicated 20% validation split / OOF fold for fitting the Isotonic Calibrator and optimizing decision threshold $\tau$.
