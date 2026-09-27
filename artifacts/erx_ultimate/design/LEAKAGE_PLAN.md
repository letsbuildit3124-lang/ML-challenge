# ER-X Ultimate: Data Leakage Prevention Plan

## 1. Leakage Prevention Protocol

To guarantee zero leakage:
1. **Feature Transformation & IDF Calculation**:
   - IDF weights and token frequencies are computed strictly on the training partition.
   - Test partition never updates or influences IDF dictionaries.
2. **Learned Rules Extraction**:
   - Typo, alias, and OCR rules are extracted strictly from `train_ground_truth.tsv` associated with the training fold.
3. **Probability Calibration**:
   - Out-of-fold predictions from K-fold cross-validation are used to train the Isotonic Regression calibrators.
   - Decision thresholds $\tau$ are computed on OOF sets to maximize Expected-F0.5 without touching the test data.
4. **Target Matching Exclusivity**:
   - Target assignments are resolved purely by rank and calibrated probability margin.
