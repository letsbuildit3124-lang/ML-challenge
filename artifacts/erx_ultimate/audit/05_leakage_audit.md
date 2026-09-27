# ER-X Ultimate: Data Leakage Prevention & Generalization Audit

## 1. Leakage Vectors & Audit Findings
In high-cardinality entity resolution pipelines, data leakage can artificially inflate validation metrics while causing catastrophic degradation on unseen test sets.

### Identified Vectors:
1. **Target ID / Entity Overlap in Cross-Validation**:
   - Random record splitting allows records belonging to the same underlying entity cluster to appear in both training and validation splits.
   - *Fix*: Pure Disjoint Entity Splitting. S1 entities are hashed into 5 folds ($80/20$). All target records (S2, S3) mapped to an S1 cluster must strictly stay within that fold.
2. **Frequency Statistics Leakage**:
   - Global inverse document frequency (IDF) and rare token tables calculated over the entire dataset before splitting.
   - *Fix*: Calculate all IDF weights, token frequencies, and learned typo/alias dictionaries ONLY on the training split.
3. **Target Exclusivity Overfitting**:
   - Threshold tuning on validation set using identical candidate pools as training.
   - *Fix*: Calibration and margin threshold optimization performed on held-out Out-Of-Fold (OOF) predictions across the 5 disjoint folds.

---

## 2. In-Fold Calibration Invariants
- LightGBM models trained on 4 folds ($80\%$).
- Isotonic regression calibrators fitted strictly on OOF predictions of the 5th fold ($20\%$).
- Threshold search for Expected-F0.5 maximization runs exclusively on OOF probability outputs.
- Test inference applies the ensemble calibrator without reference to ground truth statistics.
