# ER-X Ultimate: Validation & Evaluation Protocol

## 1. Disjoint 5-Fold Entity Cross-Validation Scheme

To rigorously mirror the official test distribution:
1. All unique S1 cluster IDs from `train_ground_truth.tsv` are hashed modulo 5 into 5 disjoint validation splits (80% Train, 20% Val).
2. Any S2/S3 record that is connected to an S1 cluster in Fold $k$ is strictly assigned to Fold $k$.
3. Unmatched/singleton S2 and S3 records in train are partitioned uniformly across the 5 folds.

```
Total S1 Entities (Train: 2,206,821)
├── Fold 0 (20% Val: 441,364 S1s)  <-- Validation Set
└── Folds 1-4 (80% Train: 1,765,457 S1s) <-- Model Training Set
```

## 2. Official Evaluation Metrics

### 2.1 Macro F0.5 Formulation
For each cluster $C_i$:
$$\text{Precision}_i = \frac{|T_{\text{pred}, i} \cap T_{\text{true}, i}|}{|T_{\text{pred}, i}|}, \quad \text{Recall}_i = \frac{|T_{\text{pred}, i} \cap T_{\text{true}, i}|}{|T_{\text{true}, i}|}$$
$$F_{0.5, i} = \frac{1.25 \times \text{Precision}_i \times \text{Recall}_i}{0.25 \times \text{Precision}_i + \text{Recall}_i}$$
$$\text{Macro } F_{0.5} = \frac{1}{N} \sum_{i=1}^N F_{0.5, i}$$

### 2.2 Invariant Checks on Validation Outputs
- Validation pipeline computes precision, recall, and Macro F0.5 on held-out 20% entity fold.
- Tracks candidate recall ladder: Channel 1 $\to$ Channel 6, ensuring overall candidate recall $\ge 98.5\%$.
