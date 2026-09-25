"""
Model Comparison Module for Entity Resolution.
Trains and compares LightGBM vs XGBoost on the exact same V2 candidate pool and 29 features.
Uses memory-safe chunked target streaming to run reliably on 2 CPU / 8 GB RAM EC2 instances.
Finds the optimal validation threshold for both models, records metrics, and selects the winner.
"""

import os
import gc
import json
import time
import random
from typing import Dict, List, Tuple, Any, Set
import numpy as np
import polars as pl

from src.config import Config, get_config
from src.data_loader import load_source_file, load_ground_truth
from src.dataset_builder import extract_record_dict_from_df
from src.features import compute_pairwise_features
from src.evaluation import evaluate_predictions
from src.model import LightGBMERModel, XGBoostERModel
from src.blocking_v2 import add_v2_blocking_columns, block_s1_against_target_file_chunked


def run_model_comparison():
    config = get_config()
    print("=" * 80)
    print("RUNNING MODEL COMPARISON: LIGHTGBM VS XGBOOST (V2 CANDIDATES & 29 FEATURES)")
    print("=" * 80)

    # 1. Load Data
    t_start = time.time()
    total_requested = config.train_sample_s1_count + config.val_sample_s1_count
    
    # Load only the required sample of S1 rows for validation benchmark
    s1_df = load_source_file(config.train_s1_path, expected_prefix="S1-", n_rows=total_requested * 2)
    gt_df = load_ground_truth(config.train_gt_path, n_rows=total_requested * 2)

    # Ground Truth Mapping
    gt_map: Dict[str, List[str]] = {}
    for row in gt_df.iter_rows():
        s1_id = str(row[0])
        m_str = str(row[1]) if row[1] is not None else ""
        gt_map[s1_id] = [x.strip() for x in m_str.split(",") if x.strip()]
    del gt_df
    gc.collect()

    # Deterministic S1 split (Seed 42)
    all_s1_ids = list(s1_df["entity_id"].to_list())
    random.seed(config.seed)
    random.shuffle(all_s1_ids)

    selected_s1_ids = all_s1_ids[:total_requested]
    split_idx = config.train_sample_s1_count
    train_s1_ids = set(selected_s1_ids[:split_idx])
    val_s1_ids = set(selected_s1_ids[split_idx:total_requested])
    val_s1_id_list = list(selected_s1_ids[split_idx:total_requested])

    print(f"Selected {len(train_s1_ids):,} Train S1 entities and {len(val_s1_ids):,} Validation S1 entities (Seed: {config.seed})")

    val_gt_map = {s1_id: gt_map.get(s1_id, []) for s1_id in val_s1_ids}
    total_val_gt_pairs = sum(len(v) for v in val_gt_map.values())
    print(f"Total Validation Ground-Truth Pairs: {total_val_gt_pairs:,}")

    # Prepare S1 DataFrame
    s1_sub_df = s1_df.filter(pl.col("entity_id").is_in(selected_s1_ids))
    del s1_df
    gc.collect()

    s1_p = add_v2_blocking_columns(s1_sub_df)
    s1_records = extract_record_dict_from_df(s1_p)
    target_records: Dict[str, Dict[str, Any]] = {}

    # 2. Block against S2 in memory-safe streaming chunks of 100k
    print("\nStreaming and Blocking against Train Source 2 in chunks of 100k rows...", flush=True)
    t0 = time.time()
    cands_s2, records_s2 = block_s1_against_target_file_chunked(
        s1_p,
        config.train_s2_path,
        target_chunk_size=100000,
        max_cands_per_s1=40,
        extract_matched_records=True
    )
    target_records.update(records_s2)
    del records_s2
    gc.collect()
    print(f"S2 Blocking complete in {time.time() - t0:.2f}s (Found {len(cands_s2):,} S2 candidate pairs)", flush=True)

    # 3. Block against S3 in memory-safe streaming chunks of 100k
    print("\nStreaming and Blocking against Train Source 3 in chunks of 100k rows...", flush=True)
    t0 = time.time()
    cands_s3, records_s3 = block_s1_against_target_file_chunked(
        s1_p,
        config.train_s3_path,
        target_chunk_size=100000,
        max_cands_per_s1=40,
        extract_matched_records=True
    )
    target_records.update(records_s3)
    del records_s3
    gc.collect()
    print(f"S3 Blocking complete in {time.time() - t0:.2f}s (Found {len(cands_s3):,} S3 candidate pairs)", flush=True)

    # 4. Candidate pairs union and split
    all_cand_df = pl.concat([cands_s2, cands_s3]).unique()
    train_cand_df = all_cand_df.filter(pl.col("s1_id").is_in(list(train_s1_ids)))
    val_cand_df = all_cand_df.filter(pl.col("s1_id").is_in(list(val_s1_ids)))

    print(f"\nGenerated {len(train_cand_df):,} Train candidate pairs and {len(val_cand_df):,} Validation candidate pairs.")

    # 5. Extract features for Train and Val
    print("\nExtracting Pairwise Features for Training Set...", flush=True)
    t0 = time.time()
    X_train_list, y_train_list = [], []
    for row in train_cand_df.iter_rows():
        s1_id, cand_id = str(row[0]), str(row[1])
        if s1_id in s1_records and cand_id in target_records:
            f = compute_pairwise_features(s1_records[s1_id], target_records[cand_id], cand_id)
            label = 1.0 if cand_id in set(gt_map.get(s1_id, [])) else 0.0
            X_train_list.append(f)
            y_train_list.append(label)

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.float32)
    print(f"Train Matrix: {X_train.shape} (Positives: {int(y_train.sum()):,}, Negatives: {len(y_train) - int(y_train.sum()):,}) in {time.time() - t0:.2f}s")

    print("\nExtracting Pairwise Features for Validation Set...", flush=True)
    t0 = time.time()
    X_val_list, y_val_list = [], []
    val_pairs = []
    for row in val_cand_df.iter_rows():
        s1_id, cand_id = str(row[0]), str(row[1])
        if s1_id in s1_records and cand_id in target_records:
            f = compute_pairwise_features(s1_records[s1_id], target_records[cand_id], cand_id)
            label = 1.0 if cand_id in set(gt_map.get(s1_id, [])) else 0.0
            X_val_list.append(f)
            y_val_list.append(label)
            val_pairs.append((s1_id, cand_id))

    X_val = np.array(X_val_list, dtype=np.float32)
    y_val = np.array(y_val_list, dtype=np.float32)
    print(f"Val Matrix: {X_val.shape} (Positives: {int(y_val.sum()):,}, Negatives: {len(y_val) - int(y_val.sum()):,}) in {time.time() - t0:.2f}s")

    threshold_grid = [round(t, 2) for t in np.arange(0.10, 1.00, 0.05)]

    # =========================================================================
    # 6. TRAIN & EVALUATE LIGHTGBM
    # =========================================================================
    print("\n" + "=" * 80)
    print("TRAINING & EVALUATING LIGHTGBM")
    print("=" * 80)
    lgb_model = LightGBMERModel(config)
    lgb_train_res = lgb_model.train(X_train, y_train, X_val, y_val)

    t_inf_lgb = time.time()
    lgb_probs = lgb_model.predict_proba(X_val)
    lgb_inf_time = time.time() - t_inf_lgb

    print(f"[LightGBM] Scored {len(X_val):,} validation pairs in {lgb_inf_time:.3f}s")

    lgb_grid_results = []
    best_lgb_f05 = -1.0
    best_lgb_res = {}
    best_lgb_thresh = 0.50

    for thresh in threshold_grid:
        pred_map: Dict[str, List[str]] = {s1: [] for s1 in val_s1_id_list}
        for (s1_id, cand_id), p in zip(val_pairs, lgb_probs):
            if p >= thresh:
                pred_map[s1_id].append(cand_id)

        eval_res = evaluate_predictions(val_gt_map, pred_map)
        matches_count = sum(len(v) for v in pred_map.values())
        empty_count = sum(1 for v in pred_map.values() if not v)

        entry = {
            "threshold": thresh,
            "macro_f05": eval_res["macro_f05"],
            "precision": eval_res["macro_precision"],
            "recall": eval_res["macro_recall"],
            "singleton_accuracy": eval_res["singleton_accuracy"],
            "predicted_matches": matches_count,
            "empty_predictions": empty_count
        }
        lgb_grid_results.append(entry)

        if (eval_res["macro_f05"] > best_lgb_f05 + 1e-5) or (abs(eval_res["macro_f05"] - best_lgb_f05) <= 1e-5 and thresh > best_lgb_thresh):
            best_lgb_f05 = eval_res["macro_f05"]
            best_lgb_thresh = thresh
            best_lgb_res = entry

    print(f"[LightGBM Best] Threshold: {best_lgb_thresh:.2f} | Macro F0.5: {best_lgb_res['macro_f05']:.4f} | Prec: {best_lgb_res['precision']:.4f} | Rec: {best_lgb_res['recall']:.4f} | Singleton Acc: {best_lgb_res['singleton_accuracy']*100:.2f}%")

    # =========================================================================
    # 7. TRAIN & EVALUATE XGBOOST
    # =========================================================================
    print("\n" + "=" * 80)
    print("TRAINING & EVALUATING XGBOOST")
    print("=" * 80)
    xgb_model = XGBoostERModel(config)
    xgb_train_res = xgb_model.train(X_train, y_train, X_val, y_val)

    t_inf_xgb = time.time()
    xgb_probs = xgb_model.predict_proba(X_val)
    xgb_inf_time = time.time() - t_inf_xgb

    print(f"[XGBoost] Scored {len(X_val):,} validation pairs in {xgb_inf_time:.3f}s")

    xgb_grid_results = []
    best_xgb_f05 = -1.0
    best_xgb_res = {}
    best_xgb_thresh = 0.50

    for thresh in threshold_grid:
        pred_map: Dict[str, List[str]] = {s1: [] for s1 in val_s1_id_list}
        for (s1_id, cand_id), p in zip(val_pairs, xgb_probs):
            if p >= thresh:
                pred_map[s1_id].append(cand_id)

        eval_res = evaluate_predictions(val_gt_map, pred_map)
        matches_count = sum(len(v) for v in pred_map.values())
        empty_count = sum(1 for v in pred_map.values() if not v)

        entry = {
            "threshold": thresh,
            "macro_f05": eval_res["macro_f05"],
            "precision": eval_res["macro_precision"],
            "recall": eval_res["macro_recall"],
            "singleton_accuracy": eval_res["singleton_accuracy"],
            "predicted_matches": matches_count,
            "empty_predictions": empty_count
        }
        xgb_grid_results.append(entry)

        if (eval_res["macro_f05"] > best_xgb_f05 + 1e-5) or (abs(eval_res["macro_f05"] - best_xgb_f05) <= 1e-5 and thresh > best_xgb_thresh):
            best_xgb_f05 = eval_res["macro_f05"]
            best_xgb_thresh = thresh
            best_xgb_res = entry

    print(f"[XGBoost Best]  Threshold: {best_xgb_thresh:.2f} | Macro F0.5: {best_xgb_res['macro_f05']:.4f} | Prec: {best_xgb_res['precision']:.4f} | Rec: {best_xgb_res['recall']:.4f} | Singleton Acc: {best_xgb_res['singleton_accuracy']*100:.2f}%")

    # =========================================================================
    # 8. WINNER DETERMINATION & SAVE ARTIFACTS
    # =========================================================================
    winner_name = "lightgbm" if best_lgb_res["macro_f05"] >= best_xgb_res["macro_f05"] else "xgboost"
    winner_f05 = max(best_lgb_res["macro_f05"], best_xgb_res["macro_f05"])
    winner_thresh = best_lgb_thresh if winner_name == "lightgbm" else best_xgb_thresh

    # Save models
    lgb_save_path = os.path.join(config.models_dir, "lightgbm", "model.txt")
    xgb_save_path = os.path.join(config.models_dir, "xgboost", "model.json")
    lgb_model.save(lgb_save_path)
    xgb_model.save(xgb_save_path)

    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": config.seed,
        "feature_names": config.feature_names,
        "feature_count": len(config.feature_names),
        "train_pairs_count": len(X_train),
        "val_pairs_count": len(X_val),
        "train_s1_count": len(train_s1_ids),
        "val_s1_count": len(val_s1_ids),
        "selected_winner": winner_name,
        "winner_threshold": winner_thresh,
        "winner_validation_macro_f05": winner_f05,
        "lightgbm": {
            "model_path": lgb_save_path,
            "best_threshold": best_lgb_thresh,
            "metrics": best_lgb_res,
            "training_time_s": lgb_train_res["training_time"],
            "inference_time_s": lgb_inf_time,
            "best_iteration": lgb_train_res["best_iteration"],
            "grid_results": lgb_grid_results
        },
        "xgboost": {
            "model_path": xgb_save_path,
            "best_threshold": best_xgb_thresh,
            "metrics": best_xgb_res,
            "training_time_s": xgb_train_res["training_time"],
            "inference_time_s": xgb_inf_time,
            "best_iteration": xgb_train_res["best_iteration"],
            "grid_results": xgb_grid_results
        }
    }

    meta_path = os.path.join(config.models_dir, "model_metadata.json")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved model metadata to {meta_path}")

    # Write model_comparison.md
    report_path = os.path.join(config.reports_dir, "model_comparison.md")
    os.makedirs(config.reports_dir, exist_ok=True)

    report_lines = [
        "# Model Comparison Report: LightGBM vs XGBoost",
        "",
        "## 1. Executive Summary & Validation Benchmark",
        "",
        f"- **Best Validation Model**: `{winner_name.upper()}`",
        f"- **Winning Threshold**: `{winner_thresh:.2f}`",
        f"- **Winning Macro F0.5**: `{winner_f05:.4f}`",
        "",
        "| Model | Optimal Threshold | Macro F0.5 | Macro Precision | Macro Recall | Singleton Accuracy | Predicted Matches | Empty Predictions | Training Time | Inference Time |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **LightGBM** | `{best_lgb_thresh:.2f}` | **{best_lgb_res['macro_f05']:.4f}** | {best_lgb_res['precision']:.4f} | {best_lgb_res['recall']:.4f} | {best_lgb_res['singleton_accuracy']*100:.2f}% | {best_lgb_res['predicted_matches']:,} | {best_lgb_res['empty_predictions']:,} | {lgb_train_res['training_time']:.2f}s | {lgb_inf_time:.3f}s |",
        f"| **XGBoost**  | `{best_xgb_thresh:.2f}` | **{best_xgb_res['macro_f05']:.4f}** | {best_xgb_res['precision']:.4f} | {best_xgb_res['recall']:.4f} | {best_xgb_res['singleton_accuracy']*100:.2f}% | {best_xgb_res['predicted_matches']:,} | {best_xgb_res['empty_predictions']:,} | {xgb_train_res['training_time']:.2f}s | {xgb_inf_time:.3f}s |",
        "",
        "---",
        "",
        "## 2. Threshold Search Grid Comparison",
        "",
        "| Threshold | LightGBM F0.5 | LightGBM Prec | LightGBM Rec | XGBoost F0.5 | XGBoost Prec | XGBoost Rec |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for l_entry, x_entry in zip(lgb_grid_results, xgb_grid_results):
        t = l_entry["threshold"]
        report_lines.append(
            f"| `{t:.2f}` | {l_entry['macro_f05']:.4f} | {l_entry['precision']:.4f} | {l_entry['recall']:.4f} | "
            f"{x_entry['macro_f05']:.4f} | {x_entry['precision']:.4f} | {x_entry['recall']:.4f} |"
        )

    report_lines.extend([
        "",
        "---",
        "",
        "## 3. Top Feature Importances",
        "",
        "### LightGBM (Gain):",
    ])
    for fname, imp in list(lgb_train_res["feature_importances"].items())[:8]:
        report_lines.append(f"- `{fname}`: {imp:,.1f}")

    report_lines.extend([
        "",
        "### XGBoost (Importance):",
    ])
    for fname, imp in list(xgb_train_res["feature_importances"].items())[:8]:
        report_lines.append(f"- `{fname}`: {imp:.4f}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Wrote full comparison report to {report_path}")

    # =========================================================================
    # 9. PRINT FINAL CONSOLE SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(f"LightGBM:")
    print(f"  Threshold:          {best_lgb_thresh:.2f}")
    print(f"  Macro F0.5:         {best_lgb_res['macro_f05']:.4f}")
    print(f"  Precision:          {best_lgb_res['precision']:.4f}")
    print(f"  Recall:             {best_lgb_res['recall']:.4f}")
    print(f"  Singleton Accuracy: {best_lgb_res['singleton_accuracy']*100:.2f}%")
    print(f"  Predicted Matches:  {best_lgb_res['predicted_matches']:,}")
    print(f"  Training Time:      {lgb_train_res['training_time']:.2f}s")
    print("")
    print(f"XGBoost:")
    print(f"  Threshold:          {best_xgb_thresh:.2f}")
    print(f"  Macro F0.5:         {best_xgb_res['macro_f05']:.4f}")
    print(f"  Precision:          {best_xgb_res['precision']:.4f}")
    print(f"  Recall:             {best_xgb_res['recall']:.4f}")
    print(f"  Singleton Accuracy: {best_xgb_res['singleton_accuracy']*100:.2f}%")
    print(f"  Predicted Matches:  {best_xgb_res['predicted_matches']:,}")
    print(f"  Training Time:      {xgb_train_res['training_time']:.2f}s")
    print("")
    print(f"Selected Winner:      {winner_name.upper()} (Validation Macro F0.5 = {winner_f05:.4f} @ Threshold {winner_thresh:.2f})")
    print("=" * 80)


if __name__ == "__main__":
    run_model_comparison()
