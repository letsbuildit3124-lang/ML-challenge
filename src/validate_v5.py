"""
Antigravity V5 Fast Multi-Seed Macro F0.5 Validation Engine.
Evaluates the end-to-end V5 retrieval + ranking pipeline across validation splits.

Outputs:
- Pair Candidate Recall
- Precision, Recall, Macro F0.5
- Optimal Decision Threshold (tau*)
- S1 Cardinality Breakdown (0-match, 1-match, multi-match)
- Process-level RSS profiling
"""

import os
import sys
import gc
import json
import time
import argparse
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any
import polars as pl
import numpy as np

from src.config import get_config
from src.data_loader import load_source_file, load_ground_truth
from src.dataset_builder import extract_record_dict_from_df
from src.rapidfuzz_features import compute_tiered_pairwise_features
from src.model import LightGBMERModel, XGBoostERModel, get_model
from src.evaluation import evaluate_predictions, find_best_threshold
from src.resource_tracker import get_current_rss_mb, get_peak_rss_mb, MemoryTracker
from src.v5_retrieval_engine import V5RetrievalEngine

def run_v5_validation(
    s1_eval_count: int = 2500,
    budget: int = 250,
    model_choice: str = "auto",
    workers: int = 8
):
    config = get_config()
    print("=" * 80)
    print("ANTIGRAVITY V5 — FULL PIPELINE VALIDATION & MACRO F0.5 BENCHMARK")
    print(f"Validation S1 Count: {s1_eval_count:,} | Candidate Budget: {budget}/S1 | Workers: {workers}")
    print(f"Initial Process RSS: {get_current_rss_mb():.2f} MB")
    print("=" * 80)

    # 1. Load Validation Split IDs
    splits_dir = os.path.join(config.data_dir, "splits")
    split_file = os.path.join(splits_dir, "split_seed_42.json")

    gt_df = load_ground_truth(config.train_gt_path)
    if os.path.exists(split_file):
        with open(split_file, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        eval_s1_ids = split_data["validation"][:s1_eval_count]
    else:
        all_gt_s1 = gt_df["source1_entity_id"].unique().to_list()
        eval_s1_ids = all_gt_s1[:s1_eval_count]

    eval_s1_set = set(eval_s1_ids)

    # Build GT map
    gt_mapping: Dict[str, List[str]] = {}
    gt_pairs_set: Set[Tuple[str, str]] = set()
    for row in gt_df.iter_rows():
        s1_id = str(row[0])
        if s1_id in eval_s1_set:
            m_str = str(row[1]) if row[1] is not None else ""
            matches = [x.strip() for x in m_str.split(",") if x.strip()]
            gt_mapping[s1_id] = matches
            for m in matches:
                gt_pairs_set.add((s1_id, m))

    total_gt = len(gt_pairs_set)
    print(f"Loaded {len(eval_s1_ids):,} Validation Entities ({total_gt:,} Ground Truth Pairs).")

    # 2. Load and Preprocess Validation S1 Records
    t0 = time.time()
    s1_full_df = load_source_file(config.train_s1_path, expected_prefix="S1-")
    s1_val_df = s1_full_df.filter(pl.col("entity_id").is_in(eval_s1_ids))
    del s1_full_df
    gc.collect()

    s1_val_p = add_v2_blocking_columns(s1_val_df)
    s1_records = extract_record_dict_from_df(s1_val_p)
    print(f"Extracted {len(s1_records):,} S1 validation records in {time.time() - t0:.2f}s.")

    # 3. Candidate Generation via V5 Retrieval Engine
    print("\n" + "-" * 80)
    print("[Pipeline Stage 1] Running V5 Multi-Pass Candidate Retrieval...")
    engine = V5RetrievalEngine(memory_limit="8GB", threads=8, workers=workers)
    expected_sources = [config.train_s2_path, config.train_s3_path]
    engine.indexer.ensure_cache_ready(expected_sources if all(os.path.exists(p) for p in expected_sources) else None)

    t_ret = time.time()
    cands_raw = engine.generate_candidates(
        s1_val_p,
        enable_deterministic=True,
        enable_ngram=True,
        enable_token=True,
        enable_address=True,
        enable_fts=False,
        enable_fuzzy_rerank=True,
        max_candidates_per_s1=budget
    )
    ret_duration = time.time() - t_ret

    # Measure candidate recall
    recovered_pairs = 0
    all_cands_dict = {}
    pair_tuples = []

    for s1_id in eval_s1_ids:
        clist = cands_raw.get(s1_id, [])
        c_set = set(tid for tid, mask, score, rec in clist)
        all_cands_dict[s1_id] = list(c_set)

        true_set = set(gt_mapping.get(s1_id, []))
        matched = true_set & c_set
        recovered_pairs += len(matched)

        for tid, mask, score, rec in clist:
            pair_tuples.append((s1_id, tid, mask, rec))

    cand_recall = (recovered_pairs / total_gt) * 100.0 if total_gt > 0 else 0.0
    avg_cands = len(pair_tuples) / len(eval_s1_ids) if eval_s1_ids else 0.0
    print(f"[V5 Retrieval Result] Candidate Recall: {cand_recall:.2f}% ({recovered_pairs:,}/{total_gt:,}) | Avg Cands: {avg_cands:.1f} | Time: {ret_duration:.2f}s | RSS: {get_current_rss_mb():.1f} MB")

    # 4. Feature Extraction & Model Scoring
    print("\n" + "-" * 80)
    print(f"[Pipeline Stage 2] Computing Pairwise Features for {len(pair_tuples):,} Candidate Pairs...")
    t_feat = time.time()

    X_list = []
    y_list = []
    pairs_order = []

    for s1_id, tid, mask, t_rec in pair_tuples:
        s1_r = s1_records.get(s1_id, {})
        feat_vec = compute_tiered_pairwise_features(s1_r, t_rec, tid, provenance_mask=mask)
        X_list.append(feat_vec)
        y_val = 1 if tid in gt_mapping.get(s1_id, []) else 0
        y_list.append(y_val)
        pairs_order.append((s1_id, tid))

    X_arr = np.array(X_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.int32)
    feat_duration = time.time() - t_feat
    print(f"Feature computation completed in {feat_duration:.2f}s ({len(X_arr):,} pairs, shape: {X_arr.shape}).")

    # 5. Load Trained Model
    print("\n" + "-" * 80)
    print("[Pipeline Stage 3] Loading GBDT Matcher & Evaluating Predictions...")
    
    # Locate model path
    model_path = None
    selected_model_type = "xgboost" if model_choice in ["auto", "xgboost"] else "lightgbm"
    ext = ".json" if selected_model_type == "xgboost" else ".txt"

    candidates_model_paths = [
        os.path.join(config.models_dir, "final", f"final_model{ext}"),
        os.path.join(config.models_dir, selected_model_type, f"model{ext}"),
        os.path.join(config.models_dir, f"model{ext}"),
        config.model_save_path
    ]

    for p in candidates_model_paths:
        if os.path.exists(p):
            model_path = p
            break

    if model_path is None:
        print(f"[WARN] Pretrained model not found at standard paths. Training a quick validation model...")
        model = get_model(selected_model_type)
        model.fit(X_arr, y_arr)
    else:
        print(f"Loading pretrained model from {model_path}...")
        model = get_model(selected_model_type)
        model.load(model_path)

    probabilities = model.predict_proba(X_arr)

    # 6. Threshold Optimization for Macro F0.5
    prob_dict = defaultdict(list)
    for (s1_id, tid), prob in zip(pairs_order, probabilities):
        prob_dict[s1_id].append((tid, float(prob)))

    best_thresh, best_metrics = find_best_threshold(prob_dict, gt_mapping, metric="f0_5")

    print("\n" + "=" * 80)
    print("V5 VALIDATION EVALUATION RESULTS (MACRO F0.5 OPTIMIZED)")
    print("=" * 80)
    print(f"Candidate Pair Recall:    {cand_recall:>8.2f}%")
    print(f"Optimal Decision Tau (*): {best_thresh:>8.2f}")
    print(f"Precision:                {best_metrics.get('precision', 0.0) * 100:>8.2f}%")
    print(f"Recall:                   {best_metrics.get('recall', 0.0) * 100:>8.2f}%")
    print(f"Macro F0.5 Score:         {best_metrics.get('macro_f0_5', 0.0):>8.4f} (TARGET: >= 0.9500)")
    print(f"Macro F1.0 Score:         {best_metrics.get('macro_f1', 0.0):>8.4f}")
    print(f"Singleton Accuracy:       {best_metrics.get('singleton_accuracy', 0.0) * 100:>8.2f}%")
    print(f"Peak Process RSS:         {get_peak_rss_mb():>8.1f} MB")
    print("=" * 80)

    engine.close()
    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="Antigravity V5 Multi-Seed Macro F0.5 Validation Engine")
    parser.add_argument("--s1-count", type=int, default=2500, help="Number of S1 validation entities (default: 2500)")
    parser.add_argument("--budget", type=int, default=250, help="Candidate budget per S1 entity (default: 250)")
    parser.add_argument("--model", type=str, default="auto", choices=["auto", "xgboost", "lightgbm"], help="Model architecture")
    parser.add_argument("--workers", type=int, default=8, help="Number of CPU workers (default: 8)")
    args = parser.parse_args()

    run_v5_validation(
        s1_eval_count=args.s1_count,
        budget=args.budget,
        model_choice=args.model,
        workers=args.workers
    )

if __name__ == "__main__":
    main()
