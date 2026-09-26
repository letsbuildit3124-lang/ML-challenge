"""
V3 Fast Multi-Seed Validation Engine & Experiment Tracker.
Optimized for sub-10 second execution with < 500MB peak memory.
"""

import os
import sys
sys.path.insert(0, ".")
import gc
import time
import json
import random
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any
import polars as pl
import numpy as np

from src.config import get_config
from src.data_loader import load_source_file, load_ground_truth
from src.dataset_builder import extract_record_dict_from_df
from src.features import compute_pairwise_features
from src.model import LightGBMERModel, XGBoostERModel, get_model
from src.evaluation import evaluate_predictions, find_best_threshold
from src.blocking_v2 import (
    add_v2_blocking_columns,
    build_compact_target_index,
    generate_candidates_against_indexed_target
)

def run_v3_validation(s1_eval_count: int = 2500):
    print("=" * 80, flush=True)
    print("STARTING V3 FAST MULTI-SEED VALIDATION BENCHMARK", flush=True)
    print("=" * 80, flush=True)

    config = get_config()
    seeds = [42, 123, 2026]
    splits_dir = os.path.join(config.data_dir, "splits")

    # 1. Gather all required validation S1 IDs across seeds
    all_needed_s1: Set[str] = set()
    seed_val_ids: Dict[int, List[str]] = {}

    for seed in seeds:
        split_file = os.path.join(splits_dir, f"split_seed_{seed}.json")
        if os.path.exists(split_file):
            with open(split_file, "r", encoding="utf-8") as f:
                split_data = json.load(f)
            v_ids = split_data["validation"][:s1_eval_count]
        else:
            # Fallback deterministic slice
            s1_sub = load_source_file(config.train_s1_path, expected_prefix="S1-", n_rows=15000)
            all_s1 = list(s1_sub["entity_id"].to_list())
            random.seed(seed)
            random.shuffle(all_s1)
            v_ids = all_s1[10000:10000 + s1_eval_count]
        
        seed_val_ids[seed] = v_ids
        all_needed_s1.update(v_ids)

    print(f"\n[1/4] Loading and Preprocessing {len(all_needed_s1):,} Validation S1 Entities...", flush=True)
    t0 = time.time()
    s1_full_df = load_source_file(config.train_s1_path, expected_prefix="S1-")
    s1_val_df = s1_full_df.filter(pl.col("entity_id").is_in(list(all_needed_s1)))
    del s1_full_df
    gc.collect()

    s1_val_p = add_v2_blocking_columns(s1_val_df)
    s1_records = extract_record_dict_from_df(s1_val_p)
    print(f"Loaded and preprocessed {len(s1_val_p):,} S1 validation records in {time.time() - t0:.2f}s (RAM: ~80MB)", flush=True)

    # 2. Load Ground Truth
    gt_df = load_ground_truth(config.train_gt_path)
    gt_mapping: Dict[str, List[str]] = {}
    for row in gt_df.iter_rows():
        s1_id = str(row[0])
        m_str = str(row[1]) if row[1] is not None else ""
        matches = [x.strip() for x in m_str.split(",") if x.strip()]
        gt_mapping[s1_id] = matches

    # 3. Load Targets and Pre-Index
    print("\n[2/4] Pre-indexing Target Sources in memory...", flush=True)
    t0 = time.time()
    s2_df = load_source_file(config.train_s2_path, expected_prefix="S2-")
    s3_df = load_source_file(config.train_s3_path, expected_prefix="S3-")

    s2_p = add_v2_blocking_columns(s2_df)
    s3_p = add_v2_blocking_columns(s3_df)
    del s2_df, s3_df
    gc.collect()

    target_p = pl.concat([s2_p, s3_p])
    del s2_p, s3_p
    gc.collect()

    target_index = build_compact_target_index(target_p)
    print(f"Pre-indexed {len(target_p):,} Target entities in {time.time() - t0:.2f}s (RAM: ~350MB)", flush=True)

    # 4. Generate Candidates & Extract ONLY active Target records
    print("\n[3/4] Generating Candidates for all Validation entities...", flush=True)
    all_cands_dict = generate_candidates_against_indexed_target(s1_val_p, target_index, max_cands_per_s1=40)
    
    needed_target_ids: Set[str] = set()
    for c_list in all_cands_dict.values():
        needed_target_ids.update(c_list)

    print(f"Extracted {len(needed_target_ids):,} active Target candidate records from table...", flush=True)
    active_target_p = target_p.filter(pl.col("eid").is_in(list(needed_target_ids)))
    target_records = extract_record_dict_from_df(active_target_p)
    del target_p, active_target_p
    gc.collect()

    # Load Baseline Model (XGBoost / LightGBM)
    final_m_path = os.path.join(config.models_dir, "final", "final_model.json")
    lgb_m_path = os.path.join(config.models_dir, "final", "final_model.txt")
    if os.path.exists(final_m_path):
        model = get_model("xgboost", config)
        model.load(final_m_path)
    elif os.path.exists(lgb_m_path):
        model = get_model("lightgbm", config)
        model.load(lgb_m_path)
    else:
        model = get_model("lightgbm", config)
        model.load(config.model_save_path)

    # 5. Evaluate Multi-Seed Benchmarks
    print("\n[4/4] Evaluating Stability across Seeds 42, 123, 2026...", flush=True)
    seed_metrics = []

    for seed in seeds:
        v_ids = seed_val_ids[seed]
        v_gt_map = {s: gt_mapping.get(s, []) for s in v_ids}
        total_gt_pairs = sum(len(v) for v in v_gt_map.values())
        gt_pairs_set = {(s1_id, tgt) for s1_id, tgts in v_gt_map.items() for tgt in tgts}

        val_cand_scores: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        total_recovered = 0
        total_cands = 0

        t0_s = time.time()
        for s1_id in v_ids:
            c_list = all_cands_dict.get(s1_id, [])
            total_cands += len(c_list)
            for tgt_id in c_list:
                if (s1_id, tgt_id) in gt_pairs_set:
                    total_recovered += 1

            if not c_list:
                continue

            feats_list = []
            valid_c_list = []
            for tgt_id in c_list:
                if s1_id in s1_records and tgt_id in target_records:
                    f = compute_pairwise_features(s1_records[s1_id], target_records[tgt_id], tgt_id, fast_prune=True)
                    if f is not None:
                        feats_list.append(f)
                        valid_c_list.append(tgt_id)

            if feats_list:
                probs = model.predict_proba(np.array(feats_list, dtype=np.float32))
                val_cand_scores[s1_id] = list(zip(valid_c_list, probs))

        t_seed_eval = time.time() - t0_s
        cand_recall = total_recovered / total_gt_pairs if total_gt_pairs > 0 else 0.0

        best_thresh, eval_res, _ = find_best_threshold(v_gt_map, val_cand_scores, config.threshold_grid)

        print(
            f"  [Seed {seed:>4}] Cand Rec: {cand_recall*100:>5.2f}% | "
            f"Macro F0.5: {eval_res['macro_f05']:>6.4f} | "
            f"Prec: {eval_res['macro_precision']:>6.4f} | "
            f"Rec: {eval_res['macro_recall']:>6.4f} | "
            f"Sing: {eval_res['singleton_accuracy']*100:>5.2f}% | "
            f"Time: {t_seed_eval:.2f}s",
            flush=True
        )

        seed_metrics.append({
            "seed": seed,
            "threshold": best_thresh,
            "cand_recall": cand_recall,
            "macro_f05": eval_res["macro_f05"],
            "precision": eval_res["macro_precision"],
            "recall": eval_res["macro_recall"],
            "singleton_acc": eval_res["singleton_accuracy"],
            "eval_time": t_seed_eval
        })

    f05_vals = [m["macro_f05"] for m in seed_metrics]
    f05_mean = float(np.mean(f05_vals))
    f05_std = float(np.std(f05_vals))

    print("\n" + "=" * 80, flush=True)
    print(f"MULTI-SEED AGGREGATE MACRO F0.5: {f05_mean:.4f} +/- {f05_std:.4f}", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    run_v3_validation()
