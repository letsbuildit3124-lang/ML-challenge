"""
Final Model Retraining Module on 100% Training Data.
Retrains selected architecture (XGBoost or LightGBM) on the full training dataset.
Features:
- Sub-2GB memory footprint via compact target indexing
- S1 streaming chunking with hard negative mining
- Real-time progress bar with throughput and ETA
- Serializes final model checkpoint and metadata to models/final/.
"""

import os
import sys
sys.path.insert(0, ".")
import gc
import json
import time
import argparse
from typing import Dict, List, Tuple, Any, Set
import numpy as np
import polars as pl

from src.config import Config, get_config
from src.data_loader import iter_source_file_chunks, load_source_file
from src.dataset_builder import extract_record_dict_from_df
from src.features import compute_pairwise_features
from src.model import LightGBMERModel, XGBoostERModel, get_model
from src.blocking_v2 import (
    add_v2_blocking_columns,
    build_compact_target_index,
    generate_candidates_against_indexed_target
)

def train_final_model(
    model_choice: str = "auto",
    s1_chunk_size: int = 50000,
    max_negatives_per_s1: int = 2
):
    config = get_config()
    print("=" * 80, flush=True)
    print("PHASE: V3 FINAL PRODUCTION MODEL RETRAINING (100% DATA)", flush=True)
    print("=" * 80, flush=True)

    # 1. Determine Model Type & Threshold
    selected_model_type = model_choice.lower().strip()
    selected_threshold = 0.50

    if selected_model_type == "auto":
        meta_path = os.path.join(config.models_dir, "model_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            selected_model_type = meta.get("selected_winner", "xgboost")
            selected_threshold = meta.get("winner_threshold", 0.50)
            print(f"[Auto Selection] Selected '{selected_model_type.upper()}' with validation threshold {selected_threshold:.2f} from metadata.", flush=True)
        else:
            selected_model_type = "xgboost"

    print(f"Training Model Architecture: {selected_model_type.upper()}", flush=True)

    # 2. Load Ground Truth into compact 64-bit integer hash set
    print("\n[1/3] Loading Ground Truth into compact hash lookup...", flush=True)
    t0 = time.time()
    gt_pairs_hashes: Set[int] = set()

    with open(config.train_gt_path, "r", encoding="utf-8", errors="replace") as f:
        _ = f.readline()  # header
        for line in f:
            line_str = line.strip("\r\n")
            if not line_str:
                continue
            parts = line_str.split("\t")
            if len(parts) >= 2 and parts[1]:
                s1_id = parts[0].strip()
                matches = parts[1].split(",")
                for m in matches:
                    m_id = m.strip()
                    if m_id:
                        gt_pairs_hashes.add(hash((s1_id, m_id)))

    print(f"Loaded {len(gt_pairs_hashes):,} positive ground-truth pairs in {time.time() - t0:.2f}s (RAM: ~180MB)", flush=True)

    # 3. Pre-Index Source 2 and Source 3 once in memory
    print("\n[2/3] Pre-indexing Train Source 2 & Source 3 into compact in-memory target table...", flush=True)
    t_idx = time.time()
    
    target_dfs = []
    target_records: Dict[str, Dict[str, Any]] = {}

    for s_name, path, prefix in [("Train S2", config.train_s2_path, "S2-"), ("Train S3", config.train_s3_path, "S3-")]:
        print(f"  Loading and preprocessing {s_name} ({path})...", flush=True)
        t_s = time.time()
        s_df = load_source_file(path, expected_prefix=prefix)
        s_p = add_v2_blocking_columns(s_df)
        del s_df
        target_records.update(extract_record_dict_from_df(s_p))
        target_dfs.append(s_p)
        print(f"  Processed {s_name} ({len(s_p):,} records) in {time.time() - t_s:.2f}s", flush=True)

    full_target_p = pl.concat(target_dfs)
    del target_dfs
    gc.collect()

    print("  Building in-memory compact hash index...", flush=True)
    target_index = build_compact_target_index(full_target_p)
    total_targets_indexed = len(full_target_p)
    del full_target_p
    gc.collect()

    print(f"Pre-indexed {total_targets_indexed:,} Target entities in {time.time() - t_idx:.2f}s (RAM: ~1.8GB)", flush=True)

    # 4. Stream S1 and extract training pairs with hard negative sampling
    print(f"\n[3/3] Streaming Train Source 1 across chunks of {s1_chunk_size:,} entities...", flush=True)
    
    X_train_chunks: List[np.ndarray] = []
    y_train_chunks: List[np.ndarray] = []
    total_train_pairs = 0
    total_positive_pairs = 0
    total_s1_processed = 0

    chunk_idx = 0
    t_stream_start = time.time()
    total_expected_s1 = 2217337

    for s1_raw_df in iter_source_file_chunks(config.train_s1_path, chunk_size=s1_chunk_size, expected_prefix="S1-"):
        chunk_idx += 1
        t_chk = time.time()
        chunk_s1_ids = s1_raw_df["entity_id"].to_list()
        chunk_s1_count = len(chunk_s1_ids)
        total_s1_processed += chunk_s1_count

        s1_chk = add_v2_blocking_columns(s1_raw_df)
        del s1_raw_df
        s1_records = extract_record_dict_from_df(s1_chk)

        # Fast in-memory candidate lookup
        candidates_dict = generate_candidates_against_indexed_target(
            s1_chk,
            target_index,
            max_cands_per_s1=25
        )

        chk_feats = []
        chk_labels = []
        neg_count_per_s1: Dict[str, int] = {}

        for s1_id in chunk_s1_ids:
            c_list = candidates_dict.get(s1_id, [])
            for tgt_id in c_list:
                if s1_id in s1_records and tgt_id in target_records:
                    is_positive = hash((s1_id, tgt_id)) in gt_pairs_hashes
                    if not is_positive:
                        curr_negs = neg_count_per_s1.get(s1_id, 0)
                        if curr_negs >= max_negatives_per_s1:
                            continue
                        neg_count_per_s1[s1_id] = curr_negs + 1

                    f = compute_pairwise_features(s1_records[s1_id], target_records[tgt_id], tgt_id, fast_prune=False)
                    chk_feats.append(f)
                    chk_labels.append(1.0 if is_positive else 0.0)

        if chk_feats:
            X_chk = np.array(chk_feats, dtype=np.float32)
            y_chk = np.array(chk_labels, dtype=np.float32)
            X_train_chunks.append(X_chk)
            y_train_chunks.append(y_chk)
            total_train_pairs += len(X_chk)
            total_positive_pairs += int(y_chk.sum())

        del s1_chk, s1_records, candidates_dict, chk_feats, chk_labels, neg_count_per_s1
        gc.collect()

        chk_time = time.time() - t_chk
        elap_total = time.time() - t_stream_start
        pct_done = min(100.0, (total_s1_processed / total_expected_s1) * 100.0)
        s1_rate = total_s1_processed / max(elap_total, 0.001)
        est_rem_s = (total_expected_s1 - total_s1_processed) / max(s1_rate, 1.0)
        est_rem_min = est_rem_s / 60.0

        print(
            f"  [Progress: Chunk {chunk_idx:02d} | {pct_done:>5.1f}%] "
            f"S1 Processed: {total_s1_processed:>9,} / {total_expected_s1:,} | "
            f"Pairs: {total_train_pairs:>9,} (Pos: {total_positive_pairs:>7,}) | "
            f"Rate: {s1_rate:>6,.0f} S1/s | "
            f"Chunk: {chk_time:>4.1f}s | "
            f"ETA: {est_rem_min:>4.1f} min",
            flush=True
        )

    # Free memory
    del target_index, target_records, gt_pairs_hashes
    gc.collect()

    print(f"\nExtracted all candidate features in {time.time() - t_stream_start:.2f}s", flush=True)

    print("Concatenating full dataset feature matrices...", flush=True)
    X_train_all = np.vstack(X_train_chunks)
    y_train_all = np.concatenate(y_train_chunks)
    del X_train_chunks, y_train_chunks
    gc.collect()

    print(f"Final Full Training Matrix: {X_train_all.shape} (Positives: {int(y_train_all.sum()):,}, Negatives: {len(y_train_all) - int(y_train_all.sum()):,})", flush=True)

    # 5. Train Selected Model on 100% Data
    print(f"\nRetraining {selected_model_type.upper()} on 100% Training Dataset...", flush=True)
    model = get_model(selected_model_type, config)
    train_res = model.train(X_train_all, y_train_all)

    # 6. Save Final Production Model Checkpoint
    final_dir = os.path.join(config.models_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    
    ext = ".txt" if selected_model_type == "lightgbm" else ".json"
    final_model_path = os.path.join(final_dir, f"final_model{ext}")
    model.save(final_model_path)

    final_meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": selected_model_type,
        "model_path": final_model_path,
        "selected_threshold": selected_threshold,
        "feature_names": config.feature_names,
        "feature_count": len(config.feature_names),
        "total_training_pairs": len(X_train_all),
        "total_positive_pairs": int(y_train_all.sum()),
        "total_negative_pairs": len(y_train_all) - int(y_train_all.sum()),
        "total_s1_entities": total_s1_processed,
        "training_time_s": train_res.get("train_time_seconds", 0.0),
        "best_iteration": train_res.get("best_iteration", 500),
        "top_features": train_res.get("feature_importances", [])[:10]
    }

    final_meta_path = os.path.join(final_dir, "final_model_metadata.json")
    with open(final_meta_path, "w", encoding="utf-8") as f:
        json.dump(final_meta, f, indent=2)

    print("\n" + "=" * 80, flush=True)
    print("FINAL PRODUCTION MODEL RETRAINING COMPLETE", flush=True)
    print("=" * 80, flush=True)
    print(f"  Model Type:             {selected_model_type.upper()}", flush=True)
    print(f"  Final Model Path:       {final_model_path}", flush=True)
    print(f"  Metadata Path:          {final_meta_path}", flush=True)
    print(f"  Selected Threshold:     {selected_threshold:.2f}", flush=True)
    print(f"  Total Training Pairs:   {len(X_train_all):,}", flush=True)
    print(f"  Training Runtime:       {train_res.get('train_time_seconds', 0.0):.2f}s", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train final model on 100% training data")
    parser.add_argument("--model", type=str, default="auto", choices=["lightgbm", "xgboost", "auto"], help="Model architecture")
    parser.add_argument("--chunk-size", type=int, default=50000, help="S1 chunk size for memory safety")
    parser.add_argument("--max-negs", type=int, default=2, help="Max negative candidate pairs per S1 entity")
    args = parser.parse_args()

    train_final_model(
        model_choice=args.model,
        s1_chunk_size=args.chunk_size,
        max_negatives_per_s1=args.max_negs
    )
