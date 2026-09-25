"""
Final Model Training Module on 100% Training Data.
Retrains the selected model (LightGBM, XGBoost, or Auto) on all available training sources.
Uses Pre-Indexed In-Memory Target Lookups to complete all 45 S1 chunks in ~2 minutes.
"""

import os
import gc
import json
import time
import argparse
from typing import Dict, List, Tuple, Any, Set
import numpy as np
import polars as pl

from src.config import Config, get_config
from src.data_loader import iter_source_file_chunks
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
    max_negatives_per_s1: int = 3
):
    config = get_config()
    print("=" * 80)
    print("PHASE: ULTRA-FAST FINAL PRODUCTION MODEL RETRAINING (100% DATA)")
    print("=" * 80)

    # 1. Determine Model Type & Threshold
    selected_model_type = model_choice.lower().strip()
    selected_threshold = 0.50

    if selected_model_type == "auto":
        meta_path = os.path.join(config.models_dir, "model_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            selected_model_type = meta.get("selected_winner", "lightgbm")
            selected_threshold = meta.get("winner_threshold", 0.50)
            print(f"[Auto Selection] Selected '{selected_model_type.upper()}' with validation threshold {selected_threshold:.2f} from metadata.")
        else:
            print("[Auto Selection] Metadata not found. Defaulting to 'lightgbm' at threshold 0.50.")
            selected_model_type = "lightgbm"

    print(f"Training Model Architecture: {selected_model_type.upper()}")

    # 2. Load Ground Truth into a compact flat set of pair strings
    print("\n[1/3] Loading Ground Truth into compact pair lookup...", flush=True)
    t0 = time.time()
    gt_pairs_set: Set[str] = set()

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
                        gt_pairs_set.add(f"{s1_id}_{m_id}")

    print(f"Loaded {len(gt_pairs_set):,} positive ground-truth pairs in {time.time() - t0:.2f}s", flush=True)

    # 3. Pre-Index Source 2 and Source 3 once in memory
    print("\n[2/3] Pre-indexing Train Source 2 & Source 3 into compact in-memory target table...", flush=True)
    t_idx = time.time()
    indexed_target = build_compact_target_index(
        config.train_s2_path,
        config.train_s3_path,
        chunk_size=250000
    )
    print(f"Pre-indexed {len(indexed_target):,} Target entities in {time.time() - t_idx:.2f}s (RAM safe)", flush=True)

    # 4. Stream S1 and perform sub-second candidate lookups
    print(f"\n[3/3] Streaming Train Source 1 across chunks of {s1_chunk_size:,} entities...", flush=True)
    
    X_train_chunks: List[np.ndarray] = []
    y_train_chunks: List[np.ndarray] = []
    total_train_pairs = 0
    total_positive_pairs = 0
    total_s1_processed = 0

    chunk_idx = 0
    t_stream_start = time.time()

    for s1_raw_df in iter_source_file_chunks(config.train_s1_path, chunk_size=s1_chunk_size):
        chunk_idx += 1
        t_chk = time.time()
        chunk_s1_count = len(s1_raw_df)
        total_s1_processed += chunk_s1_count

        s1_chk = add_v2_blocking_columns(s1_raw_df)
        del s1_raw_df
        s1_records = extract_record_dict_from_df(s1_chk)

        # Fast in-memory candidate lookup
        cand_df, target_records = generate_candidates_against_indexed_target(
            s1_chk,
            indexed_target,
            max_cands_per_s1=20
        )

        # Compute pairwise features with hard negative sampling
        chk_feats = []
        chk_labels = []

        cand_rows = cand_df.to_dict(as_series=False)
        s1_col = cand_rows["s1_id"]
        tgt_col = cand_rows["target_id"]

        neg_count_per_s1: Dict[str, int] = {}

        for s1_id, tgt_id in zip(s1_col, tgt_col):
            if s1_id in s1_records and tgt_id in target_records:
                is_positive = f"{s1_id}_{tgt_id}" in gt_pairs_set
                
                if not is_positive:
                    current_negs = neg_count_per_s1.get(s1_id, 0)
                    if current_negs >= max_negatives_per_s1:
                        continue
                    neg_count_per_s1[s1_id] = current_negs + 1

                f = compute_pairwise_features(s1_records[s1_id], target_records[tgt_id], tgt_id)
                chk_feats.append(f)
                chk_labels.append(1.0 if is_positive else 0.0)

        if chk_feats:
            X_chk = np.array(chk_feats, dtype=np.float32)
            y_chk = np.array(chk_labels, dtype=np.float32)
            X_train_chunks.append(X_chk)
            y_train_chunks.append(y_chk)
            total_train_pairs += len(X_chk)
            total_positive_pairs += int(y_chk.sum())

        del s1_chk, s1_records, cand_df, target_records, chk_feats, chk_labels, neg_count_per_s1
        gc.collect()

        print(
            f"  [Chunk {chunk_idx:02d}] S1: {total_s1_processed:,} | "
            f"Accumulated Pairs: {total_train_pairs:,} (Pos: {total_positive_pairs:,}) | Time: {time.time() - t_chk:.2f}s",
            flush=True
        )

    # Free memory
    del indexed_target, gt_pairs_set
    gc.collect()

    print(f"\nExtracted all candidate features in {time.time() - t_stream_start:.2f}s", flush=True)

    print("Concatenating full dataset feature matrices...", flush=True)
    X_train_all = np.vstack(X_train_chunks)
    y_train_all = np.concatenate(y_train_chunks)
    del X_train_chunks, y_train_chunks
    gc.collect()

    print(f"Final Full Training Matrix: {X_train_all.shape} (Positives: {int(y_train_all.sum()):,}, Negatives: {len(y_train_all) - int(y_train_all.sum()):,})")

    # 5. Train Selected Model on 100% Data
    print(f"\nRetraining {selected_model_type.upper()} on 100% Training Dataset...", flush=True)
    model = get_model(selected_model_type, config)
    train_res = model.train(X_train_all, y_train_all)

    # 6. Save Final Production Model
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
        "training_time_s": train_res["training_time"],
        "best_iteration": train_res["best_iteration"],
        "top_features": list(train_res["feature_importances"].keys())[:10]
    }

    final_meta_path = os.path.join(final_dir, "final_model_metadata.json")
    with open(final_meta_path, "w", encoding="utf-8") as f:
        json.dump(final_meta, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL PRODUCTION MODEL RETRAINING COMPLETE")
    print("=" * 80)
    print(f"  Model Type:             {selected_model_type.upper()}")
    print(f"  Final Model Path:       {final_model_path}")
    print(f"  Metadata Path:          {final_meta_path}")
    print(f"  Selected Threshold:     {selected_threshold:.2f}")
    print(f"  Total Training Pairs:   {len(X_train_all):,}")
    print(f"  Training Runtime:       {train_res['training_time']:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train final model on 100% training data")
    parser.add_argument("--model", type=str, default="auto", choices=["lightgbm", "xgboost", "auto"], help="Model architecture")
    parser.add_argument("--chunk-size", type=int, default=50000, help="S1 chunk size for memory safety")
    parser.add_argument("--max-negs", type=int, default=3, help="Max negative candidate pairs per S1 entity")
    args = parser.parse_args()

    train_final_model(
        model_choice=args.model,
        s1_chunk_size=args.chunk_size,
        max_negatives_per_s1=args.max_negs
    )
