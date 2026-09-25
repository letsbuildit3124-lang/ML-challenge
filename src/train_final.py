"""
Final Model Training Module on 100% Training Data.
Retrains the selected model (LightGBM, XGBoost, or Auto) on all available training sources.
Processes data in memory-safe chunks and saves to models/final/.
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
from src.data_loader import load_source_file, load_ground_truth
from src.dataset_builder import extract_record_dict_from_df
from src.features import compute_pairwise_features
from src.model import LightGBMERModel, XGBoostERModel, get_model
from src.blocking_v2 import add_v2_blocking_columns, block_s1_against_target_file_chunked


def train_final_model(model_choice: str = "auto", s1_chunk_size: int = 50000, target_chunk_size: int = 250000):
    config = get_config()
    print("=" * 80)
    print("PHASE: FINAL PRODUCTION MODEL RETRAINING (100% TRAINING DATA)")
    print("=" * 80)

    # 1. Determine Model Type
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

    # 2. Load Ground Truth
    print("\nLoading complete Ground Truth dataset...", flush=True)
    gt_df = load_ground_truth(config.train_gt_path)
    gt_map: Dict[str, Set[str]] = {}
    gt_pairs_all: Set[Tuple[str, str]] = set()

    for row in gt_df.iter_rows():
        s1_id = str(row[0])
        m_str = str(row[1]) if row[1] is not None else ""
        matches = [x.strip() for x in m_str.split(",") if x.strip()]
        gt_map[s1_id] = set(matches)
        for m in matches:
            gt_pairs_all.add((s1_id, m))

    print(f"Total Ground-Truth Positive Pairs: {len(gt_pairs_all):,} across {len(gt_map):,} S1 entities.")

    # 3. Load S1
    print("\nLoading and indexing 100% Train Source 1...", flush=True)
    t0 = time.time()
    s1_df = load_source_file(config.train_s1_path, expected_prefix="S1-")
    total_s1 = len(s1_df)
    s1_p = add_v2_blocking_columns(s1_df)
    del s1_df
    gc.collect()
    print(f"Loaded {total_s1:,} Train S1 entities in {time.time() - t0:.2f}s", flush=True)

    # 4. Process S1 in memory-safe chunks and accumulate training feature vectors
    num_chunks = (total_s1 + s1_chunk_size - 1) // s1_chunk_size
    print(f"\nGenerating V2 Candidates and extracting features across {num_chunks} chunks...", flush=True)

    X_train_chunks: List[np.ndarray] = []
    y_train_chunks: List[np.ndarray] = []
    total_train_pairs = 0
    total_positive_pairs = 0

    for c_idx in range(num_chunks):
        c_start = c_idx * s1_chunk_size
        c_len = min(s1_chunk_size, total_s1 - c_start)
        t_chk = time.time()

        s1_chk = s1_p.slice(c_start, c_len)
        s1_records = extract_record_dict_from_df(s1_chk)
        target_records: Dict[str, Dict[str, Any]] = {}

        # Stream block against S2
        cands_s2, recs_s2 = block_s1_against_target_file_chunked(
            s1_chk,
            config.train_s2_path,
            target_chunk_size=target_chunk_size,
            max_cands_per_s1=30,
            extract_matched_records=True
        )
        target_records.update(recs_s2)
        del recs_s2

        # Stream block against S3
        cands_s3, recs_s3 = block_s1_against_target_file_chunked(
            s1_chk,
            config.train_s3_path,
            target_chunk_size=target_chunk_size,
            max_cands_per_s1=30,
            extract_matched_records=True
        )
        target_records.update(recs_s3)
        del recs_s3

        cand_df = pl.concat([cands_s2, cands_s3]).unique()

        # Compute pairwise features
        chk_feats = []
        chk_labels = []

        cand_rows = cand_df.to_dict(as_series=False)
        s1_col = cand_rows["s1_id"]
        tgt_col = cand_rows["target_id"]

        for s1_id, tgt_id in zip(s1_col, tgt_col):
            if s1_id in s1_records and tgt_id in target_records:
                f = compute_pairwise_features(s1_records[s1_id], target_records[tgt_id], tgt_id)
                label = 1.0 if tgt_id in gt_map.get(s1_id, set()) else 0.0
                chk_feats.append(f)
                chk_labels.append(label)

        if chk_feats:
            X_chk = np.array(chk_feats, dtype=np.float32)
            y_chk = np.array(chk_labels, dtype=np.float32)
            X_train_chunks.append(X_chk)
            y_train_chunks.append(y_chk)
            total_train_pairs += len(X_chk)
            total_positive_pairs += int(y_chk.sum())

        del s1_chk, s1_records, cands_s2, cands_s3, cand_df, target_records, chk_feats, chk_labels
        gc.collect()

        print(
            f"  [Chunk {c_idx + 1:02d}/{num_chunks:02d}] Processed S1: {c_start + c_len:,}/{total_s1:,} | "
            f"Pairs: {total_train_pairs:,} (Pos: {total_positive_pairs:,}) | Time: {time.time() - t_chk:.2f}s",
            flush=True
        )

    del s1_p
    gc.collect()

    print("\nConcatenating full dataset feature matrices...", flush=True)
    X_train_all = np.vstack(X_train_chunks)
    y_train_all = np.concatenate(y_train_chunks)
    del X_train_chunks, y_train_chunks
    gc.collect()

    print(f"Final Full Training Matrix: {X_train_all.shape} (Positives: {int(y_train_all.sum()):,}, Negatives: {len(y_train_all) - int(y_train_all.sum()):,})")

    # 5. Train Selected Model on 100% Data
    print(f"\nRetraining {selected_model_type.upper()} on 100% of Training Data...", flush=True)
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
        "total_s1_entities": total_s1,
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
    parser.add_argument("--target-chunk-size", type=int, default=250000, help="Target chunk size")
    args = parser.parse_args()

    train_final_model(model_choice=args.model, s1_chunk_size=args.chunk_size, target_chunk_size=args.target_chunk_size)
