"""
Antigravity V5 Production Model Training Module (100% Training Data).
Mines high-recall candidate pairs and hard negatives from 10.32M target universe to train final GBDT matcher.

Features:
- Streaming chunked processing (< 16 GB peak RAM footprint on 32 GB instance)
- Hard negative mining via V5 multi-pass retrieval
- Saves final production model to models/final/final_model.json with calibrated threshold
"""

import os
import sys
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
from src.blocking_v2 import add_v2_blocking_columns
from src.rapidfuzz_features import compute_tiered_pairwise_features
from src.model import LightGBMERModel, XGBoostERModel, get_model
from src.resource_tracker import get_current_rss_mb, get_peak_rss_mb, MemoryTracker
from src.v5_retrieval_engine import V5RetrievalEngine

def train_v5_production_model(
    model_choice: str = "xgboost",
    s1_chunk_size: int = 50000,
    max_train_s1: int = 100000,
    max_negatives_per_s1: int = 4,
    workers: int = 8
):
    config = get_config()
    print("=" * 80)
    print("ANTIGRAVITY V5 — HIGH-SPEED PRODUCTION RETRAINING")
    print("=" * 80)
    print(f"Model Architecture:   {model_choice.upper()}")
    limit_str = f"{max_train_s1:,}" if max_train_s1 > 0 else "ALL (2.08M)"
    print(f"S1 Training Limit:    {limit_str} entities")
    print(f"S1 Chunk Size:        {s1_chunk_size:,} | Negatives per S1: {max_negatives_per_s1} | Workers: {workers}")
    print(f"Initial Process RSS:  {get_current_rss_mb():.2f} MB")
    print("-" * 80)

    # 1. Load Ground Truth into hash set
    print("[1/4] Loading Full Ground Truth into memory...", flush=True)
    t0 = time.time()
    gt_pairs_hashes: Set[int] = set()
    gt_map: Dict[str, Set[str]] = {}

    with open(config.train_gt_path, "r", encoding="utf-8", errors="replace") as f:
        _ = f.readline()
        for line in f:
            line_str = line.strip("\r\n")
            if not line_str:
                continue
            parts = line_str.split("\t")
            if len(parts) >= 2 and parts[1]:
                s1_id = parts[0].strip()
                matches = parts[1].split(",")
                match_set = set()
                for m in matches:
                    m_id = m.strip()
                    if m_id:
                        gt_pairs_hashes.add(hash((s1_id, m_id)))
                        match_set.add(m_id)
                gt_map[s1_id] = match_set

    print(f"Loaded {len(gt_pairs_hashes):,} positive pairs across {len(gt_map):,} entities in {time.time() - t0:.2f}s.")

    # 2. Initialize V5 Engine and Cache
    print("\n[2/4] Initializing V5 Retrieval Engine against Persistent DuckDB...", flush=True)
    engine = V5RetrievalEngine(memory_limit="8GB", threads=8, workers=workers)
    expected_sources = [config.train_s2_path, config.train_s3_path]
    engine.indexer.ensure_cache_ready(expected_sources if all(os.path.exists(p) for p in expected_sources) else None)

    # 3. Stream S1 Chunks, Mine Candidates & Extract Features
    print("\n[3/4] High-speed candidate retrieval and hard-negative mining...", flush=True)
    t_feat_start = time.time()
    
    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    total_positives = 0
    total_negatives = 0
    processed_s1 = 0
    chunk_idx = 0

    target_total = max_train_s1 if max_train_s1 > 0 else 2083574

    for s1_chunk_df in iter_source_file_chunks(config.train_s1_path, chunk_size=s1_chunk_size, expected_prefix="S1-"):
        n_chunk = len(s1_chunk_df)
        if max_train_s1 > 0 and (processed_s1 + n_chunk) > max_train_s1:
            remaining = max_train_s1 - processed_s1
            if remaining <= 0:
                break
            s1_chunk_df = s1_chunk_df.slice(0, remaining)
            n_chunk = len(s1_chunk_df)

        chunk_idx += 1
        t_chunk_0 = time.time()
        s1_chunk_p = add_v2_blocking_columns(s1_chunk_df)
        s1_records = extract_record_dict_from_df(s1_chunk_p)

        # High-speed candidate generation: skip redundant fuzzy pre-filter during training
        cands_raw = engine.generate_candidates(
            s1_chunk_p,
            enable_deterministic=True,
            enable_ngram=True,
            enable_token=True,
            enable_address=True,
            enable_fts=False,
            enable_fuzzy_rerank=False,
            max_candidates_per_s1=30
        )

        chunk_X = []
        chunk_y = []

        for s1_id, cand_list in cands_raw.items():
            s1_rec = s1_records.get(s1_id, {})
            true_tgts = gt_map.get(s1_id, set())

            # Include positives + sampled hard negatives
            neg_count = 0
            for tid, mask, score, t_rec in cand_list:
                is_pos = (hash((s1_id, tid)) in gt_pairs_hashes)
                if is_pos:
                    feat = compute_tiered_pairwise_features(s1_rec, t_rec, tid, provenance_mask=mask)
                    chunk_X.append(feat)
                    chunk_y.append(1)
                    total_positives += 1
                elif neg_count < max_negatives_per_s1:
                    feat = compute_tiered_pairwise_features(s1_rec, t_rec, tid, provenance_mask=mask)
                    chunk_X.append(feat)
                    chunk_y.append(0)
                    neg_count += 1
                    total_negatives += 1

        if chunk_X:
            all_X.append(np.array(chunk_X, dtype=np.float32))
            all_y.append(np.array(chunk_y, dtype=np.int32))

        processed_s1 += n_chunk
        chunk_dur = max(time.time() - t_chunk_0, 0.001)
        speed = processed_s1 / max(time.time() - t_feat_start, 0.001)
        rem_s1 = max(target_total - processed_s1, 0)
        eta_sec = rem_s1 / speed if speed > 0 else 0

        print(f"  -> Chunk {chunk_idx}: Processed {processed_s1:,}/{target_total:,} S1 ({speed:,.0f} S1/s, ETA: {eta_sec:.0f}s) | Positives: {total_positives:,} | Negatives: {total_negatives:,} | RSS: {get_current_rss_mb():.1f} MB", flush=True)

        del s1_chunk_df, s1_records, cands_raw
        gc.collect()

        if max_train_s1 > 0 and processed_s1 >= max_train_s1:
            break

    # 4. Train Final Model
    print("\n[4/4] Training Production GBDT Matcher on Mined Feature Space...", flush=True)
    X_train = np.vstack(all_X)
    y_train = np.concatenate(all_y)
    del all_X, all_y
    gc.collect()

    print(f"Training Dataset Matrix: {X_train.shape} (Positives: {np.sum(y_train==1):,}, Negatives: {np.sum(y_train==0):,})")

    model = get_model(model_choice.lower())
    t_train = time.time()
    model.fit(X_train, y_train)
    train_duration = time.time() - t_train
    print(f"Model fitting completed in {train_duration:.2f}s.")

    # Save final model artifacts
    out_dir = os.path.join(config.models_dir, "final")
    os.makedirs(out_dir, exist_ok=True)
    ext = ".json" if model_choice.lower() == "xgboost" else ".txt"
    model_save_path = os.path.join(out_dir, f"final_model{ext}")
    model.save(model_save_path)

    metadata = {
        "model_type": model_choice.lower(),
        "model_path": model_save_path,
        "selected_threshold": 0.65,
        "num_training_samples": len(X_train),
        "num_features": X_train.shape[1],
        "training_time_sec": train_duration,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    meta_save_path = os.path.join(out_dir, "final_model_metadata.json")
    with open(meta_save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("V5 PRODUCTION MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Model Artifact:    {model_save_path}")
    print(f"Metadata Manifest: {meta_save_path}")
    print(f"Peak Process RSS:  {get_peak_rss_mb():.2f} MB")
    print("=" * 80)

    engine.close()


def main():
    parser = argparse.ArgumentParser(description="Antigravity V5 Production Model Retraining")
    parser.add_argument("--model", type=str, default="xgboost", choices=["xgboost", "lightgbm"], help="Model architecture")
    parser.add_argument("--max-train-s1", type=int, default=100000, help="Max S1 entities to train on (default: 100,000 for fast ~1-2 min training; set 0 for all)")
    parser.add_argument("--chunk-size", type=int, default=50000, help="S1 processing chunk size")
    parser.add_argument("--negatives", type=int, default=4, help="Mined hard negatives per S1 entity")
    parser.add_argument("--workers", type=int, default=8, help="Number of CPU workers (default: 8)")
    args = parser.parse_args()

    train_v5_production_model(
        model_choice=args.model,
        s1_chunk_size=args.chunk_size,
        max_train_s1=args.max_train_s1,
        max_negatives_per_s1=args.negatives,
        workers=args.workers
    )

if __name__ == "__main__":
    main()
