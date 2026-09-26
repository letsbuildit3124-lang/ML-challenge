"""
Antigravity V5 Full Test Set Inference & Submission Deliverables Generator.
Generates official submission deliverables:
- output/matching_results.tsv
- output/candidate_pairs.tsv

Features:
- Streaming direct-to-disk chunked inference (< 16 GB peak RAM footprint on 32 GB instance)
- V5 Multi-Pass Candidate Generation against 10.32M target universe in persistent DuckDB
- Macro F0.5 optimized decision thresholding and singleton protection
- Validates output format against competition requirements
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

def generate_v5_submission(
    model_choice: str = "auto",
    user_threshold: float = None,
    s1_chunk_size: int = 50000,
    max_cands_per_s1: int = 50,
    workers: int = 8
):
    config = get_config()
    print("=" * 80)
    print("ANTIGRAVITY V5 — FULL TEST SET INFERENCE & SUBMISSION GENERATOR")
    print("=" * 80)
    print(f"Chunk Size:           {s1_chunk_size:,} | Max Candidates: {max_cands_per_s1} | Workers: {workers}")
    print(f"Initial Process RSS:  {get_current_rss_mb():.2f} MB")
    print("-" * 80)

    # 1. Locate Model and Threshold
    final_meta_path = os.path.join(config.models_dir, "final", "final_model_metadata.json")
    val_meta_path = os.path.join(config.models_dir, "model_metadata.json")

    selected_model_type = model_choice.lower().strip()
    selected_threshold = user_threshold
    model_path = None

    if os.path.exists(final_meta_path):
        with open(final_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        selected_model_type = meta.get("model_type", "xgboost")
        if selected_threshold is None:
            selected_threshold = meta.get("selected_threshold", 0.65)
        model_path = meta.get("model_path")
    elif os.path.exists(val_meta_path):
        with open(val_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        selected_model_type = meta.get("selected_winner", "xgboost")
        if selected_threshold is None:
            selected_threshold = meta.get("winner_threshold", 0.65)

    if selected_threshold is None:
        selected_threshold = 0.65

    ext = ".json" if selected_model_type == "xgboost" else ".txt"
    if model_path is None or not os.path.exists(model_path):
        candidates = [
            os.path.join(config.models_dir, "final", "final_model.json"),
            os.path.join(config.models_dir, "final", "final_model.txt"),
            os.path.join(config.models_dir, selected_model_type, "model.json"),
            os.path.join(config.models_dir, selected_model_type, "model.txt"),
        ]
        for p in candidates:
            if os.path.exists(p):
                model_path = p
                break

    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not locate trained model file! Run train_v5_production first.")

    m_type = "lightgbm" if model_path.endswith(".txt") else "xgboost"
    print(f"Loaded Model:         {model_path} ({m_type.upper()})")
    print(f"Decision Threshold:   {selected_threshold:.2f}")

    model = get_model(m_type, config)
    model.load(model_path)

    # 2. Prepare Output Deliverable Files
    os.makedirs(config.output_dir, exist_ok=True)
    matching_path = getattr(config, "output_matching_path", getattr(config, "matching_results_path", os.path.join(config.output_dir, "matching_results.tsv")))
    candidate_path = getattr(config, "output_candidates_path", getattr(config, "candidate_pairs_path", os.path.join(config.output_dir, "candidate_pairs.tsv")))

    with open(matching_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")

    with open(candidate_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")

    print(f"Streaming Output:     {matching_path}")
    print(f"                      {candidate_path}")

    # 3. Initialize V5 Retrieval Engine against Persistent DuckDB
    print("\n[Stage 1] Initializing V5 Engine against 10.32M Target Cache...", flush=True)
    engine = V5RetrievalEngine(memory_limit="8GB", threads=8, workers=workers)
    expected_sources = [config.train_s2_path, config.train_s3_path] if (os.path.exists(config.train_s2_path) and os.path.exists(config.train_s3_path)) else None
    engine.indexer.ensure_cache_ready(expected_sources)

    # 4. Stream Test S1 in Chunks & Generate Outputs
    print("\n[Stage 2] Streaming Test S1 records and predicting matches...", flush=True)
    t_start = time.time()
    total_s1 = 0
    total_matches = 0
    total_candidates_saved = 0

    for chunk_df in iter_source_file_chunks(config.test_s1_path, chunk_size=s1_chunk_size, expected_prefix="S1-"):
        n_chunk = len(chunk_df)
        s1_chunk_p = add_v2_blocking_columns(chunk_df)
        s1_records = extract_record_dict_from_df(s1_chunk_p)

        cands_raw = engine.generate_candidates(
            s1_chunk_p,
            enable_deterministic=True,
            enable_ngram=True,
            enable_token=True,
            enable_address=True,
            enable_fts=False,
            enable_fuzzy_rerank=True,
            max_candidates_per_s1=max_cands_per_s1
        )

        match_lines = []
        cand_lines = []

        # Prepare feature matrix for chunk
        pair_metadata = []
        feat_list = []

        for s1_id, cand_list in cands_raw.items():
            s1_rec = s1_records.get(s1_id, {})
            cand_tids = [tid for tid, mask, score, rec in cand_list]
            cand_lines.append(f"{s1_id}\t{','.join(cand_tids)}\n")
            total_candidates_saved += len(cand_tids)

            for tid, mask, score, t_rec in cand_list:
                feat = compute_tiered_pairwise_features(s1_rec, t_rec, tid, provenance_mask=mask)
                feat_list.append(feat)
                pair_metadata.append((s1_id, tid))

        # Predict probabilities
        matched_per_s1 = defaultdict(list)
        if feat_list:
            X_chunk = np.array(feat_list, dtype=np.float32)
            probs = model.predict_proba(X_chunk)

            for (s1_id, tid), prob in zip(pair_metadata, probs):
                if prob >= selected_threshold:
                    matched_per_s1[s1_id].append((tid, float(prob)))

        # Format matches with cardinality & competitive filtering
        for row in chunk_df.iter_rows(named=True):
            s1_id = str(row.get("entity_id") or row.get("eid") or list(row.values())[0])
            m_list = matched_per_s1.get(s1_id, [])

            if m_list:
                # Sort by probability descending
                m_list.sort(key=lambda x: x[1], reverse=True)
                top_prob = m_list[0][1]
                # Keep candidates within 0.15 margin of top candidate
                selected_tids = [tid for tid, p in m_list if p >= max(selected_threshold, top_prob - 0.15)]
                match_lines.append(f"{s1_id}\t{','.join(selected_tids)}\n")
                total_matches += len(selected_tids)
            else:
                match_lines.append(f"{s1_id}\t\n")

        # Stream directly to disk
        with open(matching_path, "a", encoding="utf-8") as f:
            f.writelines(match_lines)

        with open(candidate_path, "a", encoding="utf-8") as f:
            f.writelines(cand_lines)

        total_s1 += n_chunk
        elapsed = time.time() - t_start
        speed = total_s1 / elapsed if elapsed > 0 else 0
        print(f"  -> Processed {total_s1:,} Test S1 entities | Matches: {total_matches:,} | Speed: {speed:.1f} S1/sec | RSS: {get_current_rss_mb():.1f} MB", flush=True)

        del chunk_df, s1_records, cands_raw, feat_list, pair_metadata, match_lines, cand_lines
        gc.collect()

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print("V5 TEST INFERENCE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Total Test S1 Processed: {total_s1:,}")
    print(f"Total Matches Predicted: {total_matches:,} (Avg: {total_matches/total_s1:.3f} matches/S1)")
    print(f"Total Candidates Saved:  {total_candidates_saved:,} (Avg: {total_candidates_saved/total_s1:.1f} cands/S1)")
    print(f"Execution Time:          {total_time:.2f}s ({total_s1/total_time:.1f} S1/sec)")
    print(f"Matching Results File:   {matching_path} ({os.path.getsize(matching_path)/(1024*1024):.2f} MB)")
    print(f"Candidate Pairs File:    {candidate_path} ({os.path.getsize(candidate_path)/(1024*1024):.2f} MB)")
    print(f"Peak Process RSS:        {get_peak_rss_mb():.2f} MB")
    print("=" * 80)

    # Validate output deliverables
    print("\nValidating Deliverable TSV format...")
    from src.submission import validate_submission
    is_valid = validate_submission(matching_path, candidate_path)
    if is_valid:
        print("[SUCCESS] Submission files verified compliant with competition schema!")
    else:
        print("[WARNING] Submission validation reported warnings. Please inspect output files.")

    engine.close()


def main():
    parser = argparse.ArgumentParser(description="Antigravity V5 Test Set Inference & Submission Generator")
    parser.add_argument("--model", type=str, default="auto", choices=["auto", "xgboost", "lightgbm"], help="Model architecture")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold (default: auto from metadata)")
    parser.add_argument("--chunk-size", type=int, default=50000, help="S1 chunk size (default: 50,000)")
    parser.add_argument("--max-candidates", type=int, default=50, help="Max candidates per S1 entity (default: 50)")
    parser.add_argument("--workers", type=int, default=8, help="Number of CPU workers (default: 8)")
    args = parser.parse_args()

    generate_v5_submission(
        model_choice=args.model,
        user_threshold=args.threshold,
        s1_chunk_size=args.chunk_size,
        max_cands_per_s1=args.max_candidates,
        workers=args.workers
    )

if __name__ == "__main__":
    main()
