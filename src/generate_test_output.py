"""
Final Test Inference and Submission Generation Module.
Generates official competition deliverables:
  - output/matching_results.tsv
  - output/candidate_pairs.tsv
Uses streaming S1 chunk processing and direct-to-disk TSV writing for strictly sub-2GB RAM usage.
Adheres strictly to all official formatting and integrity rules and executes official validation.
"""

import os
import gc
import json
import time
import argparse
import subprocess
import sys
from typing import Dict, List, Tuple, Any, Set
import numpy as np
import polars as pl

from src.config import Config, get_config
from src.data_loader import iter_source_file_chunks
from src.dataset_builder import extract_record_dict_from_df
from src.features import compute_pairwise_features
from src.model import LightGBMERModel, XGBoostERModel, get_model, BaseERModel
from src.blocking_v2 import (
    add_v2_blocking_columns,
    build_compact_target_index,
    generate_candidates_against_indexed_target
)


def generate_test_output(
    model_choice: str = "auto",
    user_threshold: float = None,
    s1_chunk_size: int = 50000
):
    config = get_config()
    print("=" * 80)
    print("PHASE: FINAL TEST INFERENCE & OFFICIAL SUBMISSION GENERATION")
    print("=" * 80)

    # 1. Determine Model & Threshold
    selected_model_type = model_choice.lower().strip()
    selected_threshold = user_threshold

    final_meta_path = os.path.join(config.models_dir, "final", "final_model_metadata.json")
    val_meta_path = os.path.join(config.models_dir, "model_metadata.json")

    model_path = None

    if selected_model_type == "auto":
        if os.path.exists(final_meta_path):
            with open(final_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            selected_model_type = meta.get("model_type", "lightgbm")
            if selected_threshold is None:
                selected_threshold = meta.get("selected_threshold", 0.50)
            model_path = meta.get("model_path")
        elif os.path.exists(val_meta_path):
            with open(val_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            selected_model_type = meta.get("selected_winner", "lightgbm")
            if selected_threshold is None:
                selected_threshold = meta.get("winner_threshold", 0.50)
        else:
            selected_model_type = "lightgbm"
            if selected_threshold is None:
                selected_threshold = 0.50

    if selected_threshold is None:
        if os.path.exists(val_meta_path):
            with open(val_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            selected_threshold = meta.get(selected_model_type, {}).get("best_threshold", 0.50)
        else:
            selected_threshold = 0.50

    # Locate Model File
    ext = ".txt" if selected_model_type == "lightgbm" else ".json"
    if model_path is None or not os.path.exists(model_path):
        final_candidate = os.path.join(config.models_dir, "final", f"final_model{ext}")
        stage1_candidate = os.path.join(config.models_dir, selected_model_type, f"model{ext}")
        if os.path.exists(final_candidate):
            model_path = final_candidate
        elif os.path.exists(stage1_candidate):
            model_path = stage1_candidate
        elif os.path.exists(config.model_save_path):
            model_path = config.model_save_path
        else:
            raise FileNotFoundError(f"Could not locate trained model file for {selected_model_type}. Please run compare_models or train_final first.")

    print(f"Selected Model Type:      {selected_model_type.upper()}")
    print(f"Selected Model Path:      {model_path}")
    print(f"Selected Threshold:       {selected_threshold:.2f}")

    # Load Model
    model = get_model(selected_model_type, config)
    model.load(model_path)

    # 2. Pre-Index Test Source 2 & Test Source 3 once in memory
    print("\n[1/3] Pre-indexing Test Source 2 & Test Source 3 into compact in-memory table...", flush=True)
    t0 = time.time()
    indexed_target = build_compact_target_index(
        config.test_s2_path,
        config.test_s3_path,
        chunk_size=250000
    )
    print(f"Indexed {len(indexed_target):,} Test Targets in {time.time() - t0:.2f}s (RAM: ~1.5GB)", flush=True)

    # 3. Prepare Deliverable Output TSVs
    os.makedirs(config.output_dir, exist_ok=True)
    match_out_path = config.matching_results_path
    cand_out_path = config.candidate_pairs_path

    # Initialize TSV files with exact required official headers
    with open(match_out_path, "w", encoding="utf-8") as f_m:
        f_m.write("source1_entity_id\tmatched_entity_ids\n")

    with open(cand_out_path, "w", encoding="utf-8") as f_c:
        f_c.write("source1_entity_id\tcandidate_entity_ids\n")

    # 4. Stream Test Source 1 in Chunks and Write Directly to Disk
    print(f"\n[2/3] Streaming Test Source 1 in chunks of {s1_chunk_size:,} entities...", flush=True)

    total_test_s1 = 0
    total_candidates_found = 0
    total_matches_found = 0
    total_singletons_found = 0
    chunk_idx = 0
    t_inf_start = time.time()

    for s1_raw_df in iter_source_file_chunks(config.test_s1_path, chunk_size=s1_chunk_size):
        chunk_idx += 1
        t_chk = time.time()
        chunk_s1_ids = s1_raw_df["entity_id"].to_list()
        chunk_s1_count = len(chunk_s1_ids)
        total_test_s1 += chunk_s1_count

        s1_chk = add_v2_blocking_columns(s1_raw_df)
        del s1_raw_df
        s1_records = extract_record_dict_from_df(s1_chk)

        # Fast in-memory candidate lookup
        cand_df, target_records = generate_candidates_against_indexed_target(
            s1_chk,
            indexed_target,
            max_cands_per_s1=40
        )

        chunk_cand_count = len(cand_df)
        total_candidates_found += chunk_cand_count

        # Tracking for this chunk only
        chunk_cand_map: Dict[str, List[str]] = {s1: [] for s1 in chunk_s1_ids}
        chunk_match_map: Dict[str, List[str]] = {s1: [] for s1 in chunk_s1_ids}

        if chunk_cand_count > 0:
            cand_rows = cand_df.to_dict(as_series=False)
            s1_col = cand_rows["s1_id"]
            tgt_col = cand_rows["target_id"]

            chk_feats = []
            chk_pairs = []

            for s1_id, tgt_id in zip(s1_col, tgt_col):
                chunk_cand_map[s1_id].append(tgt_id)
                if s1_id in s1_records and tgt_id in target_records:
                    f = compute_pairwise_features(s1_records[s1_id], target_records[tgt_id], tgt_id)
                    chk_feats.append(f)
                    chk_pairs.append((s1_id, tgt_id))

            # Score candidates
            if chk_feats:
                X_chk = np.array(chk_feats, dtype=np.float32)
                probs = model.predict_proba(X_chk)
                for (s1_id, tgt_id), prob in zip(chk_pairs, probs):
                    if prob >= selected_threshold:
                        chunk_match_map[s1_id].append(tgt_id)

        # Append this chunk's predictions directly to output files on disk
        with open(match_out_path, "a", encoding="utf-8") as f_m, open(cand_out_path, "a", encoding="utf-8") as f_c:
            for s1_id in chunk_s1_ids:
                raw_cands = chunk_cand_map.get(s1_id, [])
                dedup_cands = list(dict.fromkeys(raw_cands))
                f_c.write(f"{s1_id}\t{','.join(dedup_cands)}\n")

                cand_set = set(dedup_cands)
                raw_matches = chunk_match_map.get(s1_id, [])
                valid_matches = [m for m in dict.fromkeys(raw_matches) if m in cand_set and not m.startswith("S1-")]
                
                if not valid_matches:
                    total_singletons_found += 1
                else:
                    total_matches_found += len(valid_matches)
                
                f_m.write(f"{s1_id}\t{','.join(valid_matches)}\n")

        del s1_chk, s1_records, cand_df, target_records, chunk_cand_map, chunk_match_map
        gc.collect()

        print(
            f"  [Chunk {chunk_idx:02d}] S1: {total_test_s1:,} | "
            f"Cands: {chunk_cand_count:,} | Cumulative Matches: {total_matches_found:,} | Time: {time.time() - t_chk:.2f}s",
            flush=True
        )

    # Free memory
    del indexed_target
    gc.collect()

    print(f"\n[3/3] Completed test inference for {total_test_s1:,} S1 entities in {time.time() - t_inf_start:.2f}s")
    print(f"Wrote {match_out_path}")
    print(f"Wrote {cand_out_path}")

    # 5. Execute Official Submission Validator
    print("\n" + "=" * 80)
    print("RUNNING OFFICIAL SUBMISSION VALIDATOR (utils/validate_submission.py)")
    print("=" * 80)
    cmd = [
        sys.executable,
        config.validator_path,
        "--matching", match_out_path,
        "--candidate", cand_out_path,
        "--test-dir", config.test_dir
    ]
    val_proc = subprocess.run(cmd, capture_output=True, text=True)
    print(val_proc.stdout)
    if val_proc.stderr:
        print(val_proc.stderr)

    if val_proc.returncode != 0:
        print("\n❌ SUBMISSION VALIDATION FAILED! Check error messages above.")
        sys.exit(1)

    print("=" * 80)
    print("FINAL SUBMISSION GENERATION SUMMARY")
    print("=" * 80)
    print(f"  Selected Model:             {selected_model_type.upper()}")
    print(f"  Applied Decision Threshold: {selected_threshold:.2f}")
    print(f"  Total Test S1 Entities:     {total_test_s1:,}")
    print(f"  Total Candidates Pool:      {total_candidates_found:,} (Avg {total_candidates_found/total_test_s1:.2f} cands/S1)")
    print(f"  Total Predicted Matches:    {total_matches_found:,} (Avg {total_matches_found/total_test_s1:.2f} matches/S1)")
    print(f"  Total Singletons:           {total_singletons_found:,} ({total_singletons_found/total_test_s1*100:.2f}%)")
    print(f"  Inference Runtime:          {time.time() - t_inf_start:.2f}s")
    print(f"  Deliverables Created:")
    print(f"    - {match_out_path}")
    print(f"    - {cand_out_path}")
    print(f"  Official Validation Result: PASS ✅")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate official test submission deliverables")
    parser.add_argument("--model", type=str, default="auto", choices=["lightgbm", "xgboost", "auto"], help="Model architecture")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold (default: auto from validation)")
    parser.add_argument("--chunk-size", type=int, default=50000, help="S1 chunk size for test inference")
    args = parser.parse_args()

    generate_test_output(
        model_choice=args.model,
        user_threshold=args.threshold,
        s1_chunk_size=args.chunk_size
    )
