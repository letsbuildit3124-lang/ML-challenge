"""
Final Test Inference and Submission Generation Module.
Generates official competition deliverables:
  - output/matching_results.tsv
  - output/candidate_pairs.tsv
Uses streaming chunked target blocking for strictly controlled sub-1GB RAM usage.
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
from src.data_loader import load_source_file
from src.dataset_builder import extract_record_dict_from_df
from src.features import compute_pairwise_features
from src.model import LightGBMERModel, XGBoostERModel, get_model, BaseERModel
from src.blocking_v2 import add_v2_blocking_columns, block_s1_against_target_file_chunked
from src.submission import write_submission_files


def run_sanity_checks(
    matching_path: str,
    candidate_path: str,
    all_s1_ids: List[str],
    valid_target_ids: Set[str]
) -> bool:
    """Performs 13 internal sanity checks on the generated TSV deliverables."""
    print("\n" + "=" * 80)
    print("RUNNING 13 INTERNAL SUBMISSION SANITY CHECKS")
    print("=" * 80)

    errors = []
    total_expected_s1 = len(all_s1_ids)
    expected_s1_set = set(all_s1_ids)

    # 1 & 2: Load TSVs
    matching_df = pl.read_csv(matching_path, separator="\t", null_values=["", "NULL", "null", "None", "NaN"], truncate_ragged_lines=True).with_columns(pl.col("matched_entity_ids").fill_null(""))
    candidate_df = pl.read_csv(candidate_path, separator="\t", null_values=["", "NULL", "null", "None", "NaN"], truncate_ragged_lines=True).with_columns(pl.col("candidate_entity_ids").fill_null(""))

    # 3 & 4: Exact column names
    if matching_df.columns != ["source1_entity_id", "matched_entity_ids"]:
        errors.append(f"matching_results.tsv has invalid headers: {matching_df.columns}")
    if candidate_df.columns != ["source1_entity_id", "candidate_entity_ids"]:
        errors.append(f"candidate_pairs.tsv has invalid headers: {candidate_df.columns}")

    # 5 & 6: Row counts
    if len(matching_df) != total_expected_s1:
        errors.append(f"matching_results.tsv row count ({len(matching_df):,}) != expected ({total_expected_s1:,})")
    if len(candidate_df) != total_expected_s1:
        errors.append(f"candidate_pairs.tsv row count ({len(candidate_df):,}) != expected ({total_expected_s1:,})")

    # 7 & 8: S1 set match & no duplicates
    m_s1_list = matching_df["source1_entity_id"].to_list()
    c_s1_list = candidate_df["source1_entity_id"].to_list()

    if len(m_s1_list) != len(set(m_s1_list)):
        errors.append("matching_results.tsv contains duplicate Source 1 rows.")
    if len(c_s1_list) != len(set(c_s1_list)):
        errors.append("candidate_pairs.tsv contains duplicate Source 1 rows.")

    if set(m_s1_list) != expected_s1_set:
        errors.append("matching_results.tsv S1 ID set does not match test_source1.tsv.")
    if set(c_s1_list) != expected_s1_set:
        errors.append("candidate_pairs.tsv S1 ID set does not match test_source1.tsv.")

    # 9-13: ID validity & subset checks
    cand_map = {}
    for row in candidate_df.iter_rows():
        s1 = str(row[0])
        c_str = str(row[1]) if row[1] else ""
        cand_map[s1] = [x.strip() for x in c_str.split(",") if x.strip()]

    match_map = {}
    for row in matching_df.iter_rows():
        s1 = str(row[0])
        m_str = str(row[1]) if row[1] else ""
        match_map[s1] = [x.strip() for x in m_str.split(",") if x.strip()]

    invalid_targets = 0
    s1_in_matches = 0
    candidate_violations = 0
    dup_matches = 0
    dup_cands = 0

    for s1, m_list in match_map.items():
        c_list = cand_map.get(s1, [])
        c_set = set(c_list)

        if len(m_list) != len(set(m_list)):
            dup_matches += 1
        if len(c_list) != len(set(c_list)):
            dup_cands += 1

        for m in m_list:
            if m.startswith("S1-"):
                s1_in_matches += 1
            if m not in valid_target_ids:
                invalid_targets += 1
            if m not in c_set:
                candidate_violations += 1

        for c in c_list:
            if c not in valid_target_ids:
                invalid_targets += 1

    if s1_in_matches > 0:
        errors.append(f"Found {s1_in_matches} matches containing S1- IDs.")
    if invalid_targets > 0:
        errors.append(f"Found {invalid_targets} IDs not present in test S2 or S3.")
    if candidate_violations > 0:
        errors.append(f"CRITICAL: Found {candidate_violations} matches not present in candidate_pairs.tsv!")
    if dup_matches > 0:
        errors.append(f"Found {dup_matches} entities with duplicate matched IDs.")
    if dup_cands > 0:
        errors.append(f"Found {dup_cands} entities with duplicate candidate IDs.")

    if errors:
        for err in errors:
            print(f"  ❌ {err}")
        return False
    else:
        print("  ✅ All 13 internal sanity checks passed perfectly!")
        return True


def generate_test_output(
    model_choice: str = "auto",
    user_threshold: float = None,
    s1_chunk_size: int = 50000,
    target_chunk_size: int = 250000
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

    # 2. Get Valid Target IDs for Sanity Checks
    print("\nReading test target IDs...", flush=True)
    valid_target_ids = set()
    if os.path.exists(config.test_s2_path):
        s2_ids = pl.read_csv(config.test_s2_path, separator="\t", columns=["entity_id"], truncate_ragged_lines=True)["entity_id"].to_list()
        valid_target_ids.update(s2_ids)
    if os.path.exists(config.test_s3_path):
        s3_ids = pl.read_csv(config.test_s3_path, separator="\t", columns=["entity_id"], truncate_ragged_lines=True)["entity_id"].to_list()
        valid_target_ids.update(s3_ids)

    # 3. Load Test Source 1
    t0 = time.time()
    print("\nLoading and indexing Test Source 1...", flush=True)
    test_s1_df = load_source_file(config.test_s1_path, expected_prefix="S1-")
    all_test_s1_ids = test_s1_df["entity_id"].to_list()
    total_test_s1 = len(all_test_s1_ids)
    test_s1_p = add_v2_blocking_columns(test_s1_df)
    del test_s1_df
    gc.collect()
    print(f"Loaded {total_test_s1:,} Test S1 entities in {time.time() - t0:.2f}s", flush=True)

    # Global tracking maps
    matching_map: Dict[str, List[str]] = {s1_id: [] for s1_id in all_test_s1_ids}
    candidate_map: Dict[str, List[str]] = {s1_id: [] for s1_id in all_test_s1_ids}

    # 4. Process Test S1 in memory-safe chunks
    num_chunks = (total_test_s1 + s1_chunk_size - 1) // s1_chunk_size
    print(f"\nProcessing {total_test_s1:,} Test S1 entities across {num_chunks} chunks...", flush=True)

    total_candidates_found = 0
    total_matches_found = 0
    t_inf_start = time.time()

    for c_idx in range(num_chunks):
        c_start = c_idx * s1_chunk_size
        c_len = min(s1_chunk_size, total_test_s1 - c_start)
        t_chk = time.time()

        s1_chk = test_s1_p.slice(c_start, c_len)
        s1_records = extract_record_dict_from_df(s1_chk)
        target_records: Dict[str, Dict[str, Any]] = {}

        # Stream block against Test S2
        cands_s2, recs_s2 = block_s1_against_target_file_chunked(
            s1_chk,
            config.test_s2_path,
            target_chunk_size=target_chunk_size,
            max_cands_per_s1=40,
            extract_matched_records=True
        )
        target_records.update(recs_s2)
        del recs_s2

        # Stream block against Test S3
        cands_s3, recs_s3 = block_s1_against_target_file_chunked(
            s1_chk,
            config.test_s3_path,
            target_chunk_size=target_chunk_size,
            max_cands_per_s1=40,
            extract_matched_records=True
        )
        target_records.update(recs_s3)
        del recs_s3

        cand_df = pl.concat([cands_s2, cands_s3]).unique()
        chunk_cand_count = len(cand_df)
        total_candidates_found += chunk_cand_count

        if chunk_cand_count > 0:
            cand_rows = cand_df.to_dict(as_series=False)
            s1_col = cand_rows["s1_id"]
            tgt_col = cand_rows["target_id"]

            chk_feats = []
            chk_pairs = []

            for s1_id, tgt_id in zip(s1_col, tgt_col):
                candidate_map[s1_id].append(tgt_id)
                if s1_id in s1_records and tgt_id in target_records:
                    f = compute_pairwise_features(s1_records[s1_id], target_records[tgt_id], tgt_id)
                    chk_feats.append(f)
                    chk_pairs.append((s1_id, tgt_id))

            # Score candidates
            if chk_feats:
                X_chk = np.array(chk_feats, dtype=np.float32)
                probs = model.predict_proba(X_chk)
                chunk_match_count = 0
                for (s1_id, tgt_id), prob in zip(chk_pairs, probs):
                    if prob >= selected_threshold:
                        matching_map[s1_id].append(tgt_id)
                        chunk_match_count += 1
                total_matches_found += chunk_match_count

            del target_records, chk_feats, chk_pairs

        del s1_chk, s1_records, cands_s2, cands_s3, cand_df
        gc.collect()

        print(
            f"  [Chunk {c_idx + 1:02d}/{num_chunks:02d}] S1: {c_start + c_len:,}/{total_test_s1:,} | "
            f"Candidates: {chunk_cand_count:,} | Chunk Matches: {total_matches_found:,} total | "
            f"Time: {time.time() - t_chk:.2f}s",
            flush=True
        )

    # Free memory
    del test_s1_p
    gc.collect()

    # 5. Write Deliverables to output/
    print("\nWriting official competition TSV files...", flush=True)
    match_path, cand_path = write_submission_files(config, all_test_s1_ids, matching_map, candidate_map)

    # 6. Execute 13 Internal Sanity Checks
    sanity_passed = run_sanity_checks(match_path, cand_path, all_test_s1_ids, valid_target_ids)

    # 7. Execute Official Submission Validator
    print("\n" + "=" * 80)
    print("RUNNING OFFICIAL SUBMISSION VALIDATOR (utils/validate_submission.py)")
    print("=" * 80)
    cmd = [
        sys.executable,
        config.validator_path,
        "--matching", match_path,
        "--candidate", cand_path,
        "--test-dir", config.test_dir
    ]
    val_proc = subprocess.run(cmd, capture_output=True, text=True)
    print(val_proc.stdout)
    if val_proc.stderr:
        print(val_proc.stderr)

    if val_proc.returncode != 0 or not sanity_passed:
        print("\n❌ SUBMISSION VALIDATION FAILED! Check error messages above.")
        sys.exit(1)

    # Summary Statistics
    total_assigned_matches = sum(len(v) for v in matching_map.values())
    total_singletons = sum(1 for v in matching_map.values() if not v)
    total_cands = sum(len(v) for v in candidate_map.values())

    print("=" * 80)
    print("FINAL SUBMISSION GENERATION SUMMARY")
    print("=" * 80)
    print(f"  Selected Model:             {selected_model_type.upper()}")
    print(f"  Applied Decision Threshold: {selected_threshold:.2f}")
    print(f"  Total Test S1 Entities:     {total_test_s1:,}")
    print(f"  Total Candidates Pool:      {total_cands:,} (Avg {total_cands/total_test_s1:.2f} cands/S1)")
    print(f"  Total Predicted Matches:    {total_assigned_matches:,} (Avg {total_assigned_matches/total_test_s1:.2f} matches/S1)")
    print(f"  Total Singletons:           {total_singletons:,} ({total_singletons/total_test_s1*100:.2f}%)")
    print(f"  Inference Runtime:          {time.time() - t_inf_start:.2f}s")
    print(f"  Deliverables Created:")
    print(f"    - {match_path}")
    print(f"    - {cand_path}")
    print(f"  Official Validation Result: PASS ✅")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate official test submission deliverables")
    parser.add_argument("--model", type=str, default="auto", choices=["lightgbm", "xgboost", "auto"], help="Model architecture")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold (default: auto from validation)")
    parser.add_argument("--chunk-size", type=int, default=50000, help="S1 chunk size for test inference")
    parser.add_argument("--target-chunk-size", type=int, default=250000, help="Target chunk size")
    args = parser.parse_args()

    generate_test_output(
        model_choice=args.model,
        user_threshold=args.threshold,
        s1_chunk_size=args.chunk_size,
        target_chunk_size=args.target_chunk_size
    )
