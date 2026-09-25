"""
Inference module for test dataset entity resolution with sequential memory management.
"""

import gc
import time
from typing import Dict, List, Tuple, Any, Set
import numpy as np
import polars as pl
from src.config import Config
from src.data_loader import load_source_file
from src.blocking import add_blocking_columns, generate_candidates_for_targets
from src.features import compute_pairwise_features
from src.model import ERModel
from src.submission import write_submission_files
from src.dataset_builder import extract_record_dict_from_df

def run_test_inference(
    config: Config,
    model: ERModel,
    threshold: float,
    s1_chunk_size: int = 200000
) -> Tuple[str, str]:
    """
    Executes end-to-end inference on the official test set in streaming memory-safe chunks.
    """
    print("=" * 70, flush=True)
    print("PHASE: TEST SET INFERENCE", flush=True)
    print("=" * 70, flush=True)
    print(f"Applying frozen decision threshold: {threshold:.4f}", flush=True)

    start_time = time.time()

    # 1. Load and prepare S2 and S3 tables
    print("\nLoading and preparing Test Source 2...", flush=True)
    t0 = time.time()
    test_s2_df = load_source_file(config.test_s2_path, expected_prefix="S2-")
    test_s2_p = add_blocking_columns(test_s2_df)
    del test_s2_df
    gc.collect()
    print(f"Loaded and indexed {len(test_s2_p):,} Test S2 entities in {time.time() - t0:.2f}s", flush=True)

    print("\nLoading and preparing Test Source 3...", flush=True)
    t0 = time.time()
    test_s3_df = load_source_file(config.test_s3_path, expected_prefix="S3-")
    test_s3_p = add_blocking_columns(test_s3_df)
    del test_s3_df
    gc.collect()
    print(f"Loaded and indexed {len(test_s3_p):,} Test S3 entities in {time.time() - t0:.2f}s", flush=True)

    # 2. Load and prepare S1
    print("\nLoading and preparing Test Source 1...", flush=True)
    t0 = time.time()
    test_s1_df = load_source_file(config.test_s1_path, expected_prefix="S1-")
    all_test_s1_ids = test_s1_df["entity_id"].to_list()
    total_test_s1 = len(all_test_s1_ids)
    test_s1_p = add_blocking_columns(test_s1_df)
    del test_s1_df
    gc.collect()
    print(f"Loaded and indexed {total_test_s1:,} Test S1 entities in {time.time() - t0:.2f}s", flush=True)

    # Global tracking dictionaries
    matching_map: Dict[str, List[str]] = {s1_id: [] for s1_id in all_test_s1_ids}
    candidate_map: Dict[str, List[str]] = {s1_id: [] for s1_id in all_test_s1_ids}

    # 3. Process S1 in chunks
    num_chunks = (total_test_s1 + s1_chunk_size - 1) // s1_chunk_size
    print(f"\nProcessing {total_test_s1:,} Test S1 entities across {num_chunks} chunks...", flush=True)

    total_candidates_found = 0
    total_matches_found = 0

    for c_idx in range(num_chunks):
        c_start = c_idx * s1_chunk_size
        c_len = min(s1_chunk_size, total_test_s1 - c_start)
        t_chunk = time.time()

        s1_chunk_p = test_s1_p.slice(c_start, c_len)
        s1_chunk_records = extract_record_dict_from_df(s1_chunk_p)

        # Block against S2 and S3 for this chunk
        pairs_s2 = generate_candidates_for_targets(
            s1_chunk_p, test_s2_p, max_cands_per_s1=config.max_total_cands_per_s1 // 2 + 5
        )
        pairs_s3 = generate_candidates_for_targets(
            s1_chunk_p, test_s3_p, max_cands_per_s1=config.max_total_cands_per_s1 // 2 + 5
        )

        cand_pairs_df = pl.concat([pairs_s2, pairs_s3]).unique()
        chunk_cand_count = len(cand_pairs_df)
        total_candidates_found += chunk_cand_count

        if chunk_cand_count > 0:
            # Separate active S2 and S3 target IDs
            s2_cand_ids = [t for t in cand_pairs_df["target_id"].to_list() if t.startswith("S2-")]
            s3_cand_ids = [t for t in cand_pairs_df["target_id"].to_list() if t.startswith("S3-")]

            target_records: Dict[str, Dict[str, Any]] = {}
            if s2_cand_ids:
                s2_matched = test_s2_p.filter(pl.col("eid").is_in(list(set(s2_cand_ids))))
                target_records.update(extract_record_dict_from_df(s2_matched))
                del s2_matched

            if s3_cand_ids:
                s3_matched = test_s3_p.filter(pl.col("eid").is_in(list(set(s3_cand_ids))))
                target_records.update(extract_record_dict_from_df(s3_matched))
                del s3_matched

            # Build candidate list and compute pairwise features
            cand_rows = cand_pairs_df.to_dict(as_series=False)
            s1_col = cand_rows["s1_id"]
            tgt_col = cand_rows["target_id"]

            chunk_feats = []
            chunk_pairs = []

            for s1_id, cand_id in zip(s1_col, tgt_col):
                candidate_map[s1_id].append(cand_id)
                if s1_id in s1_chunk_records and cand_id in target_records:
                    f = compute_pairwise_features(s1_chunk_records[s1_id], target_records[cand_id], cand_id)
                    chunk_feats.append(f)
                    chunk_pairs.append((s1_id, cand_id))

            # LightGBM scoring
            if chunk_feats:
                X_chunk = np.array(chunk_feats, dtype=np.float32)
                probs = model.predict_proba(X_chunk)
                chunk_match_count = 0
                for (s1_id, cand_id), prob in zip(chunk_pairs, probs):
                    if prob >= threshold:
                        matching_map[s1_id].append(cand_id)
                        chunk_match_count += 1
                total_matches_found += chunk_match_count

            del target_records, chunk_feats, chunk_pairs

        del s1_chunk_p, s1_chunk_records, pairs_s2, pairs_s3, cand_pairs_df
        gc.collect()

        print(
            f"  [Chunk {c_idx + 1:02d}/{num_chunks:02d}] S1: {c_start + c_len:,}/{total_test_s1:,} | "
            f"Candidates: {chunk_cand_count:,} | Chunk matches: {total_matches_found:,} total | "
            f"Time: {time.time() - t_chunk:.2f}s",
            flush=True
        )

    # Free large tables
    del test_s1_p, test_s2_p, test_s3_p
    gc.collect()

    # 4. Write deliverables
    print("\nWriting official deliverables to output/ ...", flush=True)
    match_path, cand_path = write_submission_files(config, all_test_s1_ids, matching_map, candidate_map)

    total_matches = sum(len(v) for v in matching_map.values())
    singletons = sum(1 for v in matching_map.values() if not v)
    total_cands = sum(len(v) for v in candidate_map.values())

    print("-" * 75, flush=True)
    print("TEST INFERENCE SUMMARY:", flush=True)
    print(f"  Total Test S1 Entities:     {total_test_s1:,}", flush=True)
    print(f"  Total Matches Assigned:     {total_matches:,} (Avg {total_matches/total_test_s1:.2f} matches/S1)", flush=True)
    print(f"  Predicted Singletons:       {singletons:,} ({singletons/total_test_s1*100:.2f}%)", flush=True)
    print(f"  Total Candidates Evaluated: {total_cands:,} (Avg {total_cands/total_test_s1:.2f} cands/S1)", flush=True)
    print(f"  Total Inference Runtime:    {time.time() - start_time:.2f}s", flush=True)
    print("-" * 75, flush=True)

    return match_path, cand_path
