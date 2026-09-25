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
    batch_size: int = 500000
) -> Tuple[str, str]:
    """
    Executes end-to-end inference on the official test set sequentially.
    """
    print("=" * 70, flush=True)
    print("PHASE: TEST SET INFERENCE", flush=True)
    print("=" * 70, flush=True)
    print(f"Applying frozen decision threshold: {threshold:.4f}", flush=True)

    # 1. Load Test S1
    print("\nLoading Test Source 1...", flush=True)
    t0 = time.time()
    test_s1_df = load_source_file(config.test_s1_path, expected_prefix="S1-")
    all_test_s1_ids = test_s1_df["entity_id"].to_list()
    total_test_s1 = len(all_test_s1_ids)
    print(f"Loaded {total_test_s1:,} Test S1 entities in {time.time() - t0:.2f}s", flush=True)

    test_s1_p = add_blocking_columns(test_s1_df)
    s1_records = extract_record_dict_from_df(test_s1_p)
    del test_s1_df
    gc.collect()

    candidate_tables = []
    target_records: Dict[str, Dict[str, Any]] = {}

    # 2. Block against Test S2
    print("\nLoading and Blocking against Test Source 2...", flush=True)
    t0 = time.time()
    test_s2_df = load_source_file(config.test_s2_path, expected_prefix="S2-")
    test_s2_p = add_blocking_columns(test_s2_df)
    del test_s2_df
    gc.collect()

    pairs_s2 = generate_candidates_for_targets(test_s1_p, test_s2_p, max_cands_per_s1=config.max_total_cands_per_s1 // 2 + 5)
    candidate_tables.append(pairs_s2)

    needed_s2 = set(pairs_s2["target_id"].to_list())
    print(f"Test S2 Blocking: {len(pairs_s2):,} candidate pairs in {time.time() - t0:.2f}s. Extracting {len(needed_s2):,} active S2 records...", flush=True)
    
    s2_matched = test_s2_p.filter(pl.col("eid").is_in(list(needed_s2)))
    target_records.update(extract_record_dict_from_df(s2_matched))

    del test_s2_p, s2_matched, pairs_s2
    gc.collect()

    # 3. Block against Test S3
    print("\nLoading and Blocking against Test Source 3...", flush=True)
    t0 = time.time()
    test_s3_df = load_source_file(config.test_s3_path, expected_prefix="S3-")
    test_s3_p = add_blocking_columns(test_s3_df)
    del test_s3_df
    gc.collect()

    pairs_s3 = generate_candidates_for_targets(test_s1_p, test_s3_p, max_cands_per_s1=config.max_total_cands_per_s1 // 2 + 5)
    candidate_tables.append(pairs_s3)

    needed_s3 = set(pairs_s3["target_id"].to_list())
    print(f"Test S3 Blocking: {len(pairs_s3):,} candidate pairs in {time.time() - t0:.2f}s. Extracting {len(needed_s3):,} active S3 records...", flush=True)

    s3_matched = test_s3_p.filter(pl.col("eid").is_in(list(needed_s3)))
    target_records.update(extract_record_dict_from_df(s3_matched))

    del test_s3_p, s3_matched, pairs_s3, test_s1_p
    gc.collect()

    # 4. Combine all candidate pairs
    cand_pairs_df = pl.concat(candidate_tables).unique()
    print(f"\nTotal Test Candidate Pairs: {len(cand_pairs_df):,} (Avg: {len(cand_pairs_df)/total_test_s1:.2f}/S1)", flush=True)

    # 5. Build candidate map
    candidate_map: Dict[str, List[str]] = {s1_id: [] for s1_id in all_test_s1_ids}
    for row in cand_pairs_df.iter_rows():
        candidate_map[str(row[0])].append(str(row[1]))

    # 6. Score candidate pairs with model
    print(f"\nScoring {len(cand_pairs_df):,} candidate pairs in batches of {batch_size:,}...", flush=True)
    
    matching_map: Dict[str, List[str]] = {s1_id: [] for s1_id in all_test_s1_ids}
    total_pairs = len(cand_pairs_df)
    num_batches = (total_pairs + batch_size - 1) // batch_size
    
    start_time = time.time()
    
    for b_idx in range(num_batches):
        b_start = b_idx * batch_size
        b_end = min(b_start + batch_size, total_pairs)
        batch_slice = cand_pairs_df.slice(b_start, b_end - b_start)
        
        batch_feats = []
        batch_pairs = []
        
        for row in batch_slice.iter_rows():
            s1_id, cand_id = str(row[0]), str(row[1])
            if s1_id in s1_records and cand_id in target_records:
                feats = compute_pairwise_features(s1_records[s1_id], target_records[cand_id], cand_id)
                batch_feats.append(feats)
                batch_pairs.append((s1_id, cand_id))

        if batch_feats:
            X_batch = np.array(batch_feats, dtype=np.float32)
            probs = model.predict_proba(X_batch)
            
            for (s1_id, cand_id), prob in zip(batch_pairs, probs):
                if prob >= threshold:
                    matching_map[s1_id].append(cand_id)

        print(f"  Batch {b_idx + 1}/{num_batches} scored ({b_end:,}/{total_pairs:,} pairs) - Elapsed: {time.time() - start_time:.1f}s", flush=True)

    # 7. Write deliverables
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
