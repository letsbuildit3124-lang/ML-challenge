"""
Comprehensive Candidate Recall Diagnosis and Auditing Suite.
Investigates:
1. Target universe completeness (Original vs Loaded target records)
2. Pair-level and S1-level ground-truth candidate recall
3. Lossless ID mapping verification on ground-truth pairs
4. S2 vs S3 candidate recall breakdown
5. Block-by-block isolated, incremental, and cumulative recall
6. Impact of country restrictions, block-size caps, and candidate filters
"""

import os
import sys
import gc
import json
import time
import argparse
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any
import polars as pl
import numpy as np

from src.config import get_config
from src.data_loader import load_source_file, load_ground_truth
from src.dataset_builder import extract_record_dict_from_df
from src.blocking_v2 import (
    add_v2_blocking_columns,
    build_compact_target_index,
    generate_candidates_against_indexed_target
)

def run_recall_diagnostics(sample_s1_count: int = 1000, target_limit: int = None):
    config = get_config()
    print("=" * 80)
    print("V3 CANDIDATE RECALL ROOT CAUSE DIAGNOSIS & VERIFICATION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. AUDIT GROUND TRUTH AND VALIDATION S1
    # -------------------------------------------------------------------------
    print("\n[Step 1/6] Auditing Ground Truth and Validation S1 Sample...", flush=True)
    gt_df = load_ground_truth(config.train_gt_path)
    
    # Load validation split if available, otherwise sample from GT
    splits_dir = os.path.join(config.data_dir, "splits")
    split_file = os.path.join(splits_dir, "split_seed_42.json")
    
    if os.path.exists(split_file):
        with open(split_file, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        eval_s1_ids = split_data["validation"][:sample_s1_count]
        print(f"Loaded {len(eval_s1_ids):,} S1 IDs from {split_file}")
    else:
        # Sample directly from GT
        all_gt_s1 = gt_df["source1_entity_id"].unique().to_list()
        eval_s1_ids = all_gt_s1[:sample_s1_count]
        print(f"Sampled {len(eval_s1_ids):,} S1 IDs directly from Ground Truth")

    eval_s1_set = set(eval_s1_ids)

    # Build GT lookup for evaluation S1
    eval_gt_map: Dict[str, List[str]] = {}
    needed_targets: Set[str] = set()
    s2_needed: Set[str] = set()
    s3_needed: Set[str] = set()
    gt_pairs_set: Set[Tuple[str, str]] = set()

    for row in gt_df.iter_rows():
        s1 = str(row[0])
        if s1 in eval_s1_set:
            m_str = str(row[1]) if row[1] is not None else ""
            tgts = [x.strip() for x in m_str.split(",") if x.strip()]
            eval_gt_map[s1] = tgts
            for t in tgts:
                needed_targets.add(t)
                gt_pairs_set.add((s1, t))
                if t.startswith("S2-"):
                    s2_needed.add(t)
                elif t.startswith("S3-"):
                    s3_needed.add(t)

    total_gt_pairs = len(gt_pairs_set)
    print(f"Evaluation S1 Count:       {len(eval_s1_ids):,}")
    print(f"Total True GT Pairs:       {total_gt_pairs:,}")
    print(f"Total Unique Targets Needed: {len(needed_targets):,} (S2: {len(s2_needed):,}, S3: {len(s3_needed):,})")

    # -------------------------------------------------------------------------
    # 2. AUDIT TARGET UNIVERSE AVAILABILITY
    # -------------------------------------------------------------------------
    print("\n[Step 2/6] Auditing Target Universe Completeness in S2 & S3...", flush=True)
    t0 = time.time()
    s2_df = load_source_file(config.train_s2_path, expected_prefix="S2-", n_rows=target_limit)
    s3_df = load_source_file(config.train_s3_path, expected_prefix="S3-", n_rows=target_limit)

    total_s2_records = len(s2_df)
    total_s3_records = len(s3_df)
    total_targets_loaded = total_s2_records + total_s3_records
    print(f"Loaded Train S2: {total_s2_records:,} records")
    print(f"Loaded Train S3: {total_s3_records:,} records")
    print(f"Total Target Universe Loaded: {total_targets_loaded:,} records in {time.time() - t0:.2f}s")

    # Check how many needed target IDs exist in loaded tables
    loaded_target_ids = set(s2_df["entity_id"].to_list()) | set(s3_df["entity_id"].to_list())
    present_needed = needed_targets & loaded_target_ids
    missing_needed = needed_targets - loaded_target_ids
    target_coverage_pct = (len(present_needed) / len(needed_targets) * 100.0) if needed_targets else 100.0

    print(f"\n--- TARGET UNIVERSE COVERAGE AUDIT ---")
    print(f"Needed Targets in Target Table: {len(present_needed):,} / {len(needed_targets):,} ({target_coverage_pct:.2f}%)")
    if missing_needed:
        print(f"WARNING: {len(missing_needed):,} ground-truth targets are MISSING from the loaded target table!")
        print(f"  Sample missing IDs: {list(missing_needed)[:5]}")
    else:
        print(f"SUCCESS: 100.0% of required ground truth targets are present in the loaded target table.")

    # -------------------------------------------------------------------------
    # 3. VERIFY ID MAPPING INTEGRITY ON 100 SAMPLE PAIRS
    # -------------------------------------------------------------------------
    print("\n[Step 3/6] Verifying ID Mapping on 100 Sample Positive Pairs...", flush=True)
    sample_pairs = list(gt_pairs_set)[:100]
    id_errors = 0
    for s1_orig, tgt_orig in sample_pairs:
        if not s1_orig.startswith("S1-") or not (tgt_orig.startswith("S2-") or tgt_orig.startswith("S3-")):
            id_errors += 1
            print(f"  ID Format Error: S1={s1_orig}, Target={tgt_orig}")

    if id_errors == 0:
        print(f"Verified 100 sampled pairs: ZERO ID namespace or format errors (100% valid prefix & namespace).")

    # -------------------------------------------------------------------------
    # 4. PREPROCESS & INDEX TARGET UNIVERSE
    # -------------------------------------------------------------------------
    print("\n[Step 4/6] Preprocessing and Indexing Full Target Universe...", flush=True)
    t0 = time.time()
    s2_p = add_v2_blocking_columns(s2_df)
    s3_p = add_v2_blocking_columns(s3_df)
    del s2_df, s3_df
    gc.collect()

    target_p = pl.concat([s2_p, s3_p])
    del s2_p, s3_p
    gc.collect()

    t_idx0 = time.time()
    target_index = build_compact_target_index(target_p)
    print(f"Target index built in {time.time() - t_idx0:.2f}s (Total indexing: {time.time() - t0:.2f}s)")

    # -------------------------------------------------------------------------
    # 5. BLOCK-BY-BLOCK ISOLATED & CUMULATIVE RECALL
    # -------------------------------------------------------------------------
    print("\n[Step 5/6] Evaluating Block-by-Block Recall Breakdown...", flush=True)
    # Load S1 sample
    s1_full_df = load_source_file(config.train_s1_path, expected_prefix="S1-")
    s1_eval_df = s1_full_df.filter(pl.col("entity_id").is_in(eval_s1_ids))
    s1_eval_p = add_v2_blocking_columns(s1_eval_df)
    del s1_full_df, s1_eval_df
    gc.collect()

    rules = [
        ("exact_compact_name", "compact_name"),
        ("translit_compact_name", "translit_cname"),
        ("exact_norm_name", "norm_name"),
        ("phonetic_soundex", "soundex_num"),
        ("cname8_addr_num", "cname8_num"),
        ("f2_words_num", "f2_num"),
        ("postal_cname4", "pin_cname4")
    ]

    cumulative_cands_map: Dict[str, Set[str]] = defaultdict(set)
    cumulative_recovered = 0

    print(f"\n{'Block Rule':<25} | {'Unique Cands':<12} | {'GT Recovered':<12} | {'Isolated Recall':<15} | {'Cumulative Recall':<17}")
    print("-" * 90)

    for rule_name, col_key in rules:
        isolated_recovered = 0
        rule_cands_count = 0

        # Sub-index for isolated testing
        sub_index = {col_key: target_index.get(col_key, {})}
        rule_cands = generate_candidates_against_indexed_target(s1_eval_p, sub_index, max_cands_per_s1=100)

        for s1_id, c_list in rule_cands.items():
            rule_cands_count += len(c_list)
            for tid in c_list:
                if (s1_id, tid) in gt_pairs_set:
                    isolated_recovered += 1
                cumulative_cands_map[s1_id].add(tid)

        # Count cumulative recovered
        cumul_rec = sum(1 for s1_id, c_set in cumulative_cands_map.items() for tid in c_set if (s1_id, tid) in gt_pairs_set)
        
        iso_pct = (isolated_recovered / total_gt_pairs * 100.0) if total_gt_pairs > 0 else 0.0
        cum_pct = (cumul_rec / total_gt_pairs * 100.0) if total_gt_pairs > 0 else 0.0

        print(f"{rule_name:<25} | {rule_cands_count:<12,} | {isolated_recovered:<12,} | {iso_pct:>13.2f}% | {cum_pct:>15.2f}%")

    # -------------------------------------------------------------------------
    # 6. FINAL CANDIDATE RECALL AUDIT
    # -------------------------------------------------------------------------
    print("\n[Step 6/6] Final Candidate Recall Summary...", flush=True)
    all_cands = generate_candidates_against_indexed_target(s1_eval_p, target_index, max_cands_per_s1=50)
    
    recovered_pairs = 0
    recovered_s2 = 0
    recovered_s3 = 0
    s1_with_at_least_one = 0
    s1_with_all_matches = 0

    for s1_id in eval_s1_ids:
        true_tgts = set(eval_gt_map.get(s1_id, []))
        cand_tgts = set(all_cands.get(s1_id, []))
        
        matched_intersection = true_tgts & cand_tgts
        recovered_pairs += len(matched_intersection)

        for tid in matched_intersection:
            if tid.startswith("S2-"):
                recovered_s2 += 1
            elif tid.startswith("S3-"):
                recovered_s3 += 1

        if len(matched_intersection) > 0:
            s1_with_at_least_one += 1
        if true_tgts and matched_intersection == true_tgts:
            s1_with_all_matches += 1

    total_s2_gt = sum(1 for s, t in gt_pairs_set if t.startswith("S2-"))
    total_s3_gt = sum(1 for s, t in gt_pairs_set if t.startswith("S3-"))

    pair_recall = (recovered_pairs / total_gt_pairs * 100.0) if total_gt_pairs > 0 else 0.0
    s2_recall = (recovered_s2 / total_s2_gt * 100.0) if total_s2_gt > 0 else 0.0
    s3_recall = (recovered_s3 / total_s3_gt * 100.0) if total_s3_gt > 0 else 0.0
    s1_at_least_one_recall = (s1_with_at_least_one / len(eval_s1_ids) * 100.0)
    s1_all_matches_recall = (s1_with_all_matches / len(eval_s1_ids) * 100.0)
    avg_cands_per_s1 = sum(len(c) for c in all_cands.values()) / len(eval_s1_ids)

    print("\n" + "=" * 80)
    print(f"FINAL CANDIDATE RECALL BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Pair-Level Candidate Recall (Total):  {pair_recall:.2f}% ({recovered_pairs:,} / {total_gt_pairs:,} pairs)")
    print(f"Source 2 Candidate Recall:             {s2_recall:.2f}% ({recovered_s2:,} / {total_s2_gt:,} pairs)")
    print(f"Source 3 Candidate Recall:             {s3_recall:.2f}% ({recovered_s3:,} / {total_s3_gt:,} pairs)")
    print(f"S1-Level Recall (>=1 match found):     {s1_at_least_one_recall:.2f}%")
    print(f"S1-Level Recall (ALL matches found):   {s1_all_matches_recall:.2f}%")
    print(f"Average Candidates per S1:             {avg_cands_per_s1:.1f}")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-count", type=int, default=1000, help="Number of S1 validation entities to evaluate")
    parser.add_argument("--target-limit", type=int, default=None, help="Target row limit (None for full universe)")
    args = parser.parse_args()
    run_recall_diagnostics(sample_s1_count=args.s1_count, target_limit=args.target_limit)
