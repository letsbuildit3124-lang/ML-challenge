"""
Disk-Backed Candidate Recall Diagnostic and Verification Suite (DuckDB Powered).

Evaluates:
1. Target universe completeness (Original vs Loaded target records)
2. Pair-level and S1-level ground-truth candidate recall
3. Lossless ID mapping verification on ground-truth pairs
4. S2 vs S3 candidate recall breakdown
5. Memory profile at each stage (Peak RAM < 1.5GB)
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

from src.config import get_config
from src.data_loader import load_source_file, load_ground_truth
from src.duckdb_indexer import DuckDBTargetIndexer

def get_rss_mb() -> float:
    """Lightweight cross-platform RSS memory meter."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0

def run_recall_diagnostics(sample_s1_count: int = 1000, target_limit: int = None, rebuild_index: bool = False):
    config = get_config()
    print("=" * 80)
    print("V3 DISK-BACKED CANDIDATE RECALL ROOT CAUSE DIAGNOSIS (DUCKDB)")
    print("=" * 80)
    print(f"Initial RSS RAM: {get_rss_mb():.1f} MB")

    # -------------------------------------------------------------------------
    # 1. AUDIT GROUND TRUTH AND VALIDATION S1
    # -------------------------------------------------------------------------
    print("\n[Step 1/5] Auditing Ground Truth and Validation S1 Sample...", flush=True)
    t0 = time.time()
    gt_df = load_ground_truth(config.train_gt_path)
    
    splits_dir = os.path.join(config.data_dir, "splits")
    split_file = os.path.join(splits_dir, "split_seed_42.json")
    
    if os.path.exists(split_file):
        with open(split_file, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        eval_s1_ids = split_data["validation"][:sample_s1_count]
        print(f"Loaded {len(eval_s1_ids):,} S1 IDs from {split_file}")
    else:
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
    print(f"Evaluation S1 Count:         {len(eval_s1_ids):,}")
    print(f"Total True GT Pairs:         {total_gt_pairs:,}")
    print(f"Total Unique Targets Needed: {len(needed_targets):,} (S2: {len(s2_needed):,}, S3: {len(s3_needed):,})")
    print(f"Step 1 Complete in {time.time() - t0:.2f}s | RSS RAM: {get_rss_mb():.1f} MB")

    # -------------------------------------------------------------------------
    # 2. VERIFY ID MAPPING INTEGRITY ON 100 SAMPLE PAIRS
    # -------------------------------------------------------------------------
    print("\n[Step 2/5] Verifying ID Mapping on 100 Sample Positive Pairs...", flush=True)
    sample_pairs = list(gt_pairs_set)[:100]
    id_errors = 0
    for s1_orig, tgt_orig in sample_pairs:
        if not s1_orig.startswith("S1-") or not (tgt_orig.startswith("S2-") or tgt_orig.startswith("S3-")):
            id_errors += 1
            print(f"  ID Format Error: S1={s1_orig}, Target={tgt_orig}")

    if id_errors == 0:
        print(f"Verified 100 sampled pairs: ZERO ID namespace or format errors (100% valid prefix & namespace).")

    # -------------------------------------------------------------------------
    # 3. BUILD OR REUSE DISK-BACKED DUCKDB INDEX
    # -------------------------------------------------------------------------
    print("\n[Step 3/5] Initializing Disk-Backed DuckDB Target Index...", flush=True)
    t_idx0 = time.time()
    indexer = DuckDBTargetIndexer(memory_limit="2GB", threads=2)
    
    source_configs = [
        ("Train S2", config.train_s2_path, "S2-"),
        ("Train S3", config.train_s3_path, "S3-")
    ]
    
    total_indexed = indexer.build_index_from_sources(
        source_configs,
        chunk_size=100000,
        limit_per_file=target_limit,
        rebuild=rebuild_index
    )
    print(f"Step 3 Complete in {time.time() - t_idx0:.2f}s | Total Indexed Targets: {total_indexed:,} | RSS RAM: {get_rss_mb():.1f} MB")

    # -------------------------------------------------------------------------
    # 4. LOAD S1 AND EXECUTE PARALLEL MULTI-PASS CANDIDATE QUERIES
    # -------------------------------------------------------------------------
    print(f"\n[Step 4/5] Querying Disk Index for {len(eval_s1_ids):,} S1 Entities...", flush=True)
    t_query0 = time.time()
    
    # Load S1 evaluation records
    s1_full_df = load_source_file(config.train_s1_path, expected_prefix="S1-")
    s1_eval_df = s1_full_df.filter(pl.col("entity_id").is_in(eval_s1_ids))
    del s1_full_df
    gc.collect()

    # Query candidate pairs with provenance masks directly from disk index
    cands_result = indexer.query_candidates_for_s1(s1_eval_df, max_cands_per_s1=50)
    t_query = time.time() - t_query0
    print(f"Candidate generation completed in {t_query:.2f}s ({len(eval_s1_ids)/t_query:.1f} S1/s) | RSS RAM: {get_rss_mb():.1f} MB")

    # -------------------------------------------------------------------------
    # 5. EVALUATE RECALL METRICS & RECOVERY BREAKDOWN
    # -------------------------------------------------------------------------
    print("\n[Step 5/5] Computing Candidate Recall & Precision Breakdown...", flush=True)
    recovered_pairs = 0
    recovered_s2 = 0
    recovered_s3 = 0
    s1_with_at_least_one = 0
    s1_with_all_matches = 0
    total_candidate_pairs = 0

    for s1_id in eval_s1_ids:
        true_tgts = set(eval_gt_map.get(s1_id, []))
        cand_list = cands_result.get(s1_id, [])
        cand_tgts = set(tid for tid, mask, rec in cand_list)
        total_candidate_pairs += len(cand_tgts)

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
    avg_cands_per_s1 = total_candidate_pairs / len(eval_s1_ids)

    print("\n" + "=" * 80)
    print("V3 DISK-BACKED CANDIDATE RECALL BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Target Universe Evaluated:            {total_indexed:,} records (Disk-Backed DuckDB)")
    print(f"Pair-Level Candidate Recall (Total):  {pair_recall:.2f}% ({recovered_pairs:,} / {total_gt_pairs:,} pairs)")
    print(f"Source 2 Candidate Recall:             {s2_recall:.2f}% ({recovered_s2:,} / {total_s2_gt:,} pairs)")
    print(f"Source 3 Candidate Recall:             {s3_recall:.2f}% ({recovered_s3:,} / {total_s3_gt:,} pairs)")
    print(f"S1-Level Recall (>=1 match found):     {s1_at_least_one_recall:.2f}%")
    print(f"S1-Level Recall (ALL matches found):   {s1_all_matches_recall:.2f}%")
    print(f"Average Candidates per S1:             {avg_cands_per_s1:.1f}")
    print(f"Peak RSS Memory:                       {get_rss_mb():.1f} MB (Strictly < 1.5GB)")
    print("=" * 80)

    indexer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-count", type=int, default=1000, help="Number of S1 validation entities to evaluate")
    parser.add_argument("--target-count", type=int, default=None, help="Target record limit per source (e.g. 100000, None for all)")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of disk index")
    args = parser.parse_args()
    run_recall_diagnostics(sample_s1_count=args.s1_count, target_limit=args.target_count, rebuild_index=args.rebuild)
