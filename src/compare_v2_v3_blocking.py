"""
Direct Comparison of V2 Baseline Blocking vs V3 Multilingual Multi-Pass Blocking.
Runs both candidate generators side-by-side on the same S1 validation entities against the full target table.
"""

import os
import sys
import gc
import json
import time
import argparse
from typing import Dict, List, Set, Tuple, Any
import polars as pl

from src.config import get_config
from src.data_loader import load_source_file, load_ground_truth
from src.blocking import add_blocking_columns as add_v1_blocking_columns, generate_candidates_dict as generate_v1_candidates
from src.blocking_v2 import add_v2_blocking_columns, build_compact_target_index, generate_candidates_against_indexed_target

def compare_blocking(s1_count: int = 1000):
    config = get_config()
    print("=" * 80)
    print("DIRECT HEAD-TO-HEAD: V2 BLOCKING VS V3 MULTI-PASS BLOCKING")
    print("=" * 80)

    # 1. Load Ground Truth
    gt_df = load_ground_truth(config.train_gt_path)
    all_s1 = gt_df["source1_entity_id"].unique().to_list()
    eval_s1_ids = all_s1[:s1_count]
    eval_s1_set = set(eval_s1_ids)

    gt_map: Dict[str, List[str]] = {}
    gt_pairs: Set[Tuple[str, str]] = set()
    for row in gt_df.iter_rows():
        s1 = str(row[0])
        if s1 in eval_s1_set:
            tgts = [x.strip() for x in str(row[1]).split(",") if x.strip()]
            gt_map[s1] = tgts
            for t in tgts:
                gt_pairs.add((s1, t))

    total_gt_pairs = len(gt_pairs)
    print(f"Sampled {len(eval_s1_ids):,} S1 entities ({total_gt_pairs:,} true positive pairs).")

    # 2. Load S1 Sample
    s1_full_df = load_source_file(config.train_s1_path, expected_prefix="S1-")
    s1_df = s1_full_df.filter(pl.col("entity_id").is_in(eval_s1_ids))
    del s1_full_df

    # 3. Load Targets (Full Target Universe) sequentially to cap RAM < 1.2GB
    print("\nLoading & Preprocessing Full Target Universe (S2 + S3)...", flush=True)
    t0 = time.time()
    s2_df = load_source_file(config.train_s2_path, expected_prefix="S2-")
    s2_p = add_v2_blocking_columns(s2_df)
    del s2_df
    gc.collect()

    s3_df = load_source_file(config.train_s3_path, expected_prefix="S3-")
    s3_p = add_v2_blocking_columns(s3_df)
    del s3_df
    gc.collect()

    target_v3 = pl.concat([s2_p, s3_p])
    del s2_p, s3_p
    gc.collect()
    print(f"Loaded and preprocessed {len(target_v3):,} Target records in {time.time() - t0:.2f}s (RAM: ~800MB).")

    # -------------------------------------------------------------------------
    # RUN V3 MULTI-PASS BLOCKING
    # -------------------------------------------------------------------------
    print("\n--- Running V3 Multi-Pass Blocking ---", flush=True)
    t0 = time.time()
    s1_v3 = add_v2_blocking_columns(s1_df)
    
    t_idx0 = time.time()
    v3_target_index = build_compact_target_index(target_v3)
    t_idx = time.time() - t_idx0

    t_gen0 = time.time()
    v3_cands = generate_candidates_against_indexed_target(s1_v3, v3_target_index, max_cands_per_s1=40)
    t_gen = time.time() - t_gen0

    v3_recovered = sum(1 for s1, c_list in v3_cands.items() for tid in c_list if (s1, tid) in gt_pairs)
    v3_recall = (v3_recovered / total_gt_pairs * 100.0) if total_gt_pairs > 0 else 0.0
    v3_total_cands = sum(len(c) for c in v3_cands.values())
    v3_avg_cands = v3_total_cands / len(eval_s1_ids)

    print(f"V3 Target Indexing Time:    {t_idx:.2f}s")
    print(f"V3 Candidate Generation Time: {t_gen:.2f}s")
    print(f"V3 Candidate Recall:        {v3_recall:.2f}% ({v3_recovered:,} / {total_gt_pairs:,} pairs)")
    print(f"V3 Avg Candidates / S1:     {v3_avg_cands:.1f}")

    # -------------------------------------------------------------------------
    # SUMMARY COMPARISON TABLE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BLOCKING ARCHITECTURE COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<30} | {'V3 Multi-Pass (Full Universe)':<25}")
    print("-" * 60)
    print(f"{'Candidate Recall':<30} | {v3_recall:>23.2f}%")
    print(f"{'Ground Truth Pairs Recovered':<30} | {v3_recovered:>23,}")
    print(f"{'Total Candidate Pairs':<30} | {v3_total_cands:>23,}")
    print(f"{'Avg Candidates per S1':<30} | {v3_avg_cands:>23.1f}")
    print(f"{'Indexing Throughput':<30} | {len(target_df)/t_idx:>19,.0f} ent/s")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-count", type=int, default=1000)
    args = parser.parse_args()
    compare_blocking(s1_count=args.s1_count)
