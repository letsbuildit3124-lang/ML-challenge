"""
Antigravity V5 Candidate Recall Ladder.
Evaluates recall vs candidate budget curve across K = 100, 250, 500, 1000, 2000.
"""

import os
import sys
import gc
import json
import time
import argparse
from typing import Dict, List, Set, Tuple, Any
import polars as pl
import numpy as np

from src.config import get_config
from src.data_loader import load_source_file, load_ground_truth
from src.resource_tracker import get_current_rss_mb, get_peak_rss_mb
from src.v5_retrieval_engine import V5RetrievalEngine
from src.v5_recall_benchmark import compute_recall_metrics

def run_v5_recall_ladder(s1_count: int = 1000, workers: int = 8):
    config = get_config()
    print("=" * 80)
    print("ANTIGRAVITY V5 — CANDIDATE RECALL LADDER")
    print(f"Controlled Evaluation Set: {s1_count:,} S1 Entities | Workers: {workers}")
    print(f"Initial Process RSS:       {get_current_rss_mb():.2f} MB")
    print("=" * 80)

    gt_df = load_ground_truth(config.train_gt_path)
    splits_dir = os.path.join(config.data_dir, "splits")
    split_file = os.path.join(splits_dir, "split_seed_42.json")

    if os.path.exists(split_file):
        with open(split_file, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        eval_s1_ids = split_data["validation"][:s1_count]
    else:
        all_gt_s1 = gt_df["source1_entity_id"].unique().to_list()
        eval_s1_ids = all_gt_s1[:s1_count]

    eval_s1_set = set(eval_s1_ids)
    eval_gt_map: Dict[str, List[str]] = {}
    gt_pairs_set: Set[Tuple[str, str]] = set()

    for row in gt_df.iter_rows():
        s1 = str(row[0])
        if s1 in eval_s1_set:
            m_str = str(row[1]) if row[1] is not None else ""
            tgts = [x.strip() for x in m_str.split(",") if x.strip()]
            eval_gt_map[s1] = tgts
            for t in tgts:
                gt_pairs_set.add((s1, t))

    total_gt_pairs = len(gt_pairs_set)
    total_s2_gt = sum(1 for s, t in gt_pairs_set if t.startswith("S2-"))
    total_s3_gt = sum(1 for s, t in gt_pairs_set if t.startswith("S3-"))

    s1_full_df = load_source_file(config.train_s1_path, expected_prefix="S1-")
    s1_eval_df = s1_full_df.filter(pl.col("entity_id").is_in(eval_s1_ids))
    del s1_full_df
    gc.collect()

    engine = V5RetrievalEngine(memory_limit="8GB", threads=8, workers=workers)
    expected_sources = [config.train_s2_path, config.train_s3_path]
    engine.indexer.ensure_cache_ready(expected_sources if all(os.path.exists(p) for p in expected_sources) else None)

    k_values = [100, 250, 500, 1000, 2000]

    print(f"\n{'K Budget':<10} | {'Pair Rec':<10} | {'S2 Rec':<8} | {'S3 Rec':<8} | {'S1 >=1':<8} | {'S1 ALL':<8} | {'Avg Cands':<10} | {'Time':<7} | {'Peak RSS':<9}")
    print("-" * 105)

    for k in k_values:
        t0 = time.time()
        cands_raw = engine.generate_candidates(
            s1_eval_df,
            enable_deterministic=True,
            enable_ngram=True,
            enable_token=True,
            enable_address=True,
            enable_fts=False,
            enable_fuzzy_rerank=True,
            max_candidates_per_s1=k
        )
        duration = time.time() - t0

        cands_set = {
            s1_id: set(tid for tid, mask, score, rec in cand_list)
            for s1_id, cand_list in cands_raw.items()
        }
        m = compute_recall_metrics(eval_s1_ids, eval_gt_map, cands_set, total_gt_pairs, total_s2_gt, total_s3_gt)

        print(f"{k:<10d} | {m['pair_recall']:>9.2f}% | {m['s2_recall']:>7.2f}% | {m['s3_recall']:>7.2f}% | {m['s1_ge1']:>7.2f}% | {m['s1_all']:>7.2f}% | {m['avg_cands']:>10.1f} | {duration:>6.2f}s | {get_peak_rss_mb():>7.1f} MB")

    print("=" * 105)
    engine.close()


def main():
    parser = argparse.ArgumentParser(description="Antigravity V5 Candidate Recall Ladder")
    parser.add_argument("--s1-count", type=int, default=1000, help="Number of S1 validation entities to benchmark")
    parser.add_argument("--workers", type=int, default=8, help="Number of CPU workers for RapidFuzz (default: 8)")
    args = parser.parse_args()

    run_v5_recall_ladder(s1_count=args.s1_count, workers=args.workers)

if __name__ == "__main__":
    main()
