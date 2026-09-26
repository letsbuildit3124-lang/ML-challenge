"""
Antigravity V3 Recall Gate Safety Check.
Asserts that pair-level candidate recall meets the target threshold (default >= 99.0%).
Fails execution with exit code 1 if threshold is not satisfied.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Set, Tuple
import polars as pl

from src.config import get_config
from src.data_loader import load_source_file, load_ground_truth
from src.duckdb_indexer import DuckDBTargetIndexer

def check_recall_gate(s1_count: int = 1000, target_recall_pct: float = 99.0, max_cands: int = 250):
    config = get_config()
    print("=" * 80)
    print(f"ANTIGRAVITY V3 — RECALL GATE (ASSERT PAIR-LEVEL RECALL >= {target_recall_pct:.1f}%)")
    print("=" * 80)

    # 1. Load Ground Truth
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

    # 2. Query Candidates from DuckDB
    indexer = DuckDBTargetIndexer(memory_limit="2GB", threads=2)
    s1_full_df = load_source_file(config.train_s1_path, expected_prefix="S1-")
    s1_eval_df = s1_full_df.filter(pl.col("entity_id").is_in(eval_s1_ids))
    del s1_full_df

    cands_result = indexer.query_candidates_for_s1(s1_eval_df, max_cands_per_s1=max_cands)
    indexer.close()

    recovered_pairs = 0
    for s1_id in eval_s1_ids:
        cand_list = cands_result.get(s1_id, [])
        cand_tgts = set(tid for tid, mask, rec in cand_list)
        true_tgts = set(eval_gt_map.get(s1_id, []))
        recovered_pairs += len(true_tgts & cand_tgts)

    pair_recall = (recovered_pairs / total_gt_pairs * 100.0) if total_gt_pairs > 0 else 0.0
    print(f"Evaluated S1 Entities:     {len(eval_s1_ids):,}")
    print(f"Total True GT Pairs:       {total_gt_pairs:,}")
    print(f"GT Pairs Recovered:        {recovered_pairs:,}")
    print(f"Measured Candidate Recall: {pair_recall:.2f}%")
    print(f"Target Recall Threshold:   {target_recall_pct:.2f}%")

    if pair_recall >= target_recall_pct:
        print(f"\n[RECALL GATE PASSED]: Measured recall ({pair_recall:.2f}%) >= Target ({target_recall_pct:.2f}%).")
        sys.exit(0)
    else:
        print(f"\n[RECALL GATE FAILED]: Measured recall ({pair_recall:.2f}%) < Target ({target_recall_pct:.2f}%).")
        print("Candidate generation must be refined before proceeding to downstream model training.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-count", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=99.0)
    parser.add_argument("--max-cands", type=int, default=250)
    args = parser.parse_args()
    check_recall_gate(s1_count=args.s1_count, target_recall_pct=args.threshold, max_cands=args.max_cands)
