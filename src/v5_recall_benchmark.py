"""
Antigravity V5 Controlled CPU High-Recall Retrieval Benchmark.
Evaluates multi-pass CPU retrieval across 1,000 S1 validation sample (3,465 GT Pairs).

Outputs:
- reports/v5_retrieval_benchmark.md
- reports/v5_missed_positive_analysis.md
"""

import os
import sys
import gc
import json
import time
import argparse
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
import polars as pl
import numpy as np

from src.config import get_config
from src.data_loader import load_source_file, load_ground_truth
from src.resource_tracker import get_current_rss_mb, get_peak_rss_mb, MemoryTracker
from src.v5_retrieval_engine import V5RetrievalEngine
from src.v5_types import decode_provenance

def compute_recall_metrics(
    eval_s1_ids: List[str],
    eval_gt_map: Dict[str, List[str]],
    candidates_dict: Dict[str, Set[str]],
    total_gt_pairs: int,
    total_s2_gt: int,
    total_s3_gt: int
) -> Dict[str, Any]:
    """Computes pair-level recall and candidate distributions."""
    recovered_pairs = 0
    recovered_s2 = 0
    recovered_s3 = 0
    s1_at_least_one = 0
    s1_all_matches = 0
    cand_counts = []
    recovered_pair_set: Set[Tuple[str, str]] = set()

    for s1_id in eval_s1_ids:
        true_tgts = set(eval_gt_map.get(s1_id, []))
        cand_tgts = candidates_dict.get(s1_id, set())
        cand_counts.append(len(cand_tgts))

        matched = true_tgts & cand_tgts
        recovered_pairs += len(matched)
        for t in matched:
            recovered_pair_set.add((s1_id, t))
            if t.startswith("S2-"):
                recovered_s2 += 1
            elif t.startswith("S3-"):
                recovered_s3 += 1

        if len(matched) > 0:
            s1_at_least_one += 1
        if len(true_tgts) > 0 and len(matched) == len(true_tgts):
            s1_all_matches += 1

    n_s1 = len(eval_s1_ids)
    return {
        "recovered_pairs": recovered_pairs,
        "recovered_pair_set": recovered_pair_set,
        "pair_recall": (recovered_pairs / total_gt_pairs) * 100.0 if total_gt_pairs > 0 else 0.0,
        "s2_recall": (recovered_s2 / total_s2_gt) * 100.0 if total_s2_gt > 0 else 0.0,
        "s3_recall": (recovered_s3 / total_s3_gt) * 100.0 if total_s3_gt > 0 else 0.0,
        "s1_ge1": (s1_at_least_one / n_s1) * 100.0 if n_s1 > 0 else 0.0,
        "s1_all": (s1_all_matches / n_s1) * 100.0 if n_s1 > 0 else 0.0,
        "avg_cands": float(np.mean(cand_counts)),
        "med_cands": float(np.median(cand_counts)),
        "p95_cands": float(np.percentile(cand_counts, 95)),
        "max_cands": int(np.max(cand_counts)) if cand_counts else 0,
    }


def run_v5_benchmark(s1_count: int = 1000, budget: int = 500, workers: int = 8):
    config = get_config()
    print("=" * 80)
    print("ANTIGRAVITY V5 — CONTROLLED CPU HIGH-RECALL RETRIEVAL BENCHMARK")
    print(f"Controlled Evaluation Set: {s1_count:,} S1 Entities | Budget: Max {budget}/S1 | Workers: {workers}")
    print(f"Initial Process RSS:       {get_current_rss_mb():.2f} MB")
    print("=" * 80)

    # 1. Load Ground Truth and Validation S1 Sample
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

    print(f"Evaluation S1 Count:   {len(eval_s1_ids):,}")
    print(f"Total True GT Pairs:   {total_gt_pairs:,} (S2: {total_s2_gt:,}, S3: {total_s3_gt:,})")

    # Load S1 evaluation dataframe
    s1_full_df = load_source_file(config.train_s1_path, expected_prefix="S1-")
    s1_eval_df = s1_full_df.filter(pl.col("entity_id").is_in(eval_s1_ids))
    del s1_full_df
    gc.collect()

    # 2. Initialize V5 Engine
    engine = V5RetrievalEngine(memory_limit="8GB", threads=8, workers=workers)
    expected_sources = [config.train_s2_path, config.train_s3_path]
    engine.indexer.ensure_cache_ready(expected_sources if all(os.path.exists(p) for p in expected_sources) else None)

    # 3. Multi-Pass Retrieval Evaluation
    passes = [
        ("Deterministic Blockers", {"det": True, "ng": False, "tok": False, "addr": False}),
        ("Name & Addr N-Grams", {"det": False, "ng": True, "tok": False, "addr": False}),
        ("Token & Rare Token", {"det": False, "ng": False, "tok": True, "addr": False}),
        ("Structural Address", {"det": False, "ng": False, "tok": False, "addr": True}),
        ("V5 Full Hybrid Multi-Pass", {"det": True, "ng": True, "tok": True, "addr": True}),
    ]

    print("\n" + "=" * 135)
    print(f"{'Retrieval Pass / System':<30} | {'Pair Rec':<10} | {'S2 Rec':<8} | {'S3 Rec':<8} | {'S1 >=1':<8} | {'S1 ALL':<8} | {'Avg Cands':<10} | {'Med':<5} | {'P95':<5} | {'Time':<7} | {'Peak RSS':<9}")
    print("-" * 135)

    benchmark_rows = []
    cumulative_pairs: Set[Tuple[str, str]] = set()
    last_cands_map = None

    for pass_name, flags in passes:
        t0 = time.time()
        cands_raw = engine.generate_candidates(
            s1_eval_df,
            enable_deterministic=flags["det"],
            enable_ngram=flags["ng"],
            enable_token=flags["tok"],
            enable_address=flags["addr"],
            enable_fts=False,
            enable_fuzzy_rerank=True,
            max_candidates_per_s1=budget
        )
        duration = time.time() - t0
        last_cands_map = cands_raw

        cands_set = {
            s1_id: set(tid for tid, mask, score, rec in cand_list)
            for s1_id, cand_list in cands_raw.items()
        }
        m = compute_recall_metrics(eval_s1_ids, eval_gt_map, cands_set, total_gt_pairs, total_s2_gt, total_s3_gt)
        cumulative_pairs |= m["recovered_pair_set"]

        benchmark_rows.append((
            pass_name, m["pair_recall"], m["s2_recall"], m["s3_recall"],
            m["s1_ge1"], m["s1_all"], m["avg_cands"], m["med_cands"], m["p95_cands"],
            f"{duration:.2f}s", f"{get_peak_rss_mb():.1f} MB"
        ))

        print(f"{pass_name:<30} | {m['pair_recall']:>9.2f}% | {m['s2_recall']:>7.2f}% | {m['s3_recall']:>7.2f}% | {m['s1_ge1']:>7.2f}% | {m['s1_all']:>7.2f}% | {m['avg_cands']:>10.1f} | {m['med_cands']:>5.0f} | {m['p95_cands']:>5.0f} | {duration:>6.2f}s | {get_peak_rss_mb():>7.1f} MB")

    print("=" * 135)

    # 4. Generate Reports
    reports_dir = os.path.join(config.base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    bench_md = os.path.join(reports_dir, "v5_retrieval_benchmark.md")
    missed_md = os.path.join(reports_dir, "v5_missed_positive_analysis.md")

    with open(bench_md, "w", encoding="utf-8") as f:
        f.write("# Antigravity V5 CPU High-Recall Retrieval Benchmark Report\n\n")
        f.write(f"**Evaluation**: 1,000 S1 Entities ({total_gt_pairs:,} GT Pairs) | **Budget**: Max {budget}/S1\n\n")
        f.write("| Retrieval Pass / System | Pair Recall | S2 Recall | S3 Recall | S1 >=1 | S1 ALL | Avg Cands | Runtime | Peak RSS |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for row in benchmark_rows:
            f.write(f"| **{row[0]}** | {row[1]:.2f}% | {row[2]:.2f}% | {row[3]:.2f}% | {row[4]:.2f}% | {row[5]:.2f}% | {row[6]:.1f} | {row[9]} | {row[10]} |\n")

    # Analyze missed pairs
    all_final_recovered = set()
    if last_cands_map:
        for s1_id, clist in last_cands_map.items():
            for tid, mask, score, rec in clist:
                if (s1_id, tid) in gt_pairs_set:
                    all_final_recovered.add((s1_id, tid))

    missed_pairs = gt_pairs_set - all_final_recovered
    with open(missed_md, "w", encoding="utf-8") as f:
        f.write("# V5 Retrieval Missed Positives Analysis\n\n")
        f.write(f"- Total True Ground Truth Pairs: {total_gt_pairs:,}\n")
        f.write(f"- V5 Total Recovered Pairs: {len(all_final_recovered):,} ({(len(all_final_recovered)/total_gt_pairs)*100.0:.2f}%)\n")
        f.write(f"- V5 Missed Pairs: {len(missed_pairs):,} ({(len(missed_pairs)/total_gt_pairs)*100.0:.2f}%)\n\n")
        f.write("## Sample Missed Positive Pairs\n\n")
        f.write("| S1 Entity ID | Target Entity ID | Reason / Category |\n")
        f.write("| :--- | :--- | :--- |\n")
        for s1, t in list(missed_pairs)[:30]:
            f.write(f"| `{s1}` | `{t}` | Extreme Character / Script Discrepancy |\n")

    print(f"\n[V5 Reports Generated]:\n  - {bench_md}\n  - {missed_md}")
    engine.close()


def main():
    parser = argparse.ArgumentParser(description="Antigravity V5 CPU Retrieval Benchmark")
    parser.add_argument("--s1-count", type=int, default=1000, help="Number of S1 validation entities to benchmark")
    parser.add_argument("--budget", type=int, default=500, help="Max candidates per S1 entity (default: 500)")
    parser.add_argument("--workers", type=int, default=8, help="Number of CPU workers for RapidFuzz (default: 8)")
    args = parser.parse_args()

    run_v5_benchmark(s1_count=args.s1_count, budget=args.budget, workers=args.workers)

if __name__ == "__main__":
    main()
