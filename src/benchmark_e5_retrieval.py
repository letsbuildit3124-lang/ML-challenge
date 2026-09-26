"""
Antigravity E5 Dense & Hybrid Retrieval Benchmark Suite.
Measures true empirical candidate recall on the validation ground-truth split:
1. V4 Baseline Recall (Deterministic + Sparse TF-IDF)
2. E5 Dense-Only Recall (Top-K ANN)
3. V4 + E5 Hybrid Recall (Unified Candidate Pool)
4. Incremental Recall Contributed by E5 over V4
5. Average / P50 / P95 Candidate Set Cardinality per S1
6. Query Latency & FAISS Search Throughput

Focuses on the critical competition metric: Incremental Recall over V4.
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
from src.resource_tracker import get_current_rss_mb, MemoryTracker
from src.data_loader import load_ground_truth, load_source_file
from src.v4_pipeline import V4HybridPipeline
from src.e5_dense_retriever import E5DenseRetriever
from src.v5_e5_hybrid_pipeline import V5E5HybridPipeline


def run_e5_retrieval_benchmark(
    s1_count: int = 1000,
    ann_dir: str = "cache/ann/e5/production",
    candidate_budget: int = 250,
    k_dense_values: List[int] = [25, 50, 100, 250],
    output_report: str = "reports/e5_dense_benchmark.md",
    allow_incomplete: bool = False
) -> Dict[str, Any]:
    """
    Executes comparative evaluation across V4, E5, and Hybrid pipelines on GT split.
    """
    config = get_config()
    target_ann_dir = os.path.join(config.base_dir, ann_dir)
    gt_file = os.path.join(config.base_dir, "data", "train_ground_truth.tsv")
    s1_file = os.path.join(config.base_dir, "data", "train_source1.tsv")

    print("=" * 80)
    print("ANTIGRAVITY — V5 MULTILINGUAL E5 HYBRID RETRIEVAL BENCHMARK")
    print("=" * 80)
    print(f"Validation S1 Queries: {s1_count:,}")
    print(f"Candidate Budget / S1: {candidate_budget}")
    print(f"ANN Directory:         {target_ann_dir}")
    print(f"Initial RSS:           {get_current_rss_mb():.2f} MB")
    print("-" * 80)

    # 1. Load Ground Truth & Validation S1 Subset
    print("[1/4] Loading ground truth & validation queries...")
    df_gt = load_ground_truth(gt_file)
    df_s1_all = load_source_file(s1_file, expected_prefix="s1_")
    df_s1 = df_s1_all.head(s1_count)
    s1_ids = [str(x) for x in df_s1["entity_id"].to_list()]
    s1_id_set = set(s1_ids)

    # Parse GT pairs for this S1 subset
    gt_pairs: Set[Tuple[str, str]] = set()
    for row in df_gt.iter_rows(named=True):
        s1_id = str(row["source1_entity_id"])
        if s1_id in s1_id_set:
            matched_str = str(row.get("matched_entity_ids") or "")
            if matched_str and matched_str not in ("NULL", "null", "None", ""):
                for target_id in matched_str.split(";"):
                    tid = target_id.strip()
                    if tid:
                        gt_pairs.add((s1_id, tid))

    total_gt = len(gt_pairs)
    print(f"  Loaded {len(s1_ids):,} S1 queries with {total_gt:,} true positive target pairs.")

    # 2. V4 Baseline Candidate Retrieval
    print("\n[2/4] Evaluating V4 Baseline Candidate Retrieval (Deterministic + Sparse)...")
    v4_pipe = V4HybridPipeline()
    t0_v4 = time.time()
    v4_results = v4_pipe.generate_candidates(
        df_s1,
        enable_deterministic=True,
        enable_sparse_name=True,
        enable_sparse_addr=True,
        enable_sparse_translit=True,
        max_candidates_per_s1=candidate_budget
    )
    v4_duration = time.time() - t0_v4

    v4_found_pairs: Set[Tuple[str, str]] = set()
    v4_counts = []
    for s1_id, cands in v4_results.items():
        v4_counts.append(len(cands))
        for tid, mask, rec in cands:
            if (s1_id, tid) in gt_pairs:
                v4_found_pairs.add((s1_id, tid))

    v4_recall = (len(v4_found_pairs) / total_gt) * 100.0 if total_gt > 0 else 0.0
    v4_avg_cands = np.mean(v4_counts) if v4_counts else 0.0
    v4_pipe.close()

    print(f"  V4 Baseline Recall:    {v4_recall:.2f}% ({len(v4_found_pairs):,}/{total_gt:,} pairs)")
    print(f"  V4 Avg Candidates:     {v4_avg_cands:.1f} (P50: {np.median(v4_counts):.0f}, P95: {np.percentile(v4_counts, 95):.0f})")
    print(f"  V4 Retrieval Latency:  {v4_duration:.2f}s ({(v4_duration / s1_count)*1000:.1f} ms/query)")

    # 3. E5 Dense & Hybrid Evaluations
    print("\n[3/4] Evaluating E5 Dense & Hybrid Retrieval across K values...")
    try:
        e5_retriever = E5DenseRetriever(ann_dir=target_ann_dir)
        e5_retriever.load()
    except Exception as e:
        print(f"[Benchmark ERROR]: Failed to load E5 retriever from {target_ann_dir}: {e}")
        return {}

    benchmark_rows = []

    for k_dense in k_dense_values:
        t0_dense = time.time()
        dense_results = e5_retriever.retrieve_dense_candidates(df_s1, top_k=k_dense)
        dense_dur = time.time() - t0_dense

        # Dense-only Recall
        dense_found_pairs: Set[Tuple[str, str]] = set()
        dense_counts = []
        for s1_id, t_dict in dense_results.items():
            dense_counts.append(len(t_dict))
            for tid in t_dict.keys():
                if (s1_id, tid) in gt_pairs:
                    dense_found_pairs.add((s1_id, tid))

        dense_recall = (len(dense_found_pairs) / total_gt) * 100.0 if total_gt > 0 else 0.0
        dense_avg_cands = np.mean(dense_counts) if dense_counts else 0.0

        # Hybrid (V4 UNION E5)
        hybrid_found_pairs: Set[Tuple[str, str]] = set(v4_found_pairs)
        hybrid_counts = []
        newly_recovered_by_e5: Set[Tuple[str, str]] = set()

        for s1_id in s1_ids:
            v4_cands_s1 = {tid for tid, _, _ in v4_results.get(s1_id, [])}
            e5_cands_s1 = set(dense_results.get(s1_id, {}).keys())
            union_s1 = v4_cands_s1 | e5_cands_s1
            hybrid_counts.append(min(len(union_s1), candidate_budget))

            for tid in e5_cands_s1:
                if (s1_id, tid) in gt_pairs:
                    hybrid_found_pairs.add((s1_id, tid))
                    if (s1_id, tid) not in v4_found_pairs:
                        newly_recovered_by_e5.add((s1_id, tid))

        hybrid_recall = (len(hybrid_found_pairs) / total_gt) * 100.0 if total_gt > 0 else 0.0
        incremental_recall = hybrid_recall - v4_recall
        hybrid_avg_cands = np.mean(hybrid_counts) if hybrid_counts else 0.0

        print(f"\n  [E5 Top-{k_dense:03d}] Dense Recall: {dense_recall:.2f}% | Hybrid (V4+E5): {hybrid_recall:.2f}% (+{incremental_recall:.2f}% over V4) | Avg Cands: {hybrid_avg_cands:.1f}")

        benchmark_rows.append({
            "dense_k": k_dense,
            "v4_recall": round(v4_recall, 2),
            "dense_recall": round(dense_recall, 2),
            "hybrid_recall": round(hybrid_recall, 2),
            "incremental_recall": round(incremental_recall, 2),
            "new_pairs_recovered": len(newly_recovered_by_e5),
            "avg_hybrid_cands": round(hybrid_avg_cands, 1),
            "p50_cands": int(np.median(hybrid_counts)),
            "p95_cands": int(np.percentile(hybrid_counts, 95)),
            "dense_latency_sec": round(dense_dur, 2)
        })

    e5_retriever.close()

    # 4. Generate Markdown Benchmark Report
    print("\n[4/4] Writing Benchmark Report...")
    report_file = os.path.join(config.base_dir, output_report)
    os.makedirs(os.path.dirname(report_file), exist_ok=True)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Antigravity V5 — Multilingual E5 Dense & Hybrid Retrieval Benchmark\n\n")
        f.write(f"- **Evaluated S1 Queries**: `{s1_count:,}`\n")
        f.write(f"- **Total True GT Pairs**: `{total_gt:,}`\n")
        f.write(f"- **Candidate Budget**: `{candidate_budget}` per S1\n")
        f.write(f"- **V4 Baseline Recall**: `{v4_recall:.2f}%` (Avg `{v4_avg_cands:.1f}` candidates)\n\n")
        f.write("## Comparative Results\n\n")
        f.write("| Configuration | Dense Recall | Hybrid (V4 + E5) Recall | **Incremental Over V4** | New Pairs Recovered | Avg Candidates | P50 / P95 Cands |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        f.write(f"| **V4 Baseline** | — | `{v4_recall:.2f}%` | `0.00%` | 0 | `{v4_avg_cands:.1f}` | {int(np.median(v4_counts))} / {int(np.percentile(v4_counts, 95))} |\n")
        for r in benchmark_rows:
            f.write(
                f"| **V5 Hybrid (E5 Top-{r['dense_k']})** | `{r['dense_recall']:.2f}%` | "
                f"`{r['hybrid_recall']:.2f}%` | **`+{r['incremental_recall']:.2f}%`** | "
                f"`{r['new_pairs_recovered']:,}` | `{r['avg_hybrid_cands']:.1f}` | "
                f"{r['p50_cands']} / {r['p95_cands']} |\n"
            )

    print("=" * 80)
    print(f"BENCHMARK COMPLETE. Report written to: {report_file}")
    print("=" * 80)

    return {
        "v4_recall": v4_recall,
        "results": benchmark_rows
    }


def main():
    parser = argparse.ArgumentParser(description="Antigravity E5 Retrieval Benchmark")
    parser.add_argument("--s1-count", type=int, default=1000, help="Number of S1 queries to benchmark")
    parser.add_argument("--ann-dir", type=str, default="cache/ann/e5/production", help="Directory containing E5 FAISS index")
    parser.add_argument("--smoke", action="store_true", help="Benchmark against smoke index in cache/ann/e5/smoke")
    parser.add_argument("--budget", type=int, default=250, help="Candidate budget per S1 entity")
    parser.add_argument("--report", type=str, default="reports/e5_dense_benchmark.md", help="Output report path")
    args = parser.parse_args()

    ann_path = "cache/ann/e5/smoke" if args.smoke else args.ann_dir

    run_e5_retrieval_benchmark(
        s1_count=args.s1_count,
        ann_dir=ann_path,
        candidate_budget=args.budget,
        output_report=args.report
    )


if __name__ == "__main__":
    main()
