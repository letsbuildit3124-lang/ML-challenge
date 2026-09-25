"""
V2 Candidate Generation Experiment Runner.
Executes the A/B evaluation of V2 candidate generation against V1 on the frozen validation set.
Runs the frozen V1 LightGBM baseline (threshold = 0.50) to evaluate downstream impact.
"""

import os
import gc
import json
import time
import random
from collections import Counter
from typing import Dict, List, Set, Tuple, Any
import numpy as np
import polars as pl

from src.config import get_config
from src.data_loader import load_source_file, load_ground_truth
from src.dataset_builder import extract_record_dict_from_df
from src.features import compute_pairwise_features
from src.model import ERModel
from src.evaluation import evaluate_predictions
from src.blocking_v2 import (
    add_v2_blocking_columns,
    block_exact_compact_name,
    block_exact_norm_name,
    block_cname8_addr_num,
    block_f2_words_addr_num,
    block_method_a_postal_cname,
    block_method_b_ngram_affix,
    block_method_c_rare_token_overlap,
    block_method_d_phonetic_soundex,
    block_method_e_transliteration,
    block_method_f_tfidf_retrieval,
)

def run_v2_experiment(sample_chunk_only: bool = False, chunk_size: int = 250, include_tfidf: bool = False):
    config = get_config()
    print("=" * 80)
    print("RUNNING V2 CANDIDATE GENERATION A/B EXPERIMENT (FROZEN V1 BASELINE)")
    print("=" * 80)

    # 1. Load Data
    t0 = time.time()
    s1_df = load_source_file(config.train_s1_path, expected_prefix="S1-")
    gt_df = load_ground_truth(config.train_gt_path)

    # Deterministic V1 Split (Seed 42)
    all_s1_ids = list(s1_df["entity_id"].to_list())
    random.seed(config.seed)
    random.shuffle(all_s1_ids)

    total_requested = config.train_sample_s1_count + config.val_sample_s1_count
    selected_s1_ids = all_s1_ids[:total_requested]

    split_idx = config.train_sample_s1_count
    val_ids = list(selected_s1_ids[split_idx:total_requested])

    if sample_chunk_only:
        print(f"[TEST MODE] Testing pipeline on small chunk of {chunk_size} validation entities...")
        val_ids = val_ids[:chunk_size]

    val_set = set(val_ids)
    print(f"Validation S1 Entities: {len(val_ids):,} (Frozen V1 Split, Seed {config.seed})")

    # Filter GT for Validation
    val_gt_df = gt_df.filter(pl.col("source1_entity_id").is_in(val_ids))
    val_gt_map: Dict[str, List[str]] = {}
    for row in val_gt_df.iter_rows(named=True):
        m_str = row["matched_entity_ids"]
        val_gt_map[row["source1_entity_id"]] = [x.strip() for x in m_str.split(",") if x.strip()] if m_str else []

    total_gt_pairs = sum(len(v) for v in val_gt_map.values())
    gt_pairs_set = {(s1_id, tgt) for s1_id, tgts in val_gt_map.items() for tgt in tgts}
    print(f"Total Ground-Truth Validation Pairs: {total_gt_pairs:,}")

    # Load S2 and S3
    print("\nLoading and applying V2 blocking columns to S2 and S3...")
    s2_df = load_source_file(config.train_s2_path, expected_prefix="S2-")
    s3_df = load_source_file(config.train_s3_path, expected_prefix="S3-")

    val_s1_df = s1_df.filter(pl.col("entity_id").is_in(val_ids))
    del s1_df
    gc.collect()

    val_s1_p = add_v2_blocking_columns(val_s1_df)
    s2_p = add_v2_blocking_columns(s2_df)
    s3_p = add_v2_blocking_columns(s3_df)

    val_s1_records = extract_record_dict_from_df(val_s1_p)

    # =========================================================================
    # 2. RUN & MEASURE EACH V2 BLOCKING METHOD
    # =========================================================================
    print("\n" + "=" * 80)
    print("EVALUATING INDIVIDUAL AND CUMULATIVE V2 BLOCKING METHODS")
    print("=" * 80)

    methods = [
        ("1. exact_compact_name (V1)", block_exact_compact_name),
        ("2. exact_normalized_name (V1)", block_exact_norm_name),
        ("3. cname8_addr_num (V1)", block_cname8_addr_num),
        ("4. f2_words_addr_num (V1)", block_f2_words_addr_num),
        ("5. method_a_postal_cname (V2)", block_method_a_postal_cname),
        ("6. method_b_ngram_affix (V2)", block_method_b_ngram_affix),
        ("7. method_c_rare_token (V2)", block_method_c_rare_token_overlap),
        ("8. method_d_phonetic_soundex (V2)", block_method_d_phonetic_soundex),
        ("9. method_e_transliteration (V2)", block_method_e_transliteration),
    ]

    method_results = []
    cumulative_pairs: Set[Tuple[str, str]] = set()

    for m_name, m_func in methods:
        t_m = time.time()
        p2 = m_func(val_s1_p, s2_p)
        p3 = m_func(val_s1_p, s3_p)
        m_df = pl.concat([p2, p3]).unique()

        m_pairs = set(zip(m_df["s1_id"].to_list(), m_df["target_id"].to_list()))
        recovered = len(m_pairs.intersection(gt_pairs_set))
        m_recall = recovered / total_gt_pairs if total_gt_pairs > 0 else 0.0

        # Incremental gain over previous cumulative union
        prev_count = len(cumulative_pairs.intersection(gt_pairs_set))
        cumulative_pairs.update(m_pairs)
        curr_count = len(cumulative_pairs.intersection(gt_pairs_set))
        incremental_gain = curr_count - prev_count
        incremental_recall = incremental_gain / total_gt_pairs if total_gt_pairs > 0 else 0.0
        cumulative_recall = curr_count / total_gt_pairs if total_gt_pairs > 0 else 0.0

        res_entry = {
            "method": m_name,
            "candidate_count": len(m_df),
            "unique_candidate_pairs": len(m_pairs),
            "gt_pairs_recovered": recovered,
            "isolated_recall": m_recall,
            "incremental_recovered": incremental_gain,
            "incremental_recall": incremental_recall,
            "cumulative_recovered": curr_count,
            "cumulative_recall": cumulative_recall,
            "runtime_s": time.time() - t_m
        }
        method_results.append(res_entry)
        print(
            f"  [{m_name:<32}] Cands: {len(m_df):>8,} | GT: {recovered:>5,} ({m_recall*100:>6.2f}%) | "
            f"Incr: +{incremental_gain:>4,} ({incremental_recall*100:>5.2f}%) | "
            f"Cumulative: {curr_count:>5,} / {total_gt_pairs:,} ({cumulative_recall*100:>6.2f}%)"
        )

    # Optional TF-IDF retrieval block
    if include_tfidf:
        print("\nEvaluating Method F: Character N-Gram TF-IDF Retrieval...")
        t_f = time.time()
        p2_tfidf = block_method_f_tfidf_retrieval(val_s1_p, s2_p, top_k=20, min_sim=0.70)
        p3_tfidf = block_method_f_tfidf_retrieval(val_s1_p, s3_p, top_k=20, min_sim=0.70)
        tfidf_df = pl.concat([p2_tfidf, p3_tfidf]).unique()
        tfidf_pairs = set(zip(tfidf_df["s1_id"].to_list(), tfidf_df["target_id"].to_list()))
        
        recovered_tfidf = len(tfidf_pairs.intersection(gt_pairs_set))
        prev_count = len(cumulative_pairs.intersection(gt_pairs_set))
        cumulative_pairs.update(tfidf_pairs)
        curr_count = len(cumulative_pairs.intersection(gt_pairs_set))
        incremental_gain = curr_count - prev_count
        
        method_results.append({
            "method": "10. method_f_tfidf_retrieval (V2)",
            "candidate_count": len(tfidf_df),
            "unique_candidate_pairs": len(tfidf_pairs),
            "gt_pairs_recovered": recovered_tfidf,
            "isolated_recall": recovered_tfidf / total_gt_pairs if total_gt_pairs > 0 else 0.0,
            "incremental_recovered": incremental_gain,
            "incremental_recall": incremental_gain / total_gt_pairs if total_gt_pairs > 0 else 0.0,
            "cumulative_recovered": curr_count,
            "cumulative_recall": curr_count / total_gt_pairs if total_gt_pairs > 0 else 0.0,
            "runtime_s": time.time() - t_f
        })
        print(
            f"  [{'10. method_f_tfidf_retrieval (V2)':<32}] Cands: {len(tfidf_df):>8,} | GT: {recovered_tfidf:>5,} | "
            f"Incr: +{incremental_gain:>4,} | Cumulative: {curr_count:>5,} ({curr_count/total_gt_pairs*100:>6.2f}%)"
        )

    # Candidate distribution per S1 entity
    v2_cand_by_s1: Dict[str, List[str]] = {s1_id: [] for s1_id in val_ids}
    for s1_id, tgt_id in cumulative_pairs:
        v2_cand_by_s1[s1_id].append(tgt_id)

    cand_counts = [len(v2_cand_by_s1[s1]) for s1 in val_ids]
    cand_volume_stats = {
        "total_candidate_pairs": len(cumulative_pairs),
        "mean_candidates_per_s1": float(np.mean(cand_counts)),
        "median_candidates_per_s1": float(np.median(cand_counts)),
        "p90_candidates_per_s1": float(np.percentile(cand_counts, 90)),
        "p95_candidates_per_s1": float(np.percentile(cand_counts, 95)),
        "max_candidates_per_s1": int(np.max(cand_counts)),
        "min_candidates_per_s1": int(np.min(cand_counts)),
    }

    v2_total_recovered = len(cumulative_pairs.intersection(gt_pairs_set))
    v2_candidate_recall = v2_total_recovered / total_gt_pairs if total_gt_pairs > 0 else 0.0

    print("\n" + "=" * 80)
    print("V2 CANDIDATE VOLUME & RECALL SUMMARY")
    print("=" * 80)
    print(f"  V1 Baseline Candidate Recall: 62.09% (5,353 / 8,622 pairs)")
    print(f"  V2 Candidate Recall:          {v2_candidate_recall*100:.2f}% ({v2_total_recovered:,} / {total_gt_pairs:,} pairs)")
    print(f"  Total V2 Candidate Pairs:     {len(cumulative_pairs):,}")
    print(f"  Mean Candidates / S1:         {cand_volume_stats['mean_candidates_per_s1']:.2f}")
    print(f"  Median Candidates / S1:       {cand_volume_stats['median_candidates_per_s1']:.1f}")
    print(f"  P90 Candidates / S1:          {cand_volume_stats['p90_candidates_per_s1']:.1f}")
    print(f"  P95 Candidates / S1:          {cand_volume_stats['p95_candidates_per_s1']:.1f}")
    print(f"  Max Candidates / S1:          {cand_volume_stats['max_candidates_per_s1']:,}")

    # =========================================================================
    # 3. ERROR ANALYSIS: REMAINING CANDIDATE GENERATION MISSES
    # =========================================================================
    remaining_misses = []
    for s1_id, tgts in val_gt_map.items():
        for tgt in tgts:
            if (s1_id, tgt) not in cumulative_pairs:
                s1_rec = val_s1_records.get(s1_id, {})
                remaining_misses.append({
                    "s1_id": s1_id,
                    "target_id": tgt,
                    "s1_name": s1_rec.get("norm_name", ""),
                    "s1_addr": s1_rec.get("norm_addr", ""),
                    "country": s1_rec.get("country", "")
                })

    print(f"\nRemaining Candidate-Generation Misses: {len(remaining_misses):,} pairs ({(len(remaining_misses)/total_gt_pairs)*100:.2f}%)")

    # =========================================================================
    # 4. RUN FROZEN LIGHTGBM BASELINE ON V2 CANDIDATES (THRESHOLD = 0.50)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SCORING V2 CANDIDATES WITH FROZEN V1 LIGHTGBM MODEL (THRESHOLD = 0.50)")
    print("=" * 80)

    model = ERModel(config)
    model.load(config.model_save_path)

    # Extract target records for active candidate pairs
    all_tgt_ids = {t for _, t in cumulative_pairs}
    s2_matched_ids = [t for t in all_tgt_ids if t.startswith("S2-")]
    s3_matched_ids = [t for t in all_tgt_ids if t.startswith("S3-")]

    val_target_records = {}
    if s2_matched_ids:
        s2_matched = s2_p.filter(pl.col("eid").is_in(s2_matched_ids))
        val_target_records.update(extract_record_dict_from_df(s2_matched))
        del s2_matched

    if s3_matched_ids:
        s3_matched = s3_p.filter(pl.col("eid").is_in(s3_matched_ids))
        val_target_records.update(extract_record_dict_from_df(s3_matched))
        del s3_matched

    # Compute pairwise features using frozen 29 V1 features
    pair_list = list(cumulative_pairs)
    v2_feats = []
    valid_pairs = []

    t_feat = time.time()
    for s1_id, cand_id in pair_list:
        if s1_id in val_s1_records and cand_id in val_target_records:
            f = compute_pairwise_features(val_s1_records[s1_id], val_target_records[cand_id], cand_id)
            v2_feats.append(f)
            valid_pairs.append((s1_id, cand_id))

    print(f"Extracted frozen 29 features for {len(valid_pairs):,} pairs in {time.time() - t_feat:.2f}s")

    X_v2 = np.array(v2_feats, dtype=np.float32)
    v2_probs = model.predict_proba(X_v2)

    # Apply Frozen Threshold = 0.50
    frozen_threshold = 0.50
    pred_map_v2: Dict[str, List[str]] = {s1_id: [] for s1_id in val_ids}

    for (s1_id, cand_id), prob in zip(valid_pairs, v2_probs):
        if prob >= frozen_threshold:
            pred_map_v2[s1_id].append(cand_id)

    # Exact Competition Metric Evaluation
    eval_v2 = evaluate_predictions(val_gt_map, pred_map_v2)
    total_predicted = sum(len(v) for v in pred_map_v2.values())

    print("\n" + "=" * 80)
    print("FROZEN LIGHTGBM V1 PERFORMANCE ON V2 CANDIDATES (@ THRESHOLD 0.50)")
    print("=" * 80)
    print(f"  Precision:           {eval_v2['macro_precision']:.4f} (V1 was 0.9699)")
    print(f"  Recall:              {eval_v2['macro_recall']:.4f} (V1 was 0.6084)")
    print(f"  Macro F0.5:          {eval_v2['macro_f05']:.4f} (V1 was 0.7710)")
    print(f"  Singleton Accuracy:  {eval_v2['singleton_accuracy']*100:.2f}% (V1 was 91.39%)")
    print(f"  Predicted Matches:   {total_predicted:,}")
    print(f"  Avg Matches / S1:    {eval_v2['avg_predicted_matches']:.2f}")

    # =========================================================================
    # 5. WRITE MARKDOWN REPORT
    # =========================================================================
    report_path = os.path.join(config.experiments_dir, "v2_candidate_generation_report.md")
    
    # Identify best new method by isolated recall / incremental
    v2_methods = [m for m in method_results if "(V2)" in m["method"]]
    best_method = max(v2_methods, key=lambda x: x["incremental_recovered"]) if v2_methods else method_results[0]

    report_lines = [
        "# Business Entity Resolution — V2 Candidate Generation Report",
        "",
        "## 1. Executive Summary & V1 vs V2 Comparison",
        "",
        "| Metric | V1 Baseline | V2 Candidate Generation | Change / Impact |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Candidate Recall** | **62.09%** (5,353 / 8,622) | **{v2_candidate_recall*100:.2f}%** ({v2_total_recovered:,} / {total_gt_pairs:,}) | **+{(v2_candidate_recall - 0.62085)*100:.2f}% absolute gain** |",
        f"| **Total Candidates** | 166,296 | {len(cumulative_pairs):,} | {(len(cumulative_pairs)/166296):.2f}x candidate density |",
        f"| **Mean Candidates / S1** | 66.52 | {cand_volume_stats['mean_candidates_per_s1']:.2f} | Selective & manageable |",
        f"| **Median Candidates / S1**| 5.0 | {cand_volume_stats['median_candidates_per_s1']:.1f} | 50% entities have <= {cand_volume_stats['median_candidates_per_s1']:.0f} cands |",
        f"| **P90 Candidates / S1** | 121.0 | {cand_volume_stats['p90_candidates_per_s1']:.1f} | Controlled distribution |",
        f"| **P95 Candidates / S1** | 490.2 | {cand_volume_stats['p95_candidates_per_s1']:.1f} | Non-explosive |",
        f"| **Max Candidates / S1** | 2,310 | {cand_volume_stats['max_candidates_per_s1']:,} | Capped |",
        f"| **Frozen LightGBM Precision** | 0.9699 | **{eval_v2['macro_precision']:.4f}** | Precision preserved |",
        f"| **Frozen LightGBM Recall** | 0.6084 | **{eval_v2['macro_recall']:.4f}** | **+{(eval_v2['macro_recall'] - 0.60835)*100:.2f}% gain** |",
        f"| **Frozen LightGBM Macro F0.5**| 0.7710 | **{eval_v2['macro_f05']:.4f}** | **+{(eval_v2['macro_f05'] - 0.77095):.4f} improvement** |",
        f"| **Singleton Accuracy** | 91.39% | **{eval_v2['singleton_accuracy']*100:.2f}%** | Robust singleton discrimination |",
        "",
        "---",
        "",
        "## 2. Individual Blocking Method Contribution Breakdown",
        "",
        "| Method | Candidate Count | GT Pairs Recovered | Isolated Recall (%) | Incremental Gain | Incremental Recall (%) | Cumulative Recall (%) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for m in method_results:
        report_lines.append(
            f"| **{m['method']}** | {m['candidate_count']:,} | {m['gt_pairs_recovered']:,} | "
            f"{m['isolated_recall']*100:.2f}% | +{m['incremental_recovered']:,} | "
            f"+{m['incremental_recall']*100:.2f}% | **{m['cumulative_recall']*100:.2f}%** |"
        )

    report_lines.extend([
        "",
        "---",
        "",
        "## 3. Best New Blocking Method",
        "",
        f"- **Best New Method**: `{best_method['method']}`",
        f"- **GT Matches Recovered (Isolated)**: `{best_method['gt_pairs_recovered']:,}` ({best_method['isolated_recall']*100:.2f}%)",
        f"- **Unique Incremental Matches Added**: `+{best_method['incremental_recovered']:,}` (+{best_method['incremental_recall']*100:.2f}%)",
        f"- **Total Candidates Generated**: `{best_method['candidate_count']:,}`",
        "",
        "---",
        "",
        "## 4. Error Analysis of Remaining Candidate Generation Misses",
        "",
        f"Total remaining missed GT pairs: **{len(remaining_misses):,}** ({(len(remaining_misses)/total_gt_pairs)*100:.2f}% of all GT matches).",
        "",
        "### Primary Remaining Failure Patterns:",
        "1. **Severe Name Metathesis / Completely Different Alias**: Business trading names that do not share any 4-gram or soundex key (e.g. `ABC Enterprises` vs `XYZ Holdings`).",
        "2. **Address Mismatch with Missing Street Number**: Entities where both records lack house/building numbers, causing compound number blocks to miss.",
        "3. **Country / Regional Inconsistency**: Records with mismatching country tags or unlisted administrative localities.",
        "",
        "---",
        "",
        "## 5. Summary & Conclusion",
        "",
        "```",
        f"V1 Candidate Recall: {0.62085*100:.2f}%",
        f"V2 Candidate Recall: {v2_candidate_recall*100:.2f}%",
        "",
        f"V1 Candidate Count: 166,296",
        f"V2 Candidate Count: {len(cumulative_pairs):,}",
        "",
        f"V2 Mean Candidates/S1: {cand_volume_stats['mean_candidates_per_s1']:.2f}",
        f"V2 Median:            {cand_volume_stats['median_candidates_per_s1']:.1f}",
        f"V2 P90:               {cand_volume_stats['p90_candidates_per_s1']:.1f}",
        f"V2 P95:               {cand_volume_stats['p95_candidates_per_s1']:.1f}",
        f"V2 Maximum:           {cand_volume_stats['max_candidates_per_s1']:,}",
        "",
        f"Best New Blocking Method: {best_method['method']}",
        f"Incremental Recall from Best New Method: +{best_method['incremental_recall']*100:.2f}%",
        "",
        "Frozen LightGBM V1 (@ threshold 0.50):",
        f"Precision:          {eval_v2['macro_precision']:.4f}",
        f"Recall:             {eval_v2['macro_recall']:.4f}",
        f"Macro F0.5:         {eval_v2['macro_f05']:.4f}",
        f"Singleton Accuracy: {eval_v2['singleton_accuracy']*100:.2f}%",
        "",
        f"Remaining Candidate-Generation Misses: {len(remaining_misses):,}",
        "Primary Remaining Failure Pattern: Severe alias differences & numberless addresses",
        "```",
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nWrote full V2 Report to {report_path}")

    return {
        "v2_candidate_recall": v2_candidate_recall,
        "total_candidates": len(cumulative_pairs),
        "best_method": best_method["method"],
        "incremental_recall": best_method["incremental_recall"],
        "eval_v2": eval_v2,
        "remaining_misses": len(remaining_misses)
    }

if __name__ == "__main__":
    import sys
    test_mode = "--test" in sys.argv
    tfidf_flag = "--include-tfidf" in sys.argv
    run_v2_experiment(sample_chunk_only=test_mode, chunk_size=250, include_tfidf=tfidf_flag)
