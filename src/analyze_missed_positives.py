"""
Deep-Dive Missed-Positive Diagnostic & Categorization Engine.
Analyzes every missed ground-truth pair, computes exact similarity features,
and categorizes them by linguistic / structural variation.

Outputs:
- reports/missed_positive_analysis.md
"""

import os
import sys
import gc
import json
import time
import argparse
from typing import Dict, List, Set, Tuple, Any
from collections import Counter
from rapidfuzz.distance import Levenshtein, JaroWinkler

from src.config import get_config
from src.data_loader import load_source_file, load_ground_truth
from src.duckdb_indexer import DuckDBTargetIndexer
from src.blocking_v2 import add_v2_blocking_columns
from src.normalize import normalize_text, offline_transliterate

def analyze_missed(s1_count: int = 1000, max_cands: int = 100):
    config = get_config()
    print("=" * 80)
    print("ANTIGRAVITY V3 — MISSED-POSITIVE AUDIT & CATEGORIZATION")
    print("=" * 80)

    # 1. Load Ground Truth and Sample
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
    print(f"Auditing {len(eval_s1_ids):,} S1 Entities ({total_gt_pairs:,} GT Pairs)...")

    # 2. Query DuckDB Candidates
    indexer = DuckDBTargetIndexer(memory_limit="2GB", threads=2)
    source_configs = [
        ("Train S2", config.train_s2_path, "S2-"),
        ("Train S3", config.train_s3_path, "S3-")
    ]
    indexer.build_index_from_sources(source_configs, chunk_size=100000)

    s1_full_df = load_source_file(config.train_s1_path, expected_prefix="S1-")
    s1_eval_df = s1_full_df.filter(pl.col("entity_id").is_in(eval_s1_ids))
    del s1_full_df

    cands_result = indexer.query_candidates_for_s1(s1_eval_df, max_cands_per_s1=max_cands)

    # 3. Identify Missed Pairs
    recovered_pairs = set()
    for s1_id in eval_s1_ids:
        cand_list = cands_result.get(s1_id, [])
        for tid, mask, rec in cand_list:
            if (s1_id, tid) in gt_pairs_set:
                recovered_pairs.add((s1_id, tid))

    missed_pairs = gt_pairs_set - recovered_pairs
    recall_pct = (len(recovered_pairs) / total_gt_pairs * 100.0) if total_gt_pairs > 0 else 0.0
    print(f"\nCandidate Retrieval Recall: {recall_pct:.2f}% ({len(recovered_pairs):,} recovered, {len(missed_pairs):,} missed)")

    # 4. Fetch details for missed target records
    missed_target_ids = list(set(t for s, t in missed_pairs))
    target_details = {}

    if missed_target_ids:
        target_rows = indexer.conn.execute(f"""
            SELECT eid, country, norm_name, compact_name, norm_addr
            FROM targets
            WHERE eid IN {tuple(missed_target_ids) if len(missed_target_ids) > 1 else f"('{missed_target_ids[0]}')"};
        """).fetchall()

        for tid, ctry, n_name, c_name, n_addr in target_rows:
            target_details[tid] = {
                "eid": tid,
                "country": ctry or "",
                "norm_name": n_name or "",
                "compact_name": c_name or "",
                "norm_addr": n_addr or ""
            }

    s1_details = {}
    for row in s1_eval_df.iter_rows(named=True):
        s1_id = row["entity_id"]
        s1_details[s1_id] = {
            "name": row.get("business_name") or "",
            "addr": row.get("business_address") or "",
            "country": row.get("country") or "",
            "norm_name": normalize_text(row.get("business_name") or ""),
            "norm_addr": normalize_text(row.get("business_address") or "")
        }

    # 5. Categorize Missed Pairs
    category_counts = Counter()
    missed_examples = []

    for s1_id, tgt_id in missed_pairs:
        s1 = s1_details.get(s1_id, {})
        tgt = target_details.get(tgt_id, {})

        s1_n = s1.get("norm_name", "")
        tgt_n = tgt.get("norm_name", "")
        s1_a = s1.get("norm_addr", "")
        tgt_a = tgt.get("norm_addr", "")
        s1_c = str(s1.get("country", "")).upper()
        tgt_c = str(tgt.get("country", "")).upper()

        name_lev = Levenshtein.normalized_similarity(s1_n, tgt_n) if (s1_n or tgt_n) else 0.0
        addr_lev = Levenshtein.normalized_similarity(s1_a, tgt_a) if (s1_a or tgt_a) else 0.0

        cats = []
        if s1_c and tgt_c and s1_c != tgt_c:
            cats.append("Country Mismatch")
        elif name_lev < 0.40 and addr_lev > 0.60:
            cats.append("Extreme Name Variation (Strong Address)")
        elif name_lev > 0.70 and addr_lev < 0.30:
            cats.append("Strong Name (Noisy Address)")
        elif name_lev < 0.50 and addr_lev < 0.50:
            cats.append("Severe Lexical Distortion")
        elif any(w in s1_n or w in tgt_n for w in ["pvt", "ltd", "inc", "corp", "sarl", "sas"]):
            cats.append("Legal Suffix / Word Order Variation")
        else:
            cats.append("Token / Spelling Variation")

        for c in cats:
            category_counts[c] += 1

        if len(missed_examples) < 25:
            missed_examples.append({
                "s1_id": s1_id,
                "target_id": tgt_id,
                "s1_name": s1.get("name", ""),
                "target_name": tgt.get("norm_name", ""),
                "s1_addr": s1.get("addr", ""),
                "target_addr": tgt.get("norm_addr", ""),
                "name_lev": name_lev,
                "addr_lev": addr_lev,
                "category": ", ".join(cats)
            })

    indexer.close()

    # 6. Generate Markdown Report
    report_path = os.path.join(config.reports_dir, "missed_positive_analysis.md")
    generate_missed_report(report_path, total_gt_pairs, len(recovered_pairs), len(missed_pairs), category_counts, missed_examples)

def generate_missed_report(
    path: str,
    total_gt: int,
    recovered: int,
    missed_cnt: int,
    categories: Counter,
    examples: List[Dict[str, Any]]
):
    md = f"""# V3 Missed-Positive Diagnostic & Root Cause Report

**Total True Ground Truth Pairs Audited**: {total_gt:,}  
**Recovered by Blocker Ensemble**: {recovered:,} ({recovered/total_gt*100:.2f}%)  
**Missed Ground Truth Pairs**: {missed_cnt:,} ({missed_cnt/total_gt*100:.2f}%)  

---

## 1. Missed-Positive Failure Mode Breakdown

| Category / Failure Mode | Missed Pairs Count | Percentage of Total Misses |
| :--- | :---: | :---: |
"""
    for cat, count in categories.most_common():
        pct = (count / missed_cnt * 100.0) if missed_cnt > 0 else 0.0
        md += f"| **{cat}** | {count:,} | {pct:.1f}% |\n"

    md += """
---

## 2. Sample Missed Ground-Truth Pairs (First 20 Samples)

| S1 ID | Target ID | S1 Business Name | Target Name | S1 Address | Target Address | Name Sim | Addr Sim | Diagnostic Category |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :--- |
"""
    for ex in examples:
        md += f"| `{ex['s1_id']}` | `{ex['target_id']}` | {ex['s1_name'][:30]} | {ex['target_name'][:30]} | {ex['s1_addr'][:30]} | {ex['target_addr'][:30]} | {ex['name_lev']:.2f} | {ex['addr_lev']:.2f} | {ex['category']} |\n"

    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\n[Report] Saved missed-positive analysis to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-count", type=int, default=1000)
    parser.add_argument("--max-cands", type=int, default=100)
    args = parser.parse_args()
    analyze_missed(s1_count=args.s1_count, max_cands=args.max_cands)
