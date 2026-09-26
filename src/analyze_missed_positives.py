"""
Deep-Dive Missed-Positive Diagnostic & Categorization Engine.
Analyzes every missed ground-truth pair, computes exact similarity metrics,
and categorizes them by linguistic / structural variation across 15 standard categories.

Outputs:
- reports/missed_positive_analysis.md
"""

import os
import sys
import gc
import json
import time
import re
import argparse
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import Counter, defaultdict
import polars as pl
from rapidfuzz.distance import Levenshtein, JaroWinkler

from src.config import get_config
from src.data_loader import load_source_file, load_ground_truth
from src.duckdb_indexer import DuckDBTargetIndexer
from src.blocking_v2 import add_v2_blocking_columns, CORP_STOPWORDS, compute_soundex
from src.normalize import normalize_text, offline_transliterate

def extract_tokens(text: str) -> List[str]:
    return [t for t in re.sub(r"[^\w\s]", " ", text.lower()).split() if t]

def extract_informative_tokens(text: str) -> Set[str]:
    return set(t for t in extract_tokens(text) if t not in CORP_STOPWORDS and len(t) >= 2)

def extract_ngrams(text: str, n: int = 3) -> Set[str]:
    cleaned = re.sub(r"\s+", "", text.lower())
    if len(cleaned) < n:
        return {cleaned} if cleaned else set()
    return set(cleaned[i:i+n] for i in range(len(cleaned) - n + 1))

def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def extract_digits(text: str) -> List[str]:
    return re.findall(r"\b\d+\b", text)

def analyze_missed(s1_count: int = 1000, max_cands: int = 100):
    config = get_config()
    print("=" * 80)
    print("ANTIGRAVITY V3.1 — DEEP-DIVE MISSED-POSITIVE AUDIT & CATEGORIZATION")
    print("=" * 80)

    # 1. Load Ground Truth and Sample S1
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

    # 2. Query DuckDB Candidates from Persistent Cache
    indexer = DuckDBTargetIndexer(memory_limit="2GB", threads=2)
    expected_sources = [config.train_s2_path, config.train_s3_path]
    manifest = indexer.ensure_cache_ready(expected_sources if all(os.path.exists(p) for p in expected_sources) else None)

    s1_full_df = load_source_file(config.train_s1_path, expected_prefix="S1-")
    s1_eval_df = s1_full_df.filter(pl.col("entity_id").is_in(eval_s1_ids))
    del s1_full_df

    t0_q = time.time()
    cands_result = indexer.query_candidates_for_s1(s1_eval_df, max_cands_per_s1=max_cands)
    query_time = time.time() - t0_q
    print(f"Candidate query complete in {query_time:.2f}s.")

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
        # Fetch in batches if necessary
        batch_size = 5000
        for i in range(0, len(missed_target_ids), batch_size):
            chunk_ids = missed_target_ids[i:i+batch_size]
            t_rows = indexer.conn.execute(f"""
                SELECT eid, country, norm_name, compact_name, norm_addr
                FROM targets
                WHERE eid IN {tuple(chunk_ids) if len(chunk_ids) > 1 else f"('{chunk_ids[0]}')"};
            """).fetchall()

            for tid, ctry, n_name, c_name, n_addr in t_rows:
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

    # 5. Fine-Grained Categorization across 15 Formal Categories
    category_counts = Counter()
    missed_diagnostics = []

    for s1_id, tgt_id in missed_pairs:
        s1 = s1_details.get(s1_id, {})
        tgt = target_details.get(tgt_id, {})

        s1_n = s1.get("norm_name", "")
        tgt_n = tgt.get("norm_name", "")
        s1_a = s1.get("norm_addr", "")
        tgt_a = tgt.get("norm_addr", "")
        s1_c = str(s1.get("country", "")).upper()
        tgt_c = str(tgt.get("country", "")).upper()

        if not tgt:
            category_counts["O. Possible implementation/indexing issue"] += 1
            continue

        name_lev = Levenshtein.normalized_similarity(s1_n, tgt_n) if (s1_n or tgt_n) else 0.0
        name_jw = JaroWinkler.similarity(s1_n, tgt_n) if (s1_n or tgt_n) else 0.0
        addr_lev = Levenshtein.normalized_similarity(s1_a, tgt_a) if (s1_a or tgt_a) else 0.0

        s1_toks = extract_informative_tokens(s1_n)
        tgt_toks = extract_informative_tokens(tgt_n)
        tok_jaccard = jaccard_similarity(s1_toks, tgt_toks)

        s1_ngrams = extract_ngrams(s1_n, 3)
        tgt_ngrams = extract_ngrams(tgt_n, 3)
        ngram_jaccard = jaccard_similarity(s1_ngrams, tgt_ngrams)

        s1_nums = extract_digits(s1_a)
        tgt_nums = extract_digits(tgt_a)
        has_common_num = bool(set(s1_nums) & set(tgt_nums))

        # Check transliteration & soundex
        s1_trans = offline_transliterate(s1_n)
        tgt_trans = offline_transliterate(tgt_n)
        trans_lev = Levenshtein.normalized_similarity(s1_trans, tgt_trans) if (s1_trans or tgt_trans) else 0.0

        s1_snd = compute_soundex(extract_tokens(s1_trans)[0]) if extract_tokens(s1_trans) else ""
        tgt_snd = compute_soundex(extract_tokens(tgt_trans)[0]) if extract_tokens(tgt_trans) else ""
        soundex_match = bool(s1_snd and tgt_snd and s1_snd == tgt_snd)

        # Categorization Decision Logic
        cat = "N. Other"
        if not s1_n or not tgt_n:
            cat = "M. Missing name information"
        elif s1_c and tgt_c and s1_c != tgt_c:
            cat = "J. Postal / Country missing or different"
        elif s1_toks and tgt_toks and s1_toks == tgt_toks and s1_n != tgt_n:
            cat = "C. Word reordering"
        elif any(len(t) <= 3 and t in "".join(w[0] for w in tgt_n.split()) for t in s1_n.split()):
            cat = "A. Name abbreviation"
        elif (s1_trans != s1_n or tgt_trans != tgt_n) and trans_lev > 0.75 and name_lev < 0.60:
            cat = "F. Transliteration/script variation"
        elif soundex_match and name_lev < 0.60:
            cat = "G. Phonetic variation"
        elif tok_jaccard >= 0.50 and name_lev < 0.70:
            cat = "D. Token variation"
        elif name_lev >= 0.70 and name_lev < 1.0:
            cat = "E. Character typo/edit variation"
        elif addr_lev >= 0.60 and name_lev < 0.40:
            cat = "H. Address-driven match (High address similarity, low name)"
        elif s1_nums and tgt_nums and not has_common_num and name_lev > 0.70:
            cat = "I. Address number variation"
        elif any(w in s1_n or w in tgt_n for w in ["pvt", "ltd", "inc", "corp", "sarl", "sas", "llc", "gmbh"]):
            cat = "B. Legal suffix variation"
        elif name_lev < 0.30 and addr_lev < 0.30:
            cat = "L. Alias / substantially different name"
        else:
            cat = "K. Common/shared business name"

        category_counts[cat] += 1

        if len(missed_diagnostics) < 40:
            missed_diagnostics.append({
                "s1_id": s1_id,
                "target_id": tgt_id,
                "s1_name": s1.get("name", ""),
                "target_name": tgt.get("norm_name", ""),
                "s1_addr": s1.get("addr", ""),
                "target_addr": tgt.get("norm_addr", ""),
                "name_lev": name_lev,
                "addr_lev": addr_lev,
                "tok_jaccard": tok_jaccard,
                "ngram_jaccard": ngram_jaccard,
                "category": cat
            })

    indexer.close()

    # 6. Generate Markdown Report
    report_path = os.path.join(config.reports_dir, "missed_positive_analysis.md")
    generate_missed_report(report_path, total_gt_pairs, len(recovered_pairs), len(missed_pairs), category_counts, missed_diagnostics)

def generate_missed_report(
    path: str,
    total_gt: int,
    recovered: int,
    missed_cnt: int,
    categories: Counter,
    examples: List[Dict[str, Any]]
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    md = f"""# V3.1 Missed-Positive Diagnostic & Root Cause Report

**Total True Ground Truth Pairs Audited**: {total_gt:,}  
**Recovered by Blocker Ensemble**: {recovered:,} ({recovered/total_gt*100:.2f}%)  
**Missed Ground Truth Pairs**: {missed_cnt:,} ({missed_cnt/total_gt*100:.2f}%)  

---

## 1. Missed-Positive Failure Mode Breakdown (15 Standard Categories)

| Category / Failure Mode | Missed Pairs Count | % of Misses | Primary Recovery Mechanism |
| :--- | :---: | :---: | :--- |
"""
    recovery_recommendations = {
        "A. Name abbreviation": "Prefix-4 / N-Gram Retrieval",
        "B. Legal suffix variation": "Expanded Corporate Suffix Stripping",
        "C. Word reordering": "Token-Set / Sorted-Token Signature",
        "D. Token variation": "Rare-Token / Informative-Token Index",
        "E. Character typo/edit variation": "Character N-Gram Overlap & RapidFuzz",
        "F. Transliteration/script variation": "Multilingual Bidirectional Cross-Match",
        "G. Phonetic variation": "Phonetic Soundex Cross-Transliteration",
        "H. Address-driven match (High address similarity, low name)": "Address-First Retrieval (Street + Num)",
        "I. Address number variation": "Name-First Fuzzy Matching (Postal/City)",
        "J. Postal / Country missing or different": "Country-Agnostic Lexical Fallback",
        "K. Common/shared business name": "Compound Name + Address Token Matching",
        "L. Alias / substantially different name": "Arctic Embedding Semantic Retrieval",
        "M. Missing name information": "Pure Address-First Blocking",
        "N. Other": "Fuzzy Ensemble Retrieval",
        "O. Possible implementation/indexing issue": "Target Cache Integrity Audit"
    }

    for cat, count in categories.most_common():
        pct = (count / missed_cnt * 100.0) if missed_cnt > 0 else 0.0
        rec = recovery_recommendations.get(cat, "Lexical / Semantic Expansion")
        md += f"| **{cat}** | {count:,} | {pct:.1f}% | {rec} |\n"

    md += """
---

## 2. Sample Missed Ground-Truth Pairs Audit (First 30 Samples)

| S1 ID | Target ID | S1 Business Name | Target Name | S1 Address | Target Address | Name Sim | Addr Sim | N-Gram | Diagnostic Category |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
"""
    for ex in examples:
        md += f"| `{ex['s1_id']}` | `{ex['target_id']}` | {ex['s1_name'][:25]} | {ex['target_name'][:25]} | {ex['s1_addr'][:25]} | {ex['target_addr'][:25]} | {ex['name_lev']:.2f} | {ex['addr_lev']:.2f} | {ex['ngram_jaccard']:.2f} | {ex['category']} |\n"

    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\n[Report SUCCESS] Saved missed-positive analysis to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-count", type=int, default=1000)
    parser.add_argument("--max-cands", type=int, default=100)
    args = parser.parse_args()
    analyze_missed(s1_count=args.s1_count, max_cands=args.max_cands)
