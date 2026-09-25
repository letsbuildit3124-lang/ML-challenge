"""
High-Speed Vectorized Blocking & Candidate Generation Module using Polars.
"""

import time
from typing import Dict, List, Set, Tuple, Any, Optional
import polars as pl
import numpy as np

def add_blocking_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes vectorized normalized blocking attributes with compound name+number keys.
    """
    return df.with_columns([
        pl.col("entity_id").alias("eid"),
        pl.col("business_name").fill_null("").str.to_lowercase().str.replace_all(r"[^\w\s]", " ").str.replace_all(r"\s+", " ").str.strip_chars().alias("norm_name"),
        pl.col("business_address").fill_null("").str.to_lowercase().str.replace_all(r"[^\w\s]", " ").str.replace_all(r"\s+", " ").str.strip_chars().alias("norm_addr"),
        pl.col("country").fill_null("").str.to_uppercase().str.strip_chars().alias("country"),
    ]).with_columns([
        pl.col("norm_name").str.replace_all(
            r"\b(ltd|limited|pvt|private|corp|corporation|inc|incorporated|llc|llp|co|company|gmbh|sa|sarl|plc|bv|nv|assoc|associates|group|holdings|enterprises|services|solutions|technologies|international|consultants|industries|global|systems)\b",
            ""
        ).str.replace_all(r"\s+", "").alias("compact_name"),
        pl.col("norm_name").str.split(" ").list.slice(0, 2).list.join("_").alias("f2_name"),
        pl.col("norm_addr").str.extract(r"(\d+)", 1).alias("first_addr_num"),
    ]).with_columns([
        pl.when(pl.col("first_addr_num").is_not_null()).then(
            pl.concat_str([pl.col("compact_name").str.slice(0, 8), pl.lit("_"), pl.col("first_addr_num")])
        ).otherwise(None).alias("cname8_num"),
        pl.when(pl.col("first_addr_num").is_not_null()).then(
            pl.concat_str([pl.col("f2_name"), pl.lit("_"), pl.col("first_addr_num")])
        ).otherwise(None).alias("f2_num")
    ]).drop(["first_addr_num"])

def generate_candidates_for_targets(
    s1_prep_df: pl.DataFrame,
    target_prep_df: pl.DataFrame,
    max_cands_per_s1: int = 40
) -> pl.DataFrame:
    """
    Generates candidate pairs between S1 and a target source using non-explosive compound keys.
    """
    # 1. Exact compact name + country
    j1 = s1_prep_df.filter(pl.col("compact_name").str.len_chars() >= 3).select(["eid", "compact_name", "country"]).join(
        target_prep_df.filter(pl.col("compact_name").str.len_chars() >= 3).select(["eid", "compact_name", "country"]),
        on=["compact_name", "country"]
    ).select([pl.col("eid").alias("s1_id"), pl.col("eid_right").alias("target_id")])

    # 2. Exact normalized name + country
    j2 = s1_prep_df.filter(pl.col("norm_name").str.len_chars() >= 4).select(["eid", "norm_name", "country"]).join(
        target_prep_df.filter(pl.col("norm_name").str.len_chars() >= 4).select(["eid", "norm_name", "country"]),
        on=["norm_name", "country"]
    ).select([pl.col("eid").alias("s1_id"), pl.col("eid_right").alias("target_id")])

    # 3. Compact Name Prefix (8) + First Address Number + Country
    j3 = s1_prep_df.filter(pl.col("cname8_num").is_not_null()).select(["eid", "cname8_num", "country"]).join(
        target_prep_df.filter(pl.col("cname8_num").is_not_null()).select(["eid", "cname8_num", "country"]),
        on=["cname8_num", "country"]
    ).select([pl.col("eid").alias("s1_id"), pl.col("eid_right").alias("target_id")])

    # 4. First 2 Words + First Address Number + Country
    j4 = s1_prep_df.filter(pl.col("f2_num").is_not_null()).select(["eid", "f2_num", "country"]).join(
        target_prep_df.filter(pl.col("f2_num").is_not_null()).select(["eid", "f2_num", "country"]),
        on=["f2_num", "country"]
    ).select([pl.col("eid").alias("s1_id"), pl.col("eid_right").alias("target_id")])

    # Union and limit per S1 entity
    joined = pl.concat([j1, j2, j3, j4]).unique()
    
    # Cap candidate count per S1
    if max_cands_per_s1:
        joined = joined.with_columns(
            pl.int_range(pl.len()).over("s1_id").alias("rank")
        ).filter(pl.col("rank") < max_cands_per_s1).select(["s1_id", "target_id"])

    return joined

def generate_all_candidate_pairs(
    s1_df: pl.DataFrame,
    s2_df: pl.DataFrame,
    s3_df: pl.DataFrame,
    max_cands_per_s1: int = 50
) -> pl.DataFrame:
    """
    Generates candidates from both S2 and S3 for all S1 entities.
    """
    t0 = time.time()
    s1_p = add_blocking_columns(s1_df)
    s2_p = add_blocking_columns(s2_df)
    s3_p = add_blocking_columns(s3_df)

    pairs_s2 = generate_candidates_for_targets(s1_p, s2_p, max_cands_per_s1=max_cands_per_s1 // 2 + 5)
    pairs_s3 = generate_candidates_for_targets(s1_p, s3_p, max_cands_per_s1=max_cands_per_s1 // 2 + 5)

    all_pairs = pl.concat([pairs_s2, pairs_s3]).unique()
    print(f"[Blocking] Generated {len(all_pairs):,} total candidates for {len(s1_df):,} S1 entities in {time.time() - t0:.2f}s (Avg: {len(all_pairs)/max(1, len(s1_df)):.2f}/S1)", flush=True)
    return all_pairs

def evaluate_candidate_recall(
    candidate_pairs_df: pl.DataFrame,
    gt_mapping: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Calculates candidate recall against ground truth mapping.
    """
    cand_map: Dict[str, Set[str]] = {}
    for row in candidate_pairs_df.iter_rows():
        s1, t = str(row[0]), str(row[1])
        if s1 not in cand_map:
            cand_map[s1] = set()
        cand_map[s1].add(t)

    total_true = sum(len(v) for v in gt_mapping.values())
    found_true = 0
    
    for s1_id, targets in gt_mapping.items():
        found_true += len(set(targets).intersection(cand_map.get(s1_id, set())))

    recall = found_true / total_true if total_true > 0 else 0.0
    print(f"[Candidate Recall] Captured {found_true:,} / {total_true:,} true matches ({recall*100:.2f}% recall)", flush=True)

    return {
        "total_true_matches": total_true,
        "captured_true_matches": found_true,
        "candidate_recall": recall,
        "total_candidates": len(candidate_pairs_df),
        "avg_candidates_per_s1": len(candidate_pairs_df) / max(1, len(gt_mapping))
    }
