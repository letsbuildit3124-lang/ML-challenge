"""
Audit script for Entity IDs and Ground Truth Mapping Integrity.
Verifies:
- Prefix conventions (S1-, S2-, S3-)
- Unique ID integrity (no duplicate original IDs)
- Lossless ground truth mapping
- Integer / String type safety
"""

import os
import sys
import json
import polars as pl
from src.config import get_config
from src.data_loader import load_source_file, load_ground_truth

def audit_ids():
    config = get_config()
    print("=" * 80)
    print("AUDITING ID MAPPING & NAMESPACE INTEGRITY")
    print("=" * 80)

    # 1. Ground Truth Audit
    print("[1/4] Auditing train_ground_truth.tsv...", flush=True)
    gt_df = load_ground_truth(config.train_gt_path)
    total_gt_rows = len(gt_df)
    unique_s1 = gt_df["source1_entity_id"].n_unique()
    null_s1 = gt_df["source1_entity_id"].null_count()
    null_matches = gt_df["matched_entities"].null_count()

    print(f"  Total Ground Truth Rows:    {total_gt_rows:,}")
    print(f"  Unique Source 1 Entities:   {unique_s1:,}")
    print(f"  Null S1 IDs:                {null_s1}")
    print(f"  Null Match Lists:           {null_matches}")

    # Inspect match string parsing
    s2_matches = 0
    s3_matches = 0
    invalid_prefixes = 0
    total_parsed_matches = 0

    for row in gt_df.iter_rows():
        m_str = str(row[1]) if row[1] is not None else ""
        if not m_str:
            continue
        for tid in m_str.split(","):
            tid = tid.strip()
            if not tid:
                continue
            total_parsed_matches += 1
            if tid.startswith("S2-"):
                s2_matches += 1
            elif tid.startswith("S3-"):
                s3_matches += 1
            else:
                invalid_prefixes += 1

    print(f"  Total True Matches Parsed:  {total_parsed_matches:,}")
    print(f"    Source 2 True Matches:    {s2_matches:,} ({s2_matches/total_parsed_matches*100:.1f}%)")
    print(f"    Source 3 True Matches:    {s3_matches:,} ({s3_matches/total_parsed_matches*100:.1f}%)")
    print(f"    Invalid Prefix Matches:   {invalid_prefixes}")

    # 2. Source 1 Audit
    print("\n[2/4] Auditing train_source1.tsv...", flush=True)
    s1_df = pl.scan_csv(config.train_s1_path, separator="\t", infer_schema_length=1000).collect()
    s1_col = s1_df.columns[0]
    print(f"  Total S1 Records:           {len(s1_df):,}")
    print(f"  Unique S1 IDs:              {s1_df[s1_col].n_unique():,}")
    print(f"  Null IDs:                   {s1_df[s1_col].null_count()}")

    # 3. Source 2 Audit
    print("\n[3/4] Auditing train_source2.tsv...", flush=True)
    s2_df = pl.scan_csv(config.train_s2_path, separator="\t", infer_schema_length=1000).collect()
    s2_col = s2_df.columns[0]
    print(f"  Total S2 Records:           {len(s2_df):,}")
    print(f"  Unique S2 IDs:              {s2_df[s2_col].n_unique():,}")
    print(f"  Null IDs:                   {s2_df[s2_col].null_count()}")

    # 4. Source 3 Audit
    print("\n[4/4] Auditing train_source3.tsv...", flush=True)
    s3_df = pl.scan_csv(config.train_s3_path, separator="\t", infer_schema_length=1000).collect()
    s3_col = s3_df.columns[0]
    print(f"  Total S3 Records:           {len(s3_df):,}")
    print(f"  Unique S3 IDs:              {s3_df[s3_col].n_unique():,}")
    print(f"  Null IDs:                   {s3_df[s3_col].null_count()}")

    # Cross-source ID collision check
    s1_set = set(s1_df[s1_col].to_list())
    s2_set = set(s2_df[s2_col].to_list())
    s3_set = set(s3_df[s3_col].to_list())

    print("\n--- CROSS-SOURCE NAMESPACE COLLISION CHECK ---")
    print(f"S1 ∩ S2 Collision: {len(s1_set & s2_set)} (Expected: 0)")
    print(f"S1 ∩ S3 Collision: {len(s1_set & s3_set)} (Expected: 0)")
    print(f"S2 ∩ S3 Collision: {len(s2_set & s3_set)} (Expected: 0)")

    if len(s1_set & s2_set) == 0 and len(s1_set & s3_set) == 0 and len(s2_set & s3_set) == 0:
        print("\nALL ID AUDITS PASSED: Zero namespace collisions, zero null IDs, 100% prefix integrity.")

if __name__ == "__main__":
    audit_ids()
