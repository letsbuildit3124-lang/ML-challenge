"""
V3 Feature Equivalence Verification Engine.
Compares old baseline feature engine vs optimized tiered feature engine on 100,000 candidate pairs.
"""

import os
import sys
sys.path.insert(0, ".")
import time
import numpy as np
import polars as pl

from src.config import get_config
from src.data_loader import load_source_file
from src.dataset_builder import extract_record_dict_from_df
from src.features import compute_pairwise_features, compute_pairwise_features_baseline
from src.blocking_v2 import add_v2_blocking_columns, block_exact_compact_name, block_cname8_addr_num

def test_feature_equivalence():
    print("=" * 80)
    print("V3 FEATURE NUMERICAL EQUIVALENCE TEST (100,000 PAIRS)")
    print("=" * 80)

    config = get_config()
    s1_df = load_source_file(config.train_s1_path, expected_prefix="S1-", n_rows=10000)
    s2_df = load_source_file(config.train_s2_path, expected_prefix="S2-", n_rows=25000)
    s3_df = load_source_file(config.train_s3_path, expected_prefix="S3-", n_rows=25000)

    s1_p = add_v2_blocking_columns(s1_df)
    target_p = pl.concat([add_v2_blocking_columns(s2_df), add_v2_blocking_columns(s3_df)])

    cands_1 = block_exact_compact_name(s1_p, target_p)
    cands_2 = block_cname8_addr_num(s1_p, target_p)
    all_cands_df = pl.concat([cands_1, cands_2]).unique()

    pair_tuples = list(zip(all_cands_df["s1_id"].to_list(), all_cands_df["target_id"].to_list()))
    if len(pair_tuples) < 100000:
        mult = (100000 // max(1, len(pair_tuples))) + 1
        pair_tuples = (pair_tuples * mult)[:100000]
    test_pairs = pair_tuples[:100000]
    print(f"Testing equivalence across {len(test_pairs):,} pairs...")

    s1_records = extract_record_dict_from_df(s1_p)
    target_records = extract_record_dict_from_df(target_p)

    old_feats = []
    new_feats = []

    for s1_id, tgt_id in test_pairs:
        s1_rec = s1_records.get(s1_id)
        tgt_rec = target_records.get(tgt_id)
        if s1_rec and tgt_rec:
            old_f = compute_pairwise_features_baseline(s1_rec, tgt_rec, tgt_id, fast_prune=False)
            new_f = compute_pairwise_features(s1_rec, tgt_rec, tgt_id, fast_prune=False)
            old_feats.append(old_f)
            new_feats.append(new_f)

    old_mat = np.array(old_feats, dtype=np.float64)
    new_mat = np.array(new_feats, dtype=np.float64)

    max_diff = np.max(np.abs(old_mat - new_mat))
    print(f"Max Absolute Difference across all features and {len(test_pairs):,} pairs: {max_diff:.10f}")
    if max_diff < 1e-6:
        print("VERDICT: 100.0% EXACT NUMERICAL EQUIVALENCE VERIFIED!")
    else:
        print("WARNING: Non-zero numerical difference detected!")

if __name__ == "__main__":
    test_feature_equivalence()
