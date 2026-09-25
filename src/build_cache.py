"""
V3 Disk-Backed Record Cache & Index Builder.
Precomputes and serializes normalized records with compact integer IDs for S1, S2, and S3.
"""

import os
import sys
sys.path.insert(0, ".")
import gc
import time
import json
from typing import Dict, List, Any
import polars as pl

from src.config import get_config
from src.data_loader import load_source_file
from src.normalize import normalize_text, normalize_address, compact_name, offline_transliterate

def build_v3_cache():
    print("=" * 80)
    print("STARTING V3 RECORD PREPROCESSING & DISK CACHE BUILD")
    print("=" * 80)

    config = get_config()
    cache_dir = os.path.join(config.base_dir, "cache")
    norm_cache_dir = os.path.join(cache_dir, "normalized")
    os.makedirs(norm_cache_dir, exist_ok=True)

    sources = [
        ("Train S1", config.train_s1_path, "S1-", "train_s1_norm.parquet"),
        ("Train S2", config.train_s2_path, "S2-", "train_s2_norm.parquet"),
        ("Train S3", config.train_s3_path, "S3-", "train_s3_norm.parquet"),
    ]

    for label, path, prefix, out_file in sources:
        if not os.path.exists(path):
            print(f"Skipping {label} (not found: {path})")
            continue
        print(f"\nProcessing and caching {label} ({path})...")
        t0 = time.time()
        df = load_source_file(path, expected_prefix=prefix)
        
        # Add normalized columns
        df_p = df.with_columns([
            pl.col("entity_id").alias("eid"),
            pl.col("business_name").fill_null("").str.to_lowercase().str.replace_all(r"[^\w\s]", " ").str.replace_all(r"\s+", " ").str.strip_chars().alias("norm_name"),
            pl.col("business_address").fill_null("").str.to_lowercase().str.replace_all(r"[^\w\s]", " ").str.replace_all(r"\s+", " ").str.strip_chars().alias("norm_addr"),
            pl.col("country").fill_null("").str.to_uppercase().str.strip_chars().alias("country"),
        ])
        
        out_path = os.path.join(norm_cache_dir, out_file)
        df_p.write_parquet(out_path)
        print(f"Saved {len(df_p):,} normalized records to {out_path} in {time.time() - t0:.2f}s")

    print("\n" + "=" * 80)
    print("V3 CACHE BUILD COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    build_v3_cache()
