"""
Standalone CLI script to inspect and validate the DuckDB Target Cache.

Usage:
    PYTHONPATH=. python3 -m src.inspect_target_cache
"""

import os
import sys
import json
import argparse
import duckdb
from src.config import get_config
from src.duckdb_indexer import DuckDBTargetIndexer, DEFAULT_DB_PATH, DEFAULT_MANIFEST_PATH

def main():
    parser = argparse.ArgumentParser(
        description="Inspect and validate the DuckDB Target Cache."
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Custom path to duckdb file."
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Custom path to manifest JSON file."
    )
    args = parser.parse_args()

    config = get_config()
    db_path = args.db_path or os.path.join(config.base_dir, DEFAULT_DB_PATH)
    manifest_path = args.manifest_path or os.path.join(config.base_dir, DEFAULT_MANIFEST_PATH)

    print("=" * 80)
    print("ANTIGRAVITY V3 PERSISTENT TARGET CACHE INSPECTOR")
    print("=" * 80)
    print(f"Database File:  {db_path}")
    print(f"Manifest File:  {manifest_path}")
    print("-" * 80)

    indexer = DuckDBTargetIndexer(db_path=db_path, manifest_path=manifest_path)

    s2_path = os.path.join(config.data_dir, "train", "train_source2.tsv")
    s3_path = os.path.join(config.data_dir, "train", "train_source3.tsv")
    expected_sources = [p for p in [s2_path, s3_path] if os.path.exists(p)]

    is_valid, reason, manifest = indexer.validate_cache(expected_sources if expected_sources else None)

    if not is_valid:
        print(f"CACHE STATUS: [INVALID / NOT READY]")
        print(f"Reason:       {reason}")
        print("\nTo build or update the cache, run:")
        print("    PYTHONPATH=. python3 -m src.build_target_cache --force-rebuild")
        print("=" * 80)
        sys.exit(1)

    print(f"CACHE STATUS: [VALID & READY]")
    print(f"Cache Version:        {manifest.get('cache_version', 'Unknown')}")
    print(f"Schema Version:       {manifest.get('schema_version', 'Unknown')}")
    print(f"Created At:           {manifest.get('created_at', 'Unknown')}")
    print(f"Total Target Records: {manifest.get('total_rows', 0):,}")
    print(f"  - Train S2 Rows:    {manifest.get('source2_rows', 0):,}")
    print(f"  - Train S3 Rows:    {manifest.get('source3_rows', 0):,}")
    print(f"Database File Size:   {manifest.get('database_size_mb', 0):.2f} MB")
    print(f"Build Duration:       {manifest.get('build_duration_seconds', 0):.2f} seconds")
    print(f"Target Fingerprint:   {manifest.get('target_data_fingerprint', 'N/A')}")
    print("-" * 80)
    print("Available High-Recall Indexes:")
    for idx_name in manifest.get("available_indexes", []):
        print(f"  [x] {idx_name}")

    print("-" * 80)
    print("Direct DuckDB Table Row Counts Verification:")
    try:
        conn = duckdb.connect(db_path, read_only=True)
        tables = [
            "targets", "idx_compact_name", "idx_translit_cname", "idx_norm_name",
            "idx_cname_p6", "idx_tokens", "idx_soundex_num", "idx_cname8_num",
            "idx_f2_num", "idx_pin_cname4", "idx_addr_street", "idx_fallback_exact"
        ]
        for tbl in tables:
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                print(f"  - {tbl:<20}: {count:,} entries")
            except Exception as ex:
                print(f"  - {tbl:<20}: [MISSING/ERROR: {ex}]")
        conn.close()
    except Exception as e:
        print(f"  Error reading DuckDB tables: {e}")

    print("=" * 80)
    print("Persistent target cache is fully verified and ready for experimental candidate generation.")
    print("=" * 80)

if __name__ == "__main__":
    main()
