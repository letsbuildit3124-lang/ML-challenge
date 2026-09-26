"""
Standalone CLI script to build and persist the DuckDB Target Cache (Train S2 + Train S3).

Usage:
    PYTHONPATH=. python3 -m src.build_target_cache
    PYTHONPATH=. python3 -m src.build_target_cache --force-rebuild
    PYTHONPATH=. python3 -m src.build_target_cache --chunk-size 100000 --memory-limit 2GB
"""

import os
import sys
import argparse
import time
from src.config import get_config
from src.duckdb_indexer import DuckDBTargetIndexer

def main():
    parser = argparse.ArgumentParser(
        description="Build and persist the DuckDB Target Cache for Train S2 + Train S3."
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild the cache from raw TSV files even if cache already exists."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Chunk size (rows) for streaming TSV ingestion into DuckDB (default: 100,000)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows per source file for testing/dry-runs."
    )
    parser.add_argument(
        "--memory-limit",
        type=str,
        default="2GB",
        help="DuckDB memory limit (e.g., '2GB', '4GB'). Default: '2GB'."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="DuckDB thread count. Default: 2."
    )
    parser.add_argument(
        "--s2-path",
        type=str,
        default=None,
        help="Custom path to train_source2.tsv"
    )
    parser.add_argument(
        "--s3-path",
        type=str,
        default=None,
        help="Custom path to train_source3.tsv"
    )
    args = parser.parse_args()

    config = get_config()
    s2_path = args.s2_path or os.path.join(config.data_dir, "train", "train_source2.tsv")
    s3_path = args.s3_path or os.path.join(config.data_dir, "train", "train_source3.tsv")

    print("=" * 80)
    print("ANTIGRAVITY V3 TARGET CACHE BUILDER")
    print("=" * 80)
    print(f"Source 2 Path:    {s2_path} (exists: {os.path.exists(s2_path)})")
    print(f"Source 3 Path:    {s3_path} (exists: {os.path.exists(s3_path)})")
    print(f"Memory Limit:     {args.memory_limit}")
    print(f"Threads:          {args.threads}")
    print(f"Chunk Size:       {args.chunk_size:,} rows")
    print(f"Limit / Source:   {args.limit if args.limit else 'ALL (Full 10.3M dataset)'}")
    print(f"Force Rebuild:    {args.force_rebuild}")
    print("=" * 80)

    if not os.path.exists(s2_path) and not os.path.exists(s3_path):
        print(f"[ERROR] Neither Source 2 ({s2_path}) nor Source 3 ({s3_path}) exists on disk.")
        sys.exit(1)

    source_specs = []
    if os.path.exists(s2_path):
        source_specs.append(("Train S2", s2_path, "train_s2"))
    if os.path.exists(s3_path):
        source_specs.append(("Train S3", s3_path, "train_s3"))

    indexer = DuckDBTargetIndexer(
        memory_limit=args.memory_limit,
        threads=args.threads
    )

    t0 = time.time()
    try:
        manifest = indexer.build_cache_from_sources(
            source_paths=source_specs,
            chunk_size=args.chunk_size,
            limit_per_file=args.limit,
            force_rebuild=args.force_rebuild
        )
        print("\n" + "=" * 80)
        print("TARGET CACHE BUILD FINISHED SUCCESSFULLY")
        print(f"Database File:    {indexer.db_path}")
        print(f"Manifest File:    {indexer.manifest_path}")
        print(f"Total Targets:    {manifest.get('total_rows', 0):,} records")
        print(f"Time Elapsed:     {time.time() - t0:.2f}s")
        print("=" * 80)
    except Exception as e:
        print(f"\n[FATAL ERROR] Cache construction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        indexer.close()

if __name__ == "__main__":
    main()
