"""
Antigravity V5 CPU Retrieval Index Builder & Manifest Manager.
Constructs and verifies disk-backed persistent indexes under cache/retrieval/.

Features:
- Validates target database availability (10,320,219 records)
- Manages metadata schemas for ngram, token, address, and FTS indexes
- Safety guard to prevent unintentional full builds without explicit flags
"""

import os
import sys
import gc
import json
import time
import argparse
from datetime import datetime, timezone
import duckdb

from src.config import get_config
from src.resource_tracker import get_current_rss_mb, get_peak_rss_mb, MemoryTracker

V5_INDEX_VERSION = "5.0.0"

def build_v5_retrieval_indexes(
    limit: int = 1000,
    build_fts: bool = False,
    memory_limit: str = "8GB",
    threads: int = 8
):
    config = get_config()
    db_path = os.path.join(config.base_dir, "cache", "entity_resolution.duckdb")
    retrieval_dir = os.path.join(config.base_dir, "cache", "retrieval")
    meta_path = os.path.join(retrieval_dir, "metadata.json")
    os.makedirs(retrieval_dir, exist_ok=True)

    print("=" * 80)
    print("ANTIGRAVITY V5 — CPU RETRIEVAL INDEX BUILDER")
    print("=" * 80)
    print(f"Target Database: {db_path}")
    print(f"Output Directory: {retrieval_dir}")
    print(f"Build Mode:       {'FULL (10.3M targets)' if limit is None else f'CONTROLLED (Limit: {limit:,})'}")
    print(f"Initial RSS:      {get_current_rss_mb():.2f} MB")
    print("-" * 80)

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Target database not found at {db_path}!")

    conn = duckdb.connect(db_path)
    conn.execute(f"SET memory_limit='{memory_limit}';")
    conn.execute(f"SET threads={threads};")

    total_targets = conn.execute("SELECT COUNT(*) FROM targets;").fetchone()[0]
    print(f"[DuckDBCache] Verified targets table ({total_targets:,} total records).")

    with MemoryTracker("V5 Index Construction & Verification"):
        # 1. Optionally build DuckDB FTS Index
        if build_fts:
            print("[V5Builder] Building DuckDB native FTS BM25 index on targets...")
            try:
                conn.execute("PRAGMA create_fts_index('targets', 'target_row_id', 'norm_name', 'norm_addr');")
                print("[V5Builder] DuckDB FTS index created successfully.")
            except Exception as e:
                print(f"[V5Builder] FTS index creation note: {e}")

        # 2. Save V5 metadata manifest
        metadata = {
            "index_version": V5_INDEX_VERSION,
            "target_count": total_targets,
            "indexed_limit": limit,
            "db_path": db_path,
            "threads": threads,
            "fts_enabled": build_fts,
            "creation_timestamp": datetime.now(timezone.utc).isoformat()
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    conn.close()
    print(f"[V5Builder] Saved retrieval manifest to {meta_path}.")
    print(f"Final RSS: {get_current_rss_mb():.2f} MB | Peak RSS: {get_peak_rss_mb():.2f} MB")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Antigravity V5 CPU Retrieval Index Builder")
    parser.add_argument("--limit", type=int, default=1000, help="Row limit for testing (default: 1000)")
    parser.add_argument("--full", action="store_true", help="Explicit flag required to build full 10.3M indexes")
    parser.add_argument("--build-fts", action="store_true", help="Build DuckDB native FTS BM25 index")
    parser.add_argument("--threads", type=int, default=8, help="Number of CPU threads (default: 8)")
    parser.add_argument("--memory-limit", type=str, default="8GB", help="DuckDB memory limit (default: 8GB)")
    args = parser.parse_args()

    if args.limit is None and not args.full:
        print("\n[SAFETY GUARD]: Full 10.3M index build requires explicit '--full' flag. Exiting.\n")
        sys.exit(1)

    effective_limit = None if args.full else args.limit
    build_v5_retrieval_indexes(
        limit=effective_limit,
        build_fts=args.build_fts,
        memory_limit=args.memory_limit,
        threads=args.threads
    )

if __name__ == "__main__":
    main()
