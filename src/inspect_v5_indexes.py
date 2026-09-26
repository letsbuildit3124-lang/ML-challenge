"""
Antigravity V5 CPU Retrieval Index Inspection Utility.
Inspects system resources (CPU cores, RAM), persistent DuckDB cache, and V5 retrieval manifests.
"""

import os
import sys
import json
import duckdb

from src.config import get_config
from src.resource_tracker import get_current_rss_mb, get_peak_rss_mb

def inspect_v5_environment():
    config = get_config()
    db_path = os.path.join(config.base_dir, "cache", "entity_resolution.duckdb")
    retrieval_dir = os.path.join(config.base_dir, "cache", "retrieval")
    meta_path = os.path.join(retrieval_dir, "metadata.json")

    print("=" * 80)
    print("ANTIGRAVITY V5 — CPU RETRIEVAL ENGINE & INDEX INSPECTION")
    print("=" * 80)
    print(f"Python:           {sys.version.split()[0]}")
    print(f"Platform:         {sys.platform}")
    print(f"CPU Cores:        {os.cpu_count()} vCPU")
    print(f"Current RSS:      {get_current_rss_mb():.2f} MB")
    print("-" * 80)

    # 1. Target DuckDB Cache Status
    if os.path.exists(db_path):
        db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
        print(f"[OK] Target DuckDB:   {db_path} ({db_size_mb:.2f} MB)")
        try:
            conn = duckdb.connect(db_path, read_only=True)
            n_targets = conn.execute("SELECT COUNT(*) FROM targets;").fetchone()[0]
            conn.close()
            print(f"[OK] Target Universe: {n_targets:,} records ready")
        except Exception as e:
            print(f"[WARN] DuckDB query check failed: {e}")
    else:
        print(f"[ERROR] Target DuckDB NOT found at {db_path}")

    # 2. V5 Index Manifest
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            print(f"[OK] V5 Manifest:     {meta_path}")
            print(f"     Version:         {meta.get('index_version')}")
            print(f"     Target Count:    {meta.get('target_count'):,}")
            print(f"     Threads:         {meta.get('threads')}")
            print(f"     FTS Enabled:     {meta.get('fts_enabled')}")
            print(f"     Created At:      {meta.get('creation_timestamp')}")
        except Exception as e:
            print(f"[WARN] Error reading manifest: {e}")
    else:
        print(f"[INFO] V5 Manifest:   Not yet created (Run build_v5_indexes)")

    print("=" * 80)

if __name__ == "__main__":
    inspect_v5_environment()
