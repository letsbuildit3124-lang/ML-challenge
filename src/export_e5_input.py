"""
Antigravity One-Shot E5 Target Input Exporter.
Streams target records from persistent DuckDB cache and exports them as a compact,
high-throughput Parquet file for the One-Shot Kaggle GPU worker.

Exported Columns:
- target_id: Target entity identifier (e.g. s2_XXXX, s3_XXXX)
- business_name: Business name string (Unicode preserved)
- business_address: Business address string (Unicode preserved)
- country: Country code string
- source: Dataset source ("S2" or "S3")

Features:
- Streaming chunked export (Peak RAM < 150MB)
- Strict positional ordering (ORDER BY target_row_id ASC)
- Computes comprehensive row counts, S2/S3 distribution, null counts, and SHA256 checksum
- Supports small test modes (--rows 10000, --rows 100000)
"""

import os
import sys
import gc
import json
import time
import hashlib
import argparse
from typing import Optional, Dict, Any
import duckdb
import polars as pl

from src.config import get_config
from src.resource_tracker import get_current_rss_mb, MemoryTracker

EXPECTED_TOTAL_TARGETS = 10320219
DEFAULT_CHUNK_SIZE = 100000


def compute_file_sha256(file_path: str) -> str:
    """Computes SHA256 hash in 64KB streaming blocks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_export_sql(limit_rows: Optional[int] = None) -> str:
    """
    Constructs deterministic SELECT query for DuckDB export without trailing semicolons.
    Ensures safe nesting inside COPY (...) expressions.
    """
    limit_clause = f"\n        LIMIT {limit_rows}" if limit_rows else ""
    return f"""
        SELECT 
            eid AS target_id,
            norm_name AS business_name,
            norm_addr AS business_address,
            country,
            CASE WHEN eid LIKE 's2_%' THEN 'S2' ELSE 'S3' END AS source
        FROM targets
        ORDER BY target_row_id ASC{limit_clause}
    """.strip()


def export_e5_targets(
    db_path: Optional[str] = None,
    output_path: str = "cache/e5_gpu/input/targets.parquet",
    manifest_path: str = "cache/e5_gpu/input/input_manifest.json",
    limit_rows: Optional[int] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> Dict[str, Any]:
    """
    Streams DuckDB target records and exports to a single compact Parquet file.
    """
    config = get_config()
    db_file = db_path or os.path.join(config.base_dir, "cache", "entity_resolution.duckdb")
    out_file = os.path.join(config.base_dir, output_path)
    man_file = os.path.join(config.base_dir, manifest_path)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    os.makedirs(os.path.dirname(man_file), exist_ok=True)

    if not os.path.exists(db_file):
        raise FileNotFoundError(f"Target DuckDB database not found at '{db_file}'! Build cache first.")

    conn = duckdb.connect(db_file, read_only=True)
    total_in_db = conn.execute("SELECT COUNT(*) FROM targets;").fetchone()[0]
    total_to_export = min(total_in_db, limit_rows) if limit_rows else total_in_db
    is_complete_prod = (total_to_export == EXPECTED_TOTAL_TARGETS)

    print("=" * 80)
    print("ANTIGRAVITY — E5 TARGET DATA EXPORTER (DUCKDB -> PARQUET)")
    print("=" * 80)
    print(f"Source Database:     {db_file} ({total_in_db:,} records in DB)")
    print(f"Target Export Rows:  {total_to_export:,} ({'FULL PRODUCTION' if is_complete_prod else 'TEST / SUBSET'})")
    print(f"Output Parquet:      {out_file}")
    print(f"Initial RSS:         {get_current_rss_mb():.2f} MB")
    print("-" * 80)

    t0 = time.time()

    # Temporary staging file
    temp_parquet = f"{out_file}.tmp"
    if os.path.exists(temp_parquet):
        os.remove(temp_parquet)

    # Build clean SELECT subquery without trailing semicolon
    query = build_export_sql(limit_rows=total_to_export if total_to_export < total_in_db else None)

    print("[Exporter] Streaming target records from DuckDB to Parquet...")
    with MemoryTracker(f"DuckDB Target Export ({total_to_export:,} rows)"):
        # Use DuckDB COPY directly to write compact, compressed Parquet at C++ speed
        copy_sql = f"""
            COPY (
                {query}
            ) TO '{temp_parquet.replace(chr(92), '/')}' (
                FORMAT PARQUET,
                COMPRESSION 'ZSTD',
                ROW_GROUP_SIZE {chunk_size}
            );
        """
        conn.execute(copy_sql)

    conn.close()

    if os.path.exists(out_file):
        os.remove(out_file)
    os.replace(temp_parquet, out_file)

    # Compute fast metadata and column statistics via Polars metadata / scan
    df_meta = pl.scan_parquet(out_file)
    schema = df_meta.collect_schema()
    row_count = df_meta.select(pl.len()).collect().item()

    # Source & Null breakdown
    stats_df = df_meta.select([
        pl.col("source").value_counts(),
        pl.col("business_name").is_null().sum().alias("null_name"),
        pl.col("business_address").is_null().sum().alias("null_addr"),
        pl.col("country").is_null().sum().alias("null_ctry")
    ]).collect()

    sources_dict = {}
    for row in stats_df["source"][0].to_dicts():
        sources_dict[row["source"]] = row["count"]
    s2_count = sources_dict.get("S2", 0)
    s3_count = sources_dict.get("S3", 0)

    null_name_count = stats_df["null_name"][0]
    null_addr_count = stats_df["null_addr"][0]
    null_ctry_count = stats_df["null_ctry"][0]

    # File size & SHA256 checksum
    file_size_bytes = os.path.getsize(out_file)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"[Exporter] Calculating SHA256 checksum of {out_file}...")
    sha256 = compute_file_sha256(out_file)

    elapsed = time.time() - t0

    manifest = {
        "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "output_file": os.path.basename(out_file),
        "total_rows": row_count,
        "is_complete_production_universe": is_complete_prod,
        "expected_production_targets": EXPECTED_TOTAL_TARGETS,
        "source_counts": {
            "S2": s2_count,
            "S3": s3_count
        },
        "null_counts": {
            "business_name": null_name_count,
            "business_address": null_addr_count,
            "country": null_ctry_count
        },
        "file_size_mb": round(file_size_mb, 2),
        "sha256_checksum": sha256,
        "elapsed_seconds": round(elapsed, 2)
    }

    with open(man_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("-" * 80)
    print(f"[Exporter SUCCESS] Target export completed in {elapsed:.2f}s:")
    print(f"  Rows Exported:     {row_count:,}")
    print(f"  Source Breakdown:  S2 = {s2_count:,} | S3 = {s3_count:,}")
    print(f"  Null Values:       Name: {null_name_count:,} | Address: {null_addr_count:,} | Country: {null_ctry_count:,}")
    print(f"  File Size:         {file_size_mb:.2f} MB")
    print(f"  SHA256 Checksum:   {sha256}")
    print(f"  Manifest Saved:    {man_file}")
    print(f"  Final RSS:         {get_current_rss_mb():.2f} MB")
    print("=" * 80)

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Antigravity E5 Target Input Exporter")
    parser.add_argument("--rows", type=int, default=None, help="Limit number of target rows to export (e.g. 10000 for smoke, 100000 for benchmark)")
    parser.add_argument("--smoke", action="store_true", help="Shortcut for 10,000 smoke test rows")
    parser.add_argument("--benchmark", action="store_true", help="Shortcut for 100,000 benchmark rows")
    parser.add_argument("--output", type=str, default="cache/e5_gpu/input/targets.parquet", help="Output Parquet path")
    parser.add_argument("--manifest", type=str, default="cache/e5_gpu/input/input_manifest.json", help="Manifest path")
    parser.add_argument("--db-path", type=str, default=None, help="Path to DuckDB cache")
    args = parser.parse_args()

    limit = args.rows
    if args.smoke:
        limit = 10000
    elif args.benchmark:
        limit = 100000

    export_e5_targets(
        db_path=args.db_path,
        output_path=args.output,
        manifest_path=args.manifest,
        limit_rows=limit
    )


if __name__ == "__main__":
    main()
