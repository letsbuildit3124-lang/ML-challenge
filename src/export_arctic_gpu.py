"""
Antigravity V4.1 Arctic GPU Target Text Chunker & Manifest Generator.
Streams target records from the persistent DuckDB cache and exports them as
lightweight Parquet chunks for remote GPU worker execution.

Features:
- Streaming chunked reads from DuckDB (Bounded RAM footprint < 100MB)
- Strict positional target ordering (ORDER BY target_row_id ASC)
- SHA256 input checksum generation
- Durable JSON manifest for checkpointing & resumability
"""

import os
import sys
import gc
import json
import time
import hashlib
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import duckdb
import polars as pl

from src.config import get_config
from src.resource_tracker import get_current_rss_mb, MemoryTracker

EXPECTED_TOTAL_TARGETS = 10320219
DEFAULT_CHUNK_SIZE = 100000

def compute_sha256(file_path: str) -> str:
    """Computes SHA256 hash of a file in streaming 64KB blocks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class ArcticGPUExporter:
    """
    Exports DuckDB target text records into Parquet chunks with manifest tracking.
    """
    def __init__(
        self,
        db_path: Optional[str] = None,
        output_root: str = "cache/arctic_gpu",
        manifest_path: Optional[str] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        model_name: str = "themelder/arctic-embed-xs-entity-resolution",
        dimension: int = 384
    ):
        config = get_config()
        self.db_path = db_path or os.path.join(config.base_dir, "cache", "entity_resolution.duckdb")
        self.output_root = os.path.join(config.base_dir, output_root)
        self.input_dir = os.path.join(self.output_root, "input")
        self.manifest_path = manifest_path or os.path.join(self.output_root, "manifest.json")
        self.chunk_size = chunk_size
        self.model_name = model_name
        self.dimension = dimension

        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_root, "output"), exist_ok=True)
        os.makedirs(os.path.join(self.output_root, "logs"), exist_ok=True)

    def load_or_create_manifest(self, total_rows: int) -> Dict[str, Any]:
        """Loads existing manifest or creates a new one."""
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                if manifest.get("total_targets") == total_rows and manifest.get("chunk_size") == self.chunk_size:
                    return manifest
            except Exception as e:
                print(f"[Exporter] Warning: Failed to parse existing manifest ({e}). Creating new.")

        total_chunks = (total_rows + self.chunk_size - 1) // self.chunk_size
        manifest = {
            "pipeline_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "model_name": self.model_name,
            "embedding_dimension": self.dimension,
            "total_targets": total_rows,
            "chunk_size": self.chunk_size,
            "total_chunks": total_chunks,
            "source_database": self.db_path,
            "is_complete_target_universe": (total_rows == EXPECTED_TOTAL_TARGETS),
            "chunks": []
        }
        return manifest

    def save_manifest(self, manifest: Dict[str, Any]):
        """Atomically saves manifest JSON file."""
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        temp_manifest = f"{self.manifest_path}.tmp"
        with open(temp_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        os.replace(temp_manifest, self.manifest_path)

    def export_chunks(
        self,
        limit_targets: Optional[int] = None,
        limit_chunks: Optional[int] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Executes streaming export from DuckDB to Parquet chunks.
        """
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Target DuckDB cache not found at {self.db_path}!")

        conn = duckdb.connect(self.db_path, read_only=True)
        total_in_db = conn.execute("SELECT COUNT(*) FROM targets;").fetchone()[0]
        total_rows = min(total_in_db, limit_targets) if limit_targets else total_in_db
        total_chunks = (total_rows + self.chunk_size - 1) // self.chunk_size
        if limit_chunks and limit_chunks < total_chunks:
            total_chunks = limit_chunks
            total_rows = min(total_rows, total_chunks * self.chunk_size)

        print("=" * 80)
        print("ANTIGRAVITY V4.1 — ARCTIC GPU TARGET TEXT EXPORTER")
        print("=" * 80)
        print(f"Source DuckDB:       {self.db_path} ({total_in_db:,} records in DB)")
        print(f"Target Rows to Export: {total_rows:,} ({total_chunks:,} chunks of {self.chunk_size:,})")
        print(f"Input Directory:     {self.input_dir}")
        print(f"Manifest Path:       {self.manifest_path}")
        print(f"Initial RSS:         {get_current_rss_mb():.2f} MB")
        print("-" * 80)

        if dry_run:
            print("[DRY-RUN] Export plan:")
            print(f"  Total Chunks:      {total_chunks}")
            print(f"  Approx Disk/Chunk: ~10-15 MB (Compressed Parquet)")
            print(f"  Total Input Size:  ~{total_chunks * 12 / 1024:.2f} GB")
            conn.close()
            return self.load_or_create_manifest(total_rows)

        manifest = self.load_or_create_manifest(total_rows)
        existing_chunks = {c["chunk_id"]: c for c in manifest.get("chunks", [])}

        t0 = time.time()
        exported_count = 0
        skipped_count = 0

        with MemoryTracker(f"Target Text Chunk Export ({total_chunks} chunks)"):
            for chunk_id in range(total_chunks):
                offset = chunk_id * self.chunk_size
                chunk_n = min(self.chunk_size, total_rows - offset)
                chunk_filename = f"chunk_{chunk_id:06d}.parquet"
                chunk_filepath = os.path.join(self.input_dir, chunk_filename)

                # Check if already exported and valid
                if chunk_id in existing_chunks and os.path.exists(chunk_filepath):
                    existing_entry = existing_chunks[chunk_id]
                    if existing_entry.get("row_count") == chunk_n:
                        skipped_count += 1
                        continue

                # Query chunk strictly ordered by target_row_id
                query = f"""
                    SELECT target_row_id, eid, norm_name, norm_addr, country 
                    FROM targets 
                    ORDER BY target_row_id ASC 
                    LIMIT {chunk_n} OFFSET {offset};
                """
                rows = conn.execute(query).fetchall()
                if not rows:
                    break

                target_row_ids = [r[0] for r in rows]
                eids = [r[1] for r in rows]
                names = [r[2] or "" for r in rows]
                addrs = [r[3] or "" for r in rows]
                ctrys = [r[4] or "" for r in rows]

                df_chunk = pl.DataFrame({
                    "target_row_id": target_row_ids,
                    "target_id": eids,
                    "business_name": names,
                    "business_address": addrs,
                    "country": ctrys
                })

                df_chunk.write_parquet(chunk_filepath, compression="zstd")
                sha256 = compute_sha256(chunk_filepath)

                chunk_meta = {
                    "chunk_id": chunk_id,
                    "input_filename": chunk_filename,
                    "row_count": len(eids),
                    "first_target_id": eids[0] if eids else "",
                    "last_target_id": eids[-1] if eids else "",
                    "input_sha256": sha256,
                    "status": existing_chunks.get(chunk_id, {}).get("status", "pending"),
                    "output_filename": f"chunk_{chunk_id:06d}_embeddings.npy",
                    "ids_filename": f"chunk_{chunk_id:06d}_ids.parquet",
                    "output_sha256": existing_chunks.get(chunk_id, {}).get("output_sha256"),
                    "attempts": existing_chunks.get(chunk_id, {}).get("attempts", 0),
                    "completed_at": existing_chunks.get(chunk_id, {}).get("completed_at"),
                    "error_message": None
                }

                existing_chunks[chunk_id] = chunk_meta
                exported_count += 1

                pct = ((chunk_id + 1) / total_chunks) * 100.0
                print(f"  -> Exported Chunk {chunk_id:04d}/{total_chunks:04d} ({chunk_n:,} rows) [{pct:.1f}%] | SHA256: {sha256[:12]}... | RSS: {get_current_rss_mb():.1f} MB", flush=True)

                del rows, df_chunk, target_row_ids, eids, names, addrs, ctrys
                if chunk_id % 10 == 0:
                    gc.collect()

        conn.close()

        # Update and persist manifest
        manifest["chunks"] = [existing_chunks[i] for i in sorted(existing_chunks.keys()) if i < total_chunks]
        self.save_manifest(manifest)

        elapsed = time.time() - t0
        print("\n[Exporter] Target chunk export complete.")
        print(f"  Exported: {exported_count} chunks | Skipped (reused): {skipped_count} chunks | Elapsed: {elapsed:.2f}s")
        print(f"  Manifest: {self.manifest_path}")
        print(f"  Final RSS: {get_current_rss_mb():.2f} MB")
        print("=" * 80)
        return manifest


def main():
    parser = argparse.ArgumentParser(description="Antigravity V4.1 Arctic GPU Target Text Exporter")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Rows per Parquet chunk")
    parser.add_argument("--limit-targets", type=int, default=None, help="Limit total target rows to export (for testing)")
    parser.add_argument("--limit-chunks", type=int, default=None, help="Limit total chunks to export (for testing)")
    parser.add_argument("--dry-run", action="store_true", help="Print export plan without writing files")
    args = parser.parse_args()

    exporter = ArcticGPUExporter(chunk_size=args.chunk_size)
    exporter.export_chunks(
        limit_targets=args.limit_targets,
        limit_chunks=args.limit_chunks,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()
