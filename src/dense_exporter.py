"""
Antigravity V5 Multilingual E5 Target Text Exporter & Grouped Job Planner.
Streams DuckDB target records strictly ordered by target_row_id ASC and groups chunks
into ~1,000,000-row batch jobs for high-throughput Kaggle GPU execution.

Eliminates the 104-kernel bottleneck:
104 Parquet chunks -> ~11 Grouped Kaggle GPU Jobs (~1,000,000 rows/job).
"""

import os
import sys
import gc
import json
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import duckdb
import polars as pl

from src.config import get_config
from src.resource_tracker import get_current_rss_mb, MemoryTracker

EXPECTED_TOTAL_TARGETS = 10320219
DEFAULT_CHUNK_SIZE = 100000
DEFAULT_JOB_TARGET_ROWS = 1000000


def compute_sha256(file_path: str) -> str:
    """Computes SHA256 checksum of a file in 64KB streaming blocks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dict_hash(d: Dict[str, Any]) -> str:
    """Computes a deterministic hash of a dictionary."""
    encoded = json.dumps(d, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


class DenseExporter:
    """
    Exports DuckDB target text records into Parquet chunks and groups them into ~1M-row jobs.
    """
    def __init__(
        self,
        db_path: Optional[str] = None,
        output_root: str = "cache/dense_gpu",
        manifest_path: Optional[str] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        job_target_rows: int = DEFAULT_JOB_TARGET_ROWS,
        model_name: str = "intfloat/multilingual-e5-small",
        dimension: int = 384,
        precision: str = "fp16",
        dtype: str = "float16",
        max_length: int = 128
    ):
        config = get_config()
        self.db_path = db_path or os.path.join(config.base_dir, "cache", "entity_resolution.duckdb")
        self.output_root = os.path.join(config.base_dir, output_root)
        self.input_dir = os.path.join(self.output_root, "input")
        self.output_dir = os.path.join(self.output_root, "output")
        self.staging_dir = os.path.join(self.output_root, "staging")
        self.logs_dir = os.path.join(self.output_root, "logs")
        self.manifest_path = manifest_path or os.path.join(self.output_root, "manifest.json")

        self.chunk_size = chunk_size
        self.job_target_rows = job_target_rows
        self.model_name = model_name
        self.dimension = dimension
        self.precision = precision
        self.dtype = dtype
        self.max_length = max_length

        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.staging_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    def load_or_create_manifest(self, total_rows: int) -> Dict[str, Any]:
        """Loads existing master manifest or creates a new one."""
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                if (
                    manifest.get("total_targets") == total_rows
                    and manifest.get("chunk_size") == self.chunk_size
                    and manifest.get("job_target_rows") == self.job_target_rows
                    and manifest.get("model_name") == self.model_name
                ):
                    return manifest
            except Exception as e:
                print(f"[DenseExporter] Warning: Failed to parse existing manifest ({e}). Creating new.")

        total_chunks = (total_rows + self.chunk_size - 1) // self.chunk_size
        manifest = {
            "pipeline_version": "5.0_grouped",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "model_name": self.model_name,
            "model_revision": "main",
            "embedding_dimension": self.dimension,
            "precision": self.precision,
            "dtype": self.dtype,
            "max_length": self.max_length,
            "total_targets": total_rows,
            "chunk_size": self.chunk_size,
            "job_target_rows": self.job_target_rows,
            "total_chunks": total_chunks,
            "source_database": self.db_path,
            "is_complete_target_universe": (total_rows == EXPECTED_TOTAL_TARGETS),
            "chunks": [],
            "jobs": []
        }
        return manifest

    def save_manifest(self, manifest: Dict[str, Any]):
        """Atomically saves manifest JSON file."""
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        temp_manifest = f"{self.manifest_path}.tmp"
        with open(temp_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        os.replace(temp_manifest, self.manifest_path)

    def plan_grouped_jobs(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Dynamically groups chunks into ~1,000,000 row jobs with explicit manifests.
        Example: 104 chunks -> ~11 jobs of 10 chunks each (final job with remainder).
        """
        jobs: List[Dict[str, Any]] = []
        curr_job_chunks: List[Dict[str, Any]] = []
        curr_rows = 0
        job_idx = 0

        for c in sorted(chunks, key=lambda x: x["chunk_id"]):
            curr_job_chunks.append(c)
            curr_rows += c["row_count"]

            if curr_rows >= self.job_target_rows:
                job_manifest = self._create_job_dict(job_idx, curr_job_chunks, curr_rows)
                jobs.append(job_manifest)
                job_idx += 1
                curr_job_chunks = []
                curr_rows = 0

        # Remaining chunks in final job
        if curr_job_chunks:
            job_manifest = self._create_job_dict(job_idx, curr_job_chunks, curr_rows)
            jobs.append(job_manifest)

        return jobs

    def _create_job_dict(self, job_idx: int, job_chunks: List[Dict[str, Any]], total_rows: int) -> Dict[str, Any]:
        """Creates metadata dictionary for a single grouped job."""
        job_id = f"job_{job_idx:03d}"
        chunk_ids = [c["chunk_id"] for c in job_chunks]
        input_files = [c["input_filename"] for c in job_chunks]
        
        cfg_summary = {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "precision": self.precision,
            "dtype": self.dtype,
            "max_length": self.max_length,
            "chunk_ids": chunk_ids,
            "expected_row_count": total_rows
        }

        return {
            "job_id": job_id,
            "job_index": job_idx,
            "status": "pending",
            "model_name": self.model_name,
            "model_revision": "main",
            "embedding_dimension": self.dimension,
            "precision": self.precision,
            "dtype": self.dtype,
            "max_length": self.max_length,
            "chunk_ids": chunk_ids,
            "input_filenames": input_files,
            "expected_row_count": total_rows,
            "completed_chunks": 0,
            "total_chunks": len(job_chunks),
            "config_hash": compute_dict_hash(cfg_summary),
            "attempts": 0,
            "completed_at": None,
            "error_message": None
        }

    def export_chunks(
        self,
        limit_targets: Optional[int] = None,
        limit_chunks: Optional[int] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Exports Parquet chunks from DuckDB and computes grouped jobs plan.
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
        print("ANTIGRAVITY V5 — MULTILINGUAL E5 TARGET TEXT EXPORTER & JOB PLANNER")
        print("=" * 80)
        print(f"Source DuckDB:       {self.db_path} ({total_in_db:,} records in DB)")
        print(f"Target Rows to Export: {total_rows:,} ({total_chunks:,} chunks of {self.chunk_size:,})")
        print(f"Job Target Rows:     {self.job_target_rows:,} (~{(self.job_target_rows + self.chunk_size - 1)//self.chunk_size} chunks/job)")
        print(f"Model:               {self.model_name} (Dim: {self.dimension}, Precision: {self.precision})")
        print(f"Initial RSS:         {get_current_rss_mb():.2f} MB")
        print("-" * 80)

        manifest = self.load_or_create_manifest(total_rows)
        existing_chunks = {c["chunk_id"]: c for c in manifest.get("chunks", [])}

        if dry_run:
            print("[DRY-RUN] Export and Grouping Plan:")
            dummy_chunks = [
                {
                    "chunk_id": i,
                    "row_count": min(self.chunk_size, total_rows - i * self.chunk_size),
                    "input_filename": f"chunk_{i:06d}.parquet"
                }
                for i in range(total_chunks)
            ]
            planned_jobs = self.plan_grouped_jobs(dummy_chunks)
            print(f"  Total Parquet Chunks: {total_chunks}")
            print(f"  Total Grouped Jobs:   {len(planned_jobs)}")
            for j in planned_jobs:
                print(f"    - {j['job_id']}: {len(j['chunk_ids'])} chunks ({j['expected_row_count']:,} rows) -> Chunks {j['chunk_ids'][0]}..{j['chunk_ids'][-1]}")
            conn.close()
            return manifest

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

                # Query chunk strictly ordered by target_row_id ASC
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
                    "output_dir": f"chunk_{chunk_id:06d}",
                    "output_embeddings": f"chunk_{chunk_id:06d}/embeddings.npy",
                    "output_ids": f"chunk_{chunk_id:06d}/target_ids.parquet",
                    "output_meta": f"chunk_{chunk_id:06d}/metadata.json",
                    "output_sha256": existing_chunks.get(chunk_id, {}).get("output_sha256"),
                    "attempts": existing_chunks.get(chunk_id, {}).get("attempts", 0),
                    "completed_at": existing_chunks.get(chunk_id, {}).get("completed_at"),
                    "error_message": None
                }

                existing_chunks[chunk_id] = chunk_meta
                exported_count += 1

                pct = ((chunk_id + 1) / total_chunks) * 100.0
                print(f"  -> Exported Chunk {chunk_id:04d}/{total_chunks:04d} ({chunk_n:,} rows) [{pct:.1f}%] | SHA256: {sha256[:12]}...", flush=True)

                del rows, df_chunk, target_row_ids, eids, names, addrs, ctrys
                if chunk_id % 10 == 0:
                    gc.collect()

        conn.close()

        # Update manifest chunks and compute grouped jobs
        final_chunks = [existing_chunks[i] for i in sorted(existing_chunks.keys()) if i < total_chunks]
        manifest["chunks"] = final_chunks

        # Build / merge grouped jobs
        planned_jobs = self.plan_grouped_jobs(final_chunks)
        manifest["jobs"] = planned_jobs

        self.save_manifest(manifest)

        elapsed = time.time() - t0
        print("\n[DenseExporter] Target chunk export & job planning complete.")
        print(f"  Total Chunks:  {len(final_chunks)} (Exported: {exported_count}, Reused: {skipped_count})")
        print(f"  Grouped Jobs:  {len(planned_jobs)} jobs (~{self.job_target_rows:,} rows each)")
        print(f"  Elapsed:       {elapsed:.2f}s | Final RSS: {get_current_rss_mb():.2f} MB")
        print("=" * 80)
        return manifest
