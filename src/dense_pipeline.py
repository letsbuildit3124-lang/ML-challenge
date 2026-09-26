"""
Antigravity V5 Master Dense Multilingual E5 Pipeline.
Entry Point: python3 -m src.dense_pipeline

Automates the complete end-to-end GPU-first lifecycle:
1. Environment & Pre-flight Validation
2. DuckDB Streaming Export into 100k Parquet Chunks
3. Deterministic Grouping into ~1,000,000-Row Kaggle GPU Jobs (104 chunks -> ~11 jobs)
4. Kaggle GPU Worker Execution with Persistent Model Lifecycle & Queue Timeout (180s)
5. Strict EC2 Positional & Numerical Verification
6. Disk-Backed Float16 Memmap Assembly (Zero Full-Corpus RAM Allocation)
7. Memory-Bounded FAISS ANN Index Construction
8. Validation & Benchmark Summary
"""

import os
import sys
import gc
import json
import time
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
import polars as pl
import numpy as np

from src.config import get_config
from src.resource_tracker import get_current_rss_mb, get_peak_rss_mb, MemoryTracker
from src.dense_exporter import DenseExporter, EXPECTED_TOTAL_TARGETS, DEFAULT_CHUNK_SIZE, DEFAULT_JOB_TARGET_ROWS
from src.kaggle_dense_controller import KaggleDenseController
from src.build_dense_faiss import build_faiss_index
from src.dense_embeddings import DEFAULT_MODEL_NAME, EMBEDDING_DIM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("dense.pipeline")


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Loads YAML config with basic fallback if PyYAML is missing."""
    if not os.path.exists(config_path):
        return {}
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        data: Dict[str, Any] = {}
        curr_section = None
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.endswith(":") and not line.startswith("-"):
                    curr_section = line[:-1].strip()
                    data[curr_section] = {}
                elif ":" in line and curr_section:
                    k, v = line.split(":", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if v.lower() == "true":
                        v = True
                    elif v.lower() == "false":
                        v = False
                    elif v.isdigit():
                        v = int(v)
                    data[curr_section][k] = v
        return data


class DenseMasterPipeline:
    """
    Master Orchestrator for Grouped-Job Multilingual E5 GPU Pipeline.
    """
    def __init__(
        self,
        config_dict: Dict[str, Any],
        is_production: bool = False,
        is_smoke: bool = False,
        chunk_rows: int = DEFAULT_CHUNK_SIZE,
        job_target_rows: int = DEFAULT_JOB_TARGET_ROWS,
        resume: bool = True,
        max_concurrent_jobs: int = 1,
        queue_timeout_seconds: int = 180,
        model_name: Optional[str] = None,
        limit_targets: Optional[int] = None,
        limit_chunks: Optional[int] = None
    ):
        self.app_config = get_config()
        self.config_dict = config_dict
        self.is_production = is_production
        self.is_smoke = is_smoke
        self.resume = resume
        self.limit_targets = limit_targets
        self.limit_chunks = limit_chunks

        # Configure rows based on mode
        data_cfg = config_dict.get("data", {})
        if self.is_smoke:
            self.limit_targets = data_cfg.get("smoke_target_rows", 10000)
            self.chunk_rows = self.limit_targets
            self.job_target_rows = self.limit_targets
        else:
            self.chunk_rows = chunk_rows
            self.job_target_rows = job_target_rows

        # Model Config
        model_cfg = config_dict.get("model", {})
        self.model_name = model_name or model_cfg.get("name", DEFAULT_MODEL_NAME)
        self.embedding_dim = int(model_cfg.get("embedding_dim", EMBEDDING_DIM))
        self.precision = model_cfg.get("precision", "fp16")
        self.max_length = int(model_cfg.get("max_length", 128))
        self.dtype = config_dict.get("storage", {}).get("dtype", "float16")

        # Paths
        paths_cfg = config_dict.get("paths", {})
        self.root_dir = os.path.join(self.app_config.base_dir, paths_cfg.get("root", "cache/dense_gpu"))
        self.manifest_path = os.path.join(self.app_config.base_dir, paths_cfg.get("manifest", "cache/dense_gpu/manifest.json"))
        self.input_dir = os.path.join(self.root_dir, "input")
        self.output_dir = os.path.join(self.root_dir, "output")
        self.staging_dir = os.path.join(self.root_dir, "staging")
        self.logs_dir = os.path.join(self.root_dir, "logs")

        # Separate Production vs Smoke storage
        if self.is_production:
            self.merged_emb_path = os.path.join(self.app_config.base_dir, paths_cfg.get("production_embeddings", "cache/embeddings/multilingual_e5/production/target_embeddings.npy"))
            self.merged_ids_path = os.path.join(self.app_config.base_dir, paths_cfg.get("production_ids", "cache/embeddings/multilingual_e5/production/target_ids.json"))
            self.merged_meta_path = os.path.join(self.app_config.base_dir, paths_cfg.get("production_meta", "cache/embeddings/multilingual_e5/production/metadata.json"))
            self.faiss_dir = os.path.join(self.app_config.base_dir, paths_cfg.get("production_ann_dir", "cache/ann/multilingual_e5/production"))
        else:
            self.merged_emb_path = os.path.join(self.app_config.base_dir, paths_cfg.get("smoke_embeddings", "cache/embeddings/multilingual_e5/smoke/target_embeddings.npy"))
            self.merged_ids_path = os.path.join(self.app_config.base_dir, paths_cfg.get("smoke_ids", "cache/embeddings/multilingual_e5/smoke/target_ids.json"))
            self.merged_meta_path = os.path.join(self.app_config.base_dir, paths_cfg.get("smoke_meta", "cache/embeddings/multilingual_e5/smoke/metadata.json"))
            self.faiss_dir = os.path.join(self.app_config.base_dir, paths_cfg.get("smoke_ann_dir", "cache/ann/multilingual_e5/smoke"))

        # Kaggle Config
        kaggle_cfg = config_dict.get("kaggle", {})
        self.kernel_slug = kaggle_cfg.get("kernel", "rajeshshitap/arctic-entity-resolution-worker")
        self.dataset_slug = kaggle_cfg.get("dataset_slug", "rajeshshitap/arctic-er-input")
        self.accelerator = kaggle_cfg.get("accelerator", "NvidiaL4")
        self.timeout_seconds = int(kaggle_cfg.get("timeout_seconds", 3600))
        self.queue_timeout_seconds = queue_timeout_seconds or int(kaggle_cfg.get("queue_timeout_seconds", 180))
        self.poll_interval = int(kaggle_cfg.get("poll_interval_seconds", 15))
        self.max_attempts = int(kaggle_cfg.get("max_attempts", 2))
        self.max_concurrent_jobs = max_concurrent_jobs

        # Components
        self.exporter = DenseExporter(
            output_root=paths_cfg.get("root", "cache/dense_gpu"),
            manifest_path=self.manifest_path,
            chunk_size=self.chunk_rows,
            job_target_rows=self.job_target_rows,
            model_name=self.model_name,
            dimension=self.embedding_dim,
            precision=self.precision,
            dtype=self.dtype,
            max_length=self.max_length
        )

        self.kaggle_ctrl = KaggleDenseController(
            kernel_slug=self.kernel_slug,
            dataset_slug=self.dataset_slug,
            accelerator=self.accelerator,
            timeout_seconds=self.timeout_seconds,
            queue_timeout_seconds=self.queue_timeout_seconds,
            poll_interval=self.poll_interval,
            max_attempts=self.max_attempts,
            logs_dir=self.logs_dir,
            worker_src_dir=paths_cfg.get("worker_src_dir", "kaggle/dense_worker")
        )

    def print_plan(self):
        """Prints dry-run plan of grouped jobs."""
        print("=" * 80)
        print("ANTIGRAVITY V5 — MULTILINGUAL E5 PIPELINE EXECUTION PLAN")
        print("=" * 80)
        print(f"Mode:                {'PRODUCTION (10.32M Universe)' if self.is_production else 'SMOKE TEST / SANDBOX'}")
        print(f"Model Name:          {self.model_name} (Dim: {self.embedding_dim})")
        print(f"Precision:           {self.precision.upper()} | Storage: {self.dtype}")
        print(f"Target Rows:         {self.limit_targets or EXPECTED_TOTAL_TARGETS:,}")
        print(f"Chunk Size:          {self.chunk_rows:,} rows")
        print(f"Grouped Job Target:  {self.job_target_rows:,} rows (~{(self.job_target_rows + self.chunk_rows - 1)//self.chunk_rows} chunks/job)")
        print(f"Queue Timeout:       {self.queue_timeout_seconds} seconds")
        print(f"Kaggle Kernel:       {self.kernel_slug}")
        print(f"Kaggle Dataset:      {self.dataset_slug}")
        print("-" * 80)

        total_rows = self.limit_targets or EXPECTED_TOTAL_TARGETS
        total_chunks = (total_rows + self.chunk_rows - 1) // self.chunk_rows
        approx_jobs = (total_rows + self.job_target_rows - 1) // self.job_target_rows
        storage_gb = total_rows * self.embedding_dim * 2 / (1024.0 ** 3) # float16 = 2 bytes

        print(f"Total Chunks:        {total_chunks:,}")
        print(f"Grouped Kaggle Jobs: {approx_jobs:,} jobs (was previously {total_chunks:,} individual jobs!)")
        print(f"Float16 Corpus Size: ~{storage_gb:.2f} GB")
        print(f"Corpus Embeddings:   {self.merged_emb_path}")
        print(f"FAISS Index Path:    {os.path.join(self.faiss_dir, 'target.index')}")
        print("=" * 80)

    def validate_preflight(self, skip_kaggle: bool = False):
        """Strict pre-flight validation."""
        print("\n" + "=" * 80)
        print("[PRE-FLIGHT VALIDATION] Verifying Environment, Disk Space & Kaggle Access...")
        print("=" * 80)

        if not os.path.exists(self.exporter.db_path):
            print(f"[ERROR]: DuckDB target cache not found at '{self.exporter.db_path}'!")
            sys.exit(1)
        print(f"[OK] Source DuckDB:      {self.exporter.db_path}")

        try:
            _, _, free_b = shutil.disk_usage(self.root_dir if os.path.exists(self.root_dir) else ".")
            free_gb = free_b / (1024.0 ** 3)
            req_gb = 25.0 if self.is_production else 2.0
            print(f"[OK] Available Disk:     {free_gb:.2f} GB (Required: {req_gb:.2f} GB)")
            if free_gb < req_gb:
                print(f"[ERROR]: Insufficient disk space ({free_gb:.2f} GB < {req_gb:.2f} GB)!")
                sys.exit(1)
        except Exception as e:
            print(f"[WARN] Disk check: {e}")

        if not skip_kaggle:
            auth_ok, auth_msg = self.kaggle_ctrl.check_kaggle_auth()
            if not auth_ok:
                print(f"[ERROR] Kaggle Auth: {auth_msg}")
                sys.exit(1)
            print(f"[OK] Kaggle Auth:        {auth_msg}")

        print("[OK] Pre-Flight Checks Passed.\n")

    def assemble_memmap(self, manifest: Dict[str, Any]) -> Tuple[str, str]:
        """
        Assembles verified chunk embeddings directly into a disk-backed memory map.
        Uses float16 (2 bytes per element) and predetermined chunk offsets.
        """
        os.makedirs(os.path.dirname(self.merged_emb_path), exist_ok=True)
        chunks = manifest.get("chunks", [])
        completed_chunks = [c for c in chunks if c.get("status") == "completed"]

        if len(completed_chunks) != len(chunks):
            raise RuntimeError(f"Cannot assemble incomplete corpus! Completed: {len(completed_chunks)}/{len(chunks)}")

        total_rows = sum(c["row_count"] for c in completed_chunks)
        is_complete = (total_rows == EXPECTED_TOTAL_TARGETS)

        print("\n" + "=" * 80)
        print(f"[Memmap Assembly] Writing {len(completed_chunks)} chunks directly into disk-backed store...")
        print(f"Target Rows:       {total_rows:,} / {EXPECTED_TOTAL_TARGETS:,}")
        print(f"Production Universe: {'YES (10,320,219 Targets)' if is_complete else 'NO (SMOKE / SANDBOX)'}")
        print(f"Embeddings Path:   {self.merged_emb_path} (dtype: {self.dtype})")
        print("=" * 80)

        # Allocate disk-backed memmap directly
        fp_merged = np.lib.format.open_memmap(
            self.merged_emb_path,
            mode="w+",
            dtype=self.dtype,
            shape=(total_rows, self.embedding_dim)
        )

        all_target_ids: List[str] = []
        offset = 0

        with MemoryTracker(f"Memmap Assembly ({len(completed_chunks)} chunks)"):
            for chunk_meta in sorted(completed_chunks, key=lambda x: x["chunk_id"]):
                c_id = chunk_meta["chunk_id"]
                c_n = chunk_meta["row_count"]
                chunk_prefix = f"chunk_{c_id:06d}"
                chunk_dir = os.path.join(self.output_dir, chunk_prefix)
                emb_file = os.path.join(chunk_dir, "embeddings.npy")
                ids_file = os.path.join(chunk_dir, "target_ids.parquet")

                # Read chunk embeddings and IDs
                chunk_embs = np.load(emb_file)
                df_ids = pl.read_parquet(ids_file)
                chunk_ids = [str(x) for x in df_ids["target_id"].to_list()]

                assert chunk_embs.shape == (c_n, self.embedding_dim), f"Shape mismatch in chunk {chunk_prefix}"
                assert len(chunk_ids) == c_n, f"ID count mismatch in chunk {chunk_prefix}"

                # Assign directly into memmap
                fp_merged[offset : offset + c_n] = chunk_embs.astype(self.dtype)
                all_target_ids.extend(chunk_ids)

                offset += c_n
                del chunk_embs, df_ids, chunk_ids
                if c_id % 10 == 0:
                    gc.collect()

            fp_merged.flush()
            del fp_merged
            gc.collect()

        # Strict Production Assertions
        if is_complete:
            assert len(all_target_ids) == EXPECTED_TOTAL_TARGETS, f"Total IDs mismatch: {len(all_target_ids)} != {EXPECTED_TOTAL_TARGETS}"
            unique_count = len(set(all_target_ids))
            assert unique_count == EXPECTED_TOTAL_TARGETS, f"Duplicate target IDs in corpus! Unique {unique_count:,} != Expected {EXPECTED_TOTAL_TARGETS:,}"
            print(f"[Memmap Assembly] Verified 100% unique target IDs ({unique_count:,} records).")

        # Save target IDs JSON
        with open(self.merged_ids_path, "w", encoding="utf-8") as f:
            json.dump(all_target_ids, f)

        # Save metadata JSON
        meta = {
            "model_name": self.model_name,
            "model_revision": "main",
            "embedding_dimension": self.embedding_dim,
            "dtype": self.dtype,
            "precision": self.precision,
            "normalization": "L2",
            "target_count": total_rows,
            "expected_target_count": EXPECTED_TOTAL_TARGETS,
            "is_complete_target_universe": is_complete,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        with open(self.merged_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"[Memmap Assembly] Complete ({total_rows:,} rows | Complete: {is_complete}).")
        return self.merged_emb_path, self.merged_ids_path

    def run_pipeline(
        self,
        skip_export: bool = False,
        skip_kaggle: bool = False,
        skip_faiss: bool = False
    ):
        """Runs the grouped-job pipeline."""
        t0_all = time.time()
        print("=" * 80)
        print(f"ANTIGRAVITY V5 — MASTER MULTILINGUAL E5 GPU PIPELINE")
        print(f"Mode: {'PRODUCTION (10.32M Records)' if self.is_production else 'SMOKE TEST / DEV'}")
        print("=" * 80)

        self.validate_preflight(skip_kaggle=skip_kaggle)

        # Stage 1: Export & Grouped Job Planning
        if not skip_export:
            manifest = self.exporter.export_chunks(
                limit_targets=self.limit_targets,
                limit_chunks=self.limit_chunks
            )
        else:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

        jobs = manifest.get("jobs", [])
        total_jobs = len(jobs)

        # Stage 2: Grouped Kaggle GPU Execution
        if not skip_kaggle:
            print("\n" + "=" * 80)
            print(f"[Stage 2: Kaggle GPU Execution] Executing {total_jobs} Grouped Jobs (~{self.job_target_rows:,} rows/job)...")
            print("=" * 80)

            chunk_map = {c["chunk_id"]: c for c in manifest.get("chunks", [])}

            for job_idx, job in enumerate(jobs):
                job_id = job["job_id"]
                job_chunks = job["chunk_ids"]
                expected_rows = job["expected_row_count"]

                # Check if all chunks in job already completed
                all_done = True
                for c_id in job_chunks:
                    c_prefix = f"chunk_{c_id:06d}"
                    c_emb = os.path.join(self.output_dir, c_prefix, "embeddings.npy")
                    c_ids = os.path.join(self.output_dir, c_prefix, "target_ids.parquet")
                    if not (os.path.exists(c_emb) and os.path.exists(c_ids)):
                        all_done = False
                        break

                if all_done and self.resume:
                    print(f"  -> Job {job_id} ({len(job_chunks)} chunks, {expected_rows:,} rows) already COMPLETED. Skipping.")
                    job["status"] = "completed"
                    continue

                attempts = job.get("attempts", 0)
                success = False

                while attempts < self.max_attempts and not success:
                    attempts += 1
                    job["attempts"] = attempts
                    print(f"\n[Kaggle Grouped Job] {job_id} (Job {job_idx + 1}/{total_jobs} | Attempt {attempts}/{self.max_attempts})...")
                    print(f"  Chunks: {len(job_chunks)} chunks ({job_chunks[0]}..{job_chunks[-1]}) | Expected Rows: {expected_rows:,}")

                    # 1. Upload Grouped Dataset containing job_manifest.json + all chunk parquets
                    ds_ok, ds_msg, up_sec = self.kaggle_ctrl.upload_grouped_dataset(job, self.input_dir, self.staging_dir)
                    if not ds_ok:
                        print(f"  [DATASET ERROR] {ds_msg}")
                        time.sleep(5)
                        continue

                    # 2. Prepare lightweight kernel package
                    kernel_dir = self.kaggle_ctrl.prepare_kernel_package(job_id, self.staging_dir)

                    # 3. Submit Kernel
                    push_ok, push_msg = self.kaggle_ctrl.submit_job(kernel_dir, job_id)
                    if not push_ok:
                        print(f"  [SUBMISSION ERROR] {push_msg}")
                        time.sleep(5)
                        continue

                    # 4. Poll status with queue timeout (180s)
                    poll_ok, poll_msg, telemetry = self.kaggle_ctrl.poll_job_status(job_id)
                    if not poll_ok:
                        print(f"  [FAILURE] {poll_msg}")
                        # Non-retryable deterministic errors
                        if "ID_ALIGNMENT_MISMATCH" in poll_msg or "ROW_COUNT_MISMATCH" in poll_msg:
                            print("  [DETERMINISTIC FAILURE] Halting retries.")
                            break
                        time.sleep(10)
                        continue

                    # 5. Download outputs
                    dl_ok, dl_msg, dl_sec = self.kaggle_ctrl.download_output(self.output_dir, job_id)
                    if not dl_ok:
                        print(f"  [DOWNLOAD ERROR] {dl_msg}")
                        time.sleep(5)
                        continue

                    # 6. Verify grouped output row-for-row on EC2
                    v_ok, v_msg, v_summary = self.kaggle_ctrl.verify_grouped_output(
                        job,
                        self.input_dir,
                        self.output_dir
                    )
                    if not v_ok:
                        print(f"  [VERIFICATION ERROR] {v_msg}")
                        job["status"] = "failed"
                        job["error_message"] = v_msg
                        self.exporter.save_manifest(manifest)
                        # Halt on deterministic verification errors
                        if "MISMATCH" in v_msg or "VALIDATION_ERROR" in v_msg:
                            print("  [FATAL DETERMINISTIC ERROR]: Refusing blind retry.")
                            break
                        time.sleep(5)
                        continue

                    # Success! Mark job and all its chunks as completed
                    success = True
                    job["status"] = "completed"
                    job["completed_chunks"] = len(job_chunks)
                    job["completed_at"] = datetime.now(timezone.utc).isoformat()
                    job["telemetry"] = telemetry

                    for c_id in job_chunks:
                        c_meta = chunk_map[c_id]
                        c_prefix = f"chunk_{c_id:06d}"
                        c_meta["status"] = "completed"
                        c_meta["completed_at"] = datetime.now(timezone.utc).isoformat()
                        if c_prefix in v_summary.get("chunks", {}):
                            c_meta["output_sha256"] = v_summary["chunks"][c_prefix]["output_sha256"]

                    self.exporter.save_manifest(manifest)
                    print(f"  -> Job {job_id} ({expected_rows:,} rows) VERIFIED & COMPLETED successfully!")

                    # Clean staging
                    shutil.rmtree(kernel_dir, ignore_errors=True)
                    ds_staging = os.path.join(self.staging_dir, f"dataset_staging_{job_id}")
                    shutil.rmtree(ds_staging, ignore_errors=True)

                if not success:
                    print(f"\n[FATAL ERROR]: Grouped job {job_id} failed! Halting pipeline.")
                    sys.exit(1)

        # Stage 3: Direct Memmap Assembly
        self.assemble_memmap(manifest)

        # Stage 4: FAISS Index Construction
        if not skip_faiss:
            print("\n" + "=" * 80)
            print("[Stage 4: FAISS ANN Index Construction]")
            print("=" * 80)
            build_faiss_index(
                embeddings_path=self.merged_emb_path,
                ids_path=self.merged_ids_path,
                output_dir=self.faiss_dir,
                index_type="ivf_pq" if self.is_production else "flat"
            )

        total_elapsed = time.time() - t0_all
        print("\n" + "=" * 80)
        print(f"ANTIGRAVITY V5 PIPELINE FINISHED IN {total_elapsed:.2f}s.")
        print(f"  Embeddings: {self.merged_emb_path}")
        print(f"  FAISS:      {os.path.join(self.faiss_dir, 'target.index')}")
        print(f"  Peak RSS:   {get_peak_rss_mb():.2f} MB")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Antigravity V5 Master Dense Multilingual E5 Pipeline")
    parser.add_argument("--smoke", action="store_true", help="Run 10,000-record smoke test")
    parser.add_argument("--production", action="store_true", help="Run full 10,320,219 production universe")
    parser.add_argument("--gpu", action="store_true", default=True, help="Enable Kaggle GPU execution")
    parser.add_argument("--config", type=str, default="config/dense_gpu.yaml", help="Path to YAML config")
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_SIZE, help="Rows per chunk")
    parser.add_argument("--job-rows", type=int, default=DEFAULT_JOB_TARGET_ROWS, help="Rows per grouped Kaggle job")
    parser.add_argument("--batch-size", type=str, default="auto", help="Batch size or 'auto'")
    parser.add_argument("--max-concurrent-jobs", type=int, default=1, help="Max concurrent Kaggle jobs")
    parser.add_argument("--queue-timeout", type=int, default=180, help="Queue timeout seconds")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Print execution plan without running")
    parser.add_argument("--skip-export", action="store_true", help="Skip DuckDB export")
    parser.add_argument("--skip-kaggle", action="store_true", help="Skip Kaggle execution")
    parser.add_argument("--skip-faiss", action="store_true", help="Skip FAISS build")
    args = parser.parse_args()

    if not args.smoke and not args.production and not args.dry_run:
        print("[SAFETY ERROR]: You must explicitly specify either '--smoke' or '--production' (or '--dry-run').")
        print("This prevents accidental execution of the full 10.32M production workload.")
        sys.exit(1)

    config_dict = load_yaml_config(args.config)

    pipeline = DenseMasterPipeline(
        config_dict=config_dict,
        is_production=args.production,
        is_smoke=args.smoke,
        chunk_rows=args.chunk_rows,
        job_target_rows=args.job_rows,
        resume=args.resume,
        max_concurrent_jobs=args.max_concurrent_jobs,
        queue_timeout_seconds=args.queue_timeout,
        model_name=args.model
    )

    if args.dry_run:
        pipeline.print_plan()
    else:
        pipeline.run_pipeline(
            skip_export=args.skip_export,
            skip_kaggle=args.skip_kaggle,
            skip_faiss=args.skip_faiss
        )


if __name__ == "__main__":
    main()
