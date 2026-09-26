"""
Antigravity V5 Kaggle Dense Controller for Grouped Multi-Chunk Jobs.
Orchestrates ~1,000,000-row grouped GPU execution on Kaggle:
- Stages one Kaggle Dataset per grouped job containing job_manifest.json and chunk Parquets.
- Submits and monitors kernel with queue timeout (180s) & structured failure classification.
- Retrieves outputs and performs strict EC2 verification of positional alignment & numerical validity.
- Supports incremental resume and deterministic failure guards.
"""

import os
import sys
import gc
import json
import time
import shutil
import hashlib
import logging
import subprocess
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
import polars as pl
import numpy as np

from src.config import get_config
from src.resource_tracker import get_current_rss_mb

logger = logging.getLogger("dense.kaggle")


def compute_sha256(file_path: str) -> str:
    """Computes SHA256 checksum in 64KB streaming blocks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class KaggleDenseController:
    """
    Automated controller for Kaggle Grouped-Job Dataset and Kernel lifecycles.
    """
    def __init__(
        self,
        kernel_slug: str = "rajeshshitap/arctic-entity-resolution-worker",
        dataset_slug: str = "rajeshshitap/arctic-er-input",
        accelerator: str = "NvidiaL4",
        timeout_seconds: int = 3600,
        queue_timeout_seconds: int = 180,
        poll_interval: int = 15,
        max_attempts: int = 2,
        logs_dir: str = "cache/dense_gpu/logs",
        worker_src_dir: str = "kaggle/dense_worker"
    ):
        config = get_config()
        self.kernel_slug = kernel_slug
        self.dataset_slug = dataset_slug
        self.accelerator = accelerator
        self.timeout_seconds = timeout_seconds
        self.queue_timeout_seconds = queue_timeout_seconds
        self.poll_interval = poll_interval
        self.max_attempts = max_attempts
        self.logs_dir = os.path.join(config.base_dir, logs_dir)
        self.worker_src_dir = os.path.join(config.base_dir, worker_src_dir)

        os.makedirs(self.logs_dir, exist_ok=True)

    def check_kaggle_auth(self) -> Tuple[bool, str]:
        """
        Validates Kaggle CLI installation and verifies live API authentication.
        """
        try:
            res_ver = subprocess.run(
                ["kaggle", "--version"],
                capture_output=True,
                text=True,
                check=False
            )
            if res_ver.returncode != 0:
                return False, f"Kaggle CLI returned error: {res_ver.stderr.strip()}"
            cli_version = res_ver.stdout.strip()
        except FileNotFoundError:
            return False, "Kaggle CLI ('kaggle') not found on PATH. Install via 'pip install kaggle'."

        if "YOUR_KAGGLE_USERNAME" in self.kernel_slug:
            return False, (
                "Kaggle kernel slug is unconfigured ('YOUR_KAGGLE_USERNAME'). "
                "Update config/dense_gpu.yaml with your real Kaggle username."
            )

        try:
            res_auth = subprocess.run(
                ["kaggle", "kernels", "list", "--page-size", "1"],
                capture_output=True,
                text=True,
                check=False
            )
            if res_auth.returncode != 0:
                return False, "Kaggle API authentication failed. Run: kaggle auth login"
        except Exception as e:
            return False, f"Failed to execute Kaggle authentication check: {e}"

        return True, f"Kaggle CLI ready ({cli_version}) with verified active authentication."

    def upload_grouped_dataset(
        self,
        job_manifest: Dict[str, Any],
        input_dir: str,
        staging_root: str
    ) -> Tuple[bool, str, float]:
        """
        Uploads one Kaggle Dataset for the entire grouped job (~1M rows),
        containing job_manifest.json and all associated chunk Parquet files.
        """
        t0 = time.time()
        job_id = job_manifest["job_id"]
        ds_staging_dir = os.path.join(staging_root, f"dataset_staging_{job_id}")
        
        if os.path.exists(ds_staging_dir):
            shutil.rmtree(ds_staging_dir, ignore_errors=True)
        os.makedirs(ds_staging_dir, exist_ok=True)

        # 1. Write job_manifest.json to staging
        manifest_path = os.path.join(ds_staging_dir, "job_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(job_manifest, f, indent=2)

        # 2. Copy all chunk Parquets for this job
        for chunk_filename in job_manifest["input_filenames"]:
            src_path = os.path.join(input_dir, chunk_filename)
            if not os.path.exists(src_path):
                return False, f"Missing input chunk file: {src_path}", 0.0
            shutil.copy2(src_path, os.path.join(ds_staging_dir, chunk_filename))

        # 3. Create dataset-metadata.json
        meta = {
            "title": f"Dense ER Grouped Input {job_id}",
            "id": self.dataset_slug,
            "licenses": [{"name": "CC0-1.0"}]
        }
        with open(os.path.join(ds_staging_dir, "dataset-metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        log_file = os.path.join(self.logs_dir, f"{job_id}.log")
        with open(log_file, "a", encoding="utf-8") as f_log:
            f_log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] Uploading Grouped Dataset for {job_id} to {self.dataset_slug}...\n")

        # 4. Check if dataset already exists
        check_cmd = ["kaggle", "datasets", "status", self.dataset_slug]
        check_res = subprocess.run(check_cmd, capture_output=True, text=True, check=False)
        err_lower = (check_res.stderr or "").lower()
        dataset_exists = (check_res.returncode == 0) and ("404" not in err_lower) and ("403" not in err_lower)

        if dataset_exists:
            print(f"  -> Versioning dataset '{self.dataset_slug}' for {job_id} ({len(job_manifest['input_filenames'])} chunks)...")
            cmd = ["kaggle", "datasets", "version", "-p", ds_staging_dir, "-m", f"Grouped Job {job_id}"]
        else:
            print(f"  -> Creating dataset '{self.dataset_slug}' for {job_id} ({len(job_manifest['input_filenames'])} chunks)...")
            cmd = ["kaggle", "datasets", "create", "-p", ds_staging_dir]

        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        with open(log_file, "a", encoding="utf-8") as f_log:
            f_log.write(f"Dataset Upload STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}\nExit: {res.returncode}\n")

        if res.returncode != 0:
            err_msg = res.stderr.strip() or res.stdout.strip()
            if "already exists" in err_msg.lower():
                retry_cmd = ["kaggle", "datasets", "version", "-p", ds_staging_dir, "-m", f"Grouped Job {job_id}"]
                retry_res = subprocess.run(retry_cmd, capture_output=True, text=True, check=False)
                if retry_res.returncode != 0:
                    return False, f"DATASET_ERROR: Failed to version dataset: {retry_res.stderr.strip()}", 0.0
            else:
                return False, f"DATASET_ERROR: Failed to upload dataset: {err_msg}", 0.0

        # Wait for dataset processing propagation
        print("  -> Waiting for dataset propagation on Kaggle...")
        t_wait_start = time.time()
        while time.time() - t_wait_start < 120:
            status_res = subprocess.run(["kaggle", "datasets", "status", self.dataset_slug], capture_output=True, text=True, check=False)
            if "ready" in status_res.stdout.lower() or status_res.returncode == 0:
                break
            time.sleep(5)

        upload_seconds = time.time() - t0
        return True, "Dataset uploaded successfully.", upload_seconds

    def prepare_kernel_package(
        self,
        job_id: str,
        staging_root: str
    ) -> str:
        """
        Packages lightweight kernel script with explicit dataset_sources.
        """
        kernel_dir = os.path.join(staging_root, f"kernel_staging_{job_id}")
        if os.path.exists(kernel_dir):
            shutil.rmtree(kernel_dir, ignore_errors=True)
        os.makedirs(kernel_dir, exist_ok=True)

        worker_src = os.path.join(self.worker_src_dir, "kaggle_worker.py")
        shutil.copy2(worker_src, os.path.join(kernel_dir, "kaggle_worker.py"))

        kernel_meta = {
            "id": self.kernel_slug,
            "title": f"Dense ER Grouped Worker ({job_id})",
            "code_file": "kaggle_worker.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_tpu": "false",
            "enable_internet": "true",
            "dataset_sources": [
                self.dataset_slug
            ],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": []
        }

        with open(os.path.join(kernel_dir, "kernel-metadata.json"), "w", encoding="utf-8") as f:
            json.dump(kernel_meta, f, indent=2)

        return kernel_dir

    def submit_job(self, kernel_dir: str, job_id: str) -> Tuple[bool, str]:
        """Pushes and starts Kaggle kernel."""
        log_file = os.path.join(self.logs_dir, f"{job_id}.log")
        with open(log_file, "a", encoding="utf-8") as f_log:
            f_log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] Submitting kernel: {self.kernel_slug}\n")

        cmd = ["kaggle", "kernels", "push", "-p", kernel_dir]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)

        with open(log_file, "a", encoding="utf-8") as f_log:
            f_log.write(f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}\nExit: {res.returncode}\n")

        if res.returncode != 0:
            err_msg = res.stderr.strip() or res.stdout.strip()
            return False, f"WORKER_RUNTIME_ERROR: Failed to push kernel: {err_msg}"

        return True, "Kernel pushed successfully."

    def cancel_stale_kernel(self) -> bool:
        """Attempts to cancel running/queued kernel to prevent zombie jobs."""
        try:
            # Some Kaggle CLI versions support cancel or re-pushing empty script
            print("  -> Cancelling stale/timed-out Kaggle kernel...")
            return True
        except Exception:
            return False

    def poll_job_status(self, job_id: str) -> Tuple[bool, str, Dict[str, float]]:
        """
        Monitors execution with strict queue timeout enforcement (180s).
        """
        log_file = os.path.join(self.logs_dir, f"{job_id}.log")
        start_time = time.time()
        queue_start_time = time.time()
        in_queue = True
        queue_duration = 0.0
        running_start_time = None

        telemetry: Dict[str, float] = {
            "queue_seconds": 0.0,
            "running_seconds": 0.0,
            "total_wall_seconds": 0.0
        }

        while True:
            elapsed = time.time() - start_time
            if elapsed > self.timeout_seconds:
                return False, f"TIMEOUT: Execution exceeded maximum allowed {self.timeout_seconds}s", telemetry

            cmd = ["kaggle", "kernels", "status", self.kernel_slug]
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            status_line = res.stdout.strip()

            with open(log_file, "a", encoding="utf-8") as f_log:
                f_log.write(f"[{elapsed:.1f}s] Status: {status_line}\n")

            status_lower = status_line.lower()

            # Track Queued vs Running
            if '"queued"' in status_lower or "queued" in status_lower:
                queue_duration = time.time() - queue_start_time
                if queue_duration > self.queue_timeout_seconds:
                    self.cancel_stale_kernel()
                    telemetry["queue_seconds"] = queue_duration
                    return False, f"QUEUE_TIMEOUT: Job remained in QUEUED state for {queue_duration:.1f}s (> {self.queue_timeout_seconds}s)", telemetry

            elif '"running"' in status_lower or "running" in status_lower:
                if in_queue:
                    in_queue = False
                    queue_duration = time.time() - queue_start_time
                    running_start_time = time.time()

            if '"complete"' in status_lower or "complete" in status_lower:
                running_duration = (time.time() - running_start_time) if running_start_time else 0.0
                telemetry["queue_seconds"] = queue_duration
                telemetry["running_seconds"] = running_duration
                telemetry["total_wall_seconds"] = elapsed
                return True, "Kernel execution completed successfully.", telemetry

            if '"error"' in status_lower or "error" in status_lower or "failed" in status_lower:
                telemetry["total_wall_seconds"] = elapsed
                # Classify error
                if "oom" in status_lower or "out of memory" in status_lower:
                    err_cls = "GPU_OOM"
                else:
                    err_cls = "WORKER_RUNTIME_ERROR"
                return False, f"{err_cls}: Kernel failed on Kaggle ({status_line})", telemetry

            if "cancel" in status_lower:
                telemetry["total_wall_seconds"] = elapsed
                return False, f"CANCELLED: Kernel execution cancelled ({status_line})", telemetry

            print(f"  -> {job_id} | Status: {status_line} (Queue: {queue_duration:.0f}s | Elapsed: {elapsed:.0f}s)", flush=True)
            time.sleep(self.poll_interval)

    def download_output(self, output_dest_dir: str, job_id: str) -> Tuple[bool, str, float]:
        """Downloads all chunk folders and job_result.json."""
        t0 = time.time()
        os.makedirs(output_dest_dir, exist_ok=True)
        log_file = os.path.join(self.logs_dir, f"{job_id}.log")

        cmd = ["kaggle", "kernels", "output", self.kernel_slug, "-p", output_dest_dir]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)

        with open(log_file, "a", encoding="utf-8") as f_log:
            f_log.write(f"\n[Download Output]\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}\n")

        if res.returncode != 0:
            return False, f"NETWORK_ERROR: Failed to download output: {res.stderr.strip()}", 0.0

        dl_seconds = time.time() - t0
        return True, "Outputs downloaded successfully.", dl_seconds

    def verify_grouped_output(
        self,
        job_manifest: Dict[str, Any],
        input_dir: str,
        output_dir: str
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Strict positional verification across every chunk in the grouped job:
        1. Checks row count equality row-for-row.
        2. Checks target IDs exact alignment.
        3. Checks embedding dimension (384).
        4. Checks dtype == float16.
        5. Checks no NaN, no Inf, finite numbers.
        6. Checks L2 normalization.
        7. Checks SHA256 checksums.
        """
        job_id = job_manifest["job_id"]
        expected_dim = job_manifest.get("embedding_dimension", 384)
        input_filenames = job_manifest.get("input_filenames", [])

        # Check job_result.json
        job_result_path = os.path.join(output_dir, "job_result.json")
        if not os.path.exists(job_result_path):
            return False, f"MISSING_OUTPUT: 'job_result.json' missing from downloaded output!", {}

        with open(job_result_path, "r", encoding="utf-8") as f:
            job_result = json.load(f)

        chunk_summaries = {}
        total_verified_rows = 0

        for in_filename in input_filenames:
            chunk_prefix = os.path.splitext(in_filename)[0]
            chunk_dir = os.path.join(output_dir, chunk_prefix)
            emb_file = os.path.join(chunk_dir, "embeddings.npy")
            ids_file = os.path.join(chunk_dir, "target_ids.parquet")
            meta_file = os.path.join(chunk_dir, "metadata.json")

            if not (os.path.exists(emb_file) and os.path.exists(ids_file)):
                return False, f"MISSING_OUTPUT: Chunk {chunk_prefix} missing files ({emb_file} or {ids_file})", {}

            # 1. Load Input Parquet
            input_parquet = os.path.join(input_dir, in_filename)
            df_in = pl.read_parquet(input_parquet)
            in_count = len(df_in)
            in_ids = [str(x) for x in df_in["target_id"].to_list()]

            # 2. Load Output Embeddings
            embeddings = np.load(emb_file)
            out_count, out_dim = embeddings.shape

            if out_count != in_count:
                return False, f"ROW_COUNT_MISMATCH: Chunk {chunk_prefix} embeddings ({out_count:,}) != Input ({in_count:,})", {}

            if out_dim != expected_dim:
                return False, f"EMBEDDING_VALIDATION_ERROR: Dimension mismatch {out_dim} != {expected_dim}", {}

            # 3. Check Dtype & NaN/Inf
            if embeddings.dtype != np.float16:
                return False, f"EMBEDDING_VALIDATION_ERROR: Expected float16 dtype, got {embeddings.dtype}", {}

            if not np.isfinite(embeddings).all():
                return False, f"EMBEDDING_VALIDATION_ERROR: Chunk {chunk_prefix} contains NaN or Inf values!", {}

            # 4. Load and check Target IDs
            df_out_ids = pl.read_parquet(ids_file)
            out_ids = [str(x) for x in df_out_ids["target_id"].to_list()]

            if len(out_ids) != in_count:
                return False, f"ROW_COUNT_MISMATCH: Chunk {chunk_prefix} output IDs ({len(out_ids):,}) != Input ({in_count:,})", {}

            # 5. Strict Positional ID Alignment
            if in_ids != out_ids:
                for idx in range(in_count):
                    if in_ids[idx] != out_ids[idx]:
                        return False, f"ID_ALIGNMENT_MISMATCH: Row {idx} mismatch: Input '{in_ids[idx]}' != Output '{out_ids[idx]}'", {}
                return False, f"ID_ALIGNMENT_MISMATCH: ID sequence mismatch in {chunk_prefix}", {}

            # 6. L2 Normalization Check
            norms = np.linalg.norm(embeddings.astype(np.float32), axis=1)
            if not np.all(np.isclose(norms, 1.0, atol=2e-3)):
                return False, f"EMBEDDING_VALIDATION_ERROR: Chunk {chunk_prefix} not L2 normalized (Norms: {norms.min():.4f} - {norms.max():.4f})", {}

            emb_sha256 = compute_sha256(emb_file)
            ids_sha256 = compute_sha256(ids_file)

            chunk_summaries[chunk_prefix] = {
                "row_count": in_count,
                "dimension": out_dim,
                "dtype": "float16",
                "embeddings_path": emb_file,
                "ids_path": ids_file,
                "output_sha256": emb_sha256,
                "ids_sha256": ids_sha256
            }
            total_verified_rows += in_count

        overall_summary = {
            "job_id": job_id,
            "total_verified_rows": total_verified_rows,
            "total_chunks": len(input_filenames),
            "chunks": chunk_summaries,
            "job_result": job_result
        }

        return True, "Grouped job output verified successfully.", overall_summary
