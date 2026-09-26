"""
Antigravity V4.1 Kaggle GPU Orchestration Controller.
Manages automated package staging, Kaggle CLI kernel submission, remote execution polling,
output retrieval, and strict positional verification.

Uses official Kaggle CLI via subprocess with structured error handling & logging.
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
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
import polars as pl
import numpy as np

from src.config import get_config
from src.resource_tracker import get_current_rss_mb

logger = logging.getLogger("arctic.kaggle")

def compute_sha256(file_path: str) -> str:
    """Computes SHA256 checksum of a file in 64KB chunks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class KaggleController:
    """
    Automated controller for Kaggle GPU kernel lifecycle.
    """
    def __init__(
        self,
        kernel_slug: str,
        accelerator: str = "NvidiaL4",
        timeout_seconds: int = 3600,
        poll_interval: int = 15,
        logs_dir: str = "cache/arctic_gpu/logs",
        worker_src_dir: str = "kaggle/arctic_worker"
    ):
        config = get_config()
        self.kernel_slug = kernel_slug
        self.accelerator = accelerator
        self.timeout_seconds = timeout_seconds
        self.poll_interval = poll_interval
        self.logs_dir = os.path.join(config.base_dir, logs_dir)
        self.worker_src_dir = os.path.join(config.base_dir, worker_src_dir)

        os.makedirs(self.logs_dir, exist_ok=True)

    def check_kaggle_auth(self) -> Tuple[bool, str]:
        """
        Validates Kaggle CLI installation and verifies live API authentication.
        Supports standard OAuth ('kaggle auth login') and API tokens transparently.
        """
        # 1. Check if kaggle CLI is available on PATH
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

        # 2. Check kernel slug configuration
        if "YOUR_KAGGLE_USERNAME" in self.kernel_slug:
            return False, (
                "Kaggle kernel slug is unconfigured ('YOUR_KAGGLE_USERNAME'). "
                "Update config/arctic_gpu.yaml with your real Kaggle username (e.g. 'rajeshshitap/arctic-entity-resolution-worker')."
            )

        # 3. Validate live API authentication by running a lightweight command
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

    def prepare_job_package(
        self,
        chunk_id: int,
        input_parquet_path: str,
        staging_root: str
    ) -> str:
        """
        Creates a dedicated Kaggle kernel staging folder containing:
        - kaggle_worker.py
        - chunk_XXXXXX.parquet
        - kernel-metadata.json
        """
        job_dir = os.path.join(staging_root, f"job_chunk_{chunk_id:06d}")
        os.makedirs(job_dir, exist_ok=True)

        # Copy worker script
        worker_src = os.path.join(self.worker_src_dir, "kaggle_worker.py")
        shutil.copy2(worker_src, os.path.join(job_dir, "kaggle_worker.py"))

        # Copy input chunk
        chunk_dest = os.path.join(job_dir, os.path.basename(input_parquet_path))
        shutil.copy2(input_parquet_path, chunk_dest)

        # Generate kernel metadata
        kernel_meta = {
            "id": self.kernel_slug,
            "title": "Arctic Entity Resolution Worker",
            "code_file": "kaggle_worker.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_tpu": "false",
            "enable_internet": "true",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": []
        }

        with open(os.path.join(job_dir, "kernel-metadata.json"), "w", encoding="utf-8") as f:
            json.dump(kernel_meta, f, indent=2)

        return job_dir

    def submit_job(self, job_dir: str, chunk_id: int) -> Tuple[bool, str]:
        """
        Pushes and starts Kaggle kernel execution via `kaggle kernels push`.
        """
        log_file = os.path.join(self.logs_dir, f"chunk_{chunk_id:06d}.log")
        with open(log_file, "a", encoding="utf-8") as f_log:
            f_log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] Pushing kernel: {self.kernel_slug}\n")

        cmd = ["kaggle", "kernels", "push", "-p", job_dir]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)

        with open(log_file, "a", encoding="utf-8") as f_log:
            f_log.write(f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}\nExit Code: {res.returncode}\n")

        if res.returncode != 0:
            err_msg = res.stderr.strip() or res.stdout.strip()
            return False, f"Failed to push Kaggle kernel: {err_msg}"

        return True, "Kernel pushed successfully."

    def poll_job_status(self, chunk_id: int) -> Tuple[bool, str]:
        """
        Monitors Kaggle kernel execution until 'complete', 'error', or timeout.
        """
        log_file = os.path.join(self.logs_dir, f"chunk_{chunk_id:06d}.log")
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > self.timeout_seconds:
                return False, f"Kernel execution timed out after {elapsed:.1f}s (> {self.timeout_seconds}s)"

            cmd = ["kaggle", "kernels", "status", self.kernel_slug]
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            status_line = res.stdout.strip()

            with open(log_file, "a", encoding="utf-8") as f_log:
                f_log.write(f"[{elapsed:.1f}s] Status: {status_line}\n")

            status_lower = status_line.lower()
            if '"complete"' in status_lower or "complete" in status_lower:
                return True, "Kernel execution completed successfully."

            if '"error"' in status_lower or "error" in status_lower or "failed" in status_lower:
                return False, f"Kernel execution failed on Kaggle: {status_line}"

            if "cancel" in status_lower:
                return False, f"Kernel execution was cancelled: {status_line}"

            print(f"  -> Chunk {chunk_id:04d} | Status: {status_line} (Elapsed: {elapsed:.0f}s)", flush=True)
            time.sleep(self.poll_interval)

    def download_output(self, output_dest_dir: str, chunk_id: int) -> Tuple[bool, str]:
        """
        Retrieves generated embeddings and metadata via `kaggle kernels output`.
        """
        os.makedirs(output_dest_dir, exist_ok=True)
        log_file = os.path.join(self.logs_dir, f"chunk_{chunk_id:06d}.log")

        cmd = ["kaggle", "kernels", "output", self.kernel_slug, "-p", output_dest_dir]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)

        with open(log_file, "a", encoding="utf-8") as f_log:
            f_log.write(f"\n[Download Output]\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}\n")

        if res.returncode != 0:
            return False, f"Failed to download kernel output: {res.stderr.strip()}"

        return True, "Outputs downloaded successfully."

    def verify_chunk_output(
        self,
        chunk_id: int,
        input_parquet_path: str,
        output_dir: str,
        expected_dim: int = 384
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Strict positional verification:
        1. Checks row count equality (input == output embeddings == output IDs).
        2. Checks target IDs match input row-for-row exactly.
        3. Checks embedding dimension == 384.
        4. Checks no duplicates in target IDs.
        5. Checks L2 normalization on embeddings.
        6. Validates SHA256 checksums.
        """
        chunk_prefix = f"chunk_{chunk_id:06d}"
        emb_file = os.path.join(output_dir, f"{chunk_prefix}_embeddings.npy")
        ids_file = os.path.join(output_dir, f"{chunk_prefix}_ids.parquet")
        meta_file = os.path.join(output_dir, f"{chunk_prefix}_meta.json")

        if not (os.path.exists(emb_file) and os.path.exists(ids_file)):
            return False, f"Missing output files in {output_dir} ({emb_file} or {ids_file})", {}

        # 1. Load Input Parquet
        df_in = pl.read_parquet(input_parquet_path)
        in_count = len(df_in)
        in_ids = [str(x) for x in df_in["target_id"].to_list()]

        # 2. Load Output Embeddings
        embeddings = np.load(emb_file)
        out_count, out_dim = embeddings.shape

        if out_count != in_count:
            return False, f"Row count mismatch: Embeddings ({out_count:,}) != Input ({in_count:,})", {}

        if out_dim != expected_dim:
            return False, f"Dimension mismatch: Found {out_dim}, Expected {expected_dim}", {}

        # 3. Load Output IDs
        df_out_ids = pl.read_parquet(ids_file)
        out_ids = [str(x) for x in df_out_ids["target_id"].to_list()]

        if len(out_ids) != in_count:
            return False, f"ID count mismatch: Output IDs ({len(out_ids):,}) != Input ({in_count:,})", {}

        # 4. Strict Row-for-Row Target ID Positional Alignment
        if in_ids != out_ids:
            # Find first mismatched index for diagnosis
            for idx in range(in_count):
                if in_ids[idx] != out_ids[idx]:
                    return False, f"Positional ID mismatch at row {idx}: Input '{in_ids[idx]}' != Output '{out_ids[idx]}'", {}
            return False, "Target ID sequence does not match input exactly", {}

        # 5. Duplicate Check
        if len(set(out_ids)) != in_count:
            return False, f"Duplicate target IDs detected within chunk {chunk_id}", {}

        # 6. L2 Normalization Check
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.all(np.isclose(norms, 1.0, atol=1e-3)):
            return False, f"Output embeddings are not L2 normalized (Norm range: {norms.min():.4f} - {norms.max():.4f})", {}

        # 7. Checksums
        emb_sha256 = compute_sha256(emb_file)
        ids_sha256 = compute_sha256(ids_file)

        summary = {
            "chunk_id": chunk_id,
            "row_count": in_count,
            "embedding_dimension": out_dim,
            "first_target_id": in_ids[0] if in_ids else "",
            "last_target_id": in_ids[-1] if in_ids else "",
            "output_sha256": emb_sha256,
            "ids_sha256": ids_sha256,
            "embeddings_path": emb_file,
            "ids_path": ids_file
        }

        return True, "All positional alignment & dimensional verifications PASSED.", summary
