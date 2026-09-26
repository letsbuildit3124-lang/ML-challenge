"""
Antigravity One-Shot Kaggle E5 Controller.
Manages the complete remote Kaggle GPU lifecycle:
1. --create-dataset / --upload-dataset: Uploads the single targets.parquet file to Kaggle.
2. --submit: Packages and pushes the one-shot worker kernel with GPU & internet enabled.
3. --status: Monitors live remote kernel execution status.
4. --download: Resumably downloads generated embedding shards, ID parquets, and manifest.json.
"""

import os
import sys
import json
import time
import shutil
import subprocess
import argparse
from typing import Tuple, Dict, Any, Optional

from src.config import get_config


class KaggleE5Controller:
    """
    Automated controller for the One-Shot Kaggle E5 dataset and kernel lifecycle.
    """
    def __init__(
        self,
        dataset_slug: str = "rajeshshitap/e5-target-input",
        kernel_slug: str = "rajeshshitap/e5-embedding-worker",
        accelerator: str = "NvidiaL4",
        timeout_seconds: int = 7200,
        staging_dir: str = "cache/e5_gpu/staging",
        worker_src_dir: str = "kaggle/e5_worker",
        output_dir: str = "cache/e5_gpu/output",
        logs_dir: str = "cache/e5_gpu/logs"
    ):
        config = get_config()
        self.dataset_slug = dataset_slug
        self.kernel_slug = kernel_slug
        self.accelerator = accelerator
        self.timeout_seconds = timeout_seconds

        self.staging_dir = os.path.join(config.base_dir, staging_dir)
        self.worker_src_dir = os.path.join(config.base_dir, worker_src_dir)
        self.output_dir = os.path.join(config.base_dir, output_dir)
        self.logs_dir = os.path.join(config.base_dir, logs_dir)

        os.makedirs(self.staging_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    def check_kaggle_auth(self) -> Tuple[bool, str]:
        """Validates Kaggle CLI installation and OAuth authentication."""
        try:
            res_ver = subprocess.run(["kaggle", "--version"], capture_output=True, text=True, check=False)
            if res_ver.returncode != 0:
                return False, f"Kaggle CLI returned error: {res_ver.stderr.strip()}"
            cli_version = res_ver.stdout.strip()
        except FileNotFoundError:
            return False, "Kaggle CLI ('kaggle') not found on PATH. Install via 'pip install kaggle'."

        try:
            res_auth = subprocess.run(["kaggle", "kernels", "list", "--page-size", "1"], capture_output=True, text=True, check=False)
            if res_auth.returncode != 0:
                return False, "Kaggle API authentication failed. Run 'kaggle auth login' to authenticate."
        except Exception as e:
            return False, f"Failed to execute Kaggle authentication check: {e}"

        return True, f"Kaggle CLI ready ({cli_version}) with verified active authentication."

    def upload_dataset(self, input_parquet_path: str) -> bool:
        """
        Creates or versions the single Kaggle Dataset containing targets.parquet.
        """
        if not os.path.exists(input_parquet_path):
            print(f"[ERROR]: Input Parquet file not found at: {input_parquet_path}")
            print("Run 'python3 -m src.export_e5_input' first!")
            return False

        ds_staging = os.path.join(self.staging_dir, "dataset_staging")
        if os.path.exists(ds_staging):
            shutil.rmtree(ds_staging, ignore_errors=True)
        os.makedirs(ds_staging, exist_ok=True)

        # Copy target parquet
        dest_parquet = os.path.join(ds_staging, "targets.parquet")
        print(f"[KaggleController] Staging {input_parquet_path} -> {dest_parquet}...")
        shutil.copy2(input_parquet_path, dest_parquet)

        # Create dataset-metadata.json
        meta = {
            "title": "E5 Target Input Data",
            "id": self.dataset_slug,
            "licenses": [{"name": "CC0-1.0"}]
        }
        with open(os.path.join(ds_staging, "dataset-metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Check if dataset already exists
        check_cmd = ["kaggle", "datasets", "status", self.dataset_slug]
        check_res = subprocess.run(check_cmd, capture_output=True, text=True, check=False)
        err_lower = (check_res.stderr or "").lower()
        dataset_exists = (check_res.returncode == 0) and ("404" not in err_lower) and ("403" not in err_lower)

        if dataset_exists:
            print(f"[KaggleController] Versioning existing dataset '{self.dataset_slug}'...")
            cmd = ["kaggle", "datasets", "version", "-p", ds_staging, "-m", f"Target Input Upload {time.strftime('%Y-%m-%d %H:%M:%S')}"]
        else:
            print(f"[KaggleController] Creating new dataset '{self.dataset_slug}'...")
            cmd = ["kaggle", "datasets", "create", "-p", ds_staging]

        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            err_msg = res.stderr.strip() or res.stdout.strip()
            if "already exists" in err_msg.lower():
                print("[KaggleController] Dataset exists. Retrying with version...")
                retry_cmd = ["kaggle", "datasets", "version", "-p", ds_staging, "-m", "Target Input Upload"]
                retry_res = subprocess.run(retry_cmd, capture_output=True, text=True, check=False)
                if retry_res.returncode != 0:
                    print(f"[ERROR]: Failed to version dataset: {retry_res.stderr.strip()}")
                    return False
            else:
                print(f"[ERROR]: Failed to upload dataset: {err_msg}")
                return False

        print("[KaggleController] Waiting for dataset processing on Kaggle...")
        t0 = time.time()
        while time.time() - t0 < 120:
            status_res = subprocess.run(["kaggle", "datasets", "status", self.dataset_slug], capture_output=True, text=True, check=False)
            if "ready" in status_res.stdout.lower() or status_res.returncode == 0:
                break
            time.sleep(5)

        print(f"[KaggleController SUCCESS] Dataset '{self.dataset_slug}' is READY on Kaggle!")
        return True

    def submit_kernel(self) -> bool:
        """
        Prepares and pushes the one-shot kernel package to Kaggle.
        """
        kernel_staging = os.path.join(self.staging_dir, "kernel_staging")
        if os.path.exists(kernel_staging):
            shutil.rmtree(kernel_staging, ignore_errors=True)
        os.makedirs(kernel_staging, exist_ok=True)

        worker_script = os.path.join(self.worker_src_dir, "kaggle_e5_worker.py")
        if not os.path.exists(worker_script):
            print(f"[ERROR]: Worker script not found at {worker_script}!")
            return False

        shutil.copy2(worker_script, os.path.join(kernel_staging, "kaggle_e5_worker.py"))

        # Generate kernel-metadata.json
        meta = {
            "id": self.kernel_slug,
            "title": "E5 Multilingual Entity Resolution Worker",
            "code_file": "kaggle_e5_worker.py",
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
        with open(os.path.join(kernel_staging, "kernel-metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"[KaggleController] Pushing kernel '{self.kernel_slug}' to Kaggle GPU...")
        cmd = ["kaggle", "kernels", "push", "-p", kernel_staging]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if res.returncode != 0:
            print(f"[ERROR]: Failed to submit kernel: {res.stderr.strip() or res.stdout.strip()}")
            return False

        print(f"[KaggleController SUCCESS] Kernel '{self.kernel_slug}' submitted successfully!")
        print(f"Run 'PYTHONPATH=. python3 -m src.kaggle_e5_controller --status' to monitor progress.")
        return True

    def check_status(self) -> str:
        """Checks remote kernel status."""
        cmd = ["kaggle", "kernels", "status", self.kernel_slug]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        status_line = res.stdout.strip()
        print(f"[Kaggle Kernel Status] {self.kernel_slug}: {status_line}")
        return status_line

    def download_output(self, dest_dir: Optional[str] = None) -> bool:
        """Resumably downloads generated embedding shards and manifest."""
        target_dest = dest_dir or self.output_dir
        os.makedirs(target_dest, exist_ok=True)

        print(f"[KaggleController] Downloading output from '{self.kernel_slug}' to '{target_dest}'...")
        cmd = ["kaggle", "kernels", "output", self.kernel_slug, "-p", target_dest]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if res.returncode != 0:
            print(f"[ERROR]: Failed to download output: {res.stderr.strip() or res.stdout.strip()}")
            return False

        print(f"[KaggleController SUCCESS] Outputs downloaded to: {target_dest}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Antigravity One-Shot Kaggle E5 Controller")
    parser.add_argument("--create-dataset", action="store_true", help="Upload input Parquet to Kaggle Dataset")
    parser.add_argument("--submit", action="store_true", help="Submit One-Shot GPU Worker Kernel")
    parser.add_argument("--status", action="store_true", help="Check execution status of remote Kaggle Kernel")
    parser.add_argument("--download", action="store_true", help="Download embedding shards from completed Kernel")
    parser.add_argument("--input-parquet", type=str, default="cache/e5_gpu/input/targets.parquet", help="Path to input Parquet")
    parser.add_argument("--output-dir", type=str, default="cache/e5_gpu/output", help="Directory to store downloaded shards")
    parser.add_argument("--dataset-slug", type=str, default="rajeshshitap/e5-target-input", help="Kaggle Dataset Slug")
    parser.add_argument("--kernel-slug", type=str, default="rajeshshitap/e5-embedding-worker", help="Kaggle Kernel Slug")
    args = parser.parse_args()

    controller = KaggleE5Controller(
        dataset_slug=args.dataset_slug,
        kernel_slug=args.kernel_slug,
        output_dir=args.output_dir
    )

    auth_ok, auth_msg = controller.check_kaggle_auth()
    if not auth_ok:
        print(f"[KAGGLE AUTH ERROR]: {auth_msg}")
        sys.exit(1)

    if args.create_dataset:
        controller.upload_dataset(args.input_parquet)
    elif args.submit:
        controller.submit_kernel()
    elif args.status:
        controller.check_status()
    elif args.download:
        controller.download_output(args.output_dir)
    else:
        print("Specify an action: --create-dataset, --submit, --status, or --download.")


if __name__ == "__main__":
    main()
