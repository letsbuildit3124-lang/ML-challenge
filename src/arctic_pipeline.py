"""
Antigravity V4.1 Master Arctic GPU Orchestration Pipeline.
Entry Point: python3 -m src.arctic_pipeline --gpu

Automates the complete end-to-end lifecycle:
1. Environment & Pre-flight Storage Verification
2. DuckDB Target Text Chunk Export
3. Kaggle GPU Worker Staging, Submission & Polling
4. Automatic Output Retrieval & Strict Positional Verification
5. Chunked Embedding Assembly into Persistent Memmap
6. Memory-Bounded FAISS ANN Index Construction
7. Hybrid V4 + Arctic Dense Retrieval Benchmarking
8. Comprehensive Report Generation
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
from src.export_arctic_gpu import ArcticGPUExporter, EXPECTED_TOTAL_TARGETS, DEFAULT_CHUNK_SIZE
from src.kaggle_controller import KaggleController
from src.build_arctic_faiss import build_faiss_index
from src.v4_1_dense_benchmark import run_dense_benchmark
from src.arctic_embeddings import EMBEDDING_DIM, DEFAULT_MODEL_NAME

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("arctic.pipeline")

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Loads YAML config using standard dictionary fallback if PyYAML is absent."""
    if not os.path.exists(config_path):
        return {}
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # Simple fallback parser for basic key-value YAML files
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


class ArcticGPUOrchestrator:
    """
    Automated orchestrator managing EC2 DuckDB -> Kaggle GPU -> FAISS -> Hybrid Benchmark.
    """
    def __init__(
        self,
        config_dict: Dict[str, Any],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        dense_top_k: int = 50,
        chunks_per_job: int = 1,
        max_retries: int = 3,
        resume: bool = True,
        limit_targets: Optional[int] = None,
        limit_chunks: Optional[int] = None
    ):
        self.app_config = get_config()
        self.config_dict = config_dict
        self.chunk_size = chunk_size
        self.dense_top_k = dense_top_k
        self.chunks_per_job = chunks_per_job
        self.max_retries = max_retries
        self.resume = resume
        self.limit_targets = limit_targets
        self.limit_chunks = limit_chunks

        # Paths
        paths_cfg = config_dict.get("paths", {})
        self.root_dir = os.path.join(self.app_config.base_dir, paths_cfg.get("root", "cache/arctic_gpu"))
        self.manifest_path = os.path.join(self.app_config.base_dir, paths_cfg.get("manifest", "cache/arctic_gpu/manifest.json"))
        self.input_dir = os.path.join(self.root_dir, "input")
        self.output_dir = os.path.join(self.root_dir, "output")
        self.logs_dir = os.path.join(self.root_dir, "logs")
        self.staging_dir = os.path.join(self.root_dir, "jobs")

        self.merged_emb_path = os.path.join(self.app_config.base_dir, paths_cfg.get("merged_embeddings", "cache/embeddings/arctic/target_embeddings.npy"))
        self.merged_ids_path = os.path.join(self.app_config.base_dir, paths_cfg.get("merged_ids", "cache/embeddings/arctic/target_ids.json"))
        self.merged_meta_path = os.path.join(self.app_config.base_dir, paths_cfg.get("merged_meta", "cache/embeddings/arctic/metadata.json"))
        self.faiss_dir = os.path.join(self.app_config.base_dir, paths_cfg.get("faiss_dir", "cache/ann/arctic"))
        self.report_path = os.path.join(self.app_config.base_dir, paths_cfg.get("report_path", "reports/arctic_gpu_benchmark.md"))

        # Kaggle Config
        kaggle_cfg = config_dict.get("kaggle", {})
        self.kernel_slug = kaggle_cfg.get("kernel", "rajeshshitap/arctic-entity-resolution-worker")
        self.dataset_slug = kaggle_cfg.get("dataset", "rajeshshitap/arctic-er-input")
        self.accelerator = kaggle_cfg.get("accelerator", "NvidiaL4")
        self.timeout_seconds = int(kaggle_cfg.get("timeout_seconds", 3600))
        self.poll_interval = int(kaggle_cfg.get("poll_interval_seconds", 15))

        # Model Config
        emb_cfg = config_dict.get("embedding", {})
        self.model_name = emb_cfg.get("model_name", DEFAULT_MODEL_NAME)
        self.dimension = int(emb_cfg.get("dimension", EMBEDDING_DIM))

        # Controllers
        self.exporter = ArcticGPUExporter(
            output_root=paths_cfg.get("root", "cache/arctic_gpu"),
            manifest_path=self.manifest_path,
            chunk_size=self.chunk_size,
            model_name=self.model_name,
            dimension=self.dimension
        )

        self.kaggle_ctrl = KaggleController(
            kernel_slug=self.kernel_slug,
            dataset_slug=self.dataset_slug,
            accelerator=self.accelerator,
            timeout_seconds=self.timeout_seconds,
            poll_interval=self.poll_interval,
            logs_dir=self.logs_dir
        )

    def print_dry_run_plan(self):
        """Displays full execution plan without running jobs or modifying state."""
        print("=" * 80)
        print("ANTIGRAVITY V4.1 ARCTIC GPU PIPELINE — DRY-RUN EXECUTION PLAN")
        print("=" * 80)
        print(f"Kernel Slug:          {self.kernel_slug}")
        print(f"Kaggle Accelerator:   {self.accelerator}")
        print(f"Embedding Model:      {self.model_name} (Dim: {self.dimension})")
        print(f"Target DB:            {self.exporter.db_path}")
        print(f"Chunk Size:           {self.chunk_size:,} rows")
        print(f"Dense Top-K:          {self.dense_top_k}")
        print(f"Max Retries:          {self.max_retries}")
        print(f"Limit Targets:        {self.limit_targets or 'ALL (10,320,219)'}")
        print(f"Limit Chunks:         {self.limit_chunks or 'ALL'}")
        print("-" * 80)

        # Estimate chunks & disk requirements
        total_rows = self.limit_targets or EXPECTED_TOTAL_TARGETS
        total_chunks = (total_rows + self.chunk_size - 1) // self.chunk_size
        if self.limit_chunks and self.limit_chunks < total_chunks:
            total_chunks = self.limit_chunks
            total_rows = min(total_rows, total_chunks * self.chunk_size)

        input_disk_mb = total_chunks * 12.0
        output_disk_mb = total_rows * self.dimension * 4 / (1024.0 * 1024.0)

        print(f"Expected Chunks:      {total_chunks:,}")
        print(f"Input Storage (Parquet): ~{input_disk_mb / 1024.0:.2f} GB")
        print(f"Output Storage (.npy):   ~{output_disk_mb / 1024.0:.2f} GB")
        print(f"FAISS IVF-PQ Storage:    ~{0.55:.2f} GB")
        print(f"Total Disk Required:     ~{(input_disk_mb + output_disk_mb) / 1024.0 + 1.0:.2f} GB")

        # Check existing manifest if available
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            completed = sum(1 for c in m.get("chunks", []) if c.get("status") == "completed")
            pending = sum(1 for c in m.get("chunks", []) if c.get("status") != "completed")
            print(f"Existing Manifest:    {self.manifest_path} (Completed: {completed}, Pending: {pending})")
        else:
            print(f"Existing Manifest:    None (Will be created on first export)")

        print("-" * 80)
        print("Dry-run validation complete. No remote jobs submitted, no files altered.")
        print("=" * 80)

    def merge_verified_chunks(self, manifest: Dict[str, Any]) -> Tuple[str, str]:
        """
        Assembles verified chunk `.npy` embeddings into persistent disk-backed memory map.
        Guarantees strict 1-to-1 positional target ID alignment.
        """
        os.makedirs(os.path.dirname(self.merged_emb_path), exist_ok=True)
        chunks = manifest.get("chunks", [])
        completed_chunks = [c for c in chunks if c.get("status") == "completed"]

        if len(completed_chunks) != len(chunks):
            raise RuntimeError(f"Cannot merge incomplete chunks! Completed: {len(completed_chunks)}/{len(chunks)}")

        total_rows = sum(c["row_count"] for c in completed_chunks)
        is_complete = (total_rows == EXPECTED_TOTAL_TARGETS)

        print("\n" + "=" * 80)
        print(f"[Assembly Stage] Merging {len(completed_chunks)} verified chunks into persistent store...")
        print(f"Target Total Rows: {total_rows:,} | Dimension: {self.dimension}")
        print(f"Target File:       {self.merged_emb_path}")
        print(f"Target IDs:        {self.merged_ids_path}")
        print("=" * 80)

        # Allocate disk-backed memmap
        fp_merged = np.lib.format.open_memmap(
            self.merged_emb_path,
            mode="w+",
            dtype="float32",
            shape=(total_rows, self.dimension)
        )

        all_target_ids: List[str] = []
        offset = 0

        with MemoryTracker(f"Embedding Chunk Merge ({len(completed_chunks)} chunks)"):
            for chunk_meta in sorted(completed_chunks, key=lambda x: x["chunk_id"]):
                c_id = chunk_meta["chunk_id"]
                c_n = chunk_meta["row_count"]
                emb_file = os.path.join(self.output_dir, f"chunk_{c_id:06d}_embeddings.npy")
                ids_file = os.path.join(self.output_dir, f"chunk_{c_id:06d}_ids.parquet")

                # Load chunk embeddings and IDs
                chunk_embs = np.load(emb_file)
                df_ids = pl.read_parquet(ids_file)
                chunk_ids = [str(x) for x in df_ids["target_id"].to_list()]

                assert chunk_embs.shape == (c_n, self.dimension), f"Shape mismatch in chunk {c_id}"
                assert len(chunk_ids) == c_n, f"ID count mismatch in chunk {c_id}"

                # Assign directly into memmap
                fp_merged[offset : offset + c_n] = chunk_embs
                all_target_ids.extend(chunk_ids)

                offset += c_n
                del chunk_embs, df_ids, chunk_ids
                if c_id % 10 == 0:
                    gc.collect()

            fp_merged.flush()
            del fp_merged
            gc.collect()

        # Save merged target IDs JSON
        with open(self.merged_ids_path, "w", encoding="utf-8") as f:
            json.dump(all_target_ids, f)

        # Save metadata JSON
        meta = {
            "model_name": self.model_name,
            "model_revision": "main",
            "embedding_dimension": self.dimension,
            "dtype": "float32",
            "normalization": "L2",
            "target_count": total_rows,
            "expected_target_count": EXPECTED_TOTAL_TARGETS,
            "is_complete_target_universe": is_complete,
            "target_cache_version": "v4_duckdb",
            "text_representation_version": "v1_name_addr_ctry",
            "creation_timestamp": datetime.now(timezone.utc).isoformat()
        }
        with open(self.merged_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"[Assembly Stage] Assembly completed successfully ({total_rows:,} rows).")
        return self.merged_emb_path, self.merged_ids_path

    def validate_preflight(self, skip_kaggle: bool = False):
        """
        Executes strict pre-flight validation checks before altering any files or creating remote jobs.
        """
        print("\n" + "=" * 80)
        print("[PRE-FLIGHT VALIDATION] Verifying Environment, Disk Space, Configuration & API Access...")
        print("=" * 80)

        # 1. Check DuckDB Target Cache
        if not os.path.exists(self.exporter.db_path):
            print(f"\n[PRE-FLIGHT ERROR]: Target DuckDB cache not found at '{self.exporter.db_path}'!")
            print("Ensure persistent DuckDB target database exists before starting Arctic pipeline.")
            sys.exit(1)
        print(f"[OK] Source DuckDB Database: {self.exporter.db_path}")

        # 2. Check Disk Space
        try:
            total_b, used_b, free_b = shutil.disk_usage(self.root_dir if os.path.exists(self.root_dir) else ".")
            free_gb = free_b / (1024.0 ** 3)
            # Full 10.3M requires ~35GB buffer; 1 chunk requires ~2GB
            req_gb = 35.0 if (self.limit_targets is None and self.limit_chunks is None) else 2.0
            print(f"[OK] Available Disk Space:  {free_gb:.2f} GB (Required Buffer: {req_gb:.2f} GB)")
            if free_gb < req_gb:
                print(f"\n[PRE-FLIGHT ERROR]: Insufficient free disk space ({free_gb:.2f} GB < {req_gb:.2f} GB)!")
                sys.exit(1)
        except Exception as e:
            print(f"[WARN] Disk space check warning: {e}")

        # 3. Check Kernel Slug Format
        if "YOUR_KAGGLE_USERNAME" in self.kernel_slug or "/" not in self.kernel_slug:
            print(f"\n[PRE-FLIGHT ERROR]: Invalid Kaggle kernel slug: '{self.kernel_slug}'!")
            print("Please configure your real Kaggle kernel in config/arctic_gpu.yaml (e.g. 'rajeshshitap/arctic-entity-resolution-worker').")
            sys.exit(1)
        print(f"[OK] Kaggle Kernel Slug:     {self.kernel_slug}")

        # 4. Check Kaggle CLI and Live Authentication
        if not skip_kaggle:
            auth_ok, auth_msg = self.kaggle_ctrl.check_kaggle_auth()
            if not auth_ok:
                print(f"\n[PRE-FLIGHT KAGGLE ERROR]: {auth_msg}")
                print("Run 'kaggle auth login' to authenticate with Kaggle OAuth before launching GPU jobs.")
                sys.exit(1)
            print(f"[OK] Kaggle API Connection:  {auth_msg}")

        print("[OK] All Pre-Flight Validation Checks PASSED.")
        print("=" * 80 + "\n")

    def run_pipeline(
        self,
        skip_export: bool = False,
        skip_kaggle: bool = False,
        skip_faiss: bool = False,
        skip_benchmark: bool = False
    ):
        """
        Executes the master pipeline stages.
        """
        t0_all = time.time()
        print("=" * 80)
        print("ANTIGRAVITY V4.1 — MASTER ARCTIC GPU PIPELINE")
        print(f"Target Universe: {EXPECTED_TOTAL_TARGETS:,} records | Model: {self.model_name}")
        print(f"Initial Process RSS: {get_current_rss_mb():.2f} MB")
        print("=" * 80)

        # Pre-flight Validation
        self.validate_preflight(skip_kaggle=skip_kaggle)

        # 1. Export Stage
        if not skip_export:
            manifest = self.exporter.export_chunks(
                limit_targets=self.limit_targets,
                limit_chunks=self.limit_chunks
            )
        else:
            print("[Stage 1: Export] Skipped by user.")
            if not os.path.exists(self.manifest_path):
                raise FileNotFoundError(f"Manifest not found at {self.manifest_path}! Cannot skip export.")
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

        chunks = manifest.get("chunks", [])
        total_chunks = len(chunks)

        # 2. Kaggle Execution Stage
        if not skip_kaggle:
            print("\n" + "=" * 80)
            print(f"[Stage 2: Kaggle GPU Worker] Processing {total_chunks} chunks on remote GPU...")
            print("=" * 80)

            for chunk_meta in chunks:
                chunk_id = chunk_meta["chunk_id"]
                status = chunk_meta.get("status", "pending")

                if status == "completed" and self.resume:
                    # Verify output existence
                    emb_file = os.path.join(self.output_dir, chunk_meta["output_filename"])
                    ids_file = os.path.join(self.output_dir, chunk_meta["ids_filename"])
                    if os.path.exists(emb_file) and os.path.exists(ids_file):
                        print(f"  -> Chunk {chunk_id:04d}/{total_chunks:04d} already completed. Skipping.")
                        continue

                input_parquet = os.path.join(self.input_dir, chunk_meta["input_filename"])
                attempts = chunk_meta.get("attempts", 0)
                success = False

                while attempts < self.max_retries and not success:
                    attempts += 1
                    chunk_meta["attempts"] = attempts
                    print(f"\n[Kaggle Job] Processing Chunk {chunk_id:04d}/{total_chunks:04d} (Attempt {attempts}/{self.max_retries})...")

                    # 1. Stage and upload Parquet chunk to dedicated Kaggle Dataset
                    ds_ok, ds_msg = self.kaggle_ctrl.upload_dataset_chunk(chunk_id, input_parquet, self.staging_dir)
                    if not ds_ok:
                        print(f"  [DATASET ERROR] {ds_msg}")
                        time.sleep(5)
                        continue

                    # 2. Stage lightweight kernel package (code + metadata referencing dataset)
                    kernel_dir = self.kaggle_ctrl.prepare_kernel_package(chunk_id, self.staging_dir)

                    # 3. Submit kernel
                    push_ok, push_msg = self.kaggle_ctrl.submit_job(kernel_dir, chunk_id)
                    if not push_ok:
                        print(f"  [ERROR] {push_msg}")
                        time.sleep(5)
                        continue

                    # 4. Poll kernel execution
                    poll_ok, poll_msg = self.kaggle_ctrl.poll_job_status(chunk_id)
                    if not poll_ok:
                        print(f"  [ERROR] {poll_msg}")
                        time.sleep(10)
                        continue

                    # 5. Download outputs
                    dl_ok, dl_msg = self.kaggle_ctrl.download_output(self.output_dir, chunk_id)
                    if not dl_ok:
                        print(f"  [ERROR] {dl_msg}")
                        time.sleep(5)
                        continue

                    # 6. Strict row-for-row verification
                    v_ok, v_msg, v_summary = self.kaggle_ctrl.verify_chunk_output(
                        chunk_id,
                        input_parquet,
                        self.output_dir,
                        expected_dim=self.dimension
                    )
                    if not v_ok:
                        print(f"  [VERIFY ERROR] Chunk {chunk_id}: {v_msg}")
                        chunk_meta["status"] = "failed"
                        chunk_meta["error_message"] = v_msg
                        self.exporter.save_manifest(manifest)
                        time.sleep(5)
                        continue

                    # Success!
                    success = True
                    chunk_meta["status"] = "completed"
                    chunk_meta["output_sha256"] = v_summary["output_sha256"]
                    chunk_meta["completed_at"] = datetime.now(timezone.utc).isoformat()
                    chunk_meta["error_message"] = None
                    self.exporter.save_manifest(manifest)
                    print(f"  -> Chunk {chunk_id:04d} VERIFIED & COMPLETED successfully!")

                    # Clean up local staging directories
                    if os.path.exists(kernel_dir):
                        shutil.rmtree(kernel_dir, ignore_errors=True)
                    ds_staging = os.path.join(self.staging_dir, f"dataset_staging_{chunk_id:06d}")
                    if os.path.exists(ds_staging):
                        shutil.rmtree(ds_staging, ignore_errors=True)

                if not success:
                    print(f"\n[FATAL ERROR]: Chunk {chunk_id} failed after {self.max_retries} attempts! Halting pipeline.")
                    sys.exit(1)

        else:
            print("[Stage 2: Kaggle] Skipped by user.")

        # 3. Assemble Merged Embeddings
        self.merge_verified_chunks(manifest)

        # 4. Build FAISS ANN Index
        if not skip_faiss:
            print("\n" + "=" * 80)
            print("[Stage 3: FAISS ANN Index Construction]")
            print("=" * 80)
            build_faiss_index(
                embeddings_path=self.merged_emb_path,
                ids_path=self.merged_ids_path,
                output_dir=self.faiss_dir,
                nlist=4096,
                pq_m=48,
                pq_nbits=8
            )
        else:
            print("[Stage 3: FAISS] Skipped by user.")

        # 5. Hybrid Benchmark
        if not skip_benchmark:
            print("\n" + "=" * 80)
            print("[Stage 4: Hybrid Candidate Retrieval Benchmark]")
            print("=" * 80)
            run_dense_benchmark(
                s1_count=1000,
                default_k=self.dense_top_k,
                candidate_budget=250,
                allow_incomplete=(self.limit_targets is not None or self.limit_chunks is not None)
            )
        else:
            print("[Stage 4: Benchmark] Skipped by user.")

        total_elapsed = time.time() - t0_all
        print("\n" + "=" * 80)
        print(f"ANTIGRAVITY V4.1 ARCTIC PIPELINE FINISHED IN {total_elapsed:.2f}s.")
        print(f"  Merged Embeddings: {self.merged_emb_path}")
        print(f"  FAISS Index:       {os.path.join(self.faiss_dir, 'target.index')}")
        print(f"  Peak RSS:          {get_peak_rss_mb():.2f} MB")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Antigravity V4.1 Master Arctic GPU Orchestrator")
    parser.add_argument("--gpu", action="store_true", default=True, help="Enable Kaggle GPU orchestration")
    parser.add_argument("--config", type=str, default="config/arctic_gpu.yaml", help="Path to config YAML")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Target rows per Parquet chunk")
    parser.add_argument("--dense-top-k", type=int, default=50, help="Candidate budget K for dense ANN retrieval")
    parser.add_argument("--chunks-per-job", type=int, default=1, help="Chunks bundled per Kaggle execution")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retry attempts for failed chunks")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing manifest")
    parser.add_argument("--skip-export", action="store_true", help="Skip DuckDB chunk export stage")
    parser.add_argument("--skip-kaggle", action="store_true", help="Skip remote Kaggle GPU execution")
    parser.add_argument("--skip-faiss", action="store_true", help="Skip FAISS index construction")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip hybrid retrieval benchmark")
    parser.add_argument("--dry-run", action="store_true", help="Print execution plan without running any jobs")
    parser.add_argument("--limit-targets", type=int, default=None, help="Limit total target rows (for testing)")
    parser.add_argument("--limit-chunks", type=int, default=None, help="Limit total chunks (for testing)")
    args = parser.parse_args()

    config_dict = load_yaml_config(args.config)

    orchestrator = ArcticGPUOrchestrator(
        config_dict=config_dict,
        chunk_size=args.chunk_size,
        dense_top_k=args.dense_top_k,
        chunks_per_job=args.chunks_per_job,
        max_retries=args.max_retries,
        resume=args.resume,
        limit_targets=args.limit_targets,
        limit_chunks=args.limit_chunks
    )

    if args.dry_run:
        orchestrator.print_dry_run_plan()
    else:
        orchestrator.run_pipeline(
            skip_export=args.skip_export,
            skip_kaggle=args.skip_kaggle,
            skip_faiss=args.skip_faiss,
            skip_benchmark=args.skip_benchmark
        )

if __name__ == "__main__":
    main()
