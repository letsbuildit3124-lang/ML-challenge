"""
Unit tests for Antigravity V4.1 Arctic GPU Orchestration Layer.
Tests offline functionality without requiring Kaggle API credentials or GPU access.
"""

import os
import gc
import json
import tempfile
import unittest
import numpy as np
import polars as pl

from src.export_arctic_gpu import ArcticGPUExporter, compute_sha256
from src.kaggle_controller import KaggleController
from src.arctic_pipeline import ArcticGPUOrchestrator, load_yaml_config

class TestArcticGPUOrchestration(unittest.TestCase):

    def test_sha256_checksum(self):
        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            f.write("Antigravity Entity Resolution Test String")
            f_path = f.name

        try:
            h1 = compute_sha256(f_path)
            h2 = compute_sha256(f_path)
            self.assertEqual(h1, h2)
            self.assertEqual(len(h1), 64)
        finally:
            if os.path.exists(f_path):
                os.remove(f_path)

    def test_manifest_creation_and_resumability(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "manifest.json")
            exporter = ArcticGPUExporter(
                output_root=tmpdir,
                manifest_path=manifest_path,
                chunk_size=100
            )

            manifest = exporter.load_or_create_manifest(total_rows=350)
            self.assertEqual(manifest["total_targets"], 350)
            self.assertEqual(manifest["chunk_size"], 100)
            self.assertEqual(manifest["total_chunks"], 4)

            # Test atomic save
            manifest["chunks"].append({
                "chunk_id": 0,
                "input_filename": "chunk_000000.parquet",
                "status": "completed"
            })
            exporter.save_manifest(manifest)

            # Reload and verify
            reloaded = exporter.load_or_create_manifest(total_rows=350)
            self.assertEqual(len(reloaded["chunks"]), 1)
            self.assertEqual(reloaded["chunks"][0]["status"], "completed")

    def test_kernel_package_preparation_and_dataset_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctrl = KaggleController(
                kernel_slug="rajeshshitap/arctic-entity-resolution-worker",
                dataset_slug="rajeshshitap/arctic-er-input"
            )

            kernel_dir = ctrl.prepare_kernel_package(chunk_id=0, staging_root=tmpdir)
            self.assertTrue(os.path.exists(kernel_dir))
            self.assertTrue(os.path.exists(os.path.join(kernel_dir, "kaggle_worker.py")))
            self.assertTrue(os.path.exists(os.path.join(kernel_dir, "kernel-metadata.json")))

            with open(os.path.join(kernel_dir, "kernel-metadata.json"), "r") as f:
                meta = json.load(f)

            self.assertEqual(meta["id"], "rajeshshitap/arctic-entity-resolution-worker")
            self.assertEqual(meta["dataset_sources"], ["rajeshshitap/arctic-er-input"])
            self.assertEqual(meta["enable_gpu"], "true")
            self.assertEqual(meta["enable_internet"], "true")

    def test_positional_verification_and_integrity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_parquet = os.path.join(tmpdir, "chunk_000000.parquet")
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(output_dir, exist_ok=True)

            n_rows = 50
            dim = 384
            ids = [f"S2-{i:06d}" for i in range(n_rows)]

            # 1. Create Input Parquet
            pl.DataFrame({
                "target_row_id": list(range(n_rows)),
                "target_id": ids,
                "business_name": [f"Company {i}" for i in range(n_rows)],
                "business_address": [f"Address {i}" for i in range(n_rows)],
                "country": ["US"] * n_rows
            }).write_parquet(input_parquet)

            # 2. Create Output Normalized Embeddings
            np.random.seed(42)
            raw = np.random.randn(n_rows, dim).astype(np.float32)
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norm_embs = raw / norms

            emb_file = os.path.join(output_dir, "chunk_000000_embeddings.npy")
            ids_file = os.path.join(output_dir, "chunk_000000_ids.parquet")
            np.save(emb_file, norm_embs)
            pl.DataFrame({"target_id": ids}).write_parquet(ids_file)

            ctrl = KaggleController(kernel_slug="rajeshshitap/arctic-entity-resolution-worker")

            # Positive verification
            ok, msg, summary = ctrl.verify_chunk_output(
                chunk_id=0,
                input_parquet_path=input_parquet,
                output_dir=output_dir,
                expected_dim=dim
            )
            self.assertTrue(ok, f"Verification failed: {msg}")
            self.assertEqual(summary["row_count"], n_rows)

            # Negative verification: Desynchronized IDs
            bad_ids = ids.copy()
            bad_ids[0], bad_ids[1] = bad_ids[1], bad_ids[0]
            pl.DataFrame({"target_id": bad_ids}).write_parquet(ids_file)

            bad_ok, bad_msg, _ = ctrl.verify_chunk_output(
                chunk_id=0,
                input_parquet_path=input_parquet,
                output_dir=output_dir,
                expected_dim=dim
            )
            self.assertFalse(bad_ok)
            self.assertIn("Positional ID mismatch", bad_msg)

    def test_dry_run_orchestrator(self):
        cfg = {
            "kaggle": {
                "kernel": "rajeshshitap/arctic-entity-resolution-worker",
                "dataset": "rajeshshitap/arctic-er-input",
                "accelerator": "NvidiaL4"
            },
            "embedding": {"model_name": "themelder/arctic-embed-xs-entity-resolution", "dimension": 384},
            "pipeline": {"chunk_size": 50000, "dense_top_k": 50},
            "paths": {"root": "cache/arctic_gpu"}
        }
        orch = ArcticGPUOrchestrator(config_dict=cfg, limit_targets=1000)
        orch.print_dry_run_plan()

if __name__ == "__main__":
    unittest.main()
