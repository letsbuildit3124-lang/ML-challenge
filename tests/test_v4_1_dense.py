"""
Unit and integration tests for Antigravity V4.1 Dense Retrieval & Resource Tracking.
"""

import os
import gc
import json
import tempfile
import unittest
import numpy as np
import polars as pl

from src.resource_tracker import get_current_rss_mb, get_peak_rss_mb, MemoryTracker
from src.arctic_embeddings import ArcticEmbedder, format_entity_text, EMBEDDING_DIM
from src.build_arctic_embeddings import check_existing_cache, METADATA_VERSION
from src.build_arctic_faiss import build_faiss_index
from src.v4_1_dense_retriever import V41DenseRetriever, PROV_DET, PROV_SPARSE_NAME, PROV_SPARSE_ADDR, PROV_DENSE

class TestV41Dense(unittest.TestCase):

    def test_resource_tracker(self):
        rss = get_current_rss_mb()
        peak = get_peak_rss_mb()
        self.assertGreater(rss, 0.0, f"Expected positive RSS, got {rss}")
        self.assertGreaterEqual(peak, rss, f"Expected Peak RSS >= Current RSS, got {peak} vs {rss}")

        with MemoryTracker("Test Memory Stage") as tracker:
            dummy = np.ones((2500, 1000), dtype=np.float32)
            del dummy
            gc.collect()

        self.assertGreater(tracker.rss_before, 0.0)
        self.assertGreaterEqual(tracker.peak_rss, tracker.rss_before)

    def test_arctic_formatting_and_normalization(self):
        text = format_entity_text("Acme Corp", "123 Main St", "US")
        self.assertIn("Business name: Acme Corp", text)
        self.assertIn("Address: 123 Main St", text)
        self.assertIn("Country: US", text)

        embedder = ArcticEmbedder(batch_size=4)
        self.assertEqual(embedder.embedding_dim, EMBEDDING_DIM)

        texts = [
            format_entity_text("Alpha Technologies Ltd", "Sector 5, Salt Lake", "IN"),
            format_entity_text("Beta Logistics LLC", "456 Industrial Way", "US")
        ]
        embs = embedder.encode(texts, normalize_embeddings=True)
        self.assertEqual(embs.shape, (2, 384))
        self.assertEqual(embs.dtype, np.float32)

        norms = np.linalg.norm(embs, axis=1)
        self.assertTrue(np.all(np.isclose(norms, 1.0, atol=1e-4)), f"Embeddings not L2 normalized: {norms}")

    def test_embedding_cache_and_ann_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            emb_dir = os.path.join(tmpdir, "embeddings")
            ann_dir = os.path.join(tmpdir, "ann")
            os.makedirs(emb_dir, exist_ok=True)
            os.makedirs(ann_dir, exist_ok=True)

            n_targets = 100
            dim = EMBEDDING_DIM

            # Create dummy normalized target embeddings
            np.random.seed(42)
            raw_vecs = np.random.randn(n_targets, dim).astype(np.float32)
            norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
            norm_vecs = raw_vecs / norms

            emb_file = os.path.join(emb_dir, "target_embeddings.npy")
            fp = np.lib.format.open_memmap(emb_file, mode="w+", dtype="float32", shape=(n_targets, dim))
            fp[:] = norm_vecs[:]
            del fp

            target_ids = [f"S2-{i:06d}" for i in range(n_targets)]
            with open(os.path.join(emb_dir, "target_ids.json"), "w", encoding="utf-8") as f:
                json.dump(target_ids, f)

            meta = {
                "model_name": "themelder/arctic-embed-xs-entity-resolution",
                "model_revision": "main",
                "embedding_dimension": dim,
                "dtype": "float32",
                "normalization": "L2",
                "target_count": n_targets,
                "target_cache_version": "v4_duckdb",
                "text_representation_version": METADATA_VERSION,
                "creation_timestamp": "2026-09-26T00:00:00Z"
            }
            with open(os.path.join(emb_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f)

            # Validate cache check helper
            self.assertTrue(check_existing_cache(emb_dir, n_targets))
            self.assertFalse(check_existing_cache(emb_dir, n_targets + 1))

            # Build ANN index
            build_faiss_index(emb_file, os.path.join(emb_dir, "target_ids.json"), ann_dir)

            # Test Dense Retriever on synthetic query
            retriever = V41DenseRetriever(
                embeddings_dir=emb_dir,
                ann_dir=ann_dir,
                batch_size=16
            )

            s1_df = pl.DataFrame({
                "entity_id": ["S1-000001", "S1-000002"],
                "business_name": ["Alpha Tech", "Beta Logistics"],
                "business_address": ["Sector 5", "Industrial Way"],
                "country": ["IN", "US"]
            })

            results = retriever.retrieve_dense_candidates(s1_df, top_k=5)
            self.assertEqual(len(results), 2)
            for s1_id in ["S1-000001", "S1-000002"]:
                self.assertIn(s1_id, results)
                self.assertEqual(len(results[s1_id]), 5)
                for tid, (mask, score) in results[s1_id].items():
                    self.assertEqual(mask, PROV_DENSE)
                    self.assertIsInstance(score, float)

            retriever.close()
            gc.collect()

    def test_provenance_bitmask_consistency(self):
        self.assertEqual(PROV_DET, 1)
        self.assertEqual(PROV_SPARSE_NAME, 2)
        self.assertEqual(PROV_SPARSE_ADDR, 4)
        self.assertEqual(PROV_DENSE, 8)

        # Multi-source candidate combination
        combined_mask = PROV_DET | PROV_SPARSE_NAME | PROV_DENSE
        self.assertEqual(combined_mask, 11)
        self.assertTrue(bool(combined_mask & PROV_DET))
        self.assertTrue(bool(combined_mask & PROV_SPARSE_NAME))
        self.assertFalse(bool(combined_mask & PROV_SPARSE_ADDR))
        self.assertTrue(bool(combined_mask & PROV_DENSE))

if __name__ == "__main__":
    unittest.main()
