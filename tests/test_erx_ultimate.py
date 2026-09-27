"""
Unit and Integration Test Suite for ER-X Ultimate
Runs purely on synthetic in-memory fixtures using standard unittest.
"""

import unittest
import numpy as np
import tempfile
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

from src.erx_ultimate.types import EntityRecord, CandidateMatch, ScoredPair, SourceType, EntityCluster
from src.erx_ultimate.normalization import (
    normalize_business_name,
    normalize_address,
    normalize_phone,
    normalize_website,
    compute_soundex,
    extract_ngrams,
)
from src.erx_ultimate.retrieval import CSRChannelIndex, ERXRetrievalEngine
from src.erx_ultimate.features import extract_pair_features, extract_batch_features, NUM_FEATURES
from src.erx_ultimate.postprocessing import PostProcessingEngine
from src.erx_ultimate.validate import compute_cluster_f05, evaluate_macro_f05, partition_entity_fold
from src.erx_ultimate.train import run_pre_training_gate
from src.erx_ultimate.output_audit import run_output_audit


class TestERXUltimate(unittest.TestCase):

    def test_normalization_functions(self):
        # Business name normalization
        self.assertEqual(normalize_business_name("Acme Corp. LLC"), "acme")
        self.assertEqual(normalize_business_name("Google, Inc."), "google")
        self.assertEqual(normalize_business_name("STARBUCKS COFFEE"), "starbucks coffee")

        # Address normalization
        self.assertEqual(normalize_address("123 Main St. NW, Ste 400"), "123 main street northwest suite 400")
        
        # Phone normalization
        self.assertEqual(normalize_phone("+1 (800) 555-0199"), "8005550199")
        self.assertEqual(normalize_phone("9876543210"), "9876543210")

        # Website normalization
        self.assertEqual(normalize_website("https://www.google.com/search?q=test"), "google.com")

        # Soundex
        self.assertEqual(compute_soundex("Robert"), "R163")
        self.assertEqual(compute_soundex("Rupert"), "R163")

    def test_csr_channel_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            postings = {
                "amazon": [0, 2],
                "google": [1],
                "apple": [0, 1, 2],
            }
            
            csr = CSRChannelIndex("test_idx")
            csr.build_from_postings(postings)
            csr.save(tmp_path)

            # Load back and query without mmap lock in temp directory
            csr_loaded = CSRChannelIndex("test_idx")
            csr_loaded.load(tmp_path, mmap_mode=None)

            hits_amazon = csr_loaded.query("amazon")
            self.assertEqual(list(hits_amazon), [0, 2])

            hits_google = csr_loaded.query("google")
            self.assertEqual(list(hits_google), [1])

            hits_none = csr_loaded.query("microsoft")
            self.assertEqual(len(hits_none), 0)

    def test_feature_extraction(self):
        tgt = EntityRecord(
            id=101,
            source=SourceType.SOURCE2,
            name_raw="Amazon Web Services Inc",
            name_norm="amazon web",
            address_raw="410 Terry Ave N, Seattle, WA",
            address_norm="410 terry avenue north seattle wa",
            city_norm="seattle",
            state_norm="wa",
            postal_code_norm="98109",
            country_norm="us",
            phone_norm="2062661000",
            website_norm="aws.amazon.com",
        )
        s1 = EntityRecord(
            id=1,
            source=SourceType.SOURCE1,
            name_raw="Amazon Web Services",
            name_norm="amazon web",
            address_raw="410 Terry Ave N",
            address_norm="410 terry avenue north",
            city_norm="seattle",
            state_norm="wa",
            postal_code_norm="98109",
            country_norm="us",
            phone_norm="2062661000",
            website_norm="amazon.com",
        )
        cand = CandidateMatch(
            target_id=101,
            target_source=SourceType.SOURCE2,
            s1_id=1,
            rrf_score=0.033,
            channel_mask=3,
        )

        feats = extract_pair_features(tgt, s1, cand)
        self.assertEqual(len(feats), NUM_FEATURES)
        self.assertEqual(feats[0], 1.0)  # Exact name match
        self.assertEqual(feats[3], 1.0)  # Exact city match
        self.assertEqual(feats[7], 1.0)  # Exact phone match

    def test_postprocessing_target_ownership(self):
        post_engine = PostProcessingEngine()

        scored_pairs = [
            # Target 101 has candidate S1=1 (prob 0.95) and S1=2 (prob 0.70)
            ScoredPair(target_id=101, target_source=SourceType.SOURCE2, s1_id=1, raw_prob=0.95, calibrated_prob=0.95, rrf_score=0.03),
            ScoredPair(target_id=101, target_source=SourceType.SOURCE2, s1_id=2, raw_prob=0.70, calibrated_prob=0.70, rrf_score=0.02),
            
            # Target 102 has low prob candidates (should be singleton/unmatched)
            ScoredPair(target_id=102, target_source=SourceType.SOURCE2, s1_id=1, raw_prob=0.40, calibrated_prob=0.40, rrf_score=0.01),
            
            # Target 201 (Source 3) matches S1=1
            ScoredPair(target_id=201, target_source=SourceType.SOURCE3, s1_id=1, raw_prob=0.88, calibrated_prob=0.88, rrf_score=0.03),
        ]

        ownership = post_engine.resolve_target_ownership(scored_pairs)
        self.assertEqual(ownership[(101, int(SourceType.SOURCE2))], 1)
        self.assertNotIn((102, int(SourceType.SOURCE2)), ownership)
        self.assertEqual(ownership[(201, int(SourceType.SOURCE3))], 1)

        clusters = post_engine.aggregate_clusters([1, 2], ownership)
        self.assertEqual(len(clusters), 2)
        self.assertEqual(clusters[0].source1_id, 1)
        self.assertEqual(clusters[0].source2_ids, [101])
        self.assertEqual(clusters[0].source3_ids, [201])
        self.assertEqual(clusters[1].source1_id, 2)
        self.assertEqual(clusters[1].source2_ids, [])
        self.assertEqual(clusters[1].source3_ids, [])

    def test_f05_evaluation_and_singletons(self):
        # Exact match
        pred_set = {(101, 2), (201, 3)}
        true_set = {(101, 2), (201, 3)}
        self.assertEqual(compute_cluster_f05(pred_set, true_set), 1.0)

        # Partial match
        pred_set_partial = {(101, 2)}
        f05 = compute_cluster_f05(pred_set_partial, true_set)
        self.assertTrue(0.0 < f05 < 1.0)

        # True singleton correctly predicted as empty
        self.assertEqual(compute_cluster_f05(set(), set()), 1.0)

        # False positive on singleton
        self.assertEqual(compute_cluster_f05({(999, 2)}, set()), 0.0)

        # Missed ground truth matches
        self.assertEqual(compute_cluster_f05(set(), {(999, 2)}), 0.0)

    def test_disjoint_fold_partitioning(self):
        # Must be deterministic and partition across 5 folds
        f1 = partition_entity_fold(100, 5)
        f2 = partition_entity_fold(100, 5)
        self.assertEqual(f1, f2)
        self.assertTrue(0 <= f1 < 5)

    def test_pre_training_gate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            shards_dir = tmp_path / "shards"
            shards_dir.mkdir()
            artifacts_dir = tmp_path / "artifacts"
            artifacts_dir.mkdir()

            # Empty shards should fail
            with self.assertRaises(RuntimeError):
                run_pre_training_gate(shards_dir, artifacts_dir)

            # Create a valid shard
            schema = [pa.field(f"f_{j}", pa.float32()) for j in range(NUM_FEATURES)]
            schema.append(pa.field("label", pa.uint8()))
            schema.append(pa.field("target_id", pa.int64()))
            schema.append(pa.field("s1_id", pa.int64()))
            arrow_schema = pa.schema(schema)

            cols = [pa.array(np.ones(10, dtype=np.float32)) for _ in range(NUM_FEATURES)]
            cols.append(pa.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], type=pa.uint8()))
            cols.append(pa.array(list(range(10)), type=pa.int64()))
            cols.append(pa.array([100] * 10, type=pa.int64()))

            table = pa.Table.from_arrays(cols, schema=arrow_schema)
            pq.write_table(table, str(shards_dir / "shard_000.parquet"))

            gate = run_pre_training_gate(shards_dir, artifacts_dir)
            self.assertEqual(gate["status"], "PASSED")
            self.assertEqual(gate["total_training_pairs"], 10)
            self.assertEqual(gate["total_positives"], 2)
            self.assertEqual(gate["total_negatives"], 8)

    def test_output_audit_engine(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tsv_path = Path(tmpdir) / "matching_results.tsv"
            with open(tsv_path, "w", encoding="utf-8") as f:
                f.write("source1_id\tsource2_ids\tsource3_ids\n")
                f.write("1\t101,102\t201\n")
                f.write("2\t\t\n")
                f.write("3\t103\t\n")

            summary = run_output_audit(str(tsv_path), expected_rows=3)
            self.assertEqual(summary["status"], "APPROVED")
            self.assertEqual(summary["total_rows"], 3)
            self.assertEqual(summary["unique_s1_count"], 3)
            self.assertEqual(summary["singletons_count"], 1)


if __name__ == "__main__":
    unittest.main()
