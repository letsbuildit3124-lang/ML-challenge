"""
Unit Tests for ER-X Normalization, Learned Rules, Retrieval, Features, and Model Calibration.
Uses standard library unittest.
"""

import unittest
import numpy as np
from src.erx.config import ERXConfig
from src.erx.types import InternalIDMapper, MultiViewRecord, ProvenanceMask, CandidatePair
from src.erx.normalization import ERXNormalizer, normalize_text, compact_name, offline_transliterate, compute_soundex
from src.erx.learned_rules import LearnedRuleEngine
from src.erx.retrieval import ERXRetrievalEngine
from src.erx.features import ERXFeatureExtractor, FEATURE_NAMES
from src.erx.pipeline import compute_entity_level_metrics


class TestERXModules(unittest.TestCase):

    def test_normalization_and_transliteration(self):
        norm = normalize_text("Acme Corp. & Sons, Inc.")
        self.assertEqual("acme corp and sons inc", norm)

        comp = compact_name("Acme Corp. & Sons, Inc.")
        self.assertTrue("acme" in comp)

        # French legal suffix removal
        french_comp = compact_name("Société Dupuis SARL")
        self.assertNotIn("sarl", french_comp)

        # Transliteration
        indic_text = "भारत"
        trans = offline_transliterate(indic_text)
        self.assertTrue(len(trans) > 0 and trans.isascii())

        # Soundex
        snd = compute_soundex("Hospital")
        self.assertTrue(snd.startswith("H") and len(snd) == 4)

    def test_learned_rule_engine(self):
        engine = LearnedRuleEngine(min_alias_observations=2, min_alias_purity=0.8)
        mock_pairs = [
            ("Acme Hospital", "Acme Hosp"),
            ("Delta Hospital", "Delta Hosp"),
            ("Alpha Engineering", "Alpha Engg"),
            ("Beta Engineering", "Beta Engg"),
        ]
        res = engine.learn_from_pairs(mock_pairs)
        self.assertTrue("hosp" in engine.token_aliases or "engg" in engine.token_aliases)

    def test_retrieval_channels(self):
        config = ERXConfig()
        normalizer = ERXNormalizer()
        id_mapper = InternalIDMapper()

        rec1 = normalizer.normalize_record(id_mapper.get_or_add("S1-1"), "S1-1", "Apollo Hospitals Enterprise", "Greams Road, Chennai", "India")
        rec2 = normalizer.normalize_record(id_mapper.get_or_add("S1-2"), "S1-2", "Walmart Supercenter 100", "123 Main Street, Bentonville", "US")

        engine = ERXRetrievalEngine(config)
        engine.index_s1([rec1, rec2])

        # Target 1 (Exact match)
        t1 = normalizer.normalize_record(id_mapper.get_or_add("S2-1"), "S2-1", "Apollo Hospitals Enterprise", "Greams Rd", "India")
        cands1 = engine.retrieve_for_target(t1)
        s1_ids1 = [c.s1_internal_id for c in cands1]
        self.assertIn(rec1.internal_id, s1_ids1)

        # Target 2 (Rare token match)
        t2 = normalizer.normalize_record(id_mapper.get_or_add("S3-2"), "S3-2", "Bentonville Walmart", "123 Main St", "US")
        cands2 = engine.retrieve_for_target(t2)
        s1_ids2 = [c.s1_internal_id for c in cands2]
        self.assertIn(rec2.internal_id, s1_ids2)

    def test_feature_extractor_dimensions(self):
        extractor = ERXFeatureExtractor()
        self.assertEqual(extractor.feature_count, len(FEATURE_NAMES))

        normalizer = ERXNormalizer()
        s1 = normalizer.normalize_record(0, "S1-1", "Google LLC", "1600 Amphitheatre Pkwy, Mountain View", "US")
        target = normalizer.normalize_record(1, "S2-1", "Google Inc", "1600 Amphitheatre Parkway", "US")
        cand = CandidatePair(target_internal_id=1, s1_internal_id=0, retrieval_score=0.95, provenance_mask=1)

        feats = extractor.extract_features_for_target_candidates(target, [cand], {0: s1})
        self.assertEqual(feats.shape, (1, len(FEATURE_NAMES)))
        self.assertFalse(np.isnan(feats).any())

    def test_metric_computation(self):
        gt = {
            "S1-1": ["S2-1", "S3-1"],
            "S1-2": [],  # Singleton
            "S1-3": ["S2-3"],
        }
        pred = {
            "S1-1": ["S2-1", "S3-1"],
            "S1-2": [],  # Singleton correctly identified
            "S1-3": ["S2-3"],
        }
        metrics = compute_entity_level_metrics(pred, gt)
        self.assertEqual(metrics["macro_f05"], 1.0)
        self.assertEqual(metrics["singleton_accuracy"], 1.0)
        self.assertEqual(metrics["singleton_count"], 1)


if __name__ == "__main__":
    unittest.main()
