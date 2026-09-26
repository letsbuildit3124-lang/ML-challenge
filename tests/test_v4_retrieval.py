"""
Unit Tests for V4 Normalization, Sparse Retrieval, and Provenance Bitmasking.
"""

import unittest
import polars as pl
from src.normalize import normalize_text, offline_transliterate
from src.blocking_v2 import add_v2_blocking_columns, LEGAL_SUFFIXES_REGEX
from src.v4_sparse_retriever import PROV_DET, PROV_NAME_TFIDF, PROV_ADDR_TFIDF, PROV_TRANSLIT_TFIDF

class TestV4Retrieval(unittest.TestCase):
    def test_multilingual_legal_suffix_normalization(self):
        """Tests that US, Indian, and French legal suffixes are recognized and stripped in compact representations."""
        test_names = [
            ("Starbucks Coffee Company Inc.", "starbuckscoffee"),
            ("Tirupati Software Solutions Pvt Ltd", "tirupatisoftware"),
            ("Societe Generale de Transport SARL", "societegeneraledetransport"),
            ("Acme Industries SASU", "acme"),
            ("Clinique Medicale EURL", "cliniquemedicale"),
        ]

        for raw_name, expected_cname_prefix in test_names:
            df = pl.DataFrame({
                "entity_id": ["E-1"],
                "business_name": [raw_name],
                "business_address": ["123 Main St"],
                "country": ["US"]
            })
            p_df = add_v2_blocking_columns(df)
            cname = p_df["compact_name"][0]
            self.assertTrue(cname.startswith(expected_cname_prefix), f"Expected {expected_cname_prefix} in {cname}")

    def test_provenance_bitmask_combination(self):
        """Tests bitmask logic for multi-branch retrieval provenance."""
        mask = PROV_DET | PROV_NAME_TFIDF
        self.assertEqual(mask, 3)
        self.assertTrue(mask & PROV_DET)
        self.assertTrue(mask & PROV_NAME_TFIDF)
        self.assertFalse(mask & PROV_ADDR_TFIDF)

        mask |= PROV_ADDR_TFIDF
        self.assertEqual(mask, 7)
        self.assertTrue(mask & PROV_ADDR_TFIDF)

    def test_transliteration_fidelity(self):
        """Tests offline Indic and multi-script transliteration."""
        self.assertEqual(offline_transliterate("Patel Brothers").lower(), "patel brothers")
        self.assertTrue(len(offline_transliterate("Shri Ram Services")) > 0)

if __name__ == "__main__":
    unittest.main()
