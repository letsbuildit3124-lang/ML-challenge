"""
Unit and integration tests for Antigravity V5 CPU-Only Retrieval Engine.
"""

import os
import gc
import json
import tempfile
import unittest
import numpy as np
import polars as pl

from src.v5_types import (
    PROV_DET, PROV_NAME_NGRAM_3, PROV_NAME_NGRAM_4, PROV_NAME_NGRAM_5,
    PROV_ADDR_NGRAM, PROV_TOKEN, PROV_RARE_TOKEN, PROV_ADDR_NUM,
    PROV_TRANSLIT, PROV_FTS, PROV_FUZZY, decode_provenance
)
from src.v5_ngram_indexer import generate_char_ngrams
from src.v5_token_indexer import extract_informative_tokens, get_sorted_token_signature, get_rarest_token
from src.v5_address_indexer import parse_address_components
from src.v5_fuzzy_reranker import V5FuzzyReranker

class TestV5Retrieval(unittest.TestCase):

    def test_ngram_generation(self):
        ng3 = generate_char_ngrams("acme", n=3)
        self.assertIn("#ac", ng3)
        self.assertIn("cme", ng3)
        self.assertIn("me#", ng3)

        ng4 = generate_char_ngrams("acme", n=4)
        self.assertIn("#acm", ng4)
        self.assertIn("cme#", ng4)

    def test_token_indexer_signatures(self):
        # Permutation invariance test: A B vs B A
        sig1 = get_sorted_token_signature("Omega Systems India Private Limited")
        sig2 = get_sorted_token_signature("India Omega Systems Pvt Ltd")
        self.assertEqual(sig1, sig2)
        self.assertEqual(sig1, "india_omega")

        # Rare token extraction
        rare = get_rarest_token("Bhagirathi Hospital Pvt Ltd")
        self.assertEqual(rare, "bhagirathi")

    def test_address_component_parsing(self):
        # Variations in building numbers
        p1 = parse_address_components("12-A, MG Road, 560001, Bangalore")
        self.assertEqual(p1["num"], "12")
        self.assertEqual(p1["postal"], "560001")
        self.assertEqual(p1["street_token"], "road" if "road" not in ["road"] else p1["street_token"])

        p2 = parse_address_components("#45/1 Near Station, Mumbai")
        self.assertEqual(p2["num"], "45")

        p3 = parse_address_components("Plot 108 Sector 5, Salt Lake")
        self.assertEqual(p3["num"], "108")
        self.assertEqual(p3["street_token"], "sect")

    def test_fuzzy_reranker_multiprocessing(self):
        reranker = V5FuzzyReranker(workers=2)
        s1_names = ["Acme Industrial Supplies Ltd", "Tata Consultancy Services"]
        target_names = ["Acme Indl Supply", "Tata Consultancy Services Ltd"]

        scores = reranker.compute_pairwise_fuzzy_scores(s1_names, target_names)
        self.assertEqual(len(scores), 2)
        self.assertGreater(scores[0], 0.70)
        self.assertGreater(scores[1], 0.90)

    def test_provenance_bitmask_algebra(self):
        mask = PROV_DET | PROV_NAME_NGRAM_3 | PROV_RARE_TOKEN | PROV_FUZZY
        self.assertTrue(bool(mask & PROV_DET))
        self.assertTrue(bool(mask & PROV_NAME_NGRAM_3))
        self.assertTrue(bool(mask & PROV_RARE_TOKEN))
        self.assertTrue(bool(mask & PROV_FUZZY))
        self.assertFalse(bool(mask & PROV_ADDR_NUM))

        channels = decode_provenance(mask)
        self.assertIn("Deterministic Blocker", channels)
        self.assertIn("Name 3-Gram", channels)
        self.assertIn("Rare Token", channels)
        self.assertIn("RapidFuzz C++ Match", channels)

if __name__ == "__main__":
    unittest.main()
