"""
Unit & Integration Test Suite for ER-X Memory-Safe Architecture & Strict Checkpointing.
"""

import os
import shutil
import tempfile
from pathlib import Path
import numpy as np

from src.erx.types import CompactS1Record, MultiViewRecord, CandidatePair, ProvenanceMask, char_ngrams_set
from src.erx.features import ERXFeatureExtractor, FEATURE_NAMES
from src.erx.final_train import write_training_shard_parquet, is_valid_shard


def test_compact_s1_record_interface():
    s1 = CompactS1Record(
        internal_id=0,
        entity_id="S1-001",
        country="US",
        raw_name="Google Inc",
        norm_name="google inc",
        compact_name="googleinc",
        translit_name="google inc",
        translit_comp_name="googleinc",
        learned_name="google",
        sorted_token_name="google inc",
        name_phonetic_sig="G240",
        raw_addr="1600 Amphitheatre Pkwy, Mountain View, CA 94043",
        norm_addr="1600 amphitheatre pkwy mountain view ca 94043",
        translit_addr="1600 amphitheatre pkwy mountain view ca 94043",
        numeric_signature="1600_94043",
        house_numbers_str="1600",
        postal_codes_str="94043",
        name_tokens_str="google inc",
        translit_tokens_str="google inc",
        addr_tokens_str="1600 amphitheatre pkwy mountain view ca 94043",
    )

    assert s1.name_tokens == ["google", "inc"]
    assert s1.name_tok_set == {"google", "inc"}
    assert s1.house_numbers == {"1600"}
    assert s1.postal_codes == {"94043"}
    assert not s1.is_name_missing
    assert not s1.is_addr_missing
    assert not s1.is_country_missing
    assert "goo" in s1.name_char3_set
    print("[PASS] CompactS1Record property interface test passed.")


def test_feature_extraction_with_compact_s1():
    s1 = CompactS1Record(
        internal_id=0,
        entity_id="S1-001",
        country="US",
        raw_name="Google Inc",
        norm_name="google inc",
        compact_name="googleinc",
        translit_name="google inc",
        translit_comp_name="googleinc",
        learned_name="google",
        sorted_token_name="google inc",
        name_phonetic_sig="G240",
        raw_addr="1600 Amphitheatre Pkwy",
        norm_addr="1600 amphitheatre pkwy",
        translit_addr="1600 amphitheatre pkwy",
        numeric_signature="1600",
        house_numbers_str="1600",
        postal_codes_str="94043",
        name_tokens_str="google inc",
        translit_tokens_str="google inc",
        addr_tokens_str="1600 amphitheatre pkwy",
    )

    target = MultiViewRecord(
        internal_id=1,
        entity_id="S2-001",
        country="US",
        raw_name="Google, Inc.",
        norm_name="google inc",
        compact_name="googleinc",
        translit_name="google inc",
        translit_comp_name="googleinc",
        learned_name="google",
        sorted_token_name="google inc",
        name_phonetic_sig="G240",
        raw_addr="1600 Amphitheatre Parkway",
        norm_addr="1600 amphitheatre parkway",
        translit_addr="1600 amphitheatre parkway",
        numeric_signature="1600",
        name_tokens=["google", "inc"],
        name_tok_set={"google", "inc"},
        translit_tokens=["google", "inc"],
        translit_tok_set={"google", "inc"},
        name_char3_set=char_ngrams_set("google inc", 3),
        name_char4_set=char_ngrams_set("google inc", 4),
        name_char5_set=char_ngrams_set("google inc", 5),
        addr_tokens=["1600", "amphitheatre", "parkway"],
        addr_tok_set={"1600", "amphitheatre", "parkway"},
        house_numbers={"1600"},
        postal_codes={"94043"},
        is_s2=True,
    )

    cand = CandidatePair(target_internal_id=1, s1_internal_id=0, retrieval_score=1.0, provenance_mask=1)
    extractor = ERXFeatureExtractor()
    feats = extractor.compute_pair_features(s1, target, cand, {})
    assert len(feats) == len(FEATURE_NAMES) == 73
    assert feats[0] == 0.0  # name_raw_exact (Google Inc vs Google, Inc.)
    assert feats[1] == 1.0  # name_canonical_exact (google inc == google inc)
    assert feats[2] == 1.0  # name_compact_exact (googleinc == googleinc)
    assert feats[22] == 1.0 # name_phonetic_match
    print(f"[PASS] Feature extraction test passed with {len(feats)} features.")


from src.erx.final_train import write_training_shard_parquet, is_valid_shard


def test_shard_writing_and_validation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        shard_p = tmp_path / "shard_s2_0001.parquet"
        shard_m = tmp_path / "shard_s2_0001.meta.json"

        # Mock chunk data
        chunk_data = {
            "s1_ids": np.array([10, 20, 30], dtype=np.int32),
            "target_ids": np.array([1, 2, 3], dtype=np.int32),
            "labels": np.array([1, 0, 0], dtype=np.int32),
            "prov_masks": np.array([1, 2, 4], dtype=np.int32),
            "features": np.ones((3, len(FEATURE_NAMES)), dtype=np.float32),
        }
        meta_info = {
            "chunk_id": 1,
            "source": "s2",
            "rows_processed": 3,
            "retrieval_hits": 1,
        }

        write_training_shard_parquet(shard_p, shard_m, chunk_data, meta_info)
        valid, meta = is_valid_shard(shard_p, shard_m)
        assert valid, "Shard should be valid!"
        assert meta["total_pairs"] == 3
        assert meta["positives"] == 1
        assert meta["negatives"] == 2
        print("[PASS] Shard parquet writing and validation test passed.")


if __name__ == "__main__":
    test_compact_s1_record_interface()
    test_feature_extraction_with_compact_s1()
    test_shard_writing_and_validation()
    print("\nALL ER-X ARCHITECTURE & CHECKPOINT INTEGRITY TESTS PASSED!")
