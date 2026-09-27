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
from src.erx.final_train import save_feature_checkpoint, load_verified_feature_checkpoint


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


def test_checkpoint_validation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        ckpt_npz = tmp_path / "train_features.npz"
        ckpt_meta = tmp_path / "train_features.meta.json"

        # 1. Save smoke checkpoint
        X_smoke = np.ones((2500, 73), dtype=np.float32)
        y_smoke = np.zeros((2500,), dtype=np.int32)
        y_smoke[:200] = 1
        save_feature_checkpoint(ckpt_npz, ckpt_meta, X_smoke, y_smoke, s1_count=10000, target_count=2500, mode="smoke")

        # 2. Test that FULL mode rejects smoke checkpoint
        loaded = load_verified_feature_checkpoint(ckpt_npz, ckpt_meta, expected_s1_count=2206821, mode="full")
        assert loaded is None, "FULL mode must reject smoke checkpoint!"
        print("[PASS] Full mode correctly rejected smoke checkpoint.")

        # 3. Test that SMOKE mode accepts smoke checkpoint
        save_feature_checkpoint(ckpt_npz, ckpt_meta, X_smoke, y_smoke, s1_count=10000, target_count=2500, mode="smoke")
        loaded = load_verified_feature_checkpoint(ckpt_npz, ckpt_meta, expected_s1_count=10000, mode="smoke")
        assert loaded is not None, "Smoke mode should accept smoke checkpoint!"
        assert loaded[0].shape == (2500, 73)
        print("[PASS] Smoke mode correctly accepted smoke checkpoint.")

        # 4. Save full production checkpoint
        X_full = np.ones((500000, 73), dtype=np.float32)
        y_full = np.zeros((500000,), dtype=np.int32)
        y_full[:100000] = 1
        save_feature_checkpoint(ckpt_npz, ckpt_meta, X_full, y_full, s1_count=2206821, target_count=500000, mode="full")

        # 5. Test that FULL mode accepts valid full checkpoint
        loaded = load_verified_feature_checkpoint(ckpt_npz, ckpt_meta, expected_s1_count=2206821, mode="full")
        assert loaded is not None, "Full mode must accept valid full checkpoint!"
        assert loaded[0].shape == (500000, 73)
        assert int(np.sum(loaded[1])) == 100000
        print("[PASS] Full mode successfully validated and loaded genuine production checkpoint.")


if __name__ == "__main__":
    test_compact_s1_record_interface()
    test_feature_extraction_with_compact_s1()
    test_checkpoint_validation()
    print("\nALL ER-X ARCHITECTURE & CHECKPOINT INTEGRITY TESTS PASSED!")
