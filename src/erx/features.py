"""
ER-X High-Performance Tiered Feature Extraction Engine.
High-Throughput Vectorized & Batched RapidFuzz C-API Implementation.

Features (73 Total):
- Group A: Vectorized Integer / Arithmetic / Provenance / Context Features (NumPy)
- Group B: Batched Token / N-Gram / Exact String Differences
- Group C: Batched RapidFuzz C-API (process.cpdist with native OpenMP parallel workers)
- Group D: Vectorized Cross-Field Multi-Channel Interaction Features
"""

import math
from typing import Dict, List, Set, Tuple, Optional, Any, Union
import numpy as np
from rapidfuzz import process, fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler

from src.erx.types import MultiViewRecord, CompactS1Record, CandidatePair, ProvenanceMask


FEATURE_NAMES = [
    # --- Name Tier 0 & Tier 1 (14) ---
    "name_raw_exact",
    "name_canonical_exact",
    "name_compact_exact",
    "name_learned_exact",
    "name_len_diff",
    "name_rel_len_diff",
    "name_is_missing",
    "name_tok_jaccard",
    "name_tok_dice",
    "name_tok_overlap_cnt",
    "name_tok_overlap_ratio",
    "name_tokens_missing_cnt",
    "name_tokens_extra_cnt",
    "name_prefix_equal",

    # --- Name Tier 2 & Tier 3 (10) ---
    "name_levenshtein",
    "name_jaro_winkler",
    "name_fuzz_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_char3_jaccard",
    "name_char4_jaccard",
    "name_char5_jaccard",
    "name_phonetic_match",
    "name_rare_token_overlap",

    # --- Address Tier 0 & Tier 1 (13) ---
    "addr_raw_exact",
    "addr_canonical_exact",
    "addr_len_diff",
    "addr_rel_len_diff",
    "addr_is_missing",
    "addr_tok_jaccard",
    "addr_tok_dice",
    "addr_tok_overlap_cnt",
    "addr_tok_overlap_ratio",
    "addr_tokens_missing_cnt",
    "addr_tokens_extra_cnt",
    "addr_num_overlap_cnt",
    "addr_num_jaccard",

    # --- Address Tier 2 & Tier 3 (8) ---
    "addr_levenshtein",
    "addr_jaro_winkler",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_number_exact",
    "addr_number_conflict",
    "addr_postal_exact",
    "addr_postal_prefix",

    # --- Cross-field & Country (12) ---
    "country_match",
    "country_is_missing",
    "is_s2",
    "is_s3",
    "name_addr_sim_prod",
    "name_addr_sim_mean",
    "name_addr_sim_max",
    "name_addr_sim_weighted",
    "name_strong_addr_strong",
    "name_strong_addr_weak",
    "name_weak_addr_strong",
    "name_weak_addr_weak",

    # --- Candidate Provenance (7) ---
    "num_channels",
    "prov_exact_or_learned",
    "prov_char_tfidf",
    "prov_rare_token",
    "prov_address",
    "prov_phonetic",
    "prov_learned_variant",

    # --- Candidate-Context Features (9) ---
    "candidate_rank",
    "retrieval_score",
    "best_retrieval_score",
    "second_best_retrieval_score",
    "candidate_margin",
    "candidate_count",
    "count_above_05",
    "count_above_07",
    "count_above_08"
]


class ERXFeatureExtractor:
    """Computes comprehensive 73-feature vector for candidate pairs using vectorized and batched C-API operations."""

    def __init__(self, token_idf: Optional[Dict[str, float]] = None):
        self.token_idf = token_idf or {}

    @property
    def feature_count(self) -> int:
        return len(FEATURE_NAMES)

    def extract_features_batch(
        self,
        targets: List[MultiViewRecord],
        s1_dict: Dict[int, Union[MultiViewRecord, CompactS1Record]],
        cand_data: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        High-Throughput Batched Feature Extraction.
        Computes 73 features for all candidate pairs in a single vectorized pass.
        Replaces individual Python scalar loops with RapidFuzz C-API batch processing and NumPy arrays.
        """
        num_pairs = int(cand_data["total_pairs"])
        if num_pairs == 0:
            return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

        cand_s1_ids = cand_data["cand_s1_ids"]
        cand_t_indices = cand_data["cand_target_idx"]
        cand_scores = cand_data["cand_scores"]
        cand_prov_masks = cand_data["cand_prov_masks"]
        cand_ranks = cand_data["cand_ranks"]

        best_scores = cand_data["best_scores"]
        second_best_scores = cand_data["second_best_scores"]
        cand_counts = cand_data["cand_counts"]
        counts_above_05 = cand_data["counts_above_05"]
        counts_above_07 = cand_data["counts_above_07"]
        counts_above_08 = cand_data["counts_above_08"]

        X = np.zeros((num_pairs, len(FEATURE_NAMES)), dtype=np.float32)

        # -----------------------------------------------------------------
        # Group A: Context & Provenance Features (Vectorized NumPy)
        # -----------------------------------------------------------------
        # Candidate Provenance [57:64]
        pmasks = cand_prov_masks
        num_ch = np.zeros(num_pairs, dtype=np.float32)
        for bit in [1, 2, 4, 8, 16, 32]:
            num_ch += (pmasks & bit != 0).astype(np.float32)

        X[:, 57] = num_ch
        X[:, 58] = (pmasks & ProvenanceMask.EXACT_OR_LEARNED != 0).astype(np.float32)
        X[:, 59] = (pmasks & ProvenanceMask.CHAR_TFIDF != 0).astype(np.float32)
        X[:, 60] = (pmasks & ProvenanceMask.RARE_TOKEN != 0).astype(np.float32)
        X[:, 61] = (pmasks & ProvenanceMask.ADDRESS != 0).astype(np.float32)
        X[:, 62] = (pmasks & ProvenanceMask.PHONETIC != 0).astype(np.float32)
        X[:, 63] = (pmasks & ProvenanceMask.LEARNED_VARIANT != 0).astype(np.float32)

        # Candidate Context [64:73]
        X[:, 64] = cand_ranks
        X[:, 65] = cand_scores
        X[:, 66] = best_scores
        X[:, 67] = second_best_scores
        X[:, 68] = best_scores - second_best_scores
        X[:, 69] = cand_counts
        X[:, 70] = counts_above_05
        X[:, 71] = counts_above_07
        X[:, 72] = counts_above_08

        # -----------------------------------------------------------------
        # Extract Batch String Pairs for C-API & Token Processing
        # -----------------------------------------------------------------
        t_norm_names: List[str] = []
        s1_norm_names: List[str] = []
        t_norm_addrs: List[str] = []
        s1_norm_addrs: List[str] = []

        idf_map = self.token_idf

        for p_idx in range(num_pairs):
            t_idx = int(cand_t_indices[p_idx])
            s1_int = int(cand_s1_ids[p_idx])

            target = targets[t_idx]
            s1 = s1_dict.get(s1_int)
            if s1 is None:
                continue

            s1_n = s1.norm_name
            t_n = target.norm_name
            s1_a = s1.norm_addr
            t_a = target.norm_addr

            t_norm_names.append(t_n)
            s1_norm_names.append(s1_n)
            t_norm_addrs.append(t_a)
            s1_norm_addrs.append(s1_a)

            # Name Tier 0 Exact & Lengths [0:7]
            X[p_idx, 0] = 1.0 if s1.raw_name and s1.raw_name == target.raw_name else 0.0
            X[p_idx, 1] = 1.0 if s1_n and s1_n == t_n else 0.0
            X[p_idx, 2] = 1.0 if s1.compact_name and s1.compact_name == target.compact_name else 0.0
            X[p_idx, 3] = 1.0 if s1.learned_name and s1.learned_name == target.learned_name else 0.0

            len_s1_n = len(s1_n)
            len_t_n = len(t_n)
            len_diff_n = abs(len_s1_n - len_t_n)
            X[p_idx, 4] = float(len_diff_n)
            X[p_idx, 5] = float(len_diff_n) / max(len_s1_n, len_t_n, 1)
            X[p_idx, 6] = 1.0 if s1.is_name_missing or target.is_name_missing else 0.0

            # Name Tokens [7:14]
            s1_n_set = s1.name_tok_set
            t_n_set = target.name_tok_set
            s1_n_len = len(s1_n_set)
            t_n_len = len(t_n_set)
            n_inter = len(s1_n_set & t_n_set)
            n_union = s1_n_len + t_n_len - n_inter

            X[p_idx, 7] = (n_inter / n_union) if n_union > 0 else (1.0 if s1_n_len == 0 and t_n_len == 0 else 0.0)
            X[p_idx, 8] = (2.0 * n_inter / (s1_n_len + t_n_len)) if (s1_n_len + t_n_len) > 0 else 0.0
            X[p_idx, 9] = float(n_inter)
            X[p_idx, 10] = (n_inter / max(s1_n_len, 1)) if s1_n_len > 0 else 0.0
            X[p_idx, 11] = float(len(s1_n_set - t_n_set))
            X[p_idx, 12] = float(len(t_n_set - s1_n_set))
            X[p_idx, 13] = 1.0 if (s1_n and t_n and s1_n[:4] == t_n[:4]) else 0.0

            # Name Character N-Grams & Phonetics [19:24]
            if s1_n and t_n:
                if s1_n == t_n:
                    X[p_idx, 19] = 1.0
                    X[p_idx, 20] = 1.0
                    X[p_idx, 21] = 1.0
                else:
                    s1_c3 = s1.name_char3_set
                    t_c3 = target.name_char3_set
                    g3_i = len(s1_c3 & t_c3)
                    g3_u = len(s1_c3 | t_c3)
                    X[p_idx, 19] = (g3_i / g3_u) if g3_u > 0 else 0.0

                    s1_c4 = s1.name_char4_set
                    t_c4 = target.name_char4_set
                    g4_i = len(s1_c4 & t_c4)
                    g4_u = len(s1_c4 | t_c4)
                    X[p_idx, 20] = (g4_i / g4_u) if g4_u > 0 else 0.0

                    s1_c5 = s1.name_char5_set
                    t_c5 = target.name_char5_set
                    g5_i = len(s1_c5 & t_c5)
                    g5_u = len(s1_c5 | t_c5)
                    X[p_idx, 21] = (g5_i / g5_u) if g5_u > 0 else 0.0

            X[p_idx, 22] = 1.0 if (s1.name_phonetic_sig and s1.name_phonetic_sig == target.name_phonetic_sig) else 0.0

            if idf_map and n_inter > 0:
                shared = s1_n_set & t_n_set
                X[p_idx, 23] = sum(idf_map.get(t, 0.0) for t in shared)

            # Address Tier 0 & Tier 1 [24:37]
            X[p_idx, 24] = 1.0 if s1.raw_addr and s1.raw_addr == target.raw_addr else 0.0
            X[p_idx, 25] = 1.0 if s1_a and s1_a == t_a else 0.0
            len_s1_a = len(s1_a)
            len_t_a = len(t_a)
            len_diff_a = abs(len_s1_a - len_t_a)
            X[p_idx, 26] = float(len_diff_a)
            X[p_idx, 27] = float(len_diff_a) / max(len_s1_a, len_t_a, 1)
            X[p_idx, 28] = 1.0 if s1.is_addr_missing or target.is_addr_missing else 0.0

            s1_a_set = s1.addr_tok_set
            t_a_set = target.addr_tok_set
            s1_a_len = len(s1_a_set)
            t_a_len = len(t_a_set)
            a_inter = len(s1_a_set & t_a_set)
            a_union = s1_a_len + t_a_len - a_inter

            X[p_idx, 29] = (a_inter / a_union) if a_union > 0 else (1.0 if s1_a_len == 0 and t_a_len == 0 else 0.0)
            X[p_idx, 30] = (2.0 * a_inter / (s1_a_len + t_a_len)) if (s1_a_len + t_a_len) > 0 else 0.0
            X[p_idx, 31] = float(a_inter)
            X[p_idx, 32] = (a_inter / max(s1_a_len, 1)) if s1_a_len > 0 else 0.0
            X[p_idx, 33] = float(len(s1_a_set - t_a_set))
            X[p_idx, 34] = float(len(t_a_set - s1_a_set))

            s1_h = s1.house_numbers
            t_h = target.house_numbers
            num_inter = len(s1_h & t_h)
            num_union = len(s1_h | t_h)
            X[p_idx, 35] = float(num_inter)
            X[p_idx, 36] = (num_inter / num_union) if num_union > 0 else (1.0 if len(s1_h) == 0 and len(t_h) == 0 else 0.0)

            # Address Numbers & Postal [41:45]
            X[p_idx, 41] = 1.0 if (s1_h and t_h and s1_h == t_h) else 0.0
            X[p_idx, 42] = 1.0 if (s1_h and t_h and not num_inter) else 0.0

            s1_p = s1.postal_codes
            t_p = target.postal_codes
            X[p_idx, 43] = 1.0 if (s1_p and t_p and bool(s1_p & t_p)) else 0.0
            if s1_p and t_p:
                p1 = next(iter(s1_p))
                p2 = next(iter(t_p))
                if len(p1) >= 3 and len(p2) >= 3 and p1[:3] == p2[:3]:
                    X[p_idx, 44] = 1.0

            # Country & Source Flags [45:49]
            s1_c = s1.country
            t_c = target.country
            X[p_idx, 45] = 1.0 if (s1_c and t_c and s1_c == t_c) else 0.0
            X[p_idx, 46] = 1.0 if s1.is_country_missing or target.is_country_missing else 0.0
            X[p_idx, 47] = 1.0 if target.is_s2 else 0.0
            X[p_idx, 48] = 1.0 if target.is_s3 else 0.0

        # -----------------------------------------------------------------
        # Group C: Batched RapidFuzz C-API Process Processors (process.cpdist)
        # -----------------------------------------------------------------
        if t_norm_names and s1_norm_names:
            # Name RapidFuzz Distances [14:19]
            X[:, 14] = process.cpdist(t_norm_names, s1_norm_names, scorer=Levenshtein.normalized_similarity, dtype=np.float32)
            X[:, 15] = process.cpdist(t_norm_names, s1_norm_names, scorer=JaroWinkler.similarity, dtype=np.float32)
            X[:, 16] = process.cpdist(t_norm_names, s1_norm_names, scorer=fuzz.ratio, dtype=np.float32) / 100.0
            X[:, 17] = process.cpdist(t_norm_names, s1_norm_names, scorer=fuzz.token_sort_ratio, dtype=np.float32) / 100.0
            X[:, 18] = process.cpdist(t_norm_names, s1_norm_names, scorer=fuzz.token_set_ratio, dtype=np.float32) / 100.0

            # Address RapidFuzz Distances [37:41]
            X[:, 37] = process.cpdist(t_norm_addrs, s1_norm_addrs, scorer=Levenshtein.normalized_similarity, dtype=np.float32)
            X[:, 38] = process.cpdist(t_norm_addrs, s1_norm_addrs, scorer=JaroWinkler.similarity, dtype=np.float32)
            X[:, 39] = process.cpdist(t_norm_addrs, s1_norm_addrs, scorer=fuzz.token_sort_ratio, dtype=np.float32) / 100.0
            X[:, 40] = process.cpdist(t_norm_addrs, s1_norm_addrs, scorer=fuzz.token_set_ratio, dtype=np.float32) / 100.0

        # -----------------------------------------------------------------
        # Group D: Cross-Field Interactions (Vectorized NumPy) [49:57]
        # -----------------------------------------------------------------
        name_lev = X[:, 14]
        addr_lev = X[:, 37]

        X[:, 49] = name_lev * addr_lev
        X[:, 50] = (name_lev + addr_lev) / 2.0
        X[:, 51] = np.maximum(name_lev, addr_lev)
        X[:, 52] = 0.65 * name_lev + 0.35 * addr_lev
        X[:, 53] = ((name_lev >= 0.85) & (addr_lev >= 0.80)).astype(np.float32)
        X[:, 54] = ((name_lev >= 0.85) & (addr_lev <= 0.50)).astype(np.float32)
        X[:, 55] = ((name_lev <= 0.60) & (addr_lev >= 0.80)).astype(np.float32)
        X[:, 56] = ((name_lev <= 0.60) & (addr_lev <= 0.50)).astype(np.float32)

        return X

    def extract_features_for_target_candidates(
        self,
        target: MultiViewRecord,
        candidates: List[CandidatePair],
        s1_records: Dict[int, Union[MultiViewRecord, CompactS1Record]]
    ) -> np.ndarray:
        """Extracts feature matrix for candidates of a single target record."""
        if not candidates:
            return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

        cand_len = len(candidates)
        cand_data = {
            "cand_s1_ids": np.array([c.s1_internal_id for c in candidates], dtype=np.uint32),
            "cand_target_idx": np.zeros(cand_len, dtype=np.int32),
            "cand_scores": np.array([c.retrieval_score for c in candidates], dtype=np.float32),
            "cand_prov_masks": np.array([c.provenance_mask for c in candidates], dtype=np.uint32),
            "cand_ranks": np.arange(cand_len, dtype=np.float32),
            "best_scores": np.full(cand_len, candidates[0].retrieval_score, dtype=np.float32),
            "second_best_scores": np.full(cand_len, candidates[1].retrieval_score if cand_len > 1 else 0.0, dtype=np.float32),
            "cand_counts": np.full(cand_len, float(cand_len), dtype=np.float32),
            "counts_above_05": np.full(cand_len, float(sum(1 for c in candidates if c.retrieval_score >= 0.5)), dtype=np.float32),
            "counts_above_07": np.full(cand_len, float(sum(1 for c in candidates if c.retrieval_score >= 0.7)), dtype=np.float32),
            "counts_above_08": np.full(cand_len, float(sum(1 for c in candidates if c.retrieval_score >= 0.8)), dtype=np.float32),
            "total_pairs": cand_len,
        }
        return self.extract_features_batch([target], s1_records, cand_data)
