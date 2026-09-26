"""
ER-X High-Performance Tiered Feature Extraction Engine.
Features:
- Tier 0: Structural, exact, missingness, and provenance flags
- Tier 1: Set-based token Jaccard/Dice, numeric overlaps, difference counts (zero-allocation)
- Tier 2: Character 3/4/5-gram Dice and Jaccard similarities
- Tier 3: RapidFuzz C++ Levenshtein, Jaro-Winkler, and token sort metrics
- Tier 4: Candidate-context relative ranking and margin features
"""

import math
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
from rapidfuzz.distance import Levenshtein, JaroWinkler
from rapidfuzz import fuzz

from src.erx.types import MultiViewRecord, CandidatePair, ProvenanceMask


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
    """Computes comprehensive feature vector for a candidate pair with candidate list context."""

    def __init__(self, token_idf: Optional[Dict[str, float]] = None):
        self.token_idf = token_idf or {}

    @property
    def feature_count(self) -> int:
        return len(FEATURE_NAMES)

    def compute_pair_features(
        self,
        s1: MultiViewRecord,
        target: MultiViewRecord,
        cand: CandidatePair,
        context_stats: Dict[str, float]
    ) -> List[float]:
        """Extracts all ~73 features for a single (S1, target) candidate pair."""

        # -------------------------------------------------------------
        # NAME FEATURES
        # -------------------------------------------------------------
        s1_n = s1.norm_name
        t_n = target.norm_name

        name_raw_exact = 1.0 if s1.raw_name and s1.raw_name == target.raw_name else 0.0
        name_canonical_exact = 1.0 if s1_n and s1_n == t_n else 0.0
        name_compact_exact = 1.0 if s1.compact_name and s1.compact_name == target.compact_name else 0.0
        name_learned_exact = 1.0 if s1.learned_name and s1.learned_name == target.learned_name else 0.0

        len_s1_n = len(s1_n)
        len_t_n = len(t_n)
        name_len_diff = float(abs(len_s1_n - len_t_n))
        name_rel_len_diff = name_len_diff / max(len_s1_n, len_t_n, 1)
        name_is_missing = 1.0 if s1.is_name_missing or target.is_name_missing else 0.0

        # Token set metrics
        s1_n_set = s1.name_tok_set
        t_n_set = target.name_tok_set
        s1_n_len = len(s1_n_set)
        t_n_len = len(t_n_set)
        n_inter = len(s1_n_set & t_n_set)
        n_union = s1_n_len + t_n_len - n_inter

        name_tok_jaccard = (n_inter / n_union) if n_union > 0 else (1.0 if s1_n_len == 0 and t_n_len == 0 else 0.0)
        name_tok_dice = (2.0 * n_inter / (s1_n_len + t_n_len)) if (s1_n_len + t_n_len) > 0 else 0.0
        name_tok_overlap_cnt = float(n_inter)
        name_tok_overlap_ratio = (n_inter / max(s1_n_len, 1)) if s1_n_len > 0 else 0.0
        name_tokens_missing_cnt = float(len(s1_n_set - t_n_set))
        name_tokens_extra_cnt = float(len(t_n_set - s1_n_set))

        name_prefix_equal = 1.0 if (s1_n and t_n and s1_n[:4] == t_n[:4]) else 0.0

        # RapidFuzz C++ metrics
        name_lev = Levenshtein.normalized_similarity(s1_n, t_n) if (s1_n or t_n) else 0.0
        name_jw = JaroWinkler.similarity(s1_n, t_n) if (s1_n or t_n) else 0.0
        name_fuzz_ratio = fuzz.ratio(s1_n, t_n) / 100.0 if (s1_n and t_n) else 0.0
        name_tok_sort = fuzz.token_sort_ratio(s1_n, t_n) / 100.0 if (s1_n and t_n) else 0.0
        name_tok_set = fuzz.token_set_ratio(s1_n, t_n) / 100.0 if (s1_n and t_n) else 0.0

        # N-gram similarities
        g3_inter = len(s1.name_char3_set & target.name_char3_set)
        g3_union = len(s1.name_char3_set | target.name_char3_set)
        name_char3_jaccard = (g3_inter / g3_union) if g3_union > 0 else 0.0

        g4_inter = len(s1.name_char4_set & target.name_char4_set)
        g4_union = len(s1.name_char4_set | target.name_char4_set)
        name_char4_jaccard = (g4_inter / g4_union) if g4_union > 0 else 0.0

        g5_inter = len(s1.name_char5_set & target.name_char5_set)
        g5_union = len(s1.name_char5_set | target.name_char5_set)
        name_char5_jaccard = (g5_inter / g5_union) if g5_union > 0 else 0.0

        name_phonetic_match = 1.0 if (s1.name_phonetic_sig and s1.name_phonetic_sig == target.name_phonetic_sig) else 0.0

        rare_token_overlap = 0.0
        if self.token_idf:
            shared = s1_n_set & t_n_set
            rare_token_overlap = sum(self.token_idf.get(t, 0.0) for t in shared)

        # -------------------------------------------------------------
        # ADDRESS FEATURES
        # -------------------------------------------------------------
        s1_a = s1.norm_addr
        t_a = target.norm_addr

        addr_raw_exact = 1.0 if s1.raw_addr and s1.raw_addr == target.raw_addr else 0.0
        addr_canonical_exact = 1.0 if s1_a and s1_a == t_a else 0.0

        len_s1_a = len(s1_a)
        len_t_a = len(t_a)
        addr_len_diff = float(abs(len_s1_a - len_t_a))
        addr_rel_len_diff = addr_len_diff / max(len_s1_a, len_t_a, 1)
        addr_is_missing = 1.0 if s1.is_addr_missing or target.is_addr_missing else 0.0

        s1_a_set = s1.addr_tok_set
        t_a_set = target.addr_tok_set
        s1_a_len = len(s1_a_set)
        t_a_len = len(t_a_set)
        a_inter = len(s1_a_set & t_a_set)
        a_union = s1_a_len + t_a_len - a_inter

        addr_tok_jaccard = (a_inter / a_union) if a_union > 0 else (1.0 if s1_a_len == 0 and t_a_len == 0 else 0.0)
        addr_tok_dice = (2.0 * a_inter / (s1_a_len + t_a_len)) if (s1_a_len + t_a_len) > 0 else 0.0
        addr_tok_overlap_cnt = float(a_inter)
        addr_tok_overlap_ratio = (a_inter / max(s1_a_len, 1)) if s1_a_len > 0 else 0.0
        addr_tokens_missing_cnt = float(len(s1_a_set - t_a_set))
        addr_tokens_extra_cnt = float(len(t_a_set - s1_a_set))

        # House & numeric overlaps
        s1_h = s1.house_numbers
        t_h = target.house_numbers
        num_inter = len(s1_h & t_h)
        num_union = len(s1_h | t_h)
        addr_num_overlap_cnt = float(num_inter)
        addr_num_jaccard = (num_inter / num_union) if num_union > 0 else (1.0 if len(s1_h) == 0 and len(t_h) == 0 else 0.0)

        addr_number_exact = 1.0 if (s1_h and t_h and s1_h == t_h) else 0.0
        addr_number_conflict = 1.0 if (s1_h and t_h and not (s1_h & t_h)) else 0.0

        # Postal overlap
        s1_p = s1.postal_codes
        t_p = target.postal_codes
        addr_postal_exact = 1.0 if (s1_p and t_p and (s1_p & t_p)) else 0.0
        addr_postal_prefix = 0.0
        if s1_p and t_p:
            p1 = next(iter(s1_p))
            p2 = next(iter(t_p))
            if len(p1) >= 3 and len(p2) >= 3 and p1[:3] == p2[:3]:
                addr_postal_prefix = 1.0

        addr_lev = Levenshtein.normalized_similarity(s1_a, t_a) if (s1_a or t_a) else 0.0
        addr_jw = JaroWinkler.similarity(s1_a, t_a) if (s1_a or t_a) else 0.0
        addr_tok_sort = fuzz.token_sort_ratio(s1_a, t_a) / 100.0 if (s1_a and t_a) else 0.0
        addr_tok_set = fuzz.token_set_ratio(s1_a, t_a) / 100.0 if (s1_a and t_a) else 0.0

        # -------------------------------------------------------------
        # CROSS-FIELD & COUNTRY
        # -------------------------------------------------------------
        s1_c = s1.country
        t_c = target.country
        country_match = 1.0 if (s1_c and t_c and s1_c == t_c) else 0.0
        country_is_missing = 1.0 if s1.is_country_missing or target.is_country_missing else 0.0

        is_s2 = 1.0 if target.is_s2 else 0.0
        is_s3 = 1.0 if target.is_s3 else 0.0

        name_addr_sim_prod = name_lev * addr_lev
        name_addr_sim_mean = (name_lev + addr_lev) / 2.0
        name_addr_sim_max = max(name_lev, addr_lev)
        name_addr_sim_weighted = 0.65 * name_lev + 0.35 * addr_lev

        name_strong_addr_strong = 1.0 if (name_lev >= 0.85 and addr_lev >= 0.80) else 0.0
        name_strong_addr_weak = 1.0 if (name_lev >= 0.85 and addr_lev <= 0.50) else 0.0
        name_weak_addr_strong = 1.0 if (name_lev <= 0.60 and addr_lev >= 0.80) else 0.0
        name_weak_addr_weak = 1.0 if (name_lev <= 0.60 and addr_lev <= 0.50) else 0.0

        # -------------------------------------------------------------
        # PROVENANCE MASK BITS
        # -------------------------------------------------------------
        pmask = cand.provenance_mask
        num_channels = float(bin(pmask).count("1"))
        prov_exact = float((pmask & ProvenanceMask.EXACT_OR_LEARNED) != 0)
        prov_tfidf = float((pmask & ProvenanceMask.CHAR_TFIDF) != 0)
        prov_rare = float((pmask & ProvenanceMask.RARE_TOKEN) != 0)
        prov_addr = float((pmask & ProvenanceMask.ADDRESS) != 0)
        prov_phon = float((pmask & ProvenanceMask.PHONETIC) != 0)
        prov_learn = float((pmask & ProvenanceMask.LEARNED_VARIANT) != 0)

        # -------------------------------------------------------------
        # CANDIDATE-CONTEXT FEATURES
        # -------------------------------------------------------------
        cand_rank = context_stats.get("candidate_rank", 0.0)
        ret_score = cand.retrieval_score
        best_ret = context_stats.get("best_score", ret_score)
        second_best_ret = context_stats.get("second_best_score", 0.0)
        cand_margin = best_ret - second_best_ret
        cand_cnt = context_stats.get("cand_count", 1.0)
        cnt_05 = context_stats.get("count_above_05", 1.0)
        cnt_07 = context_stats.get("count_above_07", 1.0)
        cnt_08 = context_stats.get("count_above_08", 1.0)

        return [
            name_raw_exact,
            name_canonical_exact,
            name_compact_exact,
            name_learned_exact,
            name_len_diff,
            name_rel_len_diff,
            name_is_missing,
            name_tok_jaccard,
            name_tok_dice,
            name_tok_overlap_cnt,
            name_tok_overlap_ratio,
            name_tokens_missing_cnt,
            name_tokens_extra_cnt,
            name_prefix_equal,
            name_lev,
            name_jw,
            name_fuzz_ratio,
            name_tok_sort,
            name_tok_set,
            name_char3_jaccard,
            name_char4_jaccard,
            name_char5_jaccard,
            name_phonetic_match,
            rare_token_overlap,
            addr_raw_exact,
            addr_canonical_exact,
            addr_len_diff,
            addr_rel_len_diff,
            addr_is_missing,
            addr_tok_jaccard,
            addr_tok_dice,
            addr_tok_overlap_cnt,
            addr_tok_overlap_ratio,
            addr_tokens_missing_cnt,
            addr_tokens_extra_cnt,
            addr_num_overlap_cnt,
            addr_num_jaccard,
            addr_lev,
            addr_jw,
            addr_tok_sort,
            addr_tok_set,
            addr_number_exact,
            addr_number_conflict,
            addr_postal_exact,
            addr_postal_prefix,
            country_match,
            country_is_missing,
            is_s2,
            is_s3,
            name_addr_sim_prod,
            name_addr_sim_mean,
            name_addr_sim_max,
            name_addr_sim_weighted,
            name_strong_addr_strong,
            name_strong_addr_weak,
            name_weak_addr_strong,
            name_weak_addr_weak,
            num_channels,
            prov_exact,
            prov_tfidf,
            prov_rare,
            prov_addr,
            prov_phon,
            prov_learn,
            cand_rank,
            ret_score,
            best_ret,
            second_best_ret,
            cand_margin,
            cand_cnt,
            cnt_05,
            cnt_07,
            cnt_08,
        ]

    def extract_features_for_target_candidates(
        self,
        target: MultiViewRecord,
        candidates: List[CandidatePair],
        s1_records: Dict[int, MultiViewRecord]
    ) -> np.ndarray:
        """Extracts feature matrix (len(candidates), num_features) for all candidate S1s of a target."""
        if not candidates:
            return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

        scores = [c.retrieval_score for c in candidates]
        best_s = scores[0] if scores else 0.0
        sec_s = scores[1] if len(scores) > 1 else 0.0
        cand_cnt = float(len(candidates))
        cnt_05 = float(sum(1 for s in scores if s >= 0.5))
        cnt_07 = float(sum(1 for s in scores if s >= 0.7))
        cnt_08 = float(sum(1 for s in scores if s >= 0.8))

        features_list = []
        for rank, cand in enumerate(candidates):
            s1 = s1_records.get(cand.s1_internal_id)
            if s1 is None:
                continue

            ctx = {
                "candidate_rank": float(rank),
                "best_score": best_s,
                "second_best_score": sec_s,
                "cand_count": cand_cnt,
                "count_above_05": cnt_05,
                "count_above_07": cnt_07,
                "count_above_08": cnt_08,
            }
            row = self.compute_pair_features(s1, target, cand, ctx)
            features_list.append(row)

        return np.array(features_list, dtype=np.float32)
