"""
V3 Tiered Feature Extraction Engine with RapidFuzz Batch / C++ Integration.
Organized into 3 computational tiers:
- Tier 1: Very cheap structural, length, country, and blocker provenance features.
- Tier 2: Precomputed integer/set n-gram and token Jaccard metrics (zero set reallocations).
- Tier 3: RapidFuzz C++ Levenshtein and Jaro-Winkler fuzzy string similarities.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import numpy as np
from rapidfuzz.distance import Levenshtein, JaroWinkler

def compute_tiered_pairwise_features(
    s1_rec: Dict[str, Any],
    target_rec: Dict[str, Any],
    target_id: str,
    provenance_mask: int = 0
) -> List[float]:
    """
    Computes all 35 V3 pairwise features in strictly optimized order.
    """
    s1_name_norm = s1_rec.get("norm_name", "")
    t_name_norm = target_rec.get("norm_name", "")
    s1_name_comp = s1_rec.get("compact_name", "")
    t_name_comp = target_rec.get("compact_name", "")
    s1_addr_norm = s1_rec.get("norm_addr", "")
    t_addr_norm = target_rec.get("norm_addr", "")

    # =========================================================================
    # TIER 1: VERY CHEAP STRUCTURAL & PROVENANCE FEATURES
    # =========================================================================
    name_exact = 1.0 if s1_name_norm and s1_name_norm == t_name_norm else 0.0
    name_comp_exact = 1.0 if s1_name_comp and s1_name_comp == t_name_comp else 0.0
    addr_exact = 1.0 if s1_addr_norm and s1_addr_norm == t_addr_norm else 0.0

    len_s1_n = len(s1_name_norm)
    len_t_n = len(t_name_norm)
    name_len_diff = float(abs(len_s1_n - len_t_n))
    name_rel_len_diff = name_len_diff / max(len_s1_n, len_t_n, 1)
    name_is_missing = 1.0 if not s1_name_norm or not t_name_norm else 0.0

    len_s1_a = len(s1_addr_norm)
    len_t_a = len(t_addr_norm)
    addr_len_diff = float(abs(len_s1_a - len_t_a))
    addr_is_missing = 1.0 if not s1_addr_norm or not t_addr_norm else 0.0

    s1_ctry = s1_rec.get("country", "")
    t_ctry = target_rec.get("country", "")
    country_match = 1.0 if s1_ctry and s1_ctry == t_ctry else 0.0
    country_is_missing = 1.0 if not s1_ctry or not t_ctry else 0.0

    is_s2 = 1.0 if target_id.startswith("S2-") else 0.0
    is_s3 = 1.0 if target_id.startswith("S3-") else 0.0

    # Blocker provenance bits
    num_blockers = float(bin(provenance_mask).count("1"))
    block_exact_name = float((provenance_mask & (1 << 0)) != 0)
    block_compact_name = float((provenance_mask & (1 << 1)) != 0)
    block_soundex = float((provenance_mask & (1 << 2)) != 0)
    block_postal = float((provenance_mask & (1 << 3)) != 0)
    block_translit = float((provenance_mask & (1 << 4)) != 0)

    # =========================================================================
    # TIER 2: PRECOMPUTED SET / N-GRAM JACCARD FEATURES (ZERO ALLOCATION)
    # =========================================================================
    s1_n_set = s1_rec.get("name_tok_set") or set(s1_rec.get("name_tokens", []))
    t_n_set = target_rec.get("name_tok_set") or set(target_rec.get("name_tokens", []))
    s1_n_len = len(s1_n_set)
    t_n_len = len(t_n_set)
    n_inter = len(s1_n_set & t_n_set)
    n_union = s1_n_len + t_n_len - n_inter

    if s1_n_len == 0 and t_n_len == 0:
        name_tok_jaccard = 1.0
    elif s1_n_len == 0 or t_n_len == 0:
        name_tok_jaccard = 0.0
    else:
        name_tok_jaccard = n_inter / n_union if n_union > 0 else 0.0

    name_tok_overlap_cnt = float(n_inter)
    name_tok_overlap_ratio = (2.0 * name_tok_overlap_cnt / (s1_n_len + t_n_len)) if (s1_n_len + t_n_len > 0) else 0.0

    # 3-gram similarity
    s1_3g_set = s1_rec.get("name_3g_set") or set()
    t_3g_set = target_rec.get("name_3g_set") or set()
    s1_3g_len = len(s1_3g_set)
    t_3g_len = len(t_3g_set)
    g_inter = len(s1_3g_set & t_3g_set)
    g_union = s1_3g_len + t_3g_len - g_inter

    if s1_3g_len == 0 and t_3g_len == 0:
        name_char_3gram_sim = 1.0
    elif s1_3g_len == 0 or t_3g_len == 0:
        name_char_3gram_sim = 0.0
    else:
        name_char_3gram_sim = g_inter / g_union if g_union > 0 else 0.0

    # Address token overlaps
    s1_a_set = s1_rec.get("addr_tok_set") or set(s1_rec.get("addr_tokens", []))
    t_a_set = target_rec.get("addr_tok_set") or set(target_rec.get("addr_tokens", []))
    s1_a_len = len(s1_a_set)
    t_a_len = len(t_a_set)
    a_inter = len(s1_a_set & t_a_set)
    a_union = s1_a_len + t_a_len - a_inter

    if s1_a_len == 0 and t_a_len == 0:
        addr_tok_jaccard = 1.0
    elif s1_a_len == 0 or t_a_len == 0:
        addr_tok_jaccard = 0.0
    else:
        addr_tok_jaccard = a_inter / a_union if a_union > 0 else 0.0

    addr_tok_overlap_cnt = float(a_inter)
    addr_tok_overlap_ratio = (2.0 * addr_tok_overlap_cnt / (s1_a_len + t_a_len)) if (s1_a_len + t_a_len > 0) else 0.0

    # Numeric overlaps
    s1_num_set = s1_rec.get("numeric_tok_set") or set(s1_rec.get("numeric_tokens", []))
    t_num_set = target_rec.get("numeric_tok_set") or set(target_rec.get("numeric_tokens", []))
    s1_num_len = len(s1_num_set)
    t_num_len = len(t_num_set)
    num_inter = len(s1_num_set & t_num_set)
    num_union = s1_num_len + t_num_len - num_inter

    addr_num_overlap_cnt = float(num_inter)
    if s1_num_len == 0 and t_num_len == 0:
        addr_num_jaccard = 1.0
    elif s1_num_len == 0 or t_num_len == 0:
        addr_num_jaccard = 0.0
    else:
        addr_num_jaccard = num_inter / num_union if num_union > 0 else 0.0

    # =========================================================================
    # TIER 3: RAPIDFUZZ C++ FUZZY STRING SIMILARITIES
    # =========================================================================
    name_lev = Levenshtein.normalized_similarity(s1_name_norm, t_name_norm) if (s1_name_norm or t_name_norm) else 0.0
    name_jw = JaroWinkler.similarity(s1_name_norm, t_name_norm) if (s1_name_norm or t_name_norm) else 0.0

    addr_lev = Levenshtein.normalized_similarity(s1_addr_norm, t_addr_norm) if (s1_addr_norm or t_addr_norm) else 0.0
    addr_jw = JaroWinkler.similarity(s1_addr_norm, t_addr_norm) if (s1_addr_norm or t_addr_norm) else 0.0

    # Interaction features
    name_addr_sim_prod = name_lev * addr_lev
    name_addr_sim_max = max(name_lev, addr_lev)
    name_addr_sim_weighted = 0.6 * name_lev + 0.4 * addr_lev
    strong_agreement = 1.0 if (name_lev > 0.80 and addr_lev > 0.80) else 0.0

    return [
        name_exact,
        name_comp_exact,
        name_lev,
        name_jw,
        name_tok_jaccard,
        name_tok_overlap_cnt,
        name_tok_overlap_ratio,
        name_char_3gram_sim,
        name_len_diff,
        name_rel_len_diff,
        name_is_missing,
        addr_exact,
        addr_lev,
        addr_jw,
        addr_tok_jaccard,
        addr_tok_overlap_cnt,
        addr_tok_overlap_ratio,
        addr_num_overlap_cnt,
        addr_num_jaccard,
        addr_len_diff,
        addr_is_missing,
        country_match,
        country_is_missing,
        is_s2,
        is_s3,
        name_addr_sim_prod,
        name_addr_sim_max,
        name_addr_sim_weighted,
        strong_agreement,
        # Blocker provenance features
        num_blockers,
        block_exact_name,
        block_compact_name,
        block_soundex,
        block_postal,
        block_translit
    ]
