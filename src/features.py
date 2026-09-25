"""
Pairwise feature extraction module for entity resolution.
Uses rapidfuzz for high-speed C++ string similarity calculations.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from rapidfuzz.distance import Levenshtein, JaroWinkler

def get_char_ngrams(text: str, n: int = 3) -> set:
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def jaccard_set_similarity(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def compute_pairwise_features(
    s1_rec: Dict[str, Any],
    target_rec: Dict[str, Any],
    target_id: str
) -> List[float]:
    """
    Computes all pairwise features between an S1 entity and a target candidate entity.
    Returns feature vector matching Config.feature_names.
    """
    # Names
    s1_name_norm = s1_rec["norm_name"]
    t_name_norm = target_rec["norm_name"]
    s1_name_comp = s1_rec["compact_name"]
    t_name_comp = target_rec["compact_name"]

    name_exact = 1.0 if s1_name_norm and s1_name_norm == t_name_norm else 0.0
    name_comp_exact = 1.0 if s1_name_comp and s1_name_comp == t_name_comp else 0.0

    name_lev = Levenshtein.normalized_similarity(s1_name_norm, t_name_norm) if (s1_name_norm or t_name_norm) else 0.0
    name_jw = JaroWinkler.similarity(s1_name_norm, t_name_norm) if (s1_name_norm or t_name_norm) else 0.0

    s1_n_toks = set(s1_rec["name_tokens"])
    t_n_toks = set(target_rec["name_tokens"])
    name_tok_jaccard = jaccard_set_similarity(s1_n_toks, t_n_toks)
    name_tok_overlap_cnt = float(len(s1_n_toks & t_n_toks))
    name_tok_overlap_ratio = (2.0 * name_tok_overlap_cnt / (len(s1_n_toks) + len(t_n_toks))) if (s1_n_toks or t_n_toks) else 0.0

    s1_3grams = get_char_ngrams(s1_name_norm, 3)
    t_3grams = get_char_ngrams(t_name_norm, 3)
    name_char_3gram_sim = jaccard_set_similarity(s1_3grams, t_3grams)

    len_s1_n = len(s1_name_norm)
    len_t_n = len(t_name_norm)
    name_len_diff = float(abs(len_s1_n - len_t_n))
    name_rel_len_diff = name_len_diff / max(len_s1_n, len_t_n, 1)
    name_is_missing = 1.0 if not s1_name_norm or not t_name_norm else 0.0

    # Addresses
    s1_addr_norm = s1_rec["norm_addr"]
    t_addr_norm = target_rec["norm_addr"]

    addr_exact = 1.0 if s1_addr_norm and s1_addr_norm == t_addr_norm else 0.0
    addr_lev = Levenshtein.normalized_similarity(s1_addr_norm, t_addr_norm) if (s1_addr_norm or t_addr_norm) else 0.0
    addr_jw = JaroWinkler.similarity(s1_addr_norm, t_addr_norm) if (s1_addr_norm or t_addr_norm) else 0.0

    s1_a_toks = set(s1_rec["addr_tokens"])
    t_a_toks = set(target_rec["addr_tokens"])
    addr_tok_jaccard = jaccard_set_similarity(s1_a_toks, t_a_toks)
    addr_tok_overlap_cnt = float(len(s1_a_toks & t_a_toks))
    addr_tok_overlap_ratio = (2.0 * addr_tok_overlap_cnt / (len(s1_a_toks) + len(t_a_toks))) if (s1_a_toks or t_a_toks) else 0.0

    s1_nums = set(s1_rec["numeric_tokens"])
    t_nums = set(target_rec["numeric_tokens"])
    addr_num_overlap_cnt = float(len(s1_nums & t_nums))
    addr_num_jaccard = jaccard_set_similarity(s1_nums, t_nums)

    len_s1_a = len(s1_addr_norm)
    len_t_a = len(t_addr_norm)
    addr_len_diff = float(abs(len_s1_a - len_t_a))
    addr_is_missing = 1.0 if not s1_addr_norm or not t_addr_norm else 0.0

    # Country & Source
    s1_ctry = s1_rec["country"]
    t_ctry = target_rec["country"]
    country_match = 1.0 if s1_ctry and s1_ctry == t_ctry else 0.0
    country_is_missing = 1.0 if not s1_ctry or not t_ctry else 0.0

    is_s2 = 1.0 if target_id.startswith("S2-") else 0.0
    is_s3 = 1.0 if target_id.startswith("S3-") else 0.0

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
        strong_agreement
    ]
