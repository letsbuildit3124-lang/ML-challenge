"""
ER-X Ultimate: Highly-Optimized Vectorized 73-Feature Extraction Matrix
Includes short-circuit exact bypass and pre-cached token/ngram set reuse.
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Dict, Any, Optional, Set

try:
    from rapidfuzz import fuzz, distance
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

from src.erx_ultimate.types import EntityRecord, CandidateMatch
from src.erx_ultimate.normalization import extract_ngrams, compute_soundex

NUM_FEATURES = 73


def compute_jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def compute_dice_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Compute Dice similarity coefficient."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    total = len(set1) + len(set2)
    return (2.0 * intersection) / total if total > 0 else 0.0


def populate_record_caches(rec: EntityRecord) -> None:
    """Pre-compute cached n-gram and token sets on EntityRecord once per record."""
    if not rec.name_tokens_set and rec.name_norm:
        toks = rec.name_norm.split()
        rec.name_tokens_set = set(toks)
        rec.phonetic_set = {compute_soundex(t) for t in toks if t}
        rec.ngrams_2 = extract_ngrams(rec.name_norm, 2)
        rec.ngrams_3 = extract_ngrams(rec.name_norm, 3)
        rec.ngrams_4 = extract_ngrams(rec.name_norm, 4)

    if not rec.address_tokens_set and rec.address_norm:
        addr_toks = rec.address_norm.split()
        rec.address_tokens_set = set(addr_toks)
        rec.numeric_tokens_set = {t for t in addr_toks if t.isdigit()}
        rec.addr_ngrams_3 = extract_ngrams(rec.address_norm, 3)


def extract_pair_features(
    tgt_rec: EntityRecord,
    s1_rec: EntityRecord,
    candidate: CandidateMatch,
) -> np.ndarray:
    """Compute 73 float32 features for a single candidate pair with short-circuit optimizations."""
    feats = np.zeros(NUM_FEATURES, dtype=np.float32)

    # 1. Exact Match Features (12)
    same_name_norm = (tgt_rec.name_norm == s1_rec.name_norm and tgt_rec.name_norm != "")
    same_addr_norm = (tgt_rec.address_norm == s1_rec.address_norm and tgt_rec.address_norm != "")

    feats[0] = 1.0 if same_name_norm else 0.0
    feats[1] = 1.0 if tgt_rec.name_raw == s1_rec.name_raw and tgt_rec.name_raw != "" else 0.0
    feats[2] = 1.0 if same_addr_norm else 0.0
    feats[3] = 1.0 if tgt_rec.city_norm == s1_rec.city_norm and tgt_rec.city_norm != "" else 0.0
    feats[4] = 1.0 if tgt_rec.state_norm == s1_rec.state_norm and tgt_rec.state_norm != "" else 0.0
    feats[5] = 1.0 if tgt_rec.postal_code_norm == s1_rec.postal_code_norm and tgt_rec.postal_code_norm != "" else 0.0
    feats[6] = 1.0 if tgt_rec.country_norm == s1_rec.country_norm and tgt_rec.country_norm != "" else 0.0
    feats[7] = 1.0 if tgt_rec.phone_norm == s1_rec.phone_norm and tgt_rec.phone_norm != "" else 0.0
    feats[8] = 1.0 if tgt_rec.website_norm == s1_rec.website_norm and tgt_rec.website_norm != "" else 0.0
    feats[9] = 1.0 if tgt_rec.name_norm and s1_rec.name_norm and (tgt_rec.name_norm in s1_rec.name_norm or s1_rec.name_norm in tgt_rec.name_norm) else 0.0
    feats[10] = 1.0 if tgt_rec.address_norm and s1_rec.address_norm and (tgt_rec.address_norm in s1_rec.address_norm or s1_rec.address_norm in tgt_rec.address_norm) else 0.0
    feats[11] = 1.0 if tgt_rec.website_norm and s1_rec.website_norm and (tgt_rec.website_norm in s1_rec.website_norm or s1_rec.website_norm in tgt_rec.website_norm) else 0.0

    # 2. N-Gram Overlap & Dice Similarities (16) (Using pre-cached sets)
    tgt_ng2 = tgt_rec.ngrams_2 if tgt_rec.ngrams_2 else extract_ngrams(tgt_rec.name_norm, 2)
    s1_ng2 = s1_rec.ngrams_2 if s1_rec.ngrams_2 else extract_ngrams(s1_rec.name_norm, 2)
    tgt_ng3 = tgt_rec.ngrams_3 if tgt_rec.ngrams_3 else extract_ngrams(tgt_rec.name_norm, 3)
    s1_ng3 = s1_rec.ngrams_3 if s1_rec.ngrams_3 else extract_ngrams(s1_rec.name_norm, 3)
    tgt_ng4 = tgt_rec.ngrams_4 if tgt_rec.ngrams_4 else extract_ngrams(tgt_rec.name_norm, 4)
    s1_ng4 = s1_rec.ngrams_4 if s1_rec.ngrams_4 else extract_ngrams(s1_rec.name_norm, 4)

    feats[12] = compute_jaccard_similarity(tgt_ng2, s1_ng2)
    feats[13] = compute_dice_similarity(tgt_ng2, s1_ng2)
    feats[14] = compute_jaccard_similarity(tgt_ng3, s1_ng3)
    feats[15] = compute_dice_similarity(tgt_ng3, s1_ng3)
    feats[16] = compute_jaccard_similarity(tgt_ng4, s1_ng4)
    feats[17] = compute_dice_similarity(tgt_ng4, s1_ng4)

    # Address n-grams
    tgt_addr_ng3 = tgt_rec.addr_ngrams_3 if tgt_rec.addr_ngrams_3 else extract_ngrams(tgt_rec.address_norm, 3)
    s1_addr_ng3 = s1_rec.addr_ngrams_3 if s1_rec.addr_ngrams_3 else extract_ngrams(s1_rec.address_norm, 3)
    feats[18] = compute_jaccard_similarity(tgt_addr_ng3, s1_addr_ng3)
    feats[19] = compute_dice_similarity(tgt_addr_ng3, s1_addr_ng3)

    # Character stats
    len1 = len(tgt_rec.name_norm)
    len2 = len(s1_rec.name_norm)
    feats[20] = abs(len1 - len2)
    feats[21] = min(len1, len2) / max(len1, len2, 1)
    feats[22] = abs(len(tgt_rec.address_norm) - len(s1_rec.address_norm))
    feats[23] = min(len(tgt_rec.address_norm), len(s1_rec.address_norm)) / max(len(tgt_rec.address_norm), len(s1_rec.address_norm), 1)
    feats[24] = 1.0 if tgt_rec.name_norm and s1_rec.name_norm and tgt_rec.name_norm[0] == s1_rec.name_norm[0] else 0.0
    feats[25] = 1.0 if tgt_rec.name_norm and s1_rec.name_norm and tgt_rec.name_norm[-1] == s1_rec.name_norm[-1] else 0.0
    feats[26] = 1.0 if tgt_rec.postal_code_norm and s1_rec.postal_code_norm and tgt_rec.postal_code_norm[:2] == s1_rec.postal_code_norm[:2] else 0.0
    feats[27] = 1.0 if tgt_rec.phone_norm and s1_rec.phone_norm and tgt_rec.phone_norm[-4:] == s1_rec.phone_norm[-4:] else 0.0

    # 3. Fuzzy & String Distance Features (20) with Fast Short-Circuiting
    if HAS_RAPIDFUZZ:
        if same_name_norm:
            feats[28:34] = 1.0
            feats[40] = 1.0
            feats[42] = 1.0
            feats[43] = 1.0
            feats[45] = 1.0
        elif not tgt_rec.name_norm or not s1_rec.name_norm:
            feats[28:34] = 0.0
            feats[40] = 0.0
            feats[42] = 0.0
            feats[43] = 0.0
            feats[45] = 0.0
        else:
            feats[28] = fuzz.ratio(tgt_rec.name_norm, s1_rec.name_norm) / 100.0
            feats[29] = fuzz.partial_ratio(tgt_rec.name_norm, s1_rec.name_norm) / 100.0
            feats[30] = fuzz.token_sort_ratio(tgt_rec.name_norm, s1_rec.name_norm) / 100.0
            feats[31] = fuzz.token_set_ratio(tgt_rec.name_norm, s1_rec.name_norm) / 100.0
            feats[32] = fuzz.WRatio(tgt_rec.name_norm, s1_rec.name_norm) / 100.0
            feats[33] = fuzz.QRatio(tgt_rec.name_norm, s1_rec.name_norm) / 100.0
            feats[40] = distance.Levenshtein.normalized_similarity(tgt_rec.name_norm, s1_rec.name_norm)
            feats[42] = distance.DamerauLevenshtein.normalized_similarity(tgt_rec.name_norm, s1_rec.name_norm)
            feats[43] = distance.JaroWinkler.similarity(tgt_rec.name_norm, s1_rec.name_norm)
            feats[45] = distance.OSA.normalized_similarity(tgt_rec.name_norm, s1_rec.name_norm)

        # Raw names
        if tgt_rec.name_raw and s1_rec.name_raw:
            feats[34] = fuzz.ratio(tgt_rec.name_raw, s1_rec.name_raw) / 100.0
            feats[35] = fuzz.token_sort_ratio(tgt_rec.name_raw, s1_rec.name_raw) / 100.0

        # Address fuzzy
        if same_addr_norm:
            feats[36:40] = 1.0
            feats[41] = 1.0
            feats[44] = 1.0
        elif not tgt_rec.address_norm or not s1_rec.address_norm:
            feats[36:40] = 0.0
            feats[41] = 0.0
            feats[44] = 0.0
        else:
            feats[36] = fuzz.ratio(tgt_rec.address_norm, s1_rec.address_norm) / 100.0
            feats[37] = fuzz.partial_ratio(tgt_rec.address_norm, s1_rec.address_norm) / 100.0
            feats[38] = fuzz.token_sort_ratio(tgt_rec.address_norm, s1_rec.address_norm) / 100.0
            feats[39] = fuzz.token_set_ratio(tgt_rec.address_norm, s1_rec.address_norm) / 100.0
            feats[41] = distance.Levenshtein.normalized_similarity(tgt_rec.address_norm, s1_rec.address_norm)
            feats[44] = distance.JaroWinkler.similarity(tgt_rec.address_norm, s1_rec.address_norm)

        feats[46] = (fuzz.ratio(tgt_rec.city_norm, s1_rec.city_norm) / 100.0) if (tgt_rec.city_norm and s1_rec.city_norm) else 0.0
        feats[47] = (fuzz.ratio(tgt_rec.website_norm, s1_rec.website_norm) / 100.0) if (tgt_rec.website_norm and s1_rec.website_norm) else 0.0

    # 4. Token Set & Overlap Statistics (12) (Using pre-cached sets)
    toks1 = tgt_rec.name_tokens_set if tgt_rec.name_tokens_set else set(tgt_rec.name_norm.split())
    toks2 = s1_rec.name_tokens_set if s1_rec.name_tokens_set else set(s1_rec.name_norm.split())
    feats[48] = len(toks1)
    feats[49] = len(toks2)
    feats[50] = len(toks1 & toks2)
    feats[51] = compute_jaccard_similarity(toks1, toks2)
    feats[52] = compute_dice_similarity(toks1, toks2)
    feats[53] = len(toks1 - toks2)
    feats[54] = len(toks2 - toks1)

    addr_toks1 = tgt_rec.address_tokens_set if tgt_rec.address_tokens_set else set(tgt_rec.address_norm.split())
    addr_toks2 = s1_rec.address_tokens_set if s1_rec.address_tokens_set else set(s1_rec.address_norm.split())
    feats[55] = len(addr_toks1 & addr_toks2)
    feats[56] = compute_jaccard_similarity(addr_toks1, addr_toks2)
    feats[57] = 1.0 if (toks1 and toks2 and list(toks1)[0] == list(toks2)[0]) else 0.0
    feats[58] = 1.0 if (toks1 and toks2 and list(toks1)[-1] == list(toks2)[-1]) else 0.0
    feats[59] = 1.0 if (toks1.issubset(toks2) or toks2.issubset(toks1)) and (toks1 and toks2) else 0.0

    # 5. Geographic, Numeric & Phonetic Alignment (8)
    num1 = tgt_rec.numeric_tokens_set if tgt_rec.numeric_tokens_set else {t for t in addr_toks1 if t.isdigit()}
    num2 = s1_rec.numeric_tokens_set if s1_rec.numeric_tokens_set else {t for t in addr_toks2 if t.isdigit()}
    feats[60] = 1.0 if num1 and num2 and num1 == num2 else 0.0
    feats[61] = 1.0 if num1 and num2 and len(num1 & num2) > 0 else 0.0
    feats[62] = 1.0 if num1 and num2 and len(num1 & num2) == 0 else 0.0
    
    ph1 = tgt_rec.phonetic_set if tgt_rec.phonetic_set else {compute_soundex(t) for t in toks1 if t}
    ph2 = s1_rec.phonetic_set if s1_rec.phonetic_set else {compute_soundex(t) for t in toks2 if t}
    feats[63] = compute_jaccard_similarity(ph1, ph2)
    feats[64] = 1.0 if ph1 and ph2 and len(ph1 & ph2) > 0 else 0.0
    feats[65] = 1.0 if tgt_rec.city_norm and s1_rec.city_norm and tgt_rec.city_norm == s1_rec.city_norm else 0.0
    feats[66] = 1.0 if tgt_rec.state_norm and s1_rec.state_norm and tgt_rec.state_norm == s1_rec.state_norm else 0.0
    feats[67] = 1.0 if tgt_rec.postal_code_norm and s1_rec.postal_code_norm and tgt_rec.postal_code_norm == s1_rec.postal_code_norm else 0.0

    # 6. Retrieval Channel & Structural Metadata (5)
    feats[68] = candidate.rrf_score
    feats[69] = float(candidate.channel_mask)
    feats[70] = 1.0 if (candidate.channel_mask & 1) else 0.0
    feats[71] = 1.0 if (candidate.channel_mask & 2) else 0.0
    feats[72] = float(int(candidate.target_source))

    return feats


def extract_batch_features(
    tgt_records: List[EntityRecord],
    s1_records: List[EntityRecord],
    candidates: List[CandidateMatch],
) -> np.ndarray:
    """Vectorized batch feature extractor returning (N, 73) float32 matrix with pre-cached sets."""
    num_pairs = len(candidates)
    feature_matrix = np.zeros((num_pairs, NUM_FEATURES), dtype=np.float32)
    
    for i in range(num_pairs):
        populate_record_caches(tgt_records[i])
        populate_record_caches(s1_records[i])
        feature_matrix[i, :] = extract_pair_features(
            tgt_records[i],
            s1_records[i],
            candidates[i]
        )
    return feature_matrix
