"""
Antigravity V3.2 Approximate Retrieval & Fuzzy Candidate Generation Engine.
Implements:
1. Weighted Character N-Gram & Substring Retrieval (Names & Addresses)
2. Address-First Street & Numeric Matching
3. Fixed Token-Set / Word-Reorder Signature Matching
4. Bounded RapidFuzz C++ Approximate Scoring
"""

import re
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
import polars as pl
from rapidfuzz import fuzz, process

from src.blocking_v2 import CORP_STOPWORDS, compute_soundex
from src.normalize import normalize_text, offline_transliterate

def extract_informative_tokens(text: str) -> List[str]:
    """Extracts non-stopword alphanumeric tokens."""
    if not text:
        return []
    toks = [t for t in re.sub(r"[^\w\s]", " ", text.lower()).split() if t not in CORP_STOPWORDS and len(t) >= 3]
    return toks

def compute_token_set_signature(norm_name: str) -> str:
    """Computes an order-independent signature from sorted informative tokens."""
    toks = extract_informative_tokens(norm_name)
    if not toks:
        return ""
    toks.sort()
    return "_".join(toks[:3])

def extract_addr_street_key(norm_addr: str) -> Tuple[str, str]:
    """Extracts (first_addr_num, first_street_token_p4)."""
    if not norm_addr:
        return ("", "")
    digits = re.findall(r"\b\d+\b", norm_addr)
    first_num = digits[0] if digits else ""
    words = [w for w in re.sub(r"[^\w\s]", " ", norm_addr.lower()).split() if w.isalpha() and w not in CORP_STOPWORDS and len(w) >= 3]
    first_street = words[0][:4] if words else ""
    return (first_num, first_street)

def run_bounded_rapidfuzz_matching(
    s1_records: Dict[str, Dict[str, Any]],
    candidate_pools_by_s1: Dict[str, List[Tuple[str, Dict[str, Any]]]],
    top_k: int = 25,
    min_score: float = 65.0
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Executes vectorized C++ RapidFuzz matching across bounded candidate pools per S1 entity.
    Returns:
        Dict[s1_id, List[(target_id, similarity_score)]]
    """
    results: Dict[str, List[Tuple[str, float]]] = {}

    for s1_id, pool in candidate_pools_by_s1.items():
        if not pool:
            continue
        s1_info = s1_records.get(s1_id, {})
        s1_name = s1_info.get("norm_name", "")
        if not s1_name:
            continue

        target_ids = [tid for tid, trec in pool]
        target_names = [trec.get("norm_name", "") for tid, trec in pool]

        if not target_names:
            continue

        extracted = process.extract(
            s1_name,
            target_names,
            scorer=fuzz.WRatio,
            limit=top_k,
            score_cutoff=min_score
        )

        matched_tuples = []
        for match_name, score, idx in extracted:
            matched_tuples.append((target_ids[idx], float(score)))

        if matched_tuples:
            results[s1_id] = matched_tuples

    return results
