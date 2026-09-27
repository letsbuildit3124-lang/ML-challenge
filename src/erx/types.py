"""
ER-X Core Types, Data Structures, and ID Mapping Infrastructure.
High-Performance Memory-Optimized Layout.
"""

from dataclasses import dataclass, field
from enum import IntFlag
from typing import Dict, List, Set, Tuple, Optional, Any, Union
import numpy as np


class ProvenanceMask(IntFlag):
    """Bitmask for candidate retrieval channels."""
    NONE = 0
    EXACT_OR_LEARNED = 1 << 0   # Bit 0: Exact canonical / learned name / compact
    CHAR_TFIDF = 1 << 1         # Bit 1: Sparse Char 3-5 TF-IDF Cosine
    RARE_TOKEN = 1 << 2         # Bit 2: Inverted Rare Token IDF
    ADDRESS = 1 << 3            # Bit 3: Address / House / Locality / Numeric
    PHONETIC = 1 << 4           # Bit 4: Soundex / Metaphone Token Signatures
    LEARNED_VARIANT = 1 << 5    # Bit 5: OCR / Learned Typo / Alias Expansions


@dataclass
class InternalIDMapper:
    """Fast two-way mapping between string entity IDs (e.g. S1-12345) and uint32 indices."""
    str_to_int: Dict[str, int] = field(default_factory=dict)
    int_to_str: List[str] = field(default_factory=list)

    def get_or_add(self, entity_id: str) -> int:
        idx = self.str_to_int.get(entity_id)
        if idx is None:
            idx = len(self.int_to_str)
            self.str_to_int[entity_id] = idx
            self.int_to_str.append(entity_id)
        return idx

    def get_int(self, entity_id: str) -> Optional[int]:
        return self.str_to_int.get(entity_id)

    def get_str(self, idx: int) -> str:
        return self.int_to_str[idx]

    def __len__(self) -> int:
        return len(self.int_to_str)


def char_ngrams_set(text: str, n: int) -> Set[str]:
    """Fast C-level character n-gram generation."""
    if not text:
        return set()
    length = len(text)
    if length < n:
        return {text}
    return {text[i : i + n] for i in range(length - n + 1)}


@dataclass(slots=True)
class MultiViewRecord:
    """
    High-performance precomputed multi-view record representation.
    Holds precomputed token lists, n-gram sets, and numeric signatures in slots.
    Provides instantaneous O(1) attribute access with ZERO inner-loop memory allocations.
    """
    internal_id: int
    entity_id: str
    country: str

    # String views
    raw_name: str
    norm_name: str
    compact_name: str
    translit_name: str
    translit_comp_name: str
    learned_name: str
    sorted_token_name: str
    name_phonetic_sig: str

    raw_addr: str
    norm_addr: str
    translit_addr: str
    numeric_signature: str

    # Precomputed token and set views (computed once on creation)
    name_tokens: List[str]
    name_tok_set: Set[str]
    translit_tokens: List[str]
    translit_tok_set: Set[str]
    name_char3_set: Set[str]
    name_char4_set: Set[str]
    name_char5_set: Set[str]

    addr_tokens: List[str]
    addr_tok_set: Set[str]
    house_numbers: Set[str]
    postal_codes: Set[str]

    # Flags
    is_s2: bool = False
    is_s3: bool = False
    is_name_missing: bool = False
    is_addr_missing: bool = False
    is_country_missing: bool = False


@dataclass(slots=True)
class CandidatePair:
    """Lightweight candidate link between a target and an S1 entity."""
    target_internal_id: int
    s1_internal_id: int
    retrieval_score: float = 0.0
    provenance_mask: int = 0


@dataclass(slots=True)
class ScoredPrediction:
    """Scored candidate pair with feature vector and model confidence."""
    target_internal_id: int
    s1_internal_id: int
    raw_probability: float
    calibrated_probability: float
    is_match: bool
    features: Optional[np.ndarray] = None


@dataclass(slots=True)
class MatchPrediction:
    """Final calibrated match decision with provenance tracking."""
    target_entity_id: str
    matched_s1_entity_id: Optional[str]
    probability: float
    is_singleton: bool
    top_candidates: List[Tuple[str, float]] = field(default_factory=list)
