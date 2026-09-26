"""
ER-X Core Types, Data Structures, and ID Mapping Infrastructure.
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


def _char_ngrams(text: str, n: int) -> Set[str]:
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


@dataclass(slots=True)
class MultiViewRecord:
    """
    Ultra-compact multi-view record representation.
    Stores string views in slots (~120 bytes per record) and computes set views on demand.
    Reduces memory from 13 GB to < 450 MB for 2.2M records.
    """
    internal_id: int
    entity_id: str
    country: str

    # Name views
    raw_name: str
    norm_name: str
    compact_name: str
    translit_name: str
    translit_comp_name: str
    learned_name: str
    sorted_token_name: str
    name_phonetic_sig: str

    # Address views
    raw_addr: str
    norm_addr: str
    translit_addr: str
    numeric_signature: str

    # Flags
    is_s2: bool = False
    is_s3: bool = False
    is_name_missing: bool = False
    is_addr_missing: bool = False
    is_country_missing: bool = False

    # Dynamic set and token properties (zero-allocation until accessed)
    @property
    def name_tokens(self) -> List[str]:
        return self.norm_name.split() if self.norm_name else []

    @property
    def name_tok_set(self) -> Set[str]:
        return set(self.name_tokens)

    @property
    def translit_tokens(self) -> List[str]:
        return self.translit_name.split() if self.translit_name else []

    @property
    def translit_tok_set(self) -> Set[str]:
        return set(self.translit_tokens)

    @property
    def name_char3_set(self) -> Set[str]:
        is_ascii = self.norm_name.isascii()
        s = _char_ngrams(self.norm_name, 3)
        if not is_ascii and self.translit_name:
            s = s | _char_ngrams(self.translit_name, 3)
        return s

    @property
    def name_char4_set(self) -> Set[str]:
        is_ascii = self.norm_name.isascii()
        s = _char_ngrams(self.norm_name, 4)
        if not is_ascii and self.translit_name:
            s = s | _char_ngrams(self.translit_name, 4)
        return s

    @property
    def name_char5_set(self) -> Set[str]:
        is_ascii = self.norm_name.isascii()
        s = _char_ngrams(self.norm_name, 5)
        if not is_ascii and self.translit_name:
            s = s | _char_ngrams(self.translit_name, 5)
        return s

    @property
    def addr_tokens(self) -> List[str]:
        return self.norm_addr.split() if self.norm_addr else []

    @property
    def addr_tok_set(self) -> Set[str]:
        return set(self.addr_tokens)

    @property
    def house_numbers(self) -> Set[str]:
        num_toks = [w for w in self.addr_tokens if w.isdigit()]
        return set(num_toks[:2]) if num_toks else set()

    @property
    def postal_codes(self) -> Set[str]:
        return {w for w in self.addr_tokens if w.isdigit() and len(w) in (5, 6)}

    @property
    def city_tokens(self) -> Set[str]:
        return set()


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

