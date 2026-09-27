"""
ER-X Ultimate: Core Data Types, Record Schemas, and Memory Structures
Includes pre-computed token/ngram caches for zero-redundancy feature extraction.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple, Dict, Any, Set
import numpy as np


class SourceType(IntEnum):
    SOURCE1 = 1
    SOURCE2 = 2
    SOURCE3 = 3


@dataclass(slots=True)
class EntityRecord:
    """Compact in-memory representation of an ingested and normalized entity record."""
    id: int
    source: SourceType
    name_raw: str
    name_norm: str
    address_raw: str
    address_norm: str
    city_norm: str
    state_norm: str
    postal_code_norm: str
    country_norm: str
    phone_norm: str
    website_norm: str
    
    # Pre-computed cached token/ngram sets to eliminate redundant extraction in hot-loops
    name_tokens_set: Set[str] = field(default_factory=set)
    address_tokens_set: Set[str] = field(default_factory=set)
    numeric_tokens_set: Set[str] = field(default_factory=set)
    ngrams_2: Set[str] = field(default_factory=set)
    ngrams_3: Set[str] = field(default_factory=set)
    ngrams_4: Set[str] = field(default_factory=set)
    addr_ngrams_3: Set[str] = field(default_factory=set)
    phonetic_set: Set[str] = field(default_factory=set)


@dataclass(slots=True)
class CandidateMatch:
    """Candidate pair retrieved for downstream scoring."""
    target_id: int
    target_source: SourceType
    s1_id: int
    rrf_score: float
    channel_mask: int = 0
    channel_ranks: Dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class ScoredPair:
    """Candidate pair with model probability and calibrated score."""
    target_id: int
    target_source: SourceType
    s1_id: int
    raw_prob: float
    calibrated_prob: float
    rrf_score: float
    features: Optional[np.ndarray] = None


@dataclass(slots=True)
class EntityCluster:
    """Final resolved entity cluster keyed by S1 ID."""
    source1_id: int
    source2_ids: List[int] = field(default_factory=list)
    source3_ids: List[int] = field(default_factory=list)
    
    def to_tsv_row(self) -> str:
        """Format as strict TSV row matching official submission specification."""
        s2_str = ",".join(map(str, sorted(self.source2_ids))) if self.source2_ids else ""
        s3_str = ",".join(map(str, sorted(self.source3_ids))) if self.source3_ids else ""
        return f"{self.source1_id}\t{s2_str}\t{s3_str}\n"
