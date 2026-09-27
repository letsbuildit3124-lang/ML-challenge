"""
ER-X Ultimate: Core Data Types, Record Schemas, and Memory Structures
Includes official submission formatting matching competition specifications.
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
    candidate_s2_ids: List[int] = field(default_factory=list)
    candidate_s3_ids: List[int] = field(default_factory=list)
    
    def to_matching_results_row(self) -> str:
        """Format as strict TSV row: source1_entity_id \t matched_entity_ids"""
        s1_str = f"S1-{self.source1_id}"
        s2_items = [f"S2-{x}" for x in sorted(self.source2_ids)]
        s3_items = [f"S3-{x}" for x in sorted(self.source3_ids)]
        matched_str = ",".join(s2_items + s3_items)
        return f"{s1_str}\t{matched_str}\n"

    def to_candidate_pairs_row(self) -> str:
        """Format as strict TSV row: source1_entity_id \t candidate_entity_ids"""
        s1_str = f"S1-{self.source1_id}"
        s2_cands = [f"S2-{x}" for x in sorted(set(self.candidate_s2_ids))]
        s3_cands = [f"S3-{x}" for x in sorted(set(self.candidate_s3_ids))]
        all_cands = sorted(set(s2_cands + s3_cands))
        cand_str = ",".join(all_cands)
        return f"{s1_str}\t{cand_str}\n"
