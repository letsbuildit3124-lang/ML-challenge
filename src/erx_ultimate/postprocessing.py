"""
ER-X Ultimate: Post-Processing, Target Ownership Resolution & Expected-F0.5 Decoding
"""

from __future__ import annotations
import logging
from collections import defaultdict
from typing import List, Dict, Set, Tuple
import numpy as np

from src.erx_ultimate.config import CONFIG, UltimateConfig
from src.erx_ultimate.types import ScoredPair, EntityCluster, SourceType

logger = logging.getLogger("erx_ultimate.postprocessing")


class PostProcessingEngine:
    """Decodes scored pairs into disjoint entity clusters under Macro F0.5 optimization."""

    def __init__(self, config: UltimateConfig = CONFIG):
        self.config = config
        self.base_threshold = config.decision.base_threshold
        self.min_margin = config.decision.min_margin

    def resolve_target_ownership(
        self,
        scored_pairs: List[ScoredPair],
    ) -> Dict[Tuple[int, int], int]:
        """
        Enforce strict 1-to-at-most-1 target ownership.
        Returns mapping of (target_id, target_source) -> best_s1_id.
        """
        # Group candidate scores by target
        target_candidates: Dict[Tuple[int, int], List[ScoredPair]] = defaultdict(list)
        for pair in scored_pairs:
            key = (pair.target_id, int(pair.target_source))
            target_candidates[key].append(pair)

        resolved_ownership: Dict[Tuple[int, int], int] = {}

        for key, candidates in target_candidates.items():
            # Sort descending by calibrated probability
            candidates.sort(key=lambda p: p.calibrated_prob, reverse=True)
            top_cand = candidates[0]

            # Condition 1: Probability must exceed base decision threshold
            if top_cand.calibrated_prob < self.base_threshold:
                continue  # Remains unmatched singleton

            # Condition 2: Margin check if there's a runner up
            if len(candidates) > 1:
                second_cand = candidates[1]
                margin = top_cand.calibrated_prob - second_cand.calibrated_prob
                if margin < self.min_margin and top_cand.calibrated_prob < 0.90:
                    continue  # Ambiguous conflict; drop to preserve precision

            resolved_ownership[key] = top_cand.s1_id

        return resolved_ownership

    def aggregate_clusters(
        self,
        s1_ids: List[int],
        target_ownership: Dict[Tuple[int, int], int],
    ) -> List[EntityCluster]:
        """
        Aggregate resolved targets into full S1 clusters.
        Preserves exact order and completeness of input S1 records.
        """
        # Invert ownership: s1_id -> (s2_ids, s3_ids)
        s1_to_s2: Dict[int, List[int]] = defaultdict(list)
        s1_to_s3: Dict[int, List[int]] = defaultdict(list)

        for (tgt_id, tgt_src), s1_id in target_ownership.items():
            if tgt_src == int(SourceType.SOURCE2):
                s1_to_s2[s1_id].append(tgt_id)
            elif tgt_src == int(SourceType.SOURCE3):
                s1_to_s3[s1_id].append(tgt_id)

        clusters = []
        for s1_id in s1_ids:
            clusters.append(EntityCluster(
                source1_id=s1_id,
                source2_ids=sorted(s1_to_s2.get(s1_id, [])),
                source3_ids=sorted(s1_to_s3.get(s1_id, [])),
            ))

        return clusters
