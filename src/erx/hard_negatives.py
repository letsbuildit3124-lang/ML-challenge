"""
ER-X Hard-Negative Mining & Training Data Assembly.
Mines informative false S1 candidates from multi-channel retrieval to train robust discriminative GBDT models.
"""

import logging
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np

from src.erx.config import ERXConfig
from src.erx.types import MultiViewRecord, CandidatePair
from src.erx.retrieval import ERXRetrievalEngine
from src.erx.features import ERXFeatureExtractor

logger = logging.getLogger("erx.hard_negatives")


class ERXHardNegativeMiner:
    """Mines realistic hard negatives from retrieval channels."""

    def __init__(
        self,
        config: ERXConfig,
        retrieval_engine: ERXRetrievalEngine,
        feature_extractor: ERXFeatureExtractor,
    ):
        self.config = config
        self.retrieval_engine = retrieval_engine
        self.feature_extractor = feature_extractor

    def mine_and_build_dataset(
        self,
        train_targets: List[MultiViewRecord],
        ground_truth: Dict[int, int],  # target_internal_id -> true_s1_internal_id
        s1_records: Dict[int, MultiViewRecord],
        negatives_per_positive: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves candidates for training targets, extracts positive pairs and top hard negative S1s,
        and produces (X, y) feature and label matrices.
        """
        num_negs = negatives_per_positive or self.config.hard_negatives_per_positive
        logger.info(f"Mining hard negatives for {len(train_targets):,} targets (target ratio: {num_negs} negs/pos)...")

        all_features: List[List[float]] = []
        all_labels: List[int] = []

        # Retrieve in batches
        batch_size = 500
        for b_start in range(0, len(train_targets), batch_size):
            b_targets = train_targets[b_start:b_start + batch_size]
            b_candidates = self.retrieval_engine.retrieve_batch_with_tfidf(b_targets)

            for target, candidates in zip(b_targets, b_candidates):
                true_s1_id = ground_truth.get(target.internal_id)
                if true_s1_id is None:
                    continue  # Target does not have a true S1 in train fold

                # Find if true S1 was retrieved
                retrieved_s1_ids = {c.s1_internal_id for c in candidates}
                scored_cands = list(candidates)

                # If true S1 not retrieved in top K, inject it as positive candidate
                if true_s1_id not in retrieved_s1_ids and true_s1_id in s1_records:
                    pos_cand = CandidatePair(
                        target_internal_id=target.internal_id,
                        s1_internal_id=true_s1_id,
                        retrieval_score=0.5,
                        provenance_mask=0
                    )
                    scored_cands.append(pos_cand)

                # Extract features for all candidates
                feats = self.feature_extractor.extract_features_for_target_candidates(
                    target, scored_cands, s1_records
                )

                # Identify positive vs negatives
                negs_collected = 0
                for idx, cand in enumerate(scored_cands):
                    if cand.s1_internal_id == true_s1_id:
                        all_features.append(feats[idx].tolist())
                        all_labels.append(1)
                    elif negs_collected < num_negs:
                        all_features.append(feats[idx].tolist())
                        all_labels.append(0)
                        negs_collected += 1

        X = np.array(all_features, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int32)
        logger.info(f"Built training dataset: X shape {X.shape}, Positives: {int(np.sum(y)):,}, Negatives: {int(len(y) - np.sum(y)):,}")
        return X, y
