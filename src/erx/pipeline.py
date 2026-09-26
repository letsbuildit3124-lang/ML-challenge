"""
ER-X End-to-End Pipeline Orchestrator.
Supports:
- 100-S1 Smoke Benchmark
- Multi-seed (42, 123, 2026) 80/20 Entity-level Validation
- Full Production Test Inference with Strict Exclusivity and Singleton Abstention
"""

import gc
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import polars as pl
import duckdb

from src.erx.config import ERXConfig
from src.erx.types import InternalIDMapper, MultiViewRecord, CandidatePair, ScoredPrediction
from src.erx.normalization import ERXNormalizer
from src.erx.learned_rules import LearnedRuleEngine
from src.erx.retrieval import ERXRetrievalEngine
from src.erx.features import ERXFeatureExtractor, FEATURE_NAMES
from src.erx.hard_negatives import ERXHardNegativeMiner
from src.erx.model import ERXModelTrainer, ERXCalibrator

logger = logging.getLogger("erx.pipeline")


def compute_entity_level_metrics(
    predictions: Dict[str, List[str]],  # s1_id -> predicted target IDs
    ground_truth: Dict[str, List[str]],  # s1_id -> true target IDs
    candidate_map: Optional[Dict[str, Set[str]]] = None,  # s1_id -> set of candidate target IDs
) -> Dict[str, float]:
    """
    Computes strict entity-level metrics:
    - Candidate Recall (proportion of true targets that were in candidate sets)
    - Precision, Recall, Entity-level Macro F0.5
    - Singleton Accuracy (accuracy on entities where ground truth is empty [])
    """
    f05_scores = []
    p_scores = []
    r_scores = []

    singleton_correct = 0
    singleton_total = 0

    true_targets_total = 0
    candidates_hit = 0

    all_s1 = set(ground_truth.keys())

    for s1_id in all_s1:
        true_set = set(ground_truth.get(s1_id, []))
        pred_set = set(predictions.get(s1_id, []))

        # Check singleton accuracy
        if len(true_set) == 0:
            singleton_total += 1
            if len(pred_set) == 0:
                singleton_correct += 1

        # Check candidate recall
        if candidate_map is not None and len(true_set) > 0:
            cands = candidate_map.get(s1_id, set())
            for t_id in true_set:
                true_targets_total += 1
                if t_id in cands:
                    candidates_hit += 1

        # Precision, Recall, Macro F0.5
        if len(pred_set) == 0 and len(true_set) == 0:
            f05_scores.append(1.0)
            p_scores.append(1.0)
            r_scores.append(1.0)
        elif len(pred_set) == 0 or len(true_set) == 0:
            f05_scores.append(0.0)
            p_scores.append(0.0)
            r_scores.append(0.0)
        else:
            tp = len(pred_set & true_set)
            p = tp / len(pred_set)
            r = tp / len(true_set)
            p_scores.append(p)
            r_scores.append(r)
            if p + r > 0 and (0.25 * p + r) > 0:
                # F0.5 = (1 + 0.5^2) * (P * R) / (0.5^2 * P + R) = 1.25 * P * R / (0.25 * P + R)
                f05 = (1.25 * p * r) / (0.25 * p + r)
                f05_scores.append(f05)
            else:
                f05_scores.append(0.0)

    macro_f05 = float(np.mean(f05_scores)) if f05_scores else 0.0
    macro_p = float(np.mean(p_scores)) if p_scores else 0.0
    macro_r = float(np.mean(r_scores)) if r_scores else 0.0
    singleton_acc = (singleton_correct / max(singleton_total, 1))

    cand_recall = (candidates_hit / max(true_targets_total, 1)) if true_targets_total > 0 else 1.0

    return {
        "macro_f05": round(macro_f05, 5),
        "macro_precision": round(macro_p, 5),
        "macro_recall": round(macro_r, 5),
        "singleton_accuracy": round(singleton_acc, 5),
        "singleton_count": singleton_total,
        "candidate_recall": round(cand_recall, 5),
        "total_true_targets": true_targets_total,
    }


class ERXPipeline:
    """Master pipeline executing high-recall entity resolution."""

    def __init__(self, config: ERXConfig):
        self.config = config
        self.config.ensure_directories()
        self.id_mapper = InternalIDMapper()
        self.normalizer = ERXNormalizer()
        self.rule_engine = LearnedRuleEngine(
            min_alias_observations=config.min_alias_observations,
            min_alias_purity=config.min_alias_purity,
        )

    def load_ground_truth(self, tsv_path: Path) -> Dict[str, List[str]]:
        """Loads ground truth mapping: s1_id -> list of target_ids."""
        gt: Dict[str, List[str]] = {}
        con = duckdb.connect()
        query = f"SELECT source1_entity_id, matched_entity_ids FROM read_csv_auto('{tsv_path}', sep='\\t', header=True)"
        for s1_id, matches in con.execute(query).fetchall():
            if matches and matches.strip():
                gt[s1_id] = [m.strip() for m in matches.split(",") if m.strip()]
            else:
                gt[s1_id] = []
        return gt

    def run_smoke_benchmark(self, num_s1: int = 100) -> Dict[str, Any]:
        """
        Runs fast 100-S1 smoke benchmark:
        - Loads first 100 S1 records and their associated targets + negative targets
        - Indexes S1 records across all 6 channels
        - Retrieves candidate S1s for all targets
        - Extracts features and trains LightGBM model
        - Evaluates Candidate Recall, Precision, Recall, Macro F0.5, Singleton Accuracy.
        """
        start_time = time.time()
        logger.info(f"=== Starting ER-X Smoke Benchmark with {num_s1} S1 Entities ===")

        con = duckdb.connect()

        # 1. Load sample S1 records
        s1_tsv = self.config.data_dir / "train" / "train_source1.tsv"
        gt_tsv = self.config.data_dir / "train" / "train_ground_truth.tsv"

        s1_rows = con.execute(f"SELECT entity_id, business_name, business_address, country FROM read_csv_auto('{s1_tsv}', sep='\\t', header=True) LIMIT {num_s1}").fetchall()
        s1_id_set = {r[0] for r in s1_rows}

        # 2. Load ground truth for sample S1 records
        gt_map: Dict[str, List[str]] = {}
        target_to_true_s1: Dict[str, str] = {}
        all_target_ids: Set[str] = set()

        gt_rows = con.execute(f"SELECT source1_entity_id, matched_entity_ids FROM read_csv_auto('{gt_tsv}', sep='\\t', header=True) WHERE source1_entity_id IN (SELECT unnest({list(s1_id_set)}))").fetchall()
        for s1_id, matches in gt_rows:
            if matches and matches.strip():
                targets = [m.strip() for m in matches.split(",") if m.strip()]
                gt_map[s1_id] = targets
                for t in targets:
                    target_to_true_s1[t] = s1_id
                    all_target_ids.add(t)
            else:
                gt_map[s1_id] = []

        # Also sample some distractor targets from S2 and S3 (not matching any S1)
        s2_tsv = self.config.data_dir / "train" / "train_source2.tsv"
        s3_tsv = self.config.data_dir / "train" / "train_source3.tsv"

        extra_s2 = con.execute(f"SELECT entity_id, business_name, business_address, country FROM read_csv_auto('{s2_tsv}', sep='\\t', header=True) LIMIT 200").fetchall()
        extra_s3 = con.execute(f"SELECT entity_id, business_name, business_address, country FROM read_csv_auto('{s3_tsv}', sep='\\t', header=True) LIMIT 200").fetchall()

        target_rows = con.execute(f"""
            SELECT entity_id, business_name, business_address, country FROM read_csv_auto('{s2_tsv}', sep='\\t', header=True) WHERE entity_id IN (SELECT unnest({list(all_target_ids)}))
            UNION ALL
            SELECT entity_id, business_name, business_address, country FROM read_csv_auto('{s3_tsv}', sep='\\t', header=True) WHERE entity_id IN (SELECT unnest({list(all_target_ids)}))
        """).fetchall()

        target_rows.extend(extra_s2)
        target_rows.extend(extra_s3)

        logger.info(f"Loaded {len(s1_rows)} S1 records and {len(target_rows)} target records.")

        # 3. Normalization
        s1_mv_records: List[MultiViewRecord] = []
        for r in s1_rows:
            int_id = self.id_mapper.get_or_add(r[0])
            mv = self.normalizer.normalize_record(int_id, r[0], r[1], r[2], r[3])
            s1_mv_records.append(mv)

        target_mv_records: List[MultiViewRecord] = []
        for r in target_rows:
            int_id = self.id_mapper.get_or_add(r[0])
            mv = self.normalizer.normalize_record(int_id, r[0], r[1], r[2], r[3])
            target_mv_records.append(mv)

        # 4. Index S1 in Retrieval Engine
        retrieval_engine = ERXRetrievalEngine(self.config)
        retrieval_engine.index_s1(s1_mv_records)

        # 5. Batch Retrieval with sparse TF-IDF (Channel B)
        t0_ret = time.time()
        retrieved_candidates = retrieval_engine.retrieve_batch_with_tfidf(target_mv_records)
        retrieval_time = time.time() - t0_ret

        # 6. Candidate Recall Evaluation
        s1_cands_map: Dict[str, Set[str]] = defaultdict(set)
        for target, cands in zip(target_mv_records, retrieved_candidates):
            for c in cands:
                s1_str = self.id_mapper.get_str(c.s1_internal_id)
                s1_cands_map[s1_str].add(target.entity_id)

        # Feature Extraction & Model Training
        s1_dict = {rec.internal_id: rec for rec in s1_mv_records}
        feature_extractor = ERXFeatureExtractor(token_idf=retrieval_engine.token_idf)

        # Ground truth integer mapping
        gt_int_map: Dict[int, int] = {}
        for t_str, s1_str in target_to_true_s1.items():
            t_int = self.id_mapper.get_int(t_str)
            s1_int = self.id_mapper.get_int(s1_str)
            if t_int is not None and s1_int is not None:
                gt_int_map[t_int] = s1_int

        miner = ERXHardNegativeMiner(self.config, retrieval_engine, feature_extractor)
        X, y = miner.mine_and_build_dataset(target_mv_records, gt_int_map, s1_dict, negatives_per_positive=5)

        # Train model
        trainer = ERXModelTrainer(self.config)
        train_stats = trainer.train(X, y)

        # Score candidates for all targets and predict
        target_predictions: Dict[str, str] = {}  # target_id -> best_s1_id
        target_candidate_scores: Dict[str, float] = {}

        for target, cands in zip(target_mv_records, retrieved_candidates):
            if not cands:
                continue
            feats = feature_extractor.extract_features_for_target_candidates(target, cands, s1_dict)
            probs = trainer.predict_proba(feats, calibrate=False)

            best_idx = int(np.argmax(probs))
            best_prob = float(probs[best_idx])
            sec_prob = float(np.partition(probs, -2)[-2]) if len(probs) > 1 else 0.0
            margin = best_prob - sec_prob

            # Apply Target Exclusivity threshold and margin defense
            if best_prob >= self.config.match_threshold and margin >= self.config.margin_threshold:
                best_s1_id = self.id_mapper.get_str(cands[best_idx].s1_internal_id)
                target_predictions[target.entity_id] = best_s1_id
                target_candidate_scores[target.entity_id] = best_prob

        # Invert target->S1 to S1->targets
        s1_predictions: Dict[str, List[str]] = {r[0]: [] for r in s1_rows}
        for t_id, s1_id in target_predictions.items():
            if s1_id in s1_predictions:
                s1_predictions[s1_id].append(t_id)

        # Compute full entity-level metrics
        metrics = compute_entity_level_metrics(s1_predictions, gt_map, s1_cands_map)
        elapsed = time.time() - start_time

        logger.info(f"=== Smoke Benchmark Completed in {elapsed:.2f}s ===")
        logger.info(f"Candidate Recall:   {metrics['candidate_recall']*100:.2f}%")
        logger.info(f"Macro F0.5:         {metrics['macro_f05']:.4f}")
        logger.info(f"Precision:          {metrics['macro_precision']*100:.2f}%")
        logger.info(f"Recall:             {metrics['macro_recall']*100:.2f}%")
        logger.info(f"Singleton Accuracy: {metrics['singleton_accuracy']*100:.2f}%")

        return {
            "num_s1": num_s1,
            "num_targets": len(target_mv_records),
            "retrieval_time_s": round(retrieval_time, 3),
            "total_elapsed_s": round(elapsed, 2),
            "metrics": metrics,
            "train_stats": train_stats,
        }
