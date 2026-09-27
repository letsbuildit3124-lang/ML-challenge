"""
ER-X MODEL SCORE EVALUATOR: IN-SAMPLE TRAINING UNIVERSE DIAGNOSTIC
===================================================================

WARNING:
IN-SAMPLE DIAGNOSTIC — OPTIMISTIC — NOT A COMPETITION SCORE
MODEL WAS TRAINED USING THIS DATA

This tool evaluates the existing final model and 6-channel retrieval pipeline
against the COMPLETE TRAINING GROUND TRUTH (2,206,821 S1 entities, 10,320,219 targets).

Strict Invariants:
1. Zero Label Leakage: Retrieval is 100% blind; GT is evaluated strictly AFTER prediction.
2. Official Metric Semantics: Entity-level Macro F0.5 across all S1 entities (including singletons).
3. Error Decomposition: Separates Retrieval Misses from Model Misses and False Positives.
4. Streaming Architecture: Memory bounded (< 4 GB RAM) via Parquet/DuckDB chunked processing.
5. Resumable Checkpointing: Granular shard caching with automatic checksum verification.

CLI Usage:
----------
1. In-Sample Training Diagnostic (Full Streaming):
   python -u -m src.erx.evaluate_final_model --mode train-diagnostic --chunk-size 100000 --resume

2. Smoke Test (Quick 10,000 S1 / 20,000 Target Verification):
   python -u -m src.erx.evaluate_final_model --smoke

3. Check Out-Of-Sample Holdout Artifacts:
   python -u -m src.erx.evaluate_final_model --mode val-check
"""

import os
import sys

# ----------------------------------------------------------------------
# Enforce Single-Threaded BLAS/OpenMP to Prevent Thread Oversubscription
# ----------------------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import gc
import time
import math
import json
import pickle
import hashlib
import argparse
import logging
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Set, Tuple, Optional, Any, Union

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler

from src.resource_tracker import get_current_rss_mb, log_memory_status
from src.erx.config import ERXConfig
from src.erx.types import InternalIDMapper, MultiViewRecord, CompactS1Record, CandidatePair, ProvenanceMask, char_ngrams_set
from src.erx.normalization import ERXNormalizer, compact_name, normalize_text, offline_transliterate
from src.erx.cache_manager import (
    get_safe_duckdb_connection,
    ensure_cached_parquet,
    ensure_ground_truth_pairs_parquet,
    load_compact_s1_records_from_parquet,
)
from src.erx.learned_rules import LearnedRuleEngine
from src.erx.retrieval import ERXRetrievalEngine, CSRChannelIndex
from src.erx.features import ERXFeatureExtractor, FEATURE_NAMES
from src.erx.model import ERXModelTrainer, ERXCalibrator

logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("erx.evaluate_final_model")

EVAL_WARNING = "IN-SAMPLE DIAGNOSTIC — OPTIMISTIC — NOT A COMPETITION SCORE"


# =====================================================================
# Integrity & Hash Helpers
# =====================================================================

def compute_file_hash(path: Path) -> str:
    """Computes SHA256 checksum of a file on disk."""
    if not path.exists():
        return "NOT_FOUND"
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def compute_retrieval_config_hash(config: ERXConfig) -> str:
    """Computes SHA256 checksum of the retrieval configuration parameters."""
    cfg_data = {
        "tfidf_ngram_range": config.tfidf_ngram_range,
        "tfidf_max_features": config.tfidf_max_features,
        "tfidf_top_k": config.tfidf_top_k,
        "rare_token_max_posting_size": config.rare_token_max_posting_size,
        "rare_token_min_idf": config.rare_token_min_idf,
        "rare_token_top_k": config.rare_token_top_k,
        "address_top_k": config.address_top_k,
        "phonetic_top_k": config.phonetic_top_k,
        "max_total_candidates_per_target": config.max_total_candidates_per_target,
    }
    serialized = json.dumps(cfg_data, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_feature_schema_hash() -> str:
    """Computes SHA256 checksum of the 73-feature schema definition."""
    serialized = json.dumps(FEATURE_NAMES)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# =====================================================================
# Out-Of-Sample Holdout Artifact Inspector
# =====================================================================

def inspect_existing_holdout(config: ERXConfig) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Inspects whether existing artifacts contain a genuine entity-level 80/20 development split
    and a model trained WITHOUT the validation S1 entities.
    """
    dev_dir = config.artifacts_dir / "dev"
    dev_model = dev_dir / "models" / "lightgbm_dev_model.txt"
    dev_cal = dev_dir / "models" / "isotonic_calibrator.pkl"
    dev_diag = dev_dir / "reports" / "validation_diagnostics.json"
    legacy_val = config.base_dir / "reports" / "val_results.json"

    # Check for genuine ER-X dev validation artifact
    if dev_model.exists() and dev_cal.exists() and dev_diag.exists():
        try:
            with open(dev_diag, "r", encoding="utf-8") as f:
                diag_data = json.load(f)
            return True, diag_data
        except Exception:
            pass

    return False, None


# =====================================================================
# Shard Management & Checkpoint Resume
# =====================================================================

def compute_shard_checksum(metadata: Dict[str, Any]) -> str:
    """Computes deterministic MD5 checksum over evaluation shard metadata."""
    serialized = json.dumps(metadata, sort_keys=True)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()


def is_eval_shard_valid(shard_parquet: Path, shard_meta: Path) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Checks whether an evaluation shard exists and matches integrity metadata."""
    if not shard_parquet.exists() or not shard_meta.exists():
        return False, None
    try:
        with open(shard_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if shard_parquet.stat().st_size < 100:
            return False, None
        expected_chk = compute_shard_checksum({k: v for k, v in meta.items() if k != "checksum"})
        if meta.get("checksum") != expected_chk:
            return False, None
        return True, meta
    except Exception:
        return False, None


def write_eval_shard(
    shard_parquet: Path,
    shard_meta: Path,
    target_ids: List[str],
    sources: List[str],
    true_s1_ids: List[str],
    pred_s1_ids: List[str],
    p_matches: np.ndarray,
    is_matches: np.ndarray,
    retrieved_true_s1: np.ndarray,
    best_probs: np.ndarray,
    second_best_probs: np.ndarray,
    retrieval_scores: np.ndarray,
    prov_masks: np.ndarray,
    metadata_info: Dict[str, Any],
):
    """Writes atomic, checksummed evaluation shard."""
    tmp_parquet = shard_parquet.with_suffix(".tmp.parquet")
    tmp_meta = shard_meta.with_suffix(".tmp.json")

    columns = {
        "target_id": pa.array(target_ids, type=pa.string()),
        "source": pa.array(sources, type=pa.string()),
        "true_s1_id": pa.array(true_s1_ids, type=pa.string()),
        "predicted_s1_id": pa.array(pred_s1_ids, type=pa.string()),
        "p_match": p_matches,
        "is_match": is_matches,
        "retrieved_true_s1": retrieved_true_s1,
        "best_prob": best_probs,
        "second_best_prob": second_best_probs,
        "retrieval_score": retrieval_scores,
        "provenance_mask": prov_masks,
    }
    table = pa.Table.from_pydict(columns)
    pq.write_table(table, tmp_parquet, compression="zstd")
    del table
    gc.collect()

    metadata_info["total_targets"] = int(len(target_ids))
    metadata_info["matches_predicted"] = int(np.sum(is_matches))
    metadata_info["file_size_bytes"] = tmp_parquet.stat().st_size
    metadata_info["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    metadata_info["checksum"] = compute_shard_checksum(metadata_info)

    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(metadata_info, f, indent=2)

    os.replace(tmp_parquet, shard_parquet)
    os.replace(tmp_meta, shard_meta)


# =====================================================================
# High-Throughput Subbatch Scoring Worker (Strictly Blind Retrieval)
# =====================================================================

def _process_target_subbatch_diagnostic(
    targets: List[MultiViewRecord],
    country_indexes: Dict[str, ERXRetrievalEngine],
    s1_dict: Dict[int, CompactS1Record],
    feat_extractor: ERXFeatureExtractor,
    num_s1: int,
) -> Dict[str, Any]:
    """
    Evaluates a subbatch of target records with STRICT BLIND CANDIDATE RETRIEVAL.
    
    RUNTIME INVARIANT:
    No ground-truth linkage is available or referenced during candidate generation or scoring.
    Candidates are retrieved purely via 6-channel index lookup over the S1 universe.
    """
    tier1_matches: List[Tuple[int, str]] = []
    tier1_candidates: List[Tuple[int, str]] = []

    tier2_targets: List[MultiViewRecord] = []
    tier2_cand_lists: List[List[CandidatePair]] = []
    target_all_retrieved: Dict[str, List[int]] = {}

    for target in targets:
        tid = target.entity_id
        c_key = target.country if (country_indexes and target.country in country_indexes) else "OTHER"
        engine = country_indexes.get(c_key)
        if engine is None:
            target_all_retrieved[tid] = []
            continue

        # 1. Tier 1 Fast-Path Check: Exact Compact Name
        exact_s1_ids = engine.index_compact_name.get(target.compact_name) if target.compact_name else None
        if exact_s1_ids is not None and len(exact_s1_ids) == 1:
            s1_int = int(exact_s1_ids[0])
            s1_cand = s1_dict.get(s1_int)
            if s1_cand is not None:
                if not target.house_numbers or not s1_cand.house_numbers or (target.house_numbers & s1_cand.house_numbers):
                    if s1_int < num_s1:
                        tier1_matches.append((s1_int, target.entity_id))
                        tier1_candidates.append((s1_int, target.entity_id))
                        target_all_retrieved[tid] = [s1_int]
                        continue

        # 2. Fast Zero-Candidate Screening
        has_exact = bool(
            (target.compact_name and target.compact_name in engine.index_compact_name)
            or (target.norm_name and target.norm_name in engine.index_norm_name)
            or (target.translit_comp_name and target.translit_comp_name in engine.index_compact_name)
            or (target.sorted_token_name and target.sorted_token_name in engine.index_sorted_tokens)
        )
        has_rare = any(tok in engine.token_postings for tok in target.name_tok_set) or any(tok in engine.token_postings for tok in target.translit_tok_set)
        has_num = bool(target.numeric_signature and target.numeric_signature in engine.index_numeric_sig)
        has_phon = bool(target.name_phonetic_sig and target.name_phonetic_sig in engine.index_phonetic)

        if not (has_exact or has_rare or has_num or has_phon):
            target_all_retrieved[tid] = []
            continue

        # 3. Blind 6-Channel Retrieval (Top 15)
        cands = engine.retrieve_for_target(target, top_k=15)
        if not cands:
            target_all_retrieved[tid] = []
            continue

        # Enforce Runtime Anti-Leakage Assertion
        for c in cands:
            assert c.provenance_mask != 0, f"FATAL: Synthetic candidate injection detected for target {tid}!"

        retrieved_s1_ints = [c.s1_internal_id for c in cands]
        target_all_retrieved[tid] = retrieved_s1_ints

        tier2_targets.append(target)
        tier2_cand_lists.append(cands)

    tier2_features = np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    if tier2_targets:
        all_s1_ids = []
        all_t_idx = []
        all_scores = []
        all_masks = []
        all_ranks = []
        all_best = []
        all_sec = []
        all_cnt = []
        all_c05 = []
        all_c07 = []
        all_c08 = []

        for t_idx, (t_rec, cands) in enumerate(zip(tier2_targets, tier2_cand_lists)):
            c_len = len(cands)
            b_s = cands[0].retrieval_score
            sec_s = cands[1].retrieval_score if c_len > 1 else 0.0
            c05 = sum(1 for c in cands if c.retrieval_score >= 0.5)
            c07 = sum(1 for c in cands if c.retrieval_score >= 0.7)
            c08 = sum(1 for c in cands if c.retrieval_score >= 0.8)

            for rank, c in enumerate(cands):
                all_s1_ids.append(c.s1_internal_id)
                all_t_idx.append(t_idx)
                all_scores.append(c.retrieval_score)
                all_masks.append(c.provenance_mask)
                all_ranks.append(float(rank))
                all_best.append(b_s)
                all_sec.append(sec_s)
                all_cnt.append(float(c_len))
                all_c05.append(float(c05))
                all_c07.append(float(c07))
                all_c08.append(float(c08))

        cand_data = {
            "cand_s1_ids": np.array(all_s1_ids, dtype=np.uint32),
            "cand_target_idx": np.array(all_t_idx, dtype=np.int32),
            "cand_scores": np.array(all_scores, dtype=np.float32),
            "cand_prov_masks": np.array(all_masks, dtype=np.uint32),
            "cand_ranks": np.array(all_ranks, dtype=np.float32),
            "best_scores": np.array(all_best, dtype=np.float32),
            "second_best_scores": np.array(all_sec, dtype=np.float32),
            "cand_counts": np.array(all_cnt, dtype=np.float32),
            "counts_above_05": np.array(all_c05, dtype=np.float32),
            "counts_above_07": np.array(all_c07, dtype=np.float32),
            "counts_above_08": np.array(all_c08, dtype=np.float32),
            "total_pairs": len(all_s1_ids),
        }
        tier2_features = feat_extractor.extract_features_batch(tier2_targets, s1_dict, cand_data)

    return {
        "tier1_matches": tier1_matches,
        "tier1_candidates": tier1_candidates,
        "tier2_targets": tier2_targets,
        "tier2_cand_lists": tier2_cand_lists,
        "tier2_features": tier2_features,
        "target_all_retrieved": target_all_retrieved,
        "target_count": len(targets),
    }


# =====================================================================
# S1 Shared Inverted Index Management
# =====================================================================

def get_or_build_training_s1_indexes(
    config: ERXConfig,
    num_workers: int = 8,
    max_s1_records: Optional[int] = None,
) -> Tuple[List[CompactS1Record], Dict[str, ERXRetrievalEngine], Dict[int, CompactS1Record], ERXFeatureExtractor, List[str], Dict[str, int]]:
    """Builds or memory-maps S1 CSR inverted indexes over the complete training S1 universe."""
    id_mapper = InternalIDMapper()
    train_s1_tsv = config.data_dir / "train" / "train_source1.tsv"
    train_s1_parquet = config.cache_dir / "train_s1_normalized.parquet"
    csr_base_dir = config.cache_dir / "indexes" / "train_s1_csr"
    csr_base_dir.mkdir(parents=True, exist_ok=True)

    ensure_cached_parquet(train_s1_tsv, train_s1_parquet, is_s2=False, is_s3=False, num_workers=num_workers)
    s1_mvs = load_compact_s1_records_from_parquet(train_s1_parquet, id_mapper, max_records=max_s1_records)
    num_s1 = len(s1_mvs)
    s1_ordered_ids = [m.entity_id for m in s1_mvs]
    s1_dict = {m.internal_id: m for m in s1_mvs}
    s1_id_to_int = {m.entity_id: m.internal_id for m in s1_mvs}

    country_indexes: Dict[str, ERXRetrievalEngine] = {
        "US": ERXRetrievalEngine(config),
        "India": ERXRetrievalEngine(config),
        "France": ERXRetrievalEngine(config),
        "OTHER": ERXRetrievalEngine(config),
    }
    s1_by_country: Dict[str, List[CompactS1Record]] = defaultdict(list)
    for mv in s1_mvs:
        c_key = mv.country if mv.country in country_indexes else "OTHER"
        s1_by_country[c_key].append(mv)

    # Check if cached CSR indexes exist on disk
    all_loaded = True
    for c_key in country_indexes:
        c_csr_dir = csr_base_dir / c_key
        if not country_indexes[c_key].load_csr_indexes(c_csr_dir):
            all_loaded = False
            break

    if all_loaded:
        logger.info(f"Loaded existing S1 CSR indexes from disk ({csr_base_dir}).")
    else:
        logger.info(f"Building 6-Channel CSR indexes for {num_s1:,} Training S1 entities...")
        for c_key, mvs in s1_by_country.items():
            if mvs:
                country_indexes[c_key].index_s1(mvs)
                country_indexes[c_key].save_csr_indexes(csr_base_dir / c_key)
                logger.info(f"  -> Persisted '{c_key}' CSR index ({len(mvs):,} entities).")

    feat_extractor = ERXFeatureExtractor(token_idf=country_indexes["US"].token_idf)
    return s1_mvs, country_indexes, s1_dict, feat_extractor, s1_ordered_ids, s1_id_to_int


# =====================================================================
# Official Entity-Level Macro F0.5 Calculation
# =====================================================================

def calculate_entity_f05(gt_set: Set[str], pred_set: Set[str]) -> Tuple[float, float, float, int, int, int, str]:
    """
    Calculates exact competition Precision, Recall, and F0.5 for a single S1 entity.
    Follows official evaluator semantics (utils/validate_submission.py and src/evaluation.py).
    """
    if not gt_set and not pred_set:
        return 1.0, 1.0, 1.0, 0, 0, 0, "SINGLETON_CORRECT"
    if not gt_set and pred_set:
        return 0.0, 0.0, 1.0, 0, len(pred_set), 0, "SINGLETON_FP"
    if gt_set and not pred_set:
        return 0.0, 1.0, 0.0, 0, 0, len(gt_set), "MISSED_MATCH"

    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    denom = (0.25 * precision + recall)
    f05 = (1.25 * precision * recall) / denom if denom > 0 else 0.0

    if fn == 0 and fp == 0:
        err_type = "CORRECT"
    elif fn > 0 and tp > 0:
        err_type = "PARTIAL_RECALL_MISS"
    elif fp > 0:
        err_type = "EXTRA_TARGET_PREDICTED"
    else:
        err_type = "MISSED_MATCH"

    return f05, precision, recall, tp, fp, fn, err_type


# =====================================================================
# Main In-Sample Diagnostic Pipeline
# =====================================================================

def run_in_sample_training_diagnostic(
    config: ERXConfig,
    chunk_size: int = 100000,
    resume: bool = True,
    smoke_test: bool = False,
    max_s1_records: Optional[int] = None,
    num_threads: int = 8,
    model_path_override: Optional[str] = None,
    calibrator_path_override: Optional[str] = None,
    output_dir_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes full in-sample diagnostic evaluation of the final model against 100% training universe.
    """
    t0_start = time.time()
    config.ensure_directories()

    eval_out_dir = Path(output_dir_override) if output_dir_override else (config.artifacts_dir / "evaluation")
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = eval_out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve Model & Calibrator Artifacts
    final_artifact_dir = config.artifacts_dir / "final"
    model_file = Path(model_path_override) if model_path_override else (final_artifact_dir / "models" / "lightgbm_final_model.txt")
    calibrator_file = Path(calibrator_path_override) if calibrator_path_override else (final_artifact_dir / "models" / "isotonic_calibrator.pkl")

    if not model_file.exists():
        fallback_model = config.base_dir / "models" / "lightgbm_baseline.txt"
        if fallback_model.exists():
            logger.warning(f"Final model {model_file} not found; falling back to baseline {fallback_model} for diagnostic.")
            model_file = fallback_model
        else:
            raise FileNotFoundError(f"FATAL: Model artifact not found at {model_file} or {fallback_model}.")

    logger.info(f"Loading LightGBM model from {model_file}...")
    trainer = ERXModelTrainer(config)
    trainer.load(model_file)

    calibrator = ERXCalibrator(method="isotonic")
    if calibrator_file.exists():
        logger.info(f"Loading Calibrator from {calibrator_file}...")
        with open(calibrator_file, "rb") as f:
            calibrator = pickle.load(f)
    else:
        logger.warning(f"Calibrator not found at {calibrator_file}; operating with raw probability passthrough.")

    model_hash = compute_file_hash(model_file)
    retrieval_hash = compute_retrieval_config_hash(config)
    schema_hash = compute_feature_schema_hash()

    # 2. Ingest S1 Inverted Index
    s1_mvs, country_indexes, s1_dict, feat_extractor, s1_ordered_ids, s1_id_to_int = get_or_build_training_s1_indexes(
        config, num_workers=num_threads, max_s1_records=max_s1_records
    )
    num_s1 = len(s1_mvs)
    s1_set = set(s1_ordered_ids)

    # 3. Ingest Authoritative Training Ground Truth
    gt_tsv = config.data_dir / "train" / "train_ground_truth.tsv"
    gt_parquet = config.cache_dir / "ground_truth_pairs.parquet"
    ensure_ground_truth_pairs_parquet(gt_tsv, gt_parquet, num_workers=num_threads)

    logger.info("Ingesting Ground Truth linkages via DuckDB Parquet cache...")
    con_gt = get_safe_duckdb_connection(num_threads=4, max_memory_gb="4GB")
    gt_rows = con_gt.execute(f"SELECT s1_id, target_id FROM read_parquet('{gt_parquet}')").fetchall()
    con_gt.close()

    s1_to_gt_targets: Dict[str, Set[str]] = defaultdict(set)
    s1_to_gt_s2: Dict[str, Set[str]] = defaultdict(set)
    s1_to_gt_s3: Dict[str, Set[str]] = defaultdict(set)
    gt_target_to_s1: Dict[str, str] = {}
    total_gt_pairs = 0
    gt_s2_pairs = 0
    gt_s3_pairs = 0

    for sid, tid in gt_rows:
        if sid in s1_set:
            s1_to_gt_targets[sid].add(tid)
            gt_target_to_s1[tid] = sid
            total_gt_pairs += 1
            if tid.startswith("S2-"):
                s1_to_gt_s2[sid].add(tid)
                gt_s2_pairs += 1
            elif tid.startswith("S3-"):
                s1_to_gt_s3[sid].add(tid)
                gt_s3_pairs += 1

    del gt_rows
    gc.collect()

    gt_singleton_s1_cnt = sum(1 for sid in s1_ordered_ids if not s1_to_gt_targets[sid])
    gt_matched_s1_cnt = num_s1 - gt_singleton_s1_cnt

    logger.info(
        f"GT Loaded: {total_gt_pairs:,} total pairs ({gt_s2_pairs:,} S2, {gt_s3_pairs:,} S3) | "
        f"{gt_matched_s1_cnt:,} matched S1, {gt_singleton_s1_cnt:,} singleton S1."
    )

    # 4. Stream Target Universe & Evaluate Blind Predictions
    sources = [
        ("s2", "Source 2", config.cache_dir / "train_s2_normalized.parquet", config.data_dir / "train" / "train_source2.tsv", True, False, 5_034_616),
        ("s3", "Source 3", config.cache_dir / "train_s3_normalized.parquet", config.data_dir / "train" / "train_source3.tsv", False, True, 5_285_603),
    ]

    all_shard_files = []
    total_targets_evaluated = 0
    id_mapper = InternalIDMapper()

    for prefix, name, parquet_path, tsv_path, is_s2, is_s3, expected_rows in sources:
        ensure_cached_parquet(tsv_path, parquet_path, is_s2=is_s2, is_s3=is_s3, num_workers=num_threads)
        pq_file = pq.ParquetFile(parquet_path)
        total_stream_rows = min(10000, pq_file.metadata.num_rows) if smoke_test else pq_file.metadata.num_rows
        chunk_idx = 0
        src_targets_done = 0

        logger.info(f"Streaming {name} ({parquet_path.name}) for In-Sample Diagnostic...")

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for batch in pq_file.iter_batches(batch_size=chunk_size):
                chunk_idx += 1
                shard_parquet = shards_dir / f"shard_{prefix}_{chunk_idx:04d}.parquet"
                shard_meta = shards_dir / f"shard_{prefix}_{chunk_idx:04d}.meta.json"

                if resume:
                    is_valid, meta_info = is_eval_shard_valid(shard_parquet, shard_meta)
                    if is_valid:
                        src_targets_done += meta_info.get("total_targets", chunk_size)
                        total_targets_evaluated += meta_info.get("total_targets", chunk_size)
                        all_shard_files.append(shard_parquet)
                        del batch
                        if smoke_test and src_targets_done >= total_stream_rows:
                            break
                        continue

                chunk_len = batch.num_rows
                chunk_t0 = time.time()
                pydict = batch.to_pydict()
                del batch

                eids = pydict["entity_id"]
                countries = pydict["country"]
                raw_names = pydict["raw_name"]
                norm_names = pydict["norm_name"]
                compact_names = pydict["compact_name"]
                translit_names = pydict["translit_name"]
                translit_comp_names = pydict["translit_comp_name"]
                learned_names = pydict["learned_name"]
                sorted_token_names = pydict["sorted_token_name"]
                name_phonetic_sigs = pydict["name_phonetic_sig"]
                raw_addrs = pydict["raw_addr"]
                norm_addrs = pydict["norm_addr"]
                translit_addrs = pydict["translit_addr"]
                numeric_signatures = pydict["numeric_signature"]
                house_numbers_strs = pydict["house_numbers_str"]
                postal_codes_strs = pydict["postal_codes_str"]
                name_tokens_strs = pydict["name_tokens_str"]
                translit_tokens_strs = pydict["translit_tokens_str"]
                addr_tokens_strs = pydict["addr_tokens_str"]

                def _build_diagnostic_slice(i_start: int, i_end: int) -> List[MultiViewRecord]:
                    recs = []
                    for i in range(i_start, i_end):
                        eid = eids[i]
                        int_id = id_mapper.get_or_add(eid)
                        norm_n = norm_names[i]
                        n_toks = name_tokens_strs[i].split() if name_tokens_strs[i] else []
                        t_toks = translit_tokens_strs[i].split() if translit_tokens_strs[i] else []
                        a_toks = addr_tokens_strs[i].split() if addr_tokens_strs[i] else []
                        hn_set = set(house_numbers_strs[i].split()) if house_numbers_strs[i] else set()
                        pc_set = set(postal_codes_strs[i].split()) if postal_codes_strs[i] else set()

                        recs.append(MultiViewRecord(
                            internal_id=int_id,
                            entity_id=eid,
                            country=countries[i],
                            raw_name=raw_names[i],
                            norm_name=norm_n,
                            compact_name=compact_names[i],
                            translit_name=translit_names[i],
                            translit_comp_name=translit_comp_names[i],
                            learned_name=learned_names[i],
                            sorted_token_name=sorted_token_names[i],
                            name_phonetic_sig=name_phonetic_sigs[i],
                            raw_addr=raw_addrs[i],
                            norm_addr=norm_addrs[i],
                            translit_addr=translit_addrs[i],
                            numeric_signature=numeric_signatures[i],
                            name_tokens=n_toks,
                            name_tok_set=set(n_toks),
                            translit_tokens=t_toks,
                            translit_tok_set=set(t_toks),
                            name_char3_set=char_ngrams_set(norm_n, 3) if norm_n else set(),
                            name_char4_set=char_ngrams_set(norm_n, 4) if norm_n else set(),
                            name_char5_set=char_ngrams_set(norm_n, 5) if norm_n else set(),
                            addr_tokens=a_toks,
                            addr_tok_set=set(a_toks),
                            house_numbers=hn_set,
                            postal_codes=pc_set,
                            is_s2=is_s2,
                            is_s3=is_s3,
                            is_name_missing=not bool(norm_n),
                        ))
                    return recs

                sub_size = max(1, math.ceil(chunk_len / num_threads))
                record_slices = []
                build_futs = [
                    executor.submit(_build_diagnostic_slice, i, min(chunk_len, i + sub_size))
                    for i in range(0, chunk_len, sub_size)
                ]
                for bf in build_futs:
                    record_slices.append(bf.result())

                del pydict
                gc.collect()

                all_tier1_matches = []
                all_tier2_targets = []
                all_tier2_cand_lists = []
                all_tier2_features = []
                target_retrieved_map: Dict[str, List[int]] = {}

                eval_futs = [
                    executor.submit(
                        _process_target_subbatch_diagnostic,
                        sb,
                        country_indexes,
                        s1_dict,
                        feat_extractor,
                        num_s1
                    )
                    for sb in record_slices
                ]
                for fut in as_completed(eval_futs):
                    res = fut.result()
                    all_tier1_matches.extend(res["tier1_matches"])
                    all_tier2_targets.extend(res["tier2_targets"])
                    all_tier2_cand_lists.extend(res["tier2_cand_lists"])
                    target_retrieved_map.update(res["target_all_retrieved"])
                    if res["tier2_features"].shape[0] > 0:
                        all_tier2_features.append(res["tier2_features"])

                del record_slices
                gc.collect()

                # Process Target Decisions
                # target_id -> (pred_s1_id, p_match, is_match, best_p, sec_p, ret_score, prov_mask)
                target_decisions: Dict[str, Tuple[str, float, int, float, float, float, int]] = {}

                # 1. Tier 1 Matches
                for s1_int, tid in all_tier1_matches:
                    if s1_int < num_s1:
                        target_decisions[tid] = (s1_ordered_ids[s1_int], 1.0, 1, 1.0, 0.0, 1.0, int(ProvenanceMask.EXACT_OR_LEARNED))

                # 2. Tier 2 Fuzzy Matches
                if all_tier2_features:
                    X_batch = np.vstack(all_tier2_features)
                    raw_probs = trainer.model.predict(X_batch, num_threads=num_threads)
                    cal_probs = calibrator.predict(raw_probs)

                    feat_offset = 0
                    for target, cands in zip(all_tier2_targets, all_tier2_cand_lists):
                        cand_len = len(cands)
                        target_probs = cal_probs[feat_offset : feat_offset + cand_len]
                        feat_offset += cand_len

                        best_idx = int(np.argmax(target_probs))
                        best_prob = float(target_probs[best_idx])
                        best_cand = cands[best_idx]

                        second_best_prob = 0.0
                        if len(target_probs) > 1:
                            target_probs_sorted = np.sort(target_probs)
                            second_best_prob = float(target_probs_sorted[-2])

                        s1_cand_rec = s1_dict[best_cand.s1_internal_id]
                        name_sim = fuzz.token_set_ratio(target.norm_name, s1_cand_rec.norm_name) / 100.0 if (target.norm_name and s1_cand_rec.norm_name) else 0.0
                        addr_sim = fuzz.token_set_ratio(target.norm_addr, s1_cand_rec.norm_addr) / 100.0 if (target.norm_addr and s1_cand_rec.norm_addr) else 0.0

                        threshold = config.s2_match_threshold if target.is_s2 else config.s3_match_threshold
                        passes_filters = (
                            (best_prob - second_best_prob >= config.margin_threshold or best_prob >= 0.85)
                            and (name_sim >= 0.40 or addr_sim >= 0.50)
                        )
                        is_match = 1 if (best_prob >= threshold and passes_filters) else 0
                        p_match = best_prob if passes_filters else 0.0
                        pred_sid = s1_ordered_ids[best_cand.s1_internal_id] if best_cand.s1_internal_id < num_s1 else ""

                        target_decisions[target.entity_id] = (
                            pred_sid,
                            p_match,
                            is_match,
                            best_prob,
                            second_best_prob,
                            best_cand.retrieval_score,
                            int(best_cand.provenance_mask)
                        )

                del all_tier2_features
                gc.collect()

                # Build Shard Arrays with Ground Truth Comparison STRICTLY AFTER PREDICTION
                shard_target_ids = []
                shard_sources = []
                shard_true_s1_ids = []
                shard_pred_s1_ids = []
                shard_p_matches = []
                shard_is_matches = []
                shard_ret_true_s1 = []
                shard_best_probs = []
                shard_sec_probs = []
                shard_ret_scores = []
                shard_prov_masks = []

                for eid in eids:
                    true_s1_str = gt_target_to_s1.get(eid, "")
                    true_s1_int = s1_id_to_int.get(true_s1_str, -1) if true_s1_str else -1
                    retrieved_ints = target_retrieved_map.get(eid, [])

                    if true_s1_int != -1:
                        ret_true = 1 if true_s1_int in retrieved_ints else 0
                    else:
                        ret_true = -1

                    dec = target_decisions.get(eid, ("", 0.0, 0, 0.0, 0.0, 0.0, 0))

                    shard_target_ids.append(eid)
                    shard_sources.append(prefix.upper())
                    shard_true_s1_ids.append(true_s1_str)
                    shard_pred_s1_ids.append(dec[0])
                    shard_p_matches.append(dec[1])
                    shard_is_matches.append(dec[2])
                    shard_ret_true_s1.append(ret_true)
                    shard_best_probs.append(dec[3])
                    shard_sec_probs.append(dec[4])
                    shard_ret_scores.append(dec[5])
                    shard_prov_masks.append(dec[6])

                del eids, target_decisions, target_retrieved_map
                gc.collect()

                # Write Shard
                write_eval_shard(
                    shard_parquet,
                    shard_meta,
                    shard_target_ids,
                    shard_sources,
                    shard_true_s1_ids,
                    shard_pred_s1_ids,
                    np.array(shard_p_matches, dtype=np.float32),
                    np.array(shard_is_matches, dtype=np.int8),
                    np.array(shard_ret_true_s1, dtype=np.int8),
                    np.array(shard_best_probs, dtype=np.float32),
                    np.array(shard_sec_probs, dtype=np.float32),
                    np.array(shard_ret_scores, dtype=np.float32),
                    np.array(shard_prov_masks, dtype=np.int32),
                    {
                        "source": prefix.upper(),
                        "shard_id": chunk_idx,
                    }
                )

                all_shard_files.append(shard_parquet)
                src_targets_done += chunk_len
                total_targets_evaluated += chunk_len

                chunk_time = time.time() - chunk_t0
                rate = chunk_len / max(chunk_time, 1e-4)
                logger.info(
                    f"[{name}] Shard {chunk_idx:03d} complete | Evaluated: {src_targets_done:,} / {total_stream_rows:,} | "
                    f"Speed: {rate:,.0f} tgts/s | RAM: {get_current_rss_mb():.1f} MB"
                )

                if smoke_test and src_targets_done >= total_stream_rows:
                    break

    logger.info(f"Target universe streaming complete. Consolidating results across {len(all_shard_files)} shards...")

    # =====================================================================
    # 5. Entity-Level Consolidation & Metric Computation
    # =====================================================================
    s1_pred_targets: Dict[str, Set[str]] = defaultdict(set)
    s1_pred_s2: Dict[str, Set[str]] = defaultdict(set)
    s1_pred_s3: Dict[str, Set[str]] = defaultdict(set)
    target_p_match_map: Dict[str, Tuple[str, float]] = {}  # target_id -> (pred_s1_id, p_match)

    total_gt_retrieved = 0
    gt_s2_retrieved = 0
    gt_s3_retrieved = 0

    total_predictions = 0
    correct_matches_cnt = 0
    false_positives_cnt = 0
    singleton_false_positives_cnt = 0

    error_records = []

    for sf in all_shard_files:
        tbl = pq.read_table(sf)
        t_ids = tbl["target_id"].to_pylist()
        srcs = tbl["source"].to_pylist()
        true_sids = tbl["true_s1_id"].to_pylist()
        pred_sids = tbl["predicted_s1_id"].to_pylist()
        p_matches = tbl["p_match"].to_numpy()
        is_matches = tbl["is_match"].to_numpy()
        ret_trues = tbl["retrieved_true_s1"].to_numpy()
        ret_scores = tbl["retrieval_score"].to_numpy()
        p_masks = tbl["provenance_mask"].to_numpy()

        for tid, src, true_sid, pred_sid, p_m, is_m, ret_t, r_score, p_mask in zip(
            t_ids, srcs, true_sids, pred_sids, p_matches, is_matches, ret_trues, ret_scores, p_masks
        ):
            if ret_t == 1:
                total_gt_retrieved += 1
                if src == "S2":
                    gt_s2_retrieved += 1
                elif src == "S3":
                    gt_s3_retrieved += 1

            if is_m == 1 and pred_sid:
                total_predictions += 1
                s1_pred_targets[pred_sid].add(tid)
                if src == "S2":
                    s1_pred_s2[pred_sid].add(tid)
                elif src == "S3":
                    s1_pred_s3[pred_sid].add(tid)

                if true_sid and pred_sid == true_sid:
                    correct_matches_cnt += 1
                else:
                    false_positives_cnt += 1

            if pred_sid and p_m > 0:
                target_p_match_map[tid] = (pred_sid, float(p_m))

            # Classify Pair-Level Error
            if true_sid:
                if is_m == 1 and pred_sid == true_sid:
                    classification = "CORRECT_MATCH"
                elif ret_t == 1:
                    classification = "MODEL_MISS"
                else:
                    classification = "RETRIEVAL_MISS"
            else:
                if is_m == 1:
                    classification = "SINGLETON_FALSE_POSITIVE"
                else:
                    classification = "CORRECT_SINGLETON_REJECTION"

            if classification != "CORRECT_SINGLETON_REJECTION" and len(error_records) < 500000:
                error_records.append({
                    "target_id": tid,
                    "source": src,
                    "true_s1_id": true_sid,
                    "predicted_s1_id": pred_sid if is_m == 1 else "",
                    "retrieved_true_s1": bool(ret_t == 1),
                    "error_classification": classification,
                    "model_probability": float(p_m),
                    "retrieval_score": float(r_score),
                    "provenance_mask": int(p_mask),
                })

        del tbl

    # Calculate Retrieval Recall
    candidate_recall = total_gt_retrieved / max(total_gt_pairs, 1)
    missed_gt_pairs = total_gt_pairs - total_gt_retrieved
    retrieval_misses_cnt = missed_gt_pairs
    model_misses_cnt = total_gt_retrieved - correct_matches_cnt

    s2_cand_recall = gt_s2_retrieved / max(gt_s2_pairs, 1)
    s3_cand_recall = gt_s3_retrieved / max(gt_s3_pairs, 1)

    # 6. Comprehensive Entity-Level Evaluation across ALL S1
    logger.info(f"Computing Entity-Level Macro F0.5 across {num_s1:,} S1 entities...")
    s1_f05_list = []
    s1_prec_list = []
    s1_rec_list = []
    s1_tp_total = 0
    s1_fp_total = 0
    s1_fn_total = 0

    s2_f05_list = []
    s2_prec_list = []
    s2_rec_list = []
    s2_tp_total = 0
    s2_fp_total = 0
    s2_fn_total = 0

    s3_f05_list = []
    s3_prec_list = []
    s3_rec_list = []
    s3_tp_total = 0
    s3_fp_total = 0
    s3_fn_total = 0

    singleton_correct_cnt = 0
    gt_match_counts = []
    pred_match_counts = []
    s1_metric_rows = []

    for sid in s1_ordered_ids:
        gt_set = s1_to_gt_targets.get(sid, set())
        pred_set = s1_pred_targets.get(sid, set())

        gt_cnt = len(gt_set)
        pred_cnt = len(pred_set)
        gt_match_counts.append(gt_cnt)
        pred_match_counts.append(pred_cnt)

        f05, prec, rec, tp, fp, fn, err_type = calculate_entity_f05(gt_set, pred_set)
        s1_f05_list.append(f05)
        s1_prec_list.append(prec)
        s1_rec_list.append(rec)
        s1_tp_total += tp
        s1_fp_total += fp
        s1_fn_total += fn

        if gt_cnt == 0:
            if pred_cnt == 0:
                singleton_correct_cnt += 1
            else:
                singleton_false_positives_cnt += 1

        # S2 Entity Metrics
        gt_s2 = s1_to_gt_s2.get(sid, set())
        pred_s2 = s1_pred_s2.get(sid, set())
        f05_2, prec_2, rec_2, tp_2, fp_2, fn_2, _ = calculate_entity_f05(gt_s2, pred_s2)
        s2_f05_list.append(f05_2)
        s2_prec_list.append(prec_2)
        s2_rec_list.append(rec_2)
        s2_tp_total += tp_2
        s2_fp_total += fp_2
        s2_fn_total += fn_2

        # S3 Entity Metrics
        gt_s3 = s1_to_gt_s3.get(sid, set())
        pred_s3 = s1_pred_s3.get(sid, set())
        f05_3, prec_3, rec_3, tp_3, fp_3, fn_3, _ = calculate_entity_f05(gt_s3, pred_s3)
        s3_f05_list.append(f05_3)
        s3_prec_list.append(prec_3)
        s3_rec_list.append(rec_3)
        s3_tp_total += tp_3
        s3_fp_total += fp_3
        s3_fn_total += fn_3

        if len(s1_metric_rows) < 500000:
            s1_metric_rows.append({
                "s1_id": sid,
                "gt_count": gt_cnt,
                "pred_count": pred_cnt,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": float(prec),
                "recall": float(rec),
                "f05": float(f05),
                "is_singleton": (gt_cnt == 0),
                "error_type": err_type,
            })

    macro_f05 = float(np.mean(s1_f05_list)) if s1_f05_list else 0.0
    macro_precision = float(np.mean(s1_prec_list)) if s1_prec_list else 0.0
    macro_recall = float(np.mean(s1_rec_list)) if s1_rec_list else 0.0
    singleton_accuracy = singleton_correct_cnt / max(gt_singleton_s1_cnt, 1)

    s2_macro_f05 = float(np.mean(s2_f05_list)) if s2_f05_list else 0.0
    s2_precision = float(np.mean(s2_prec_list)) if s2_prec_list else 0.0
    s2_recall = float(np.mean(s2_rec_list)) if s2_rec_list else 0.0

    s3_macro_f05 = float(np.mean(s3_f05_list)) if s3_f05_list else 0.0
    s3_precision = float(np.mean(s3_prec_list)) if s3_prec_list else 0.0
    s3_recall = float(np.mean(s3_rec_list)) if s3_rec_list else 0.0

    # Match Distributions
    def compute_distribution(counts: List[int]) -> Dict[str, int]:
        c = Counter(counts)
        dist = {}
        for k in range(5):
            dist[str(k)] = c.get(k, 0)
        dist["5+"] = sum(v for k, v in c.items() if k >= 5)
        return dist

    gt_distribution = compute_distribution(gt_match_counts)
    pred_distribution = compute_distribution(pred_match_counts)

    # 7. Multi-Threshold Analysis Grid
    logger.info("Executing Multi-Threshold Analysis Grid on Calibrated Probabilities...")
    threshold_grid = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    threshold_results = []

    # Map target -> assigned S1 and p_match for fast multi-threshold slicing
    for thresh in threshold_grid:
        t_s1_pred_map: Dict[str, Set[str]] = defaultdict(set)
        for tid, (psid, p_val) in target_p_match_map.items():
            if p_val >= thresh:
                t_s1_pred_map[psid].add(tid)

        t_f05_list = []
        t_prec_list = []
        t_rec_list = []
        t_tp = 0
        t_fp = 0
        t_fn = 0
        t_single_corr = 0
        t_total_preds = 0

        for sid in s1_ordered_ids:
            gt_set = s1_to_gt_targets.get(sid, set())
            pred_set = t_s1_pred_map.get(sid, set())
            t_total_preds += len(pred_set)

            f05, prec, rec, tp, fp, fn, _ = calculate_entity_f05(gt_set, pred_set)
            t_f05_list.append(f05)
            t_prec_list.append(prec)
            t_rec_list.append(rec)
            t_tp += tp
            t_fp += fp
            t_fn += fn

            if not gt_set and not pred_set:
                t_single_corr += 1

        t_m_f05 = float(np.mean(t_f05_list)) if t_f05_list else 0.0
        t_m_prec = float(np.mean(t_prec_list)) if t_prec_list else 0.0
        t_m_rec = float(np.mean(t_rec_list)) if t_rec_list else 0.0
        t_sing_acc = t_single_corr / max(gt_singleton_s1_cnt, 1)

        threshold_results.append({
            "threshold": thresh,
            "macro_f05": t_m_f05,
            "precision": t_m_prec,
            "recall": t_m_rec,
            "tp": t_tp,
            "fp": t_fp,
            "fn": t_fn,
            "singleton_accuracy": t_sing_acc,
            "total_predictions": t_total_preds,
        })

    # Save Output Files
    out_summary_json = eval_out_dir / "train_diagnostic_summary.json"
    out_s1_parquet = eval_out_dir / "train_diagnostic_s1_metrics.parquet"
    out_errors_parquet = eval_out_dir / "train_diagnostic_errors.parquet"
    out_thresh_csv = eval_out_dir / "train_diagnostic_thresholds.csv"

    summary_data = {
        "evaluation_type": "in_sample_training_diagnostic",
        "warning": EVAL_WARNING,
        "model_path": str(model_file),
        "model_hash": model_hash,
        "retrieval_config_hash": retrieval_hash,
        "feature_schema_hash": schema_hash,
        "data": {
            "s1_entities": num_s1,
            "s2_targets": 5_034_616,
            "s3_targets": 5_285_603,
            "total_targets": total_targets_evaluated,
            "gt_positive_pairs": total_gt_pairs,
            "gt_singleton_s1": gt_singleton_s1_cnt,
        },
        "retrieval": {
            "gt_pairs": total_gt_pairs,
            "retrieved_gt_pairs": total_gt_retrieved,
            "missed_gt_pairs": missed_gt_pairs,
            "candidate_recall": candidate_recall,
        },
        "entity_metrics": {
            "macro_f05": macro_f05,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "tp": s1_tp_total,
            "fp": s1_fp_total,
            "fn": s1_fn_total,
            "singleton_accuracy": singleton_accuracy,
        },
        "error_decomposition": {
            "retrieval_misses": retrieval_misses_cnt,
            "model_misses": model_misses_cnt,
            "false_positives": false_positives_cnt,
            "singleton_false_positives": singleton_false_positives_cnt,
        },
        "source_breakdown": {
            "s2": {
                "macro_f05": s2_macro_f05,
                "precision": s2_precision,
                "recall": s2_recall,
                "tp": s2_tp_total,
                "fp": s2_fp_total,
                "fn": s2_fn_total,
                "candidate_recall": s2_cand_recall,
            },
            "s3": {
                "macro_f05": s3_macro_f05,
                "precision": s3_precision,
                "recall": s3_recall,
                "tp": s3_tp_total,
                "fp": s3_fp_total,
                "fn": s3_fn_total,
                "candidate_recall": s3_cand_recall,
            },
        },
        "match_distribution": {
            "gt_matches_per_s1": gt_distribution,
            "predicted_matches_per_s1": pred_distribution,
        },
        "threshold_analysis": threshold_results,
    }

    with open(out_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    pd.DataFrame(threshold_results).to_csv(out_thresh_csv, index=False)
    if s1_metric_rows:
        pq.write_table(pa.Table.from_pandas(pd.DataFrame(s1_metric_rows)), out_s1_parquet, compression="zstd")
    if error_records:
        pq.write_table(pa.Table.from_pandas(pd.DataFrame(error_records)), out_errors_parquet, compression="zstd")

    total_time = time.time() - t0_start

    # Print Formatted Diagnostic Report
    print_diagnostic_report(summary_data, total_time)
    return summary_data


def print_diagnostic_report(summary: Dict[str, Any], elapsed_seconds: float):
    """Prints the formatted in-sample diagnostic report matching the required specification."""
    d = summary["data"]
    r = summary["retrieval"]
    e = summary["entity_metrics"]
    err = summary["error_decomposition"]
    s2 = summary["source_breakdown"]["s2"]
    s3 = summary["source_breakdown"]["s3"]
    gt_dist = summary["match_distribution"]["gt_matches_per_s1"]
    pr_dist = summary["match_distribution"]["predicted_matches_per_s1"]
    th_list = summary["threshold_analysis"]

    print("\n" + "=" * 60)
    print("ER-X FINAL MODEL — TRAINING SET DIAGNOSTIC")
    print("=" * 60)
    print("\nWARNING:")
    print("IN-SAMPLE DIAGNOSTIC")
    print("OPTIMISTIC")
    print("NOT A COMPETITION SCORE")
    print("MODEL WAS TRAINED USING THIS DATA\n")

    print("-" * 60)
    print("DATA")
    print("-" * 60)
    print(f"S1 entities:       {d['s1_entities']:,}")
    print(f"S2 targets:        {d['s2_targets']:,}")
    print(f"S3 targets:        {d['s3_targets']:,}")
    print(f"Total targets:     {d['total_targets']:,}")
    print(f"GT positive pairs: {d['gt_positive_pairs']:,}")
    print(f"GT singleton S1:   {d['gt_singleton_s1']:,}")

    print("\n" + "-" * 60)
    print("RETRIEVAL")
    print("-" * 60)
    print(f"GT pairs:          {r['gt_pairs']:,}")
    print(f"Retrieved GT pairs:{r['retrieved_gt_pairs']:,}")
    print(f"Missed GT pairs:   {r['missed_gt_pairs']:,}")
    print(f"Candidate Recall:  {r['candidate_recall']*100:.2f}%")

    print("\n" + "-" * 60)
    print("ENTITY METRICS")
    print("-" * 60)
    print(f"Macro F0.5:        {e['macro_f05']:.4f}")
    print(f"Macro Precision:   {e['macro_precision']*100:.2f}%")
    print(f"Macro Recall:      {e['macro_recall']*100:.2f}%")
    print(f"TP:                {e['tp']:,}")
    print(f"FP:                {e['fp']:,}")
    print(f"FN:                {e['fn']:,}")
    print(f"Singleton accuracy:{e['singleton_accuracy']*100:.2f}%")

    print("\n" + "-" * 60)
    print("ERROR DECOMPOSITION")
    print("-" * 60)
    print(f"Retrieval misses:          {err['retrieval_misses']:,}")
    print(f"Model misses:              {err['model_misses']:,}")
    print(f"False positives:           {err['false_positives']:,}")
    print(f"Singleton false positives: {err['singleton_false_positives']:,}")

    print("\n" + "-" * 60)
    print("SOURCE BREAKDOWN")
    print("-" * 60)
    print("S2:\n")
    print(f"Macro F0.5:        {s2['macro_f05']:.4f}")
    print(f"Precision:         {s2['precision']*100:.2f}%")
    print(f"Recall:            {s2['recall']*100:.2f}%")
    print(f"TP:                {s2['tp']:,}")
    print(f"FP:                {s2['fp']:,}")
    print(f"FN:                {s2['fn']:,}")
    print(f"Candidate Recall:  {s2['candidate_recall']*100:.2f}%\n")

    print("S3:\n")
    print(f"Macro F0.5:        {s3['macro_f05']:.4f}")
    print(f"Precision:         {s3['precision']*100:.2f}%")
    print(f"Recall:            {s3['recall']*100:.2f}%")
    print(f"TP:                {s3['tp']:,}")
    print(f"FP:                {s3['fp']:,}")
    print(f"FN:                {s3['fn']:,}")
    print(f"Candidate Recall:  {s3['candidate_recall']*100:.2f}%")

    print("\n" + "-" * 60)
    print("MATCH DISTRIBUTION")
    print("-" * 60)
    print("GT matches per S1:")
    for k, v in gt_dist.items():
        print(f"  {k:<3s}: {v:,}")
    print("\nPredicted matches per S1:")
    for k, v in pr_dist.items():
        print(f"  {k:<3s}: {v:,}")

    print("\n" + "-" * 60)
    print("THRESHOLD ANALYSIS")
    print("-" * 60)
    print(f"{'Thresh':<7} | {'Macro F0.5':<10} | {'Precision':<10} | {'Recall':<10} | {'TP':<10} | {'FP':<10} | {'FN':<10} | {'Singl Acc':<9}")
    print("-" * 88)
    for tr in th_list:
        print(
            f"{tr['threshold']:<7.2f} | {tr['macro_f05']:<10.4f} | {tr['precision']*100:<9.2f}% | {tr['recall']*100:<9.2f}% | "
            f"{tr['tp']:<10,d} | {tr['fp']:<10,d} | {tr['fn']:<10,d} | {tr['singleton_accuracy']*100:<8.2f}%"
        )
    print("-" * 88)
    print("IMPORTANT: This is diagnostic only. Do NOT automatically change the production threshold.")
    print(f"\nExecution completed in {elapsed_seconds/60:.2f} minutes.")
    print("=" * 60 + "\n")


# =====================================================================
# Main Entry Point
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="ER-X Model Score Evaluator (In-Sample Training Universe Diagnostic)")
    parser.add_argument("--mode", type=str, default="train-diagnostic", choices=["train-diagnostic", "val-check", "all"], help="Evaluation mode")
    parser.add_argument("--chunk-size", type=int, default=100000, help="Target chunk size for streaming (default: 100000)")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing cached shards")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild existing evaluation shards")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test (10k S1, 20k Targets)")
    parser.add_argument("--limit-s1", type=int, default=None, help="Optional limit for S1 records (e.g. 50000)")
    parser.add_argument("--threads", type=int, default=8, help="Number of CPU worker threads (default: 8)")
    parser.add_argument("--model-path", type=str, default=None, help="Path to LightGBM model file")
    parser.add_argument("--calibrator-path", type=str, default=None, help="Path to calibrator pkl file")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for diagnostic files")
    args = parser.parse_args()

    cfg = ERXConfig()

    # Step 1: Optional Holdout Inspection
    if args.mode in ("val-check", "all"):
        print("\n============================================================")
        print("INSPECTING OUT-OF-SAMPLE HOLDOUT ARTIFACTS")
        print("============================================================")
        has_holdout, holdout_data = inspect_existing_holdout(cfg)
        if has_holdout and holdout_data:
            print("\nOUT-OF-SAMPLE VALIDATION RESULT:")
            print(f"  Macro F0.5:         {holdout_data.get('macro_f05', 0.0):.4f}")
            print(f"  Macro Precision:    {holdout_data.get('macro_precision', 0.0)*100:.2f}%")
            print(f"  Macro Recall:       {holdout_data.get('macro_recall', 0.0)*100:.2f}%")
            print(f"  Singleton Accuracy: {holdout_data.get('singleton_accuracy', 0.0)*100:.2f}%")
            print(f"  Pair Cand Recall:   {holdout_data.get('pair_candidate_recall', 0.0)*100:.2f}%")
            print(f"  Validation S1:      {holdout_data.get('total_val_s1', 0):,}")
            print("============================================================\n")
        else:
            print("No reusable leakage-free validation artifact found.")
            print("============================================================\n")

        if args.mode == "val-check":
            return

    # Step 2: In-Sample Full Universe Diagnostic
    resume_flag = args.resume and not args.force_rebuild
    run_in_sample_training_diagnostic(
        config=cfg,
        chunk_size=args.chunk_size,
        resume=resume_flag,
        smoke_test=args.smoke,
        max_s1_records=args.limit_s1,
        num_threads=args.threads,
        model_path_override=args.model_path,
        calibrator_path_override=args.calibrator_path,
        output_dir_override=args.output_dir,
    )


if __name__ == "__main__":
    main()
