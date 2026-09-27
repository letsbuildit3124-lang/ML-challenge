"""
ER-X STAGE 3: PRODUCTION HIGH-THROUGHPUT TEST INFERENCE ENGINE (DUAL-PROCESS TURBO)
===================================================================================
Split-Process Architecture: S2 (Process A, 4 cores) & S3 (Process B, 4 cores)
Shared Read-Only Memory-Mapped S1 CSR Indexes + Batched RapidFuzz C-API + DuckDB Merge.

CLI Usage:
----------
1. Run Process A (S2 Only, 4 threads):
   python -u -m src.erx.final_inference --source S2 --threads 4

2. Run Process B (S3 Only, 4 threads):
   python -u -m src.erx.final_inference --source S3 --threads 4

3. Merge & Export Official Submission Files:
   python -u -m src.erx.final_inference --merge
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
from rapidfuzz import process, fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler

from src.resource_tracker import get_current_rss_mb, log_memory_status
from src.erx.config import ERXConfig
from src.erx.types import InternalIDMapper, MultiViewRecord, CompactS1Record, CandidatePair, ProvenanceMask, char_ngrams_set
from src.erx.normalization import ERXNormalizer, compact_name, normalize_text, offline_transliterate
from src.erx.cache_manager import (
    get_safe_duckdb_connection,
    ensure_cached_parquet,
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
logger = logging.getLogger("erx.final_inference")


# =====================================================================
# Shard Management & Checkpoint Resume
# =====================================================================

def compute_shard_checksum(metadata: Dict[str, Any]) -> str:
    """Computes deterministic MD5 checksum over shard configuration metadata."""
    serialized = json.dumps(metadata, sort_keys=True)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()


def is_shard_valid(shard_parquet: Path, shard_meta: Path) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Checks whether an inference shard exists and matches integrity metadata."""
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


def write_inference_shard(
    shard_parquet: Path,
    shard_meta: Path,
    s1_ids: np.ndarray,
    target_ids: List[str],
    is_match_flags: np.ndarray,
    probs: np.ndarray,
    metadata_info: Dict[str, Any],
):
    """Writes atomic, checksummed inference shard containing candidate and match pairs."""
    tmp_parquet = shard_parquet.with_suffix(".tmp.parquet")
    tmp_meta = shard_meta.with_suffix(".tmp.json")

    columns = {
        "s1_int_id": s1_ids,
        "target_entity_id": pa.array(target_ids, type=pa.string()),
        "is_match": is_match_flags,
        "calibrated_prob": probs,
    }
    table = pa.Table.from_pydict(columns)
    pq.write_table(table, tmp_parquet, compression="zstd")
    del table
    gc.collect()

    metadata_info["total_pairs"] = int(len(s1_ids))
    metadata_info["matches"] = int(np.sum(is_match_flags))
    metadata_info["file_size_bytes"] = tmp_parquet.stat().st_size
    metadata_info["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    metadata_info["checksum"] = compute_shard_checksum(metadata_info)

    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(metadata_info, f, indent=2)

    os.replace(tmp_parquet, shard_parquet)
    os.replace(tmp_meta, shard_meta)


# =====================================================================
# High-Throughput Subbatch Scoring Worker
# =====================================================================

def _process_target_subbatch_turbo(
    targets: List[MultiViewRecord],
    country_indexes: Dict[str, ERXRetrievalEngine],
    s1_dict: Dict[int, CompactS1Record],
    feat_extractor: ERXFeatureExtractor,
    num_s1: int,
) -> Dict[str, Any]:
    """
    Evaluates a subbatch of target records using CSR integer retrieval and batched RapidFuzz C-API:
    1. Tier 1 Fast-Path Check (Instant unique exact compact match).
    2. Zero-Candidate Screening (Eliminates non-overlapping targets before candidate retrieval).
    3. 6-Channel CSR Integer Candidate Retrieval.
    4. Batched 73-Feature Extraction (extract_features_batch).
    """
    tier1_matches: List[Tuple[int, str]] = []
    tier1_candidates: List[Tuple[int, str]] = []

    tier2_targets: List[MultiViewRecord] = []
    tier2_cand_lists: List[List[CandidatePair]] = []

    for target in targets:
        c_key = target.country if (country_indexes and target.country in country_indexes) else "OTHER"
        engine = country_indexes.get(c_key)
        if engine is None:
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
            continue

        # 3. 6-Channel Candidate Retrieval (Top 15)
        cands = engine.retrieve_for_target(target, top_k=15)
        if not cands:
            continue

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
        "target_count": len(targets),
    }


# =====================================================================
# S1 Shared Inverted Index Management (Disk-Backed Memory-Mapped CSR)
# =====================================================================

def get_or_build_shared_s1_indexes(
    config: ERXConfig,
    num_workers: int = 4,
    max_s1_records: Optional[int] = None,
) -> Tuple[List[CompactS1Record], Dict[str, ERXRetrievalEngine], Dict[int, CompactS1Record], ERXFeatureExtractor, List[str]]:
    """
    Builds or memory-maps the S1 CSR inverted indexes on disk.
    Allows multiple concurrent OS processes (S2 and S3) to share the exact same read-only memory.
    """
    id_mapper = InternalIDMapper()
    test_s1_tsv = config.data_dir / "test" / "test_source1.tsv"
    test_s1_parquet = config.cache_dir / "test_s1_normalized.parquet"
    csr_base_dir = config.artifacts_dir / "final_inference" / "s1_csr"
    csr_base_dir.mkdir(parents=True, exist_ok=True)

    ensure_cached_parquet(test_s1_tsv, test_s1_parquet, is_s2=False, is_s3=False, num_workers=num_workers)
    test_s1_mvs = load_compact_s1_records_from_parquet(test_s1_parquet, id_mapper, max_records=max_s1_records)
    num_test_s1 = len(test_s1_mvs)
    test_s1_ordered_ids = [m.entity_id for m in test_s1_mvs]
    s1_dict = {m.internal_id: m for m in test_s1_mvs}

    country_indexes: Dict[str, ERXRetrievalEngine] = {
        "US": ERXRetrievalEngine(config),
        "India": ERXRetrievalEngine(config),
        "France": ERXRetrievalEngine(config),
        "OTHER": ERXRetrievalEngine(config),
    }
    s1_by_country: Dict[str, List[CompactS1Record]] = defaultdict(list)
    for mv in test_s1_mvs:
        c_key = mv.country if mv.country in country_indexes else "OTHER"
        s1_by_country[c_key].append(mv)

    # Check if indexes are already persisted on disk
    all_loaded = True
    for c_key in country_indexes:
        c_csr_dir = csr_base_dir / c_key
        if not country_indexes[c_key].load_csr_indexes(c_csr_dir):
            all_loaded = False
            break

    if all_loaded:
        logger.info(f"Attached to shared memory-mapped S1 CSR indexes on disk ({csr_base_dir}).")
    else:
        logger.info(f"Building and persisting shared S1 CSR indexes for {num_test_s1:,} entities...")
        for c_key, mvs in s1_by_country.items():
            if mvs:
                country_indexes[c_key].index_s1(mvs)
                country_indexes[c_key].save_csr_indexes(csr_base_dir / c_key)
                logger.info(f"  -> Persisted '{c_key}' CSR index ({len(mvs):,} entities).")

    feat_extractor = ERXFeatureExtractor(token_idf=country_indexes["US"].token_idf)
    return test_s1_mvs, country_indexes, s1_dict, feat_extractor, test_s1_ordered_ids


# =====================================================================
# Single-Source Stream Evaluator (S2 or S3)
# =====================================================================

def run_single_source_inference(
    source_name: str,  # 'S2' or 'S3'
    num_threads: int = 4,
    chunk_size: int = 100000,
    force_rebuild: bool = False,
    smoke_test: bool = False,
    max_s1_records: Optional[int] = None,
):
    """
    Executes high-throughput inference for a single target source stream (S2 or S3) in its own OS process.
    """
    print("===================================================================")
    print(f"        ER-X FINAL INFERENCE: {source_name.upper()} PROCESS")
    print(f"   Source Stream: {source_name.upper()} | CPU Budget: {num_threads} Threads | Mode: Resumable Sharding")
    print("===================================================================")

    t0_start = time.time()
    config = ERXConfig()
    config.ensure_directories()

    final_artifact_dir = config.artifacts_dir / "final"
    model_file = final_artifact_dir / "models" / "lightgbm_final_model.txt"
    calibrator_file = final_artifact_dir / "models" / "isotonic_calibrator.pkl"

    if not model_file.exists():
        model_file = config.artifacts_dir / "dev" / "models" / "lightgbm_dev_model.txt"
        calibrator_file = config.artifacts_dir / "dev" / "models" / "isotonic_calibrator.pkl"

    assert model_file.exists(), f"FATAL: Model artifact not found at {model_file}."
    assert calibrator_file.exists(), f"FATAL: Calibrator artifact not found at {calibrator_file}."

    logger.info(f"[{source_name}] Loading LightGBM model from {model_file}...")
    trainer = ERXModelTrainer(config)
    trainer.load(model_file)

    logger.info(f"[{source_name}] Loading Isotonic Calibrator from {calibrator_file}...")
    with open(calibrator_file, "rb") as f:
        calibrator = pickle.load(f)

    # Ingest / Map S1 Indexes
    test_s1_mvs, country_indexes, s1_dict, feat_extractor, test_s1_ordered_ids = get_or_build_shared_s1_indexes(
        config, num_workers=num_threads, max_s1_records=max_s1_records
    )
    num_test_s1 = len(test_s1_mvs)

    is_s2 = (source_name.upper() == "S2")
    is_s3 = (source_name.upper() == "S3")
    tsv_file = config.data_dir / "test" / ("test_source2.tsv" if is_s2 else "test_source3.tsv")
    parquet_file = config.cache_dir / ("test_s2_normalized.parquet" if is_s2 else "test_s3_normalized.parquet")

    ensure_cached_parquet(tsv_file, parquet_file, is_s2=is_s2, is_s3=is_s3, num_workers=num_threads)

    # Shard directory isolation
    shards_dir = config.artifacts_dir / "final_inference" / "output" / source_name.lower()
    shards_dir.mkdir(parents=True, exist_ok=True)

    pq_file = pq.ParquetFile(parquet_file)
    total_target_count = min(20000, pq_file.metadata.num_rows) if smoke_test else pq_file.metadata.num_rows

    total_targets_processed = 0
    total_matches_selected = 0
    tier1_exact_matches = 0
    tier2_fuzzy_matches = 0

    id_mapper = InternalIDMapper()
    batch_idx = 0

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for batch in pq_file.iter_batches(batch_size=chunk_size):
            batch_idx += 1
            shard_parquet = shards_dir / f"shard_{source_name.lower()}_{batch_idx:04d}.parquet"
            shard_meta = shards_dir / f"shard_{source_name.lower()}_{batch_idx:04d}.json"

            if not force_rebuild:
                is_valid, meta_info = is_shard_valid(shard_parquet, shard_meta)
                if is_valid:
                    total_targets_processed += meta_info.get("targets_in_chunk", chunk_size)
                    total_matches_selected += meta_info.get("matches", 0)
                    logger.info(f"[{source_name}] Resumed from existing shard {shard_parquet.name} ({meta_info.get('total_pairs', 0):,} pairs).")
                    continue

            pydict = batch.to_pydict()
            del batch

            chunk_len = len(pydict["entity_id"])
            chunk_t0 = time.time()

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

            def _build_test_records_slice(i_start: int, i_end: int) -> List[MultiViewRecord]:
                recs = []
                for i in range(i_start, i_end):
                    eid = eids[i]
                    int_id = id_mapper.get_or_add(eid)
                    n_toks = name_tokens_strs[i].split() if name_tokens_strs[i] else []
                    t_toks = translit_tokens_strs[i].split() if translit_tokens_strs[i] else []
                    norm_n = norm_names[i]
                    char3 = char_ngrams_set(norm_n, 3) if norm_n else set()
                    char4 = char_ngrams_set(norm_n, 4) if norm_n else set()
                    char5 = char_ngrams_set(norm_n, 5) if norm_n else set()
                    a_toks = addr_tokens_strs[i].split() if addr_tokens_strs[i] else []
                    hn_str = house_numbers_strs[i]
                    hn_set = set(hn_str.split()) if hn_str else set()
                    pc_str = postal_codes_strs[i]
                    pc_set = set(pc_str.split()) if pc_str else set()

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
                        name_char3_set=char3,
                        name_char4_set=char4,
                        name_char5_set=char5,
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
                executor.submit(_build_test_records_slice, i, min(chunk_len, i + sub_size))
                for i in range(0, chunk_len, sub_size)
            ]
            for bf in build_futs:
                record_slices.append(bf.result())

            del pydict
            gc.collect()

            all_tier1_matches = []
            all_tier1_candidates = []
            all_tier2_targets = []
            all_tier2_cand_lists = []
            all_tier2_features = []

            eval_futs = [
                executor.submit(
                    _process_target_subbatch_turbo,
                    sb,
                    country_indexes,
                    s1_dict,
                    feat_extractor,
                    num_test_s1
                )
                for sb in record_slices
            ]
            for fut in as_completed(eval_futs):
                res = fut.result()
                all_tier1_matches.extend(res["tier1_matches"])
                all_tier1_candidates.extend(res["tier1_candidates"])
                all_tier2_targets.extend(res["tier2_targets"])
                all_tier2_cand_lists.extend(res["tier2_cand_lists"])
                if res["tier2_features"].shape[0] > 0:
                    all_tier2_features.append(res["tier2_features"])

            # Shard arrays for persistence
            shard_s1_ids = []
            shard_target_ids = []
            shard_is_matches = []
            shard_probs = []

            # 1. Tier 1 Matches
            for s1_int, tid in all_tier1_matches:
                if s1_int < num_test_s1:
                    tier1_exact_matches += 1
                    total_matches_selected += 1
                    shard_s1_ids.append(s1_int)
                    shard_target_ids.append(tid)
                    shard_is_matches.append(1)
                    shard_probs.append(1.0)

            for s1_int, tid in all_tier1_candidates:
                if s1_int < num_test_s1:
                    shard_s1_ids.append(s1_int)
                    shard_target_ids.append(tid)
                    shard_is_matches.append(0)
                    shard_probs.append(0.95)

            # 2. Tier 2 Fuzzy Candidates
            if all_tier2_features:
                X_batch = np.vstack(all_tier2_features)
                raw_probs = trainer.model.predict(X_batch, num_threads=num_threads)
                cal_probs = calibrator.predict(raw_probs)

                feat_offset = 0
                for target, cands in zip(all_tier2_targets, all_tier2_cand_lists):
                    cand_len = len(cands)
                    target_probs = cal_probs[feat_offset : feat_offset + cand_len]
                    feat_offset += cand_len

                    for c, p in zip(cands, target_probs):
                        if c.s1_internal_id < num_test_s1:
                            shard_s1_ids.append(c.s1_internal_id)
                            shard_target_ids.append(target.entity_id)
                            shard_is_matches.append(0)
                            shard_probs.append(float(p))

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
                    is_match = (
                        best_prob >= threshold
                        and (best_prob - second_best_prob >= config.margin_threshold or best_prob >= 0.85)
                        and (name_sim >= 0.40 or addr_sim >= 0.50)
                    )

                    if is_match and best_cand.s1_internal_id < num_test_s1:
                        total_matches_selected += 1
                        tier2_fuzzy_matches += 1
                        # Update match flag
                        shard_s1_ids.append(best_cand.s1_internal_id)
                        shard_target_ids.append(target.entity_id)
                        shard_is_matches.append(1)
                        shard_probs.append(best_prob)

            # Persist Shard
            write_inference_shard(
                shard_parquet,
                shard_meta,
                np.array(shard_s1_ids, dtype=np.uint32),
                shard_target_ids,
                np.array(shard_is_matches, dtype=np.int8),
                np.array(shard_probs, dtype=np.float32),
                {
                    "source": source_name,
                    "shard_id": batch_idx,
                    "targets_in_chunk": chunk_len,
                }
            )

            total_targets_processed += chunk_len
            chunk_time = time.time() - chunk_t0
            rate = chunk_len / max(chunk_time, 1e-4)
            overall_elapsed = time.time() - t0_start
            overall_rate = total_targets_processed / max(overall_elapsed, 1e-4)
            remaining = max(0, total_target_count - total_targets_processed)
            eta_mins = (remaining / max(overall_rate, 1e-4)) / 60.0
            pct_done = (total_targets_processed / total_target_count) * 100.0

            if batch_idx % 5 == 0 or total_targets_processed >= total_target_count:
                logger.info(
                    f"[{source_name}] Batch {batch_idx:3d} | Evaluated: {total_targets_processed:,} / {total_target_count:,} "
                    f"({pct_done:.1f}%) | Speed: {rate:,.0f} tgts/s (Avg: {overall_rate:,.0f}) | ETA: {eta_mins:.1f} mins | "
                    f"Matches: {total_matches_selected:,} | RAM: {get_current_rss_mb():.1f} MB"
                )

            if smoke_test and total_targets_processed >= total_target_count:
                break

    total_time = time.time() - t0_start
    print("===================================================================")
    print(f"  [SUCCESS] {source_name.upper()} INFERENCE COMPLETE IN {total_time/60:.2f} MINUTES")
    print(f"  Total Targets Evaluated: {total_targets_processed:,}")
    print(f"  Total Matches Selected:  {total_matches_selected:,}")
    print(f"  Shards Output Directory: {shards_dir}")
    print("===================================================================")


# =====================================================================
# Bulk DuckDB Output Merge & Official Deliverable Export
# =====================================================================

def run_merge_inference_outputs(config: ERXConfig):
    """
    Consolidates S2 and S3 inference shards into official output deliverables:
    - output/matching_results.tsv
    - output/candidate_pairs.tsv
    """
    print("===================================================================")
    print("        ER-X FINAL SUBMISSION EXPORT: DUCKDB MERGE ENGINE")
    print("   Aggregating S2 & S3 Shards into Official Competition Deliverables")
    print("===================================================================")

    t0_merge = time.time()
    config.ensure_directories()

    test_s1_parquet = config.cache_dir / "test_s1_normalized.parquet"
    id_mapper = InternalIDMapper()
    test_s1_mvs = load_compact_s1_records_from_parquet(test_s1_parquet, id_mapper)
    num_test_s1 = len(test_s1_mvs)
    test_s1_ordered_ids = [m.entity_id for m in test_s1_mvs]

    s2_shards_dir = config.artifacts_dir / "final_inference" / "output" / "s2"
    s3_shards_dir = config.artifacts_dir / "final_inference" / "output" / "s3"

    s2_files = sorted(s2_shards_dir.glob("*.parquet"))
    s3_files = sorted(s3_shards_dir.glob("*.parquet"))
    all_files = s2_files + s3_files

    assert len(all_files) > 0, f"FATAL: No inference shards found in {s2_shards_dir} or {s3_shards_dir}."
    logger.info(f"Merging {len(all_files)} inference shards ({len(s2_files)} S2, {len(s3_files)} S3)...")

    s1_matches: List[List[str]] = [[] for _ in range(num_test_s1)]
    s1_candidates: List[List[str]] = [[] for _ in range(num_test_s1)]

    total_pairs_scanned = 0
    total_matches_consolidated = 0

    for f_idx, sf in enumerate(all_files, 1):
        tbl = pq.read_table(sf)
        s1_col = tbl["s1_int_id"].to_numpy()
        t_col = tbl["target_entity_id"].to_pylist()
        m_col = tbl["is_match"].to_numpy()

        for s_int, t_id, is_m in zip(s1_col, t_col, m_col):
            if s_int < num_test_s1:
                total_pairs_scanned += 1
                if is_m == 1:
                    s1_matches[s_int].append(t_id)
                    total_matches_consolidated += 1
                if len(s1_candidates[s_int]) < 15:
                    s1_candidates[s_int].append(t_id)

        del tbl, s1_col, t_col, m_col
        if f_idx % 20 == 0 or f_idx == len(all_files):
            logger.info(f"  -> Merged Shards: {f_idx}/{len(all_files)} (Matches: {total_matches_consolidated:,})...")

    out_matching = config.output_dir / "matching_results.tsv"
    out_candidates = config.output_dir / "candidate_pairs.tsv"

    logger.info(f"Exporting official matching results to {out_matching}...")
    with open(out_matching, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid, matches in zip(test_s1_ordered_ids, s1_matches):
            unique_matches = list(dict.fromkeys(matches))
            match_str = ",".join(unique_matches) if unique_matches else ""
            f.write(f"{sid}\t{match_str}\n")

    logger.info(f"Exporting official candidate pairs to {out_candidates}...")
    with open(out_candidates, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid, cands in zip(test_s1_ordered_ids, s1_candidates):
            unique_cands = list(dict.fromkeys(cands))
            cand_str = ",".join(unique_cands) if unique_cands else ""
            f.write(f"{sid}\t{cand_str}\n")

    total_merge_time = time.time() - t0_merge
    matched_s1_cnt = sum(1 for m in s1_matches if m)
    singletons_cnt = num_test_s1 - matched_s1_cnt

    print("\n===================================================================")
    print(f"  [EXPORT COMPLETE] FINISHED IN {total_merge_time:.2f} SECONDS")
    print(f"  Total Test S1 Entities:     {num_test_s1:,} (100.0% Ordered Rows)")
    print(f"  Matched S1 Entities:        {matched_s1_cnt:,} ({matched_s1_cnt/num_test_s1*100:.2f}%)")
    print(f"  Singletons (Empty Matches): {singletons_cnt:,} ({singletons_cnt/num_test_s1*100:.2f}%)")
    print(f"  Total Matches Exported:     {total_matches_consolidated:,}")
    print(f"  Official Matching TSV:      {out_matching}")
    print(f"  Official Candidates TSV:    {out_candidates}")
    print("===================================================================")


# =====================================================================
# Main CLI Entry Point
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ER-X Stage 3 Final Test Inference Engine")
    parser.add_argument("--source", type=str, default=None, choices=["S2", "S3", "s2", "s3"], help="Run inference for specific target stream only (S2 or S3)")
    parser.add_argument("--merge", action="store_true", help="Merge existing S2 and S3 shards into final TSVs")
    parser.add_argument("--threads", type=int, default=None, help="Number of CPU worker threads (defaults to 4 per process)")
    parser.add_argument("--chunk-size", type=int, default=100000, help="Batch chunk size (default: 100,000)")
    parser.add_argument("--smoke", action="store_true", help="Run quick 20,000-target smoke check")
    parser.add_argument("--limit", type=int, default=None, help="Optional S1 entity limit for testing")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild existing shards")
    args = parser.parse_args()

    cfg = ERXConfig()

    if args.merge:
        run_merge_inference_outputs(cfg)
    elif args.source is not None:
        th = args.threads or 4
        run_single_source_inference(
            source_name=args.source.upper(),
            num_threads=th,
            chunk_size=args.chunk_size,
            force_rebuild=args.force_rebuild,
            smoke_test=args.smoke,
            max_s1_records=args.limit,
        )
    else:
        # Default: Run S2 then S3 sequentially or merge if both exist
        th = args.threads or 8
        logger.info("Running standalone sequential inference for S2 then S3...")
        run_single_source_inference("S2", num_threads=th, chunk_size=args.chunk_size, force_rebuild=args.force_rebuild, smoke_test=args.smoke, max_s1_records=args.limit)
        run_single_source_inference("S3", num_threads=th, chunk_size=args.chunk_size, force_rebuild=args.force_rebuild, smoke_test=args.smoke, max_s1_records=args.limit)
        run_merge_inference_outputs(cfg)
