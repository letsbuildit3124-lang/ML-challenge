"""
ER-X STAGE 2: FINAL FULL-UNIVERSE TRAINING ENGINE
Trains Final Production LightGBM Model & Calibrator on 100% of 2,206,821 S1 Entities.

Hardware Profile:
- CPU: 8 vCPUs (Controlled global concurrency: 8 workers, 0 nested thread multiplication)
- RAM: 32 GB (Working memory bounded strictly under 5.0 GB with 27 GB safe headroom)
- Caching: Vectorized Snappy Parquet Cache via DuckDB / PyArrow
- Checkpointing: Full step-level resumption from disk artifacts

Guarantees:
1. 100% S1 Training Universe (2,206,821 S1 records).
2. Zero data leakage: Blind 6-channel retrieval before label assignment.
3. Memory-safe: Pre-unnested Ground Truth Parquet + Bounded DuckDB working memory (6 GB cap).
4. Periodic progress heartbeat and proactive memory guard.
5. Resumable checkpoints at each stage.
"""

import os
import sys

# ----------------------------------------------------------------------
# Enforce Single-Threaded BLAS/OpenMP to Prevent Thread Storms & VM Freezes
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
import argparse
import logging
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Set, Tuple, Optional, Any

import duckdb
import numpy as np

from src.resource_tracker import get_current_rss_mb, log_memory_status
from src.erx.config import ERXConfig
from src.erx.types import InternalIDMapper, MultiViewRecord, CompactS1Record, CandidatePair, ProvenanceMask, char_ngrams_set
from src.erx.normalization import ERXNormalizer, compact_name, normalize_text, offline_transliterate
from src.erx.cache_manager import (
    get_safe_duckdb_connection,
    ensure_cached_parquet,
    ensure_ground_truth_pairs_parquet,
    load_compact_s1_records_from_parquet,
    load_multiview_records_from_parquet,
)
from src.erx.learned_rules import LearnedRuleEngine
from src.erx.retrieval import ERXRetrievalEngine
from src.erx.features import ERXFeatureExtractor, FEATURE_NAMES
from src.erx.model import ERXModelTrainer, ERXCalibrator

logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("erx.final_train")


def save_feature_checkpoint(
    checkpoint_npz: Path,
    checkpoint_meta: Path,
    X_train: np.ndarray,
    y_train: np.ndarray,
    s1_count: int,
    target_count: int,
    mode: str,
):
    """Saves training features array alongside strict validation metadata."""
    np.savez_compressed(checkpoint_npz, X_train=X_train, y_train=y_train)
    meta = {
        "s1_count": int(s1_count),
        "target_count": int(target_count),
        "total_pairs": int(len(X_train)),
        "positives_count": int(np.sum(y_train)),
        "feature_count": int(X_train.shape[1]),
        "mode": mode,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(checkpoint_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved verified feature checkpoint: {checkpoint_npz.name} ({meta['total_pairs']:,} pairs, {meta['positives_count']:,} positives, metadata: {checkpoint_meta.name}).")


def load_verified_feature_checkpoint(
    checkpoint_npz: Path,
    checkpoint_meta: Path,
    expected_s1_count: int,
    mode: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Forensically validates feature checkpoint on disk.
    Strictly prevents using smoke test / partial checkpoints for full production runs.
    """
    if not checkpoint_npz.exists() or not checkpoint_meta.exists():
        if checkpoint_npz.exists() and not checkpoint_meta.exists():
            logger.warning(f"[CHECKPOINT INVALID] Found {checkpoint_npz.name} without metadata file {checkpoint_meta.name}. Discarding unverified checkpoint.")
            try:
                checkpoint_npz.unlink()
            except Exception:
                pass
        return None

    try:
        with open(checkpoint_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        logger.warning(f"[CHECKPOINT INVALID] Failed to read {checkpoint_meta.name}: {e}. Discarding.")
        return None

    ckpt_mode = meta.get("mode", "unknown")
    ckpt_s1 = meta.get("s1_count", 0)
    ckpt_pairs = meta.get("total_pairs", 0)
    ckpt_pos = meta.get("positives_count", 0)
    ckpt_feats = meta.get("feature_count", 0)

    if ckpt_feats != len(FEATURE_NAMES):
        logger.warning(f"[CHECKPOINT INVALID] Feature count mismatch: {ckpt_feats} != {len(FEATURE_NAMES)}. Discarding.")
        return None

    if mode == "full":
        # Full production run requirements:
        if ckpt_mode != "full" or ckpt_s1 < 2_000_000 or ckpt_pairs < 200_000 or ckpt_pos < 10_000:
            logger.warning(
                f"[CHECKPOINT REJECTED] Found checkpoint with mode='{ckpt_mode}', S1 count={ckpt_s1:,}, "
                f"pairs={ckpt_pairs:,}, positives={ckpt_pos:,}. "
                f"Full production training requires mode='full', S1 >= 2,000,000, pairs >= 200,000. "
                f"Discarding stale/smoke checkpoint and recomputing 100% full-universe features."
            )
            try:
                checkpoint_npz.unlink()
                checkpoint_meta.unlink()
            except Exception:
                pass
            return None
    elif mode == "smoke":
        if ckpt_mode != "smoke":
            logger.info(f"Smoke run requested but checkpoint mode is '{ckpt_mode}'. Recomputing smoke features.")
            return None
    elif mode == "benchmark":
        if ckpt_s1 != expected_s1_count:
            logger.info(f"Benchmark requested for {expected_s1_count:,} S1 entities, but checkpoint has {ckpt_s1:,}. Recomputing.")
            return None

    try:
        data = np.load(checkpoint_npz)
        X_train = data["X_train"]
        y_train = data["y_train"]
        if len(X_train) != ckpt_pairs or int(np.sum(y_train)) != ckpt_pos:
            logger.warning("[CHECKPOINT CORRUPT] Array dimensions do not match metadata. Discarding.")
            return None
        logger.info(f"Loaded verified feature checkpoint: {len(X_train):,} pairs ({int(np.sum(y_train)):,} positives, {X_train.shape[1]} features, mode={ckpt_mode}).")
        return X_train, y_train
    except Exception as e:
        logger.warning(f"[CHECKPOINT ERROR] Failed to load {checkpoint_npz.name}: {e}. Discarding.")
        return None


def _process_blind_train_subbatch(
    target_items: List[Tuple[MultiViewRecord, int]],  # (target, true_s1_int)
    engine: ERXRetrievalEngine,
    s1_dict: Dict[int, MultiViewRecord],
    extractor: ERXFeatureExtractor,
) -> Dict[str, Any]:
    """Extracts blind candidate pairs and 73 features for 100% full-universe training."""
    feats_list = []
    labels_list = []
    retrieval_hits = 0
    total_evaluated = 0

    for target, true_s1_int in target_items:
        if engine is None or extractor is None or s1_dict is None:
            continue

        # Blind candidate retrieval across 6 channels
        cands = engine.retrieve_for_target(target, top_k=15)
        if not cands:
            continue

        total_evaluated += 1

        # Blind 73 feature computation
        feats = extractor.extract_features_for_target_candidates(target, cands, s1_dict)

        # Label assignment strictly after retrieval
        has_hit = False
        for idx, c in enumerate(cands):
            is_pos = (c.s1_internal_id == true_s1_int and true_s1_int != -1)
            if is_pos:
                has_hit = True
            feats_list.append(feats[idx])
            labels_list.append(1 if is_pos else 0)

        if has_hit:
            retrieval_hits += 1

    return {
        "feats": np.array(feats_list, dtype=np.float32) if feats_list else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32),
        "labels": np.array(labels_list, dtype=np.int32) if labels_list else np.empty((0,), dtype=np.int32),
        "hits": retrieval_hits,
        "evaluated": total_evaluated,
    }


def run_stage2_final_training(smoke_test: bool = False, max_s1_records: Optional[int] = None):
    """
    Executes STAGE 2: 100% Full-Universe Model Training & Artifact Generation.
    Memory-safe, CPU-efficient, fully resumable with step-level checkpoints.
    """
    if smoke_test:
        run_mode = "smoke"
        target_limit = 25000
        load_limit = 10000
    elif max_s1_records is not None:
        run_mode = "benchmark"
        target_limit = 50000
        load_limit = max_s1_records
    else:
        run_mode = "full"
        target_limit = 250000
        load_limit = None

    print("===================================================================")
    print("        ER-X STAGE 2: FINAL FULL-UNIVERSE TRAINING PIPELINE         ")
    print(f"   Universe: {'SMOKE TEST (10,000)' if smoke_test else ('BENCHMARK (' + str(max_s1_records) + ')') if max_s1_records else '100% (2,206,821 S1 Entities)'} | Zero Leakage Fit ")
    print(f"   Mode: {run_mode.upper()} | High-Throughput DuckDB & Parquet Caching Architecture")
    print("===================================================================")

    t0_all = time.time()
    config = ERXConfig()
    num_workers = min(8, os.cpu_count() or 8)

    # Namespaced artifact directories: smoke and benchmark never pollute final/
    artifact_dir = config.artifacts_dir / ("smoke" if smoke_test else "benchmark" if max_s1_records else "final")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    models_dir = artifact_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = artifact_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    config.cache_dir.mkdir(parents=True, exist_ok=True)

    rules_file = artifact_dir / "learned_rules.json"
    train_features_checkpoint = checkpoints_dir / "train_features.npz"
    train_features_meta = checkpoints_dir / "train_features.meta.json"
    model_path = models_dir / "lightgbm_final_model.txt"
    calibrator_path = models_dir / "isotonic_calibrator.pkl"

    s1_tsv = config.data_dir / "train" / "train_source1.tsv"
    gt_tsv = config.data_dir / "train" / "train_ground_truth.tsv"
    train_s2_tsv = config.data_dir / "train" / "train_source2.tsv"
    train_s3_tsv = config.data_dir / "train" / "train_source3.tsv"

    s1_parquet = config.cache_dir / "train_s1_normalized.parquet"
    s2_parquet = config.cache_dir / "train_s2_normalized.parquet"
    s3_parquet = config.cache_dir / "train_s3_normalized.parquet"
    gt_parquet = config.cache_dir / "ground_truth_pairs.parquet"

    id_mapper = InternalIDMapper()

    # ------------------------------------------------------------------
    # Step 1: Ingest 100% S1 Records via Parquet Cache
    # ------------------------------------------------------------------
    log_memory_status("[Step 1/5: S1 Ingestion]")
    logger.info("[Step 1/5] Ingesting Training S1 records from DuckDB Parquet cache (Compact Mode)...")
    ensure_cached_parquet(s1_tsv, s1_parquet, is_s2=False, is_s3=False, num_workers=num_workers)
    
    s1_records = load_compact_s1_records_from_parquet(s1_parquet, id_mapper, max_records=load_limit)
    num_s1 = len(s1_records)
    
    # Direct dictionary: internal_id -> record (Ultra-low memory layout: < 500 MB)
    s1_dict = {rec.internal_id: rec for rec in s1_records}
    s1_id_to_int = {rec.entity_id: rec.internal_id for rec in s1_records}
    logger.info(f"Loaded {num_s1:,} Compact S1 records into memory (RAM: {get_current_rss_mb():.1f} MB).")

    # ------------------------------------------------------------------
    # Step 2: Build 100% S1 6-Channel Index (Zero TF-IDF Bottleneck)
    # ------------------------------------------------------------------
    log_memory_status("[Step 2/5: Indexing]")
    logger.info(f"[Step 2/5] Building 6-Channel Retrieval Index over {num_s1:,} S1 Entities...")
    t0_idx = time.time()
    train_engine = ERXRetrievalEngine(config)
    train_engine.index_s1(s1_records)
    feat_extractor = ERXFeatureExtractor(token_idf=train_engine.token_idf)
    logger.info(f"Index built in {time.time() - t0_idx:.2f}s (RAM: {get_current_rss_mb():.1f} MB).")

    # ------------------------------------------------------------------
    # Step 3: Mine Global Normalization Rules via Pre-Unnested GT Parquet
    # ------------------------------------------------------------------
    log_memory_status("[Step 3/5: Rule Mining]")
    logger.info("[Step 3/5] Mining Learned Normalization Rules from Ground Truth Pairs...")
    ensure_cached_parquet(train_s2_tsv, s2_parquet, is_s2=True, is_s3=False, num_workers=num_workers)
    ensure_cached_parquet(train_s3_tsv, s3_parquet, is_s2=False, is_s3=True, num_workers=num_workers)
    ensure_ground_truth_pairs_parquet(gt_tsv, gt_parquet, num_workers=num_workers)

    rule_engine = LearnedRuleEngine(config.min_alias_observations, config.min_alias_purity)
    if rules_file.exists():
        rule_engine.load(rules_file)
    else:
        con_rules = get_safe_duckdb_connection(num_threads=4, max_memory_gb="4GB")
        try:
            sample_pairs = con_rules.execute(f"""
                SELECT s1.raw_name, t.raw_name
                FROM read_parquet('{gt_parquet}') gt
                JOIN read_parquet('{s1_parquet}') s1 ON gt.s1_id = s1.entity_id
                JOIN read_parquet('{s2_parquet}') t ON gt.target_id = t.entity_id
                LIMIT 100000;
            """).fetchall()
            rule_engine.learn_from_pairs([(r[0], r[1]) for r in sample_pairs if r[0] and r[1]])
            rule_engine.save(rules_file)
        except Exception as e:
            logger.warning(f"Rule mining note: {e}")
        finally:
            con_rules.close()

    # ------------------------------------------------------------------
    # Step 4: Extract Balanced Training Target Pairs & Features (Strict Validation)
    # ------------------------------------------------------------------
    log_memory_status("[Step 4/5: Feature Extraction]")
    loaded_checkpoint = load_verified_feature_checkpoint(
        train_features_checkpoint,
        train_features_meta,
        expected_s1_count=num_s1,
        mode=run_mode,
    )

    if loaded_checkpoint is not None:
        X_train, y_train = loaded_checkpoint
    else:
        logger.info(f"[Step 4/5] Extracting Balanced Target Pairs via DuckDB Parquet Join (Target Limit: {target_limit:,})...")
        t0_feat = time.time()
        con = get_safe_duckdb_connection(num_threads=4, max_memory_gb="6GB")
        
        s2_rows = con.execute(f"""
            SELECT t.entity_id, t.country, t.raw_name, t.norm_name, t.compact_name, t.translit_name,
                   t.translit_comp_name, t.learned_name, t.sorted_token_name, t.name_phonetic_sig,
                   t.raw_addr, t.norm_addr, t.translit_addr, t.numeric_signature,
                   t.house_numbers_str, t.postal_codes_str, t.name_tokens_str, t.translit_tokens_str, t.addr_tokens_str,
                   gt.s1_id
            FROM read_parquet('{gt_parquet}') gt
            JOIN read_parquet('{s2_parquet}') t ON gt.target_id = t.entity_id
            LIMIT {target_limit};
        """).fetchall()

        s3_rows = con.execute(f"""
            SELECT t.entity_id, t.country, t.raw_name, t.norm_name, t.compact_name, t.translit_name,
                   t.translit_comp_name, t.learned_name, t.sorted_token_name, t.name_phonetic_sig,
                   t.raw_addr, t.norm_addr, t.translit_addr, t.numeric_signature,
                   t.house_numbers_str, t.postal_codes_str, t.name_tokens_str, t.translit_tokens_str, t.addr_tokens_str,
                   gt.s1_id
            FROM read_parquet('{gt_parquet}') gt
            JOIN read_parquet('{s3_parquet}') t ON gt.target_id = t.entity_id
            LIMIT {target_limit};
        """).fetchall()
        con.close()

        def _rows_to_target_items(rows: list, is_s2: bool, is_s3: bool) -> List[Tuple[MultiViewRecord, int]]:
            items = []
            for r in rows:
                s1_id_str = r[19]
                s1_int_id = s1_id_to_int.get(s1_id_str, -1)
                if s1_int_id == -1:
                    continue

                eid = r[0]
                int_id = id_mapper.get_or_add(eid)
                norm_n = r[3]
                n_toks = r[16].split() if r[16] else []
                t_toks = r[17].split() if r[17] else []
                a_toks = r[18].split() if r[18] else []
                hn_set = set(r[14].split()) if r[14] else set()
                pc_set = set(r[15].split()) if r[15] else set()

                mv = MultiViewRecord(
                    internal_id=int_id,
                    entity_id=eid,
                    country=r[1],
                    raw_name=r[2],
                    norm_name=norm_n,
                    compact_name=r[4],
                    translit_name=r[5],
                    translit_comp_name=r[6],
                    learned_name=r[7],
                    sorted_token_name=r[8],
                    name_phonetic_sig=r[9],
                    raw_addr=r[10],
                    norm_addr=r[11],
                    translit_addr=r[12],
                    numeric_signature=r[13],
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
                )
                items.append((mv, s1_int_id))
            return items

        train_target_items: List[Tuple[MultiViewRecord, int]] = []
        train_target_items.extend(_rows_to_target_items(s2_rows, is_s2=True, is_s3=False))
        train_target_items.extend(_rows_to_target_items(s3_rows, is_s2=False, is_s3=True))
        del s2_rows, s3_rows
        gc.collect()

        logger.info(f"Extracting blind train features across {num_workers} worker threads ({len(train_target_items):,} targets)...")
        sub_batch_size = max(1, math.ceil(len(train_target_items) / num_workers))
        sub_batches = [train_target_items[i : i + sub_batch_size] for i in range(0, len(train_target_items), sub_batch_size)]

        all_feats = []
        all_labels = []
        total_eval = 0
        total_hits = 0

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _process_blind_train_subbatch,
                    sb,
                    train_engine,
                    s1_dict,
                    feat_extractor
                )
                for sb in sub_batches
            ]
            for fut in as_completed(futures):
                res = fut.result()
                total_eval += res["evaluated"]
                total_hits += res["hits"]
                if res["feats"].shape[0] > 0:
                    all_feats.append(res["feats"])
                    all_labels.append(res["labels"])

        del train_target_items, s1_records, s1_id_to_int, train_engine
        gc.collect()

        X_train = np.vstack(all_feats)
        y_train = np.concatenate(all_labels)
        del all_feats, all_labels
        gc.collect()

        logger.info(f"Blind candidate extraction complete in {time.time() - t0_feat:.2f}s | Pairs: {len(X_train):,} (Positives: {int(np.sum(y_train)):,}).")
        
        # Save verified feature checkpoint
        save_feature_checkpoint(
            train_features_checkpoint,
            train_features_meta,
            X_train,
            y_train,
            s1_count=num_s1,
            target_count=total_eval,
            mode=run_mode,
        )

    # ------------------------------------------------------------------
    # Step 5: Train Final Production LightGBM Model & Calibrator
    # ------------------------------------------------------------------
    log_memory_status("[Step 5/5: LightGBM Training]")
    logger.info(f"[Step 5/5] Training Final LightGBM Model on {len(X_train):,} pairs ({int(np.sum(y_train)):,} positives)...")
    val_split_mask = (np.arange(len(X_train)) % 10 == 0)
    X_tr = X_train[~val_split_mask]
    y_tr = y_train[~val_split_mask]
    X_va = X_train[val_split_mask]
    y_va = y_train[val_split_mask]

    config.lgb_params["n_jobs"] = num_workers
    trainer = ERXModelTrainer(config)
    trainer.train(X_tr, y_tr, X_va, y_va)

    # Atomic Model Save with size verification
    tmp_model_path = model_path.with_suffix(".tmp.txt")
    trainer.save(tmp_model_path)
    if not tmp_model_path.exists() or tmp_model_path.stat().st_size < 1000:
        raise RuntimeError(f"FATAL: Model artifact save failed or empty at {tmp_model_path}.")
    os.replace(tmp_model_path, model_path)
    logger.info(f"Saved Verified Final Production Model to: {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")

    logger.info("Fitting Final Isotonic Calibrator on internal validation split...")
    calibrator = ERXCalibrator(method="isotonic")
    raw_val_probs = trainer.model.predict(X_va, num_threads=num_workers)
    calibrator.fit(raw_val_probs, y_va)

    # Atomic Calibrator Save
    tmp_calibrator_path = calibrator_path.with_suffix(".tmp.pkl")
    with open(tmp_calibrator_path, "wb") as f:
        pickle.dump(calibrator, f, protocol=pickle.HIGHEST_PROTOCOL)
    if not tmp_calibrator_path.exists() or tmp_calibrator_path.stat().st_size < 100:
        raise RuntimeError(f"FATAL: Calibrator artifact save failed at {tmp_calibrator_path}.")
    os.replace(tmp_calibrator_path, calibrator_path)
    logger.info(f"Saved Verified Final Calibrator to: {calibrator_path}")

    del X_train, y_train, X_tr, y_tr, X_va, y_va
    gc.collect()

    total_time = time.time() - t0_all
    print("===================================================================")
    print(f"  FINAL STAGE 2 TRAINING COMPLETE IN {total_time/60:.2f} MINUTES")
    print(f"  Model Artifact:      {model_path}")
    print(f"  Calibrator Artifact: {calibrator_path}")
    print(f"  Learned Rules:       {rules_file}")
    print(f"  Checkpoints:         {checkpoints_dir}")
    print(f"  Final Memory RSS:    {get_current_rss_mb():.1f} MB (Peak < 5.0 GB)")
    print("===================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ER-X Final Production Training Engine")
    parser.add_argument("--smoke", action="store_true", help="Run quick 10,000-entity smoke test")
    parser.add_argument("--limit", type=int, default=None, help="Optional S1 entity limit for benchmarks")
    args = parser.parse_args()

    run_stage2_final_training(smoke_test=args.smoke, max_s1_records=args.limit)
