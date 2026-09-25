"""
Dataset builder module with sequential memory management and vectorized record extraction.
Splits S1 entities into Train / Validation, performs candidate blocking against S2 and S3 sequentially,
computes pairwise feature vectors, and formats datasets for model training.
"""

import os
import gc
import time
import random
from typing import Dict, List, Tuple, Any
import numpy as np
import polars as pl
from src.config import Config
from src.data_loader import load_source_file
from src.blocking import add_blocking_columns, generate_candidates_for_targets, evaluate_candidate_recall
from src.features import compute_pairwise_features, get_char_ngrams

def extract_record_dict_from_df(p_df: pl.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Fast conversion of already normalized DataFrame columns into record dictionaries
    with precomputed set representations for 5x accelerated feature extraction.
    """
    records = {}
    for r in p_df.select(["eid", "norm_name", "compact_name", "norm_addr", "country"]).iter_rows():
        norm_n = r[1] or ""
        norm_a = r[3] or ""
        n_toks = norm_n.split() if norm_n else []
        a_toks = norm_a.split() if norm_a else []
        num_toks = [w for w in a_toks if w.isdigit()]
        
        n_tok_set = set(n_toks)
        a_tok_set = set(a_toks)
        num_tok_set = set(num_toks)
        name_3g_set = get_char_ngrams(norm_n, 3)

        records[r[0]] = {
            "norm_name": norm_n,
            "compact_name": r[2] or "",
            "name_tokens": n_toks,
            "name_tok_set": n_tok_set,
            "name_3g_set": name_3g_set,
            "norm_addr": norm_a,
            "addr_tokens": a_toks,
            "addr_tok_set": a_tok_set,
            "numeric_tokens": num_toks,
            "numeric_tok_set": num_tok_set,
            "country": r[4] or ""
        }
    return records

def build_train_val_datasets(
    config: Config,
    train_s1_df: pl.DataFrame,
    gt_df: pl.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, List[Tuple[str, List[float]]]], Dict[str, List[str]]]:
    """
    Constructs train and validation feature matrices at the S1 entity level sequentially.
    """
    print("=" * 70, flush=True)
    print("PHASE: TRAIN/VAL DATASET CONSTRUCTION", flush=True)
    print("=" * 70, flush=True)

    # 1. Parse Ground Truth
    gt_mapping: Dict[str, List[str]] = {}
    for row in gt_df.iter_rows():
        s1_id = str(row[0])
        m_str = str(row[1]) if row[1] is not None else ""
        matches = [x.strip() for x in m_str.split(",") if x.strip()]
        gt_mapping[s1_id] = matches

    # 2. Get unique S1 IDs and perform deterministic split
    all_s1_ids = list(train_s1_df["entity_id"].to_list())
    random.seed(config.seed)
    random.shuffle(all_s1_ids)

    total_requested = config.train_sample_s1_count + config.val_sample_s1_count
    selected_s1_ids = all_s1_ids[:total_requested]

    split_idx = config.train_sample_s1_count
    train_s1_ids = set(selected_s1_ids[:split_idx])
    val_s1_ids = set(selected_s1_ids[split_idx:total_requested])

    print(f"Selected {len(train_s1_ids):,} Train S1 entities and {len(val_s1_ids):,} Validation S1 entities (Seed: {config.seed})", flush=True)

    # Filter S1 DataFrames
    train_s1_sub_df = train_s1_df.filter(pl.col("entity_id").is_in(list(train_s1_ids)))
    val_s1_sub_df = train_s1_df.filter(pl.col("entity_id").is_in(list(val_s1_ids)))

    s1_all_sub_df = pl.concat([train_s1_sub_df, val_s1_sub_df])
    s1_p = add_blocking_columns(s1_all_sub_df)
    train_s1_p = s1_p.filter(pl.col("eid").is_in(list(train_s1_ids)))
    val_s1_p = s1_p.filter(pl.col("eid").is_in(list(val_s1_ids)))

    s1_records = extract_record_dict_from_df(s1_p)
    target_records: Dict[str, Dict[str, Any]] = {}
    train_cands_list = []
    val_cands_list = []

    # 3. Process Train S2 sequentially
    print("\nLoading and Blocking against Train Source 2...", flush=True)
    t0 = time.time()
    s2_df = load_source_file(config.train_s2_path, expected_prefix="S2-")
    s2_p = add_blocking_columns(s2_df)
    del s2_df
    gc.collect()

    tr_pairs_s2 = generate_candidates_for_targets(train_s1_p, s2_p, max_cands_per_s1=config.max_total_cands_per_s1 // 2 + 5)
    va_pairs_s2 = generate_candidates_for_targets(val_s1_p, s2_p, max_cands_per_s1=config.max_total_cands_per_s1 // 2 + 5)
    train_cands_list.append(tr_pairs_s2)
    val_cands_list.append(va_pairs_s2)

    needed_s2 = set(tr_pairs_s2["target_id"].to_list()) | set(va_pairs_s2["target_id"].to_list())
    print(f"S2 Blocking finished in {time.time() - t0:.2f}s. Extracting {len(needed_s2):,} active S2 records...", flush=True)
    
    s2_matched = s2_p.filter(pl.col("eid").is_in(list(needed_s2)))
    target_records.update(extract_record_dict_from_df(s2_matched))

    del s2_p, s2_matched, tr_pairs_s2, va_pairs_s2
    gc.collect()

    # 4. Process Train S3 sequentially
    print("\nLoading and Blocking against Train Source 3...", flush=True)
    t0 = time.time()
    s3_df = load_source_file(config.train_s3_path, expected_prefix="S3-")
    s3_p = add_blocking_columns(s3_df)
    del s3_df
    gc.collect()

    tr_pairs_s3 = generate_candidates_for_targets(train_s1_p, s3_p, max_cands_per_s1=config.max_total_cands_per_s1 // 2 + 5)
    va_pairs_s3 = generate_candidates_for_targets(val_s1_p, s3_p, max_cands_per_s1=config.max_total_cands_per_s1 // 2 + 5)
    train_cands_list.append(tr_pairs_s3)
    val_cands_list.append(va_pairs_s3)

    needed_s3 = set(tr_pairs_s3["target_id"].to_list()) | set(va_pairs_s3["target_id"].to_list())
    print(f"S3 Blocking finished in {time.time() - t0:.2f}s. Extracting {len(needed_s3):,} active S3 records...", flush=True)

    s3_matched = s3_p.filter(pl.col("eid").is_in(list(needed_s3)))
    target_records.update(extract_record_dict_from_df(s3_matched))

    del s3_p, s3_matched, tr_pairs_s3, va_pairs_s3
    gc.collect()

    # 5. Combine candidate pairs
    train_cand_pairs_df = pl.concat(train_cands_list).unique()
    val_cand_pairs_df = pl.concat(val_cands_list).unique()

    train_gt_subset = {s1_id: gt_mapping.get(s1_id, []) for s1_id in train_s1_ids}
    val_gt_subset = {s1_id: gt_mapping.get(s1_id, []) for s1_id in val_s1_ids}

    print("\n--- Training Set Candidate Recall ---", flush=True)
    evaluate_candidate_recall(train_cand_pairs_df, train_gt_subset)
    print("\n--- Validation Set Candidate Recall ---", flush=True)
    evaluate_candidate_recall(val_cand_pairs_df, val_gt_subset)

    # 6. Extract Pairwise Features for Train Set
    print("\nExtracting Pairwise Features for Training Pairs...", flush=True)
    t0 = time.time()
    X_train_list = []
    y_train_list = []

    for row in train_cand_pairs_df.iter_rows():
        s1_id, cand_id = str(row[0]), str(row[1])
        if s1_id not in s1_records or cand_id not in target_records:
            continue
        
        feats = compute_pairwise_features(s1_records[s1_id], target_records[cand_id], cand_id)
        label = 1.0 if cand_id in set(gt_mapping.get(s1_id, [])) else 0.0
        
        X_train_list.append(feats)
        y_train_list.append(label)

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.float32)
    print(f"Extracted {len(X_train):,} training pairs (Positives: {int(y_train.sum()):,}, Negatives: {len(y_train) - int(y_train.sum()):,}) in {time.time() - t0:.2f}s", flush=True)

    # 7. Extract Pairwise Features for Validation Set
    print("\nExtracting Pairwise Features for Validation Pairs...", flush=True)
    t0 = time.time()
    X_val_list = []
    y_val_list = []
    val_s1_cand_pairs: Dict[str, List[Tuple[str, List[float]]]] = {s1_id: [] for s1_id in val_s1_ids}

    for row in val_cand_pairs_df.iter_rows():
        s1_id, cand_id = str(row[0]), str(row[1])
        if s1_id not in s1_records or cand_id not in target_records:
            continue
        
        feats = compute_pairwise_features(s1_records[s1_id], target_records[cand_id], cand_id)
        label = 1.0 if cand_id in set(gt_mapping.get(s1_id, [])) else 0.0
        
        X_val_list.append(feats)
        y_val_list.append(label)
        val_s1_cand_pairs[s1_id].append((cand_id, feats))

    X_val = np.array(X_val_list, dtype=np.float32)
    y_val = np.array(y_val_list, dtype=np.float32)
    print(f"Extracted {len(X_val):,} validation pairs (Positives: {int(y_val.sum()):,}, Negatives: {len(y_val) - int(y_val.sum()):,}) in {time.time() - t0:.2f}s", flush=True)

    return X_train, y_train, X_val, y_val, val_s1_cand_pairs, val_gt_subset
