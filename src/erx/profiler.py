"""
ER-X High-Throughput Stage Profiler & Performance Micro-Benchmarking Engine.
Measures isolated timings for:
1. Parquet Scanning & Column Projection
2. S1 Compact Ingestion & Inverted Index Lookup
3. Ground Truth O(1) Lookup
4. 6-Channel Candidate Retrieval
5. Hard Negative Selection & Diversification
6. 73-Feature Extraction (RapidFuzz, Set Intersections, Provenance, Ratios)
7. Parquet Shard I/O Serialization
"""

import os
import gc
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pyarrow.parquet as pq

from src.resource_tracker import get_current_rss_mb, log_memory_status
from src.erx.config import ERXConfig
from src.erx.types import InternalIDMapper, MultiViewRecord, CompactS1Record, CandidatePair, ProvenanceMask, char_ngrams_set
from src.erx.cache_manager import (
    get_safe_duckdb_connection,
    ensure_cached_parquet,
    ensure_ground_truth_pairs_parquet,
    load_compact_s1_records_from_parquet,
)
from src.erx.normalization import ERXNormalizer
from src.erx.retrieval import ERXRetrievalEngine
from src.erx.features import ERXFeatureExtractor, FEATURE_NAMES
from src.erx.final_train import write_training_shard_parquet, _process_blind_chunk_slice


def run_10k_profiling_benchmark(target_count: int = 10_000):
    """Profiles the 10,000 target hot path stage by stage with granular timer instrumentation."""
    print("===================================================================")
    print(f"        ER-X HOT-PATH STAGE PROFILER ({target_count:,} TARGETS)       ")
    print("===================================================================")

    config = ERXConfig()
    id_mapper = InternalIDMapper()
    num_workers = min(8, os.cpu_count() or 8)

    s1_parquet = config.cache_dir / "train_s1_normalized.parquet"
    s2_parquet = config.cache_dir / "train_s2_normalized.parquet"
    gt_parquet = config.cache_dir / "ground_truth_pairs.parquet"

    # Stage 1: Load S1 Index
    t0 = time.time()
    s1_records = load_compact_s1_records_from_parquet(s1_parquet, id_mapper, max_records=250_000)
    s1_dict = {rec.internal_id: rec for rec in s1_records}
    s1_id_to_int = {rec.entity_id: rec.internal_id for rec in s1_records}
    s1_load_time = time.time() - t0
    print(f"[Stage 1] Loaded {len(s1_records):,} Compact S1 records in {s1_load_time:.2f}s (RAM: {get_current_rss_mb():.1f} MB)")

    # Stage 2: Index S1
    t0 = time.time()
    engine = ERXRetrievalEngine(config)
    engine.index_s1(s1_records)
    feat_extractor = ERXFeatureExtractor(token_idf=engine.token_idf)
    s1_index_time = time.time() - t0
    print(f"[Stage 2] Built 6-Channel Index in {s1_index_time:.2f}s (RAM: {get_current_rss_mb():.1f} MB)")

    gt_tsv = config.data_dir / "train" / "train_ground_truth.tsv"
    ensure_ground_truth_pairs_parquet(gt_tsv, gt_parquet)

    # Stage 3: Ingest Ground Truth
    t0 = time.time()
    con_gt = get_safe_duckdb_connection(num_threads=4, max_memory_gb="4GB")
    gt_parquet_str = str(gt_parquet).replace("\\", "/")
    gt_rows = con_gt.execute(f"SELECT target_id, s1_id FROM read_parquet('{gt_parquet_str}') LIMIT 500000").fetchall()
    con_gt.close()
    target_to_s1_int: Dict[str, int] = {t_id: s1_id_to_int.get(s1_id, -1) for t_id, s1_id in gt_rows}
    gt_time = time.time() - t0
    print(f"[Stage 3] Ingested {len(target_to_s1_int):,} GT links in {gt_time:.2f}s (RAM: {get_current_rss_mb():.1f} MB)")

    # Stage 4: Read & Materialize 10,000 Target Records
    normalizer = ERXNormalizer()
    s2_tsv = config.data_dir / "train" / "train_source2.tsv"
    s2_tsv_str = str(s2_tsv).replace("\\", "/")

    t0 = time.time()
    con_tgt = get_safe_duckdb_connection(num_threads=4, max_memory_gb="4GB")
    raw_rows = con_tgt.execute(f"SELECT entity_id, business_name, business_address, country FROM read_csv_auto('{s2_tsv_str}', sep='\\t', header=True) LIMIT {target_count}").fetchall()
    con_tgt.close()
    scan_time = time.time() - t0

    t0 = time.time()
    target_records = []
    for r in raw_rows:
        eid, b_name, b_addr, country = r[0], r[1] or "", r[2] or "", r[3] or ""
        rec = normalizer.normalize_record(id_mapper.get_or_add(eid), eid, b_name, b_addr, country, is_s2=True, is_s3=False)
        target_records.append(rec)
    build_time = time.time() - t0

    # Stage 5: Micro-Benchmark Stage by Stage
    # 5A: Pure Candidate Retrieval
    t0 = time.time()
    all_cands = []
    for tgt in target_records:
        cands = engine.retrieve_for_target(tgt, top_k=15)
        all_cands.append(cands)
    retrieval_time = time.time() - t0
    total_cands = sum(len(c) for c in all_cands)

    # 5B: Labeling & Hard Negative Selection
    t0 = time.time()
    selected_cands_list = []
    for tgt, cands in zip(target_records, all_cands):
        true_s1_int = target_to_s1_int.get(tgt.entity_id, -1)

        pos_cands = [c for c in cands if c.s1_internal_id == true_s1_int and true_s1_int != -1]
        neg_cands = [c for c in cands if c.s1_internal_id != true_s1_int or true_s1_int == -1]

        scored_negs = sorted([(c.retrieval_score + 0.15 * c.provenance_mask.bit_count(), c) for c in neg_cands], key=lambda x: x[0], reverse=True)
        top_hard = [c for _, c in scored_negs[:3]]
        selected = pos_cands + top_hard
        selected_cands_list.append(selected)
    selection_time = time.time() - t0
    total_selected_pairs = sum(len(s) for s in selected_cands_list)

    # 5C: 73-Feature Extraction
    t0 = time.time()
    feats_list = []
    for tgt, cands in zip(target_records, selected_cands_list):
        if cands:
            feats = feat_extractor.extract_features_for_target_candidates(tgt, cands, s1_dict)
            feats_list.append(feats)
    feature_time = time.time() - t0

    # 5D: Shard Parquet Writing
    t0 = time.time()
    all_feats_mat = np.vstack(feats_list) if feats_list else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    chunk_data_dict = {
        "s1_ids": np.zeros((len(all_feats_mat),), dtype=np.int32),
        "target_ids": np.zeros((len(all_feats_mat),), dtype=np.int32),
        "labels": np.zeros((len(all_feats_mat),), dtype=np.int32),
        "prov_masks": np.zeros((len(all_feats_mat),), dtype=np.int32),
        "features": all_feats_mat,
    }
    tmp_shard_p = Path("cache/erx/prof_shard.parquet")
    tmp_shard_m = Path("cache/erx/prof_shard.meta.json")
    write_training_shard_parquet(tmp_shard_p, tmp_shard_m, chunk_data_dict, {"chunk_id": 1, "source": "s2", "rows_processed": target_count})
    write_time = time.time() - t0
    if tmp_shard_p.exists():
        tmp_shard_p.unlink()
    if tmp_shard_m.exists():
        tmp_shard_m.unlink()

    total_hot_path_time = scan_time + build_time + retrieval_time + selection_time + feature_time + write_time
    total_throughput = target_count / max(total_hot_path_time, 1e-4)

    print("\n===================================================================")
    print("                10K PROFILING BOTTLENECK BREAKDOWN                 ")
    print("===================================================================")
    print(f"  Parquet Batch Read:          {scan_time:6.3f}s ({scan_time/total_hot_path_time*100:5.1f}%)")
    print(f"  Record Materialization:      {build_time:6.3f}s ({build_time/total_hot_path_time*100:5.1f}%)")
    print(f"  6-Channel Retrieval:         {retrieval_time:6.3f}s ({retrieval_time/total_hot_path_time*100:5.1f}%) | {target_count/max(retrieval_time,1e-4):,.0f} tgts/s")
    print(f"  Hard Negative Selection:     {selection_time:6.3f}s ({selection_time/total_hot_path_time*100:5.1f}%)")
    print(f"  73-Feature Extraction:       {feature_time:6.3f}s ({feature_time/total_hot_path_time*100:5.1f}%) | {total_selected_pairs/max(feature_time,1e-4):,.0f} pairs/s")
    print(f"  Parquet Shard Disk I/O:      {write_time:6.3f}s ({write_time/total_hot_path_time*100:5.1f}%)")
    print("-------------------------------------------------------------------")
    print(f"  TOTAL 10K HOT-PATH TIME:     {total_hot_path_time:6.3f}s")
    print(f"  SINGLE-THREAD THROUGHPUT:    {total_throughput:,.0f} targets/sec")
    print(f"  PROJECTED 8-WORKER SPEED:    {total_throughput * min(num_workers, 6.5):,.0f} targets/sec")
    print(f"  CURRENT PROCESS RSS:         {get_current_rss_mb():.1f} MB")
    print("===================================================================")


if __name__ == "__main__":
    run_10k_profiling_benchmark(10_000)
