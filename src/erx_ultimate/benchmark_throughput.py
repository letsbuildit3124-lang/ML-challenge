"""
ER-X Ultimate: Real Production-Path 100K Target Throughput Benchmark
Executes the exact production pipeline on real data and outputs forensic performance metrics.
"""

from __future__ import annotations
import argparse
import gc
import os
import sys
import time
import psutil
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional, Any
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Enforce 1 BLAS thread per process
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from src.erx_ultimate.config import CONFIG, UltimateConfig
from src.erx_ultimate.types import EntityRecord, CandidateMatch, SourceType
from src.erx_ultimate.cache_manager import CacheManager
from src.erx_ultimate.retrieval import ERXRetrievalEngine
from src.erx_ultimate.features import extract_batch_features, NUM_FEATURES
from src.erx_ultimate.train import load_ground_truth_map


def get_current_rss_gb() -> float:
    """Get current process RSS memory in GB."""
    try:
        proc = psutil.Process()
        return proc.memory_info().rss / (1024 ** 3)
    except Exception:
        return 0.0


def run_benchmark(num_targets: int = 100000, chunk_size: int = 50000) -> Dict[str, Any]:
    """Execute end-to-end production benchmark on real dataset."""
    config = CONFIG
    cache_mgr = CacheManager(config)

    s1_parquet = cache_mgr.normalized_dir / "train_s1.parquet"
    s2_parquet = cache_mgr.normalized_dir / "train_s2.parquet"
    db_path = cache_mgr.db_path

    if not s1_parquet.exists() or not s2_parquet.exists() or not db_path.exists():
        print("Required dataset partitions missing in cache. Running cache_manager ingestion first...")
        from src.erx_ultimate.cache_manager import prepare_cache
        prepare_cache(config)

    print(f"=== ER-X ULTIMATE: 100K REAL PRODUCTION BENCHMARK ===")
    print(f"Target count: {num_targets:,} | Chunk size: {chunk_size:,}")

    # Load S1 CSR Retrieval Engine
    t_init_start = time.time()
    retriever = ERXRetrievalEngine(config)
    if not (cache_mgr.indices_dir / "s1_id_map.npy").exists():
        retriever.build_s1_indexes(s1_parquet)
    retriever.load_indexes(mmap_mode="r")

    # Load S1 record fast lookup map
    s1_table = pq.read_table(s1_parquet)
    s1_ids = s1_table["id"].to_numpy()
    s1_names_norm = s1_table["name_norm"].to_pylist()
    s1_names_raw = s1_table["name_raw"].to_pylist()
    s1_addrs_norm = s1_table["address_norm"].to_pylist()
    s1_addrs_raw = s1_table["address_raw"].to_pylist()
    s1_cities_norm = s1_table["city_norm"].to_pylist()
    s1_states_norm = s1_table["state_norm"].to_pylist()
    s1_zips_norm = s1_table["postal_code_norm"].to_pylist()
    s1_countries_norm = s1_table["country_norm"].to_pylist()
    s1_phones_norm = s1_table["phone_norm"].to_pylist()
    s1_webs_norm = s1_table["website_norm"].to_pylist()

    s1_records_map: Dict[int, EntityRecord] = {}
    for idx in range(len(s1_ids)):
        rec_id = int(s1_ids[idx])
        s1_records_map[rec_id] = EntityRecord(
            id=rec_id,
            source=SourceType.SOURCE1,
            name_raw=s1_names_raw[idx],
            name_norm=s1_names_norm[idx],
            address_raw=s1_addrs_raw[idx],
            address_norm=s1_addrs_norm[idx],
            city_norm=s1_cities_norm[idx],
            state_norm=s1_states_norm[idx],
            postal_code_norm=s1_zips_norm[idx],
            country_norm=s1_countries_norm[idx],
            phone_norm=s1_phones_norm[idx],
            website_norm=s1_webs_norm[idx],
        )

    # Load GT mappings
    gt_map = load_ground_truth_map(db_path)
    init_time = time.time() - t_init_start
    print(f"Initialization complete in {init_time:.2f}s | Baseline RSS: {get_current_rss_gb():.2f} GB")

    # Benchmarking Stage Timers
    t_retrieval = 0.0
    t_gt = 0.0
    t_neg = 0.0
    t_features = 0.0
    t_write = 0.0

    total_processed = 0
    total_pairs = 0
    total_pos = 0
    total_neg = 0

    benchmark_shards_dir = Path(config.paths.artifacts_dir) / "benchmark_shards"
    benchmark_shards_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = pq.ParquetFile(s2_parquet)
    t_benchmark_start = time.time()

    for batch_num, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size)):
        if total_processed >= num_targets:
            break

        pydict = batch.to_pydict()
        ids = pydict["id"]
        names_norm = pydict["name_norm"]
        names_raw = pydict["name_raw"]
        addrs_norm = pydict["address_norm"]
        addrs_raw = pydict["address_raw"]
        cities_norm = pydict["city_norm"]
        states_norm = pydict["state_norm"]
        zips_norm = pydict["postal_code_norm"]
        countries_norm = pydict["country_norm"]
        phones_norm = pydict["phone_norm"]
        webs_norm = pydict["website_norm"]
        
        take_rows = min(len(ids), num_targets - total_processed)

        batch_tgt_recs: List[EntityRecord] = []
        batch_s1_recs: List[EntityRecord] = []
        batch_cands: List[CandidateMatch] = []
        batch_labels: List[int] = []

        # 1. Retrieval & Candidate Generation
        t0 = time.time()
        for i in range(take_rows):
            tgt_id = int(ids[i])
            rec = EntityRecord(
                id=tgt_id,
                source=SourceType.SOURCE2,
                name_raw=names_raw[i],
                name_norm=names_norm[i],
                address_raw=addrs_raw[i],
                address_norm=addrs_norm[i],
                city_norm=cities_norm[i],
                state_norm=states_norm[i],
                postal_code_norm=zips_norm[i],
                country_norm=countries_norm[i],
                phone_norm=phones_norm[i],
                website_norm=webs_norm[i],
            )

            # Blind Retrieval
            cands = retriever.retrieve_candidates_for_record(rec, top_k=30)
            t_retrieval += (time.time() - t0)
            
            # GT match & Negative Sampling
            t0 = time.time()
            gt_s1_set = gt_map.get((tgt_id, int(SourceType.SOURCE2)), set())
            t_gt += (time.time() - t0)

            t0 = time.time()
            pos_cands = [c for c in cands if c.s1_id in gt_s1_set]
            neg_cands = [c for c in cands if c.s1_id not in gt_s1_set]

            selected_cands: List[Tuple[CandidateMatch, int]] = [(c, 1) for c in pos_cands]
            max_negs = max(len(pos_cands) * 4, 1 if not pos_cands else 0)
            for neg_c in neg_cands[:max_negs]:
                selected_cands.append((neg_c, 0))

            for cand, label in selected_cands:
                s1_rec = s1_records_map.get(cand.s1_id)
                if s1_rec is not None:
                    batch_tgt_recs.append(rec)
                    batch_s1_recs.append(s1_rec)
                    batch_cands.append(cand)
                    batch_labels.append(label)
            t_neg += (time.time() - t0)
            t0 = time.time()

        # 2. 73-Feature Extraction
        if batch_cands:
            t0 = time.time()
            features_matrix = extract_batch_features(batch_tgt_recs, batch_s1_recs, batch_cands)
            t_features += (time.time() - t0)

            # 3. Real Parquet Shard Serialization & Disk Write
            t0 = time.time()
            schema = [pa.field(f"f_{j}", pa.float32()) for j in range(NUM_FEATURES)]
            schema.append(pa.field("label", pa.uint8()))
            schema.append(pa.field("target_id", pa.int64()))
            schema.append(pa.field("s1_id", pa.int64()))
            arrow_schema = pa.schema(schema)

            cols = [pa.array(features_matrix[:, j]) for j in range(NUM_FEATURES)]
            cols.append(pa.array(batch_labels, type=pa.uint8()))
            cols.append(pa.array([c.target_id for c in batch_cands], type=pa.int64()))
            cols.append(pa.array([c.s1_id for c in batch_cands], type=pa.int64()))

            table_out = pa.Table.from_arrays(cols, schema=arrow_schema)
            shard_path = benchmark_shards_dir / f"benchmark_shard_{batch_num:03d}.parquet"
            pq.write_table(table_out, str(shard_path), compression="SNAPPY")
            t_write += (time.time() - t0)

            total_pairs += len(batch_cands)
            pos_count = sum(batch_labels)
            total_pos += pos_count
            total_neg += (len(batch_labels) - pos_count)

        total_processed += take_rows
        print(f"Processed {total_processed:,} / {num_targets:,} targets | Shard written: {shard_path.name}")

    total_elapsed = time.time() - t_benchmark_start
    sustained_throughput = total_processed / total_elapsed if total_elapsed > 0 else 0
    pairs_throughput = total_pairs / total_elapsed if total_elapsed > 0 else 0
    peak_rss = get_current_rss_gb()

    results = {
        "targets_processed": total_processed,
        "total_pairs": total_pairs,
        "total_positives": total_pos,
        "total_negatives": total_neg,
        "total_elapsed_sec": total_elapsed,
        "sustained_targets_per_sec": sustained_throughput,
        "pairs_per_sec": pairs_throughput,
        "retrieval_sec": t_retrieval,
        "gt_lookup_sec": t_gt,
        "negative_sampling_sec": t_neg,
        "features_sec": t_features,
        "parquet_write_sec": t_write,
        "peak_rss_gb": peak_rss,
    }

    print("\n=== BENCHMARK RESULTS ===")
    print(f"Targets Processed: {total_processed:,}")
    print(f"Total Candidate Pairs: {total_pairs:,} (Pos: {total_pos:,}, Neg: {total_neg:,})")
    print(f"Total Elapsed Time: {total_elapsed:.2f}s")
    print(f"SUSTAINED THROUGHPUT: {sustained_throughput:.1f} targets/sec")
    print(f"PAIRS THROUGHPUT: {pairs_throughput:.1f} pairs/sec")
    print(f"Stage Breakdown:")
    print(f"  - Retrieval: {t_retrieval:.2f}s ({(t_retrieval/total_elapsed)*100:.1f}%)")
    print(f"  - Features:  {t_features:.2f}s ({(t_features/total_elapsed)*100:.1f}%)")
    print(f"  - GT Lookup: {t_gt:.2f}s ({(t_gt/total_elapsed)*100:.1f}%)")
    print(f"  - Neg Sample:{t_neg:.2f}s ({(t_neg/total_elapsed)*100:.1f}%)")
    print(f"  - Disk Write:{t_write:.2f}s ({(t_write/total_elapsed)*100:.1f}%)")
    print(f"Peak RSS: {peak_rss:.2f} GB")
    print(f"Estimated Full 10.32M Runtime: {(10320219 / sustained_throughput) / 60:.1f} minutes")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ER-X Ultimate 100K Target Throughput Benchmark")
    parser.add_argument("--targets", type=int, default=100000, help="Number of real targets to benchmark")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Chunk size for batch processing")
    args = parser.parse_args()

    run_benchmark(num_targets=args.targets, chunk_size=args.chunk_size)
