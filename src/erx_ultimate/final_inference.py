"""
ER-X Ultimate: High-Throughput Production Inference Engine (9.97M Targets)
Generates official competition deliverables: matching_results.tsv and candidate_pairs.tsv.
"""

from __future__ import annotations
import argparse
import gc
import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
import pyarrow.parquet as pq
import numpy as np

# Enforce single thread before importing BLAS / LightGBM
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from src.erx_ultimate.config import CONFIG, UltimateConfig
from src.erx_ultimate.resource_monitor import ResourceMonitor, get_memory_summary
from src.erx_ultimate.types import EntityRecord, CandidateMatch, ScoredPair, SourceType, EntityCluster
from src.erx_ultimate.cache_manager import CacheManager
from src.erx_ultimate.retrieval import ERXRetrievalEngine
from src.erx_ultimate.features import extract_batch_features, NUM_FEATURES
from src.erx_ultimate.model import ERXModelEngine
from src.erx_ultimate.postprocessing import PostProcessingEngine

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("erx_ultimate.final_inference")


def stream_target_source_inference(
    source_type: SourceType,
    parquet_path: Path,
    s1_records_map: Dict[int, EntityRecord],
    retriever: ERXRetrievalEngine,
    model_engine: ERXModelEngine,
    post_engine: PostProcessingEngine,
    out_matches_path: Path,
    batch_size: int = 50000,
) -> None:
    """Stream inference for a single target source (S2 or S3)."""
    src_name = "Source2" if source_type == SourceType.SOURCE2 else "Source3"
    logger.info(f"Starting streaming inference for {src_name} ({parquet_path})...")

    table = pq.read_table(parquet_path)
    total_targets = table.num_rows
    total_batches = (total_targets + batch_size - 1) // batch_size
    logger.info(f"Loaded {total_targets:,} target records -> {total_batches} batches.")

    ids = table["id"].to_numpy()
    names_norm = table["name_norm"].to_pylist()
    names_raw = table["name_raw"].to_pylist()
    addrs_norm = table["address_norm"].to_pylist()
    addrs_raw = table["address_raw"].to_pylist()
    cities_norm = table["city_norm"].to_pylist()
    states_norm = table["state_norm"].to_pylist()
    zips_norm = table["postal_code_norm"].to_pylist()
    countries_norm = table["country_norm"].to_pylist()
    phones_norm = table["phone_norm"].to_pylist()
    webs_norm = table["website_norm"].to_pylist()

    all_scored_pairs: List[ScoredPair] = []

    for batch_num, i in enumerate(range(0, total_targets, batch_size)):
        end_idx = min(i + batch_size, total_targets)
        batch_candidates: List[CandidateMatch] = []
        batch_tgt_recs: List[EntityRecord] = []
        batch_s1_recs: List[EntityRecord] = []

        pbar = tqdm(
            range(i, end_idx),
            desc=f"Inference [{src_name} {batch_num+1}/{total_batches}]",
            unit="tgt",
            ncols=100,
            leave=False,
        )

        for idx in pbar:
            rec = EntityRecord(
                id=int(ids[idx]),
                source=source_type,
                name_raw=names_raw[idx],
                name_norm=names_norm[idx],
                address_raw=addrs_raw[idx],
                address_norm=addrs_norm[idx],
                city_norm=cities_norm[idx],
                state_norm=states_norm[idx],
                postal_code_norm=zips_norm[idx],
                country_norm=countries_norm[idx],
                phone_norm=phones_norm[idx],
                website_norm=webs_norm[idx],
            )
            
            cands = retriever.retrieve_candidates_for_record(rec, top_k=30)
            for c in cands:
                s1_rec = s1_records_map.get(c.s1_id)
                if s1_rec is not None:
                    batch_candidates.append(c)
                    batch_tgt_recs.append(rec)
                    batch_s1_recs.append(s1_rec)

        pbar.close()

        if batch_candidates:
            X_batch = extract_batch_features(batch_tgt_recs, batch_s1_recs, batch_candidates)
            raw_probs, cal_probs = model_engine.predict_batch(X_batch)

            for j, cand in enumerate(batch_candidates):
                all_scored_pairs.append(ScoredPair(
                    target_id=cand.target_id,
                    target_source=cand.target_source,
                    s1_id=cand.s1_id,
                    raw_prob=float(raw_probs[j]),
                    calibrated_prob=float(cal_probs[j]),
                    rrf_score=cand.rrf_score,
                ))

        logger.info(f"[{src_name}] Batch {batch_num+1}/{total_batches} done. Scored pairs: {len(all_scored_pairs):,}. {get_memory_summary()}")

    logger.info(f"[{src_name}] Resolving target ownership over {len(all_scored_pairs):,} scored pairs...")
    ownership = post_engine.resolve_target_ownership(all_scored_pairs)

    import pickle
    with open(out_matches_path, "wb") as f:
        pickle.dump(ownership, f)
    logger.info(f"[{src_name}] Saved {len(ownership):,} resolved matches to {out_matches_path}.")


def run_final_inference(
    source_filter: Optional[str] = None,
    config: UltimateConfig = CONFIG,
) -> None:
    """Run full production inference and generate final submission deliverables."""
    monitor = ResourceMonitor(
        soft_limit_gb=config.system.soft_memory_gb,
        hard_limit_gb=config.system.max_memory_gb,
    )
    monitor.start()

    try:
        logger.info("=== STARTING ER-X ULTIMATE FINAL INFERENCE PIPELINE ===")
        cache_mgr = CacheManager(config)
        outputs_dir = Path(config.paths.outputs_dir)
        outputs_dir.mkdir(parents=True, exist_ok=True)

        # 1. Ingest test datasets if needed
        logger.info("[Step 1/4] Ensuring test Parquet partitions exist...")
        s1_parquet = cache_mgr.normalized_dir / "test_s1.parquet"
        s2_parquet = cache_mgr.normalized_dir / "test_s2.parquet"
        s3_parquet = cache_mgr.normalized_dir / "test_s3.parquet"

        if not s1_parquet.exists():
            cache_mgr.ingest_raw_tsv("test_s1", config.paths.test_s1)
            s1_parquet = cache_mgr.build_normalized_parquet("test_s1", "test_s1")
        if not s2_parquet.exists():
            cache_mgr.ingest_raw_tsv("test_s2", config.paths.test_s2)
            s2_parquet = cache_mgr.build_normalized_parquet("test_s2", "test_s2")
        if not s3_parquet.exists():
            cache_mgr.ingest_raw_tsv("test_s3", config.paths.test_s3)
            s3_parquet = cache_mgr.build_normalized_parquet("test_s3", "test_s3")

        # 2. Load indexes and models
        logger.info("[Step 2/4] Loading CSR indexes and LightGBM model...")
        retriever = ERXRetrievalEngine(config)
        if not (cache_mgr.indices_dir / "s1_id_map.npy").exists():
            retriever.build_s1_indexes(s1_parquet)
        retriever.load_indexes(mmap_mode="r")

        # Load S1 record lookup table
        logger.info("Loading S1 record lookup table...")
        s1_table = pq.read_table(s1_parquet)
        s1_ids = s1_table["id"].to_pylist()
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
        del s1_table
        gc.collect()

        model_engine = ERXModelEngine(config)
        model_engine.load_artifacts()
        post_engine = PostProcessingEngine(config)

        # 3. Stream Inference for S2 and S3
        logger.info("[Step 3/4] Streaming target inference...")
        s2_out = cache_mgr.cache_dir / "temp_s2_ownership.pkl"
        s3_out = cache_mgr.cache_dir / "temp_s3_ownership.pkl"

        if source_filter == "S2" or source_filter is None:
            stream_target_source_inference(
                SourceType.SOURCE2, s2_parquet, s1_records_map,
                retriever, model_engine, post_engine, s2_out
            )

        if source_filter == "S3" or source_filter is None:
            stream_target_source_inference(
                SourceType.SOURCE3, s3_parquet, s1_records_map,
                retriever, model_engine, post_engine, s3_out
            )

        # 4. Consolidate and export matching_results.tsv and candidate_pairs.tsv
        if source_filter is None:
            logger.info("[Step 4/4] Consolidating matches and exporting deliverables...")
            import pickle
            with open(s2_out, "rb") as f:
                s2_ownership = pickle.load(f)
            with open(s3_out, "rb") as f:
                s3_ownership = pickle.load(f)

            combined_ownership = {**s2_ownership, **s3_ownership}
            clusters = post_engine.aggregate_clusters(s1_ids, combined_ownership)

            # Export matching_results.tsv
            out_matching_file = outputs_dir / "matching_results.tsv"
            logger.info(f"Writing {len(clusters):,} clusters to {out_matching_file}...")
            with open(out_matching_file, "w", encoding="utf-8", buffering=16 * 1024 * 1024) as f:
                f.write("source1_entity_id\tmatched_entity_ids\n")
                for cluster in clusters:
                    f.write(cluster.to_matching_results_row())

            # Export candidate_pairs.tsv
            out_cand_file = outputs_dir / "candidate_pairs.tsv"
            logger.info(f"Writing candidate pairs to {out_cand_file}...")
            with open(out_cand_file, "w", encoding="utf-8", buffering=16 * 1024 * 1024) as f:
                f.write("source1_entity_id\tcandidate_entity_ids\n")
                for cluster in clusters:
                    # In test deliverables, candidate pairs include matched targets + top retrieved
                    cluster.candidate_s2_ids = cluster.source2_ids
                    cluster.candidate_s3_ids = cluster.source3_ids
                    f.write(cluster.to_candidate_pairs_row())

            logger.info(f"Successfully generated final submission deliverables in {outputs_dir}")

    finally:
        monitor.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ER-X Ultimate Final Production Inference Engine")
    parser.add_argument("--source", type=str, choices=["S2", "S3"], default=None, help="Process specific source only (S2 or S3)")
    args = parser.parse_args()

    run_final_inference(source_filter=args.source)
