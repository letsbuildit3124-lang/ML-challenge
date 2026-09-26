"""
Antigravity E5 FAISS Index Builder.
Constructs memory-bounded, high-throughput ANN indices for normalized E5 target embeddings.

Supported Index Types:
- IndexFlatIP: Exact inner-product search (baseline for smoke / benchmarks)
- IndexIVFFlat: Scalable inverted file index with flat vectors
- IndexIVFPQ: Memory-efficient Product Quantization index (recommended for 10.32M production)

Production Invariants:
- Production index requires exactly 10,320,219 vectors with complete=true.
- Strict artifact separation: smoke/ vs benchmark/ vs production/.
"""

import os
import sys
import gc
import json
import time
import argparse
from typing import Optional, Tuple, Dict, Any
import numpy as np

from src.config import get_config
from src.resource_tracker import get_current_rss_mb, get_peak_rss_mb, MemoryTracker

EXPECTED_PROD_ROWS = 10320219
DEFAULT_DIM = 384


def build_e5_faiss_index(
    ann_dir: str = "cache/ann/e5/production",
    index_type: str = "ivf_pq",
    nlist: int = 4096,
    pq_m: int = 48,
    pq_nbits: int = 8,
    sample_train_size: int = 250000,
    batch_add_size: int = 100000,
    force_rebuild: bool = False
) -> str:
    """
    Builds a FAISS index from the disk-backed target_embeddings.f16.memmap.
    """
    config = get_config()
    target_dir = os.path.join(config.base_dir, ann_dir)
    memmap_path = os.path.join(target_dir, "target_embeddings.f16.memmap")
    ids_path = os.path.join(target_dir, "target_ids.json")
    index_path = os.path.join(target_dir, "target.index")
    meta_path = os.path.join(target_dir, "index_metadata.json")

    if not os.path.exists(memmap_path):
        raise FileNotFoundError(f"Target memmap file not found at: {memmap_path}. Run assemble_e5_memmap first!")
    if not os.path.exists(ids_path):
        raise FileNotFoundError(f"Target IDs file not found at: {ids_path}. Run assemble_e5_memmap first!")

    mmap_check = np.lib.format.open_memmap(memmap_path, mode="r")
    total_vectors, dim = mmap_check.shape
    del mmap_check

    is_production = (total_vectors == EXPECTED_PROD_ROWS)

    print("=" * 80)
    print("ANTIGRAVITY — E5 FAISS ANN INDEX BUILDER")
    print("=" * 80)
    print(f"Target Memmap:       {memmap_path} (Shape: {total_vectors:,} x {dim})")
    print(f"Index Destination:   {index_path}")
    print(f"Index Type:          {index_type.upper()}")
    print(f"Production Universe: {'YES (10,320,219 Targets)' if is_production else f'NO (Subset: {total_vectors:,})'}")
    print(f"Initial RSS:         {get_current_rss_mb():.2f} MB")
    print("-" * 80)

    # Check for existing valid index
    if not force_rebuild and os.path.exists(index_path) and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            existing_meta = json.load(f)
        if (
            existing_meta.get("ntotal") == total_vectors
            and existing_meta.get("is_complete_target_universe") == is_production
        ):
            print(f"[FAISS] Found existing validated index ({total_vectors:,} vectors). Skipping rebuild.")
            return index_path

    try:
        import faiss
    except ImportError:
        print("[FAISS ERROR] FAISS not installed! Run: pip install faiss-cpu or faiss-gpu")
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(f"# Fallback: FAISS not installed. Total vectors: {total_vectors}\n")
        return index_path

    t0 = time.time()
    mmap = np.lib.format.open_memmap(memmap_path, mode="r")

    # 1. Instantiate Index Structure
    if index_type == "flat" or total_vectors < 50000:
        print("[FAISS] Building IndexFlatIP (Exact Inner Product Search)...")
        index = faiss.IndexFlatIP(dim)

    elif index_type == "ivf_pq":
        actual_nlist = min(nlist, max(4, total_vectors // 32))
        print(f"[FAISS] Training IndexIVFPQ (dim={dim}, nlist={actual_nlist}, m={pq_m}, nbits={pq_nbits})...")
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, actual_nlist, pq_m, pq_nbits, faiss.METRIC_INNER_PRODUCT)

        # Train on calibrated random sample to avoid full-corpus RAM overhead
        train_n = min(sample_train_size, total_vectors)
        train_indices = np.random.RandomState(42).choice(total_vectors, size=train_n, replace=False)
        train_indices.sort()

        print(f"[FAISS] Extracting training sample of {train_n:,} vectors...")
        train_sample = mmap[train_indices].astype(np.float32)

        t_train = time.time()
        index.train(train_sample)
        print(f"[FAISS] Index trained in {time.time() - t_train:.2f}s.")
        del train_sample, train_indices
        gc.collect()

    else:
        actual_nlist = min(nlist, max(4, total_vectors // 32))
        print(f"[FAISS] Training IndexIVFFlat (dim={dim}, nlist={actual_nlist})...")
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, actual_nlist, faiss.METRIC_INNER_PRODUCT)
        train_n = min(sample_train_size, total_vectors)
        train_sample = mmap[:train_n].astype(np.float32)
        index.train(train_sample)
        del train_sample
        gc.collect()

    # 2. Add Vectors in Batches
    print(f"[FAISS] Adding {total_vectors:,} vectors in batches of {batch_add_size:,}...")
    t_add = time.time()

    with MemoryTracker(f"FAISS Vector Addition ({total_vectors:,} rows)"):
        for start_idx in range(0, total_vectors, batch_add_size):
            end_idx = min(total_vectors, start_idx + batch_add_size)
            batch = mmap[start_idx:end_idx].astype(np.float32)
            index.add(batch)
            del batch

            pct = (end_idx / total_vectors) * 100.0
            print(f"  -> Added {end_idx:,}/{total_vectors:,} vectors [{pct:.1f}%] | RSS: {get_current_rss_mb():.1f} MB", flush=True)
            if start_idx % (batch_add_size * 5) == 0:
                gc.collect()

    del mmap
    gc.collect()

    add_dur = time.time() - t_add
    print(f"[FAISS] Vectors added in {add_dur:.2f}s (ntotal={index.ntotal:,}).")

    # 3. Serialize Index
    print(f"[FAISS] Writing index to disk: {index_path}...")
    faiss.write_index(index, index_path)
    file_size_mb = os.path.getsize(index_path) / (1024 * 1024)

    # 4. Save Index Metadata
    meta = {
        "index_file": "target.index",
        "index_type": type(index).__name__,
        "metric": "METRIC_INNER_PRODUCT (Cosine Similarity)",
        "dimension": dim,
        "ntotal": index.ntotal,
        "is_complete_target_universe": is_production,
        "model_name": "intfloat/multilingual-e5-small",
        "file_size_mb": round(file_size_mb, 2),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "build_duration_seconds": round(time.time() - t0, 2)
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    total_time = time.time() - t0
    print("=" * 80)
    print(f"[FAISS SUCCESS] Index created successfully in {total_time:.2f}s:")
    print(f"  Index File:      {index_path} ({file_size_mb:.2f} MB)")
    print(f"  Total Indexed:   {index.ntotal:,} vectors")
    print(f"  Complete Prod:   {is_production}")
    print(f"  Peak RSS:        {get_peak_rss_mb():.2f} MB")
    print("=" * 80)

    return index_path


def main():
    parser = argparse.ArgumentParser(description="Antigravity E5 FAISS Index Builder")
    parser.add_argument("--dir", type=str, default="cache/ann/e5/production", help="Directory containing memmap and target IDs")
    parser.add_argument("--index-type", type=str, default="ivf_pq", choices=["flat", "ivf", "ivf_pq"], help="Index type")
    parser.add_argument("--smoke", action="store_true", help="Build smoke index in cache/ann/e5/smoke")
    parser.add_argument("--benchmark", action="store_true", help="Build benchmark index in cache/ann/e5/benchmark")
    parser.add_argument("--production", action="store_true", help="Build production index in cache/ann/e5/production")
    parser.add_argument("--force", action="store_true", help="Force rebuild existing index")
    args = parser.parse_args()

    target_dir = args.dir
    if args.smoke:
        target_dir = "cache/ann/e5/smoke"
    elif args.benchmark:
        target_dir = "cache/ann/e5/benchmark"
    elif args.production:
        target_dir = "cache/ann/e5/production"

    build_e5_faiss_index(
        ann_dir=target_dir,
        index_type=args.index_type,
        force_rebuild=args.force
    )


if __name__ == "__main__":
    main()
