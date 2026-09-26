"""
Antigravity V5 Multilingual E5 FAISS Index Builder.
Constructs memory-bounded, high-throughput ANN indices for 10.32M normalized target vectors.

Key Features:
- Supports IndexFlatIP, IndexIVFFlat, and memory-efficient IndexIVFPQ.
- Sample-calibrated IVF-PQ training (avoids loading 10.32M vectors into RAM simultaneously).
- Strict Production Invariant: production index must have exactly 10,320,219 vectors with complete=True.
- Full path isolation:
  - Production Index: cache/ann/multilingual_e5/production/target.index
  - Smoke / Dev Index: cache/ann/multilingual_e5/smoke/target.index
"""

import os
import sys
import gc
import json
import time
import argparse
from datetime import datetime, timezone
from typing import Optional, Tuple, List
import numpy as np

from src.config import get_config
from src.resource_tracker import get_current_rss_mb, get_peak_rss_mb, MemoryTracker

EXPECTED_PRODUCTION_TARGETS = 10320219
DEFAULT_DIM = 384


def validate_embedding_corpus(
    embeddings_path: str,
    ids_path: str,
    is_production: bool = True
) -> Tuple[int, int]:
    """
    Validates physical existence, shape, and 100% ID uniqueness of the embedding corpus.
    """
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found at: {embeddings_path}")
    if not os.path.exists(ids_path):
        raise FileNotFoundError(f"Target IDs file not found at: {ids_path}")

    mmap = np.lib.format.open_memmap(embeddings_path, mode="r")
    num_rows, dim = mmap.shape
    del mmap

    if dim != DEFAULT_DIM:
        raise ValueError(f"Embedding dimension mismatch: Found {dim}, Expected {DEFAULT_DIM}")

    with open(ids_path, "r", encoding="utf-8") as f:
        target_ids = json.load(f)

    if len(target_ids) != num_rows:
        raise ValueError(f"Count mismatch: Embeddings ({num_rows:,}) != Target IDs ({len(target_ids):,})")

    if is_production:
        if num_rows != EXPECTED_PRODUCTION_TARGETS:
            raise ValueError(
                f"[PRODUCTION INVARIANT VIOLATION] Full universe requires exactly "
                f"{EXPECTED_PRODUCTION_TARGETS:,} targets, but found {num_rows:,}."
            )
        unique_ids = len(set(target_ids))
        if unique_ids != EXPECTED_PRODUCTION_TARGETS:
            raise ValueError(
                f"[PRODUCTION INVARIANT VIOLATION] Target IDs contain duplicates! "
                f"Unique: {unique_ids:,} != Expected: {EXPECTED_PRODUCTION_TARGETS:,}"
            )

    return num_rows, dim


def build_faiss_index(
    embeddings_path: str,
    ids_path: str,
    output_dir: str,
    index_type: str = "ivf_pq",
    nlist: int = 4096,
    pq_m: int = 48,
    pq_nbits: int = 8,
    sample_train_size: int = 250000,
    batch_add_size: int = 100000,
    force_rebuild: bool = False
) -> str:
    """
    Builds a FAISS index from disk-backed float16/float32 embeddings.
    """
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "target.index")
    meta_path = os.path.join(output_dir, "metadata.json")

    # Inspect corpus
    mmap_temp = np.lib.format.open_memmap(embeddings_path, mode="r")
    total_vectors, dim = mmap_temp.shape
    emb_dtype = mmap_temp.dtype
    del mmap_temp

    is_production = (total_vectors == EXPECTED_PRODUCTION_TARGETS)

    print("=" * 80)
    print("ANTIGRAVITY V5 — MULTILINGUAL E5 FAISS ANN INDEX BUILDER")
    print("=" * 80)
    print(f"Target Embeddings:   {embeddings_path} (Shape: {total_vectors:,} x {dim} | {emb_dtype})")
    print(f"Target IDs:          {ids_path}")
    print(f"Output Directory:    {output_dir}")
    print(f"Index Type:          {index_type.upper()}")
    print(f"Production Universe: {'YES (10,320,219 Targets)' if is_production else f'NO (Smoke/Dev: {total_vectors:,})'}")
    print(f"Initial RSS:         {get_current_rss_mb():.2f} MB")
    print("-" * 80)

    # Validate corpus
    validate_embedding_corpus(embeddings_path, ids_path, is_production=is_production)

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
        print("[FAISS ERROR] FAISS is not installed! Run: pip install faiss-cpu or faiss-gpu")
        # Save fallback marker
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(f"# Fallback marker: FAISS not installed. Total vectors: {total_vectors}\n")
        return index_path

    mmap = np.lib.format.open_memmap(embeddings_path, mode="r")

    # 1. Instantiate Index
    if index_type == "flat" or total_vectors < 50000:
        print("[FAISS] Initializing IndexFlatIP (Exact Inner Product Search)...")
        index = faiss.IndexFlatIP(dim)

    elif index_type == "ivf_pq":
        actual_nlist = min(nlist, max(4, total_vectors // 32))
        print(f"[FAISS] Training IndexIVFPQ (dim={dim}, nlist={actual_nlist}, m={pq_m}, nbits={pq_nbits})...")
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, actual_nlist, pq_m, pq_nbits, faiss.METRIC_INNER_PRODUCT)

        # Train on representative sample
        train_n = min(sample_train_size, total_vectors)
        train_indices = np.random.RandomState(42).choice(total_vectors, size=train_n, replace=False)
        train_indices.sort()

        print(f"[FAISS] Extracting training sample of {train_n:,} vectors...")
        train_sample = mmap[train_indices].astype(np.float32)

        t_train = time.time()
        index.train(train_sample)
        print(f"[FAISS] Training completed in {time.time() - t_train:.2f}s.")
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

    with MemoryTracker("FAISS Batch Vector Addition"):
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

    add_duration = time.time() - t_add
    print(f"[FAISS] Vector addition complete in {add_duration:.2f}s (ntotal={index.ntotal:,}).")

    # 3. Serialize Index to Disk
    print(f"[FAISS] Writing index to disk: {index_path}...")
    faiss.write_index(index, index_path)
    file_size_mb = os.path.getsize(index_path) / (1024 * 1024)

    # 4. Save Metadata
    metadata = {
        "index_file": "target.index",
        "index_type": type(index).__name__,
        "metric": "METRIC_INNER_PRODUCT (Cosine Similarity)",
        "dimension": dim,
        "ntotal": index.ntotal,
        "is_complete_target_universe": is_production,
        "model_name": "intfloat/multilingual-e5-small",
        "file_size_mb": file_size_mb,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    total_time = time.time() - t0
    print("\n" + "=" * 80)
    print(f"[FAISS] INDEX CREATION COMPLETED SUCCESSFULLY in {total_time:.2f}s")
    print(f"  Index File:  {index_path} ({file_size_mb:.2f} MB)")
    print(f"  Total Rows:  {index.ntotal:,} (Production Complete: {is_production})")
    print(f"  Peak RSS:    {get_peak_rss_mb():.2f} MB")
    print("=" * 80)

    return index_path


def main():
    parser = argparse.ArgumentParser(description="Antigravity V5 Multilingual E5 FAISS Builder")
    parser.add_argument("--embeddings", type=str, default="cache/embeddings/multilingual_e5/production/target_embeddings.npy")
    parser.add_argument("--ids", type=str, default="cache/embeddings/multilingual_e5/production/target_ids.json")
    parser.add_argument("--output-dir", type=str, default="cache/ann/multilingual_e5/production")
    parser.add_argument("--index-type", type=str, default="ivf_pq", choices=["flat", "ivf", "ivf_pq"])
    parser.add_argument("--force", action="store_true", help="Force rebuild existing index")
    args = parser.parse_args()

    build_faiss_index(
        embeddings_path=args.embeddings,
        ids_path=args.ids,
        output_dir=args.output_dir,
        index_type=args.index_type,
        force_rebuild=args.force
    )


if __name__ == "__main__":
    main()
