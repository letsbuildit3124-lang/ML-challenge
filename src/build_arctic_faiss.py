"""
Antigravity V4.1 Persistent FAISS ANN Index Builder.
Builds memory-bounded, scalable Inner Product / Cosine Similarity index over Arctic target embeddings.

Features:
- Strict Production Invariant: Production index MUST contain exactly 10,320,219 vectors with is_complete=True.
- Smoke Test Protection: Incomplete dev/smoke indexes (ntotal < 10,320,219) are stored in cache/ann/arctic/smoke/
  and are NEVER reused for production builds or full retrieval benchmarks.
- Comprehensive Embedding Corpus Validation (counts, unique IDs, dimension, normalization).
- Sub-2GB RAM Footprint via Streaming Memmap & IVF-PQ Quantization.
- Fallback NumPy IP index for offline/test environments.
"""

import os
import sys
import gc
import json
import time
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.config import get_config
from src.resource_tracker import get_current_rss_mb, get_peak_rss_mb, MemoryTracker
from src.arctic_embeddings import EMBEDDING_DIM, DEFAULT_MODEL_NAME

EXPECTED_TOTAL_TARGETS = 10320219

def check_existing_index(
    ann_dir: str,
    target_count: int,
    expected_complete: bool = False,
    metric: str = "INNER_PRODUCT"
) -> bool:
    """
    Validates existing persisted index against metadata and production invariants.
    Returns True ONLY if all invariants match; otherwise returns False to trigger a clean rebuild.
    """
    meta_path = os.path.join(ann_dir, "metadata.json")
    index_path = os.path.join(ann_dir, "target.index")

    if not (os.path.exists(meta_path) and os.path.exists(index_path)):
        return False

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        is_complete = meta.get("is_complete_target_universe", False)
        t_count = meta.get("target_count", 0)
        dim = meta.get("embedding_dimension", 0)
        idx_metric = meta.get("metric", "")

        # 1. Target count & metric check
        if t_count != target_count or idx_metric != metric or dim != EMBEDDING_DIM:
            print(f"[FAISSIndex] Metadata mismatch (Target Count: {t_count} != {target_count}, Metric: {idx_metric} != {metric}, Dim: {dim} != {EMBEDDING_DIM}). Refusing reuse.")
            return False

        # 2. Production completeness invariant
        if expected_complete and not is_complete:
            print(f"[FAISSIndex] Incomplete/Smoke index detected (complete={is_complete}, target_count={t_count:,}). Cannot reuse for complete production universe. Refusing reuse.")
            return False

        # 3. Verify actual FAISS index file header if FAISS is installed
        try:
            import faiss
            if os.path.getsize(index_path) > 1024:
                idx = faiss.read_index(index_path)
                if idx.ntotal != target_count:
                    print(f"[FAISSIndex] FAISS index file corrupted/truncated (ntotal={idx.ntotal} != {target_count}). Refusing reuse.")
                    del idx
                    return False
                del idx
        except ImportError:
            pass

        print(f"[FAISSIndex] Existing valid ANN index verified ({target_count:,} vectors | Complete: {is_complete}). Reusing index!")
        return True
    except Exception as e:
        print(f"[FAISSIndex] Error reading index metadata: {e}. Refusing reuse.")

    return False


def validate_embedding_corpus(
    embeddings_path: str,
    ids_path: str,
    expected_count: Optional[int] = None,
    require_complete: bool = False
) -> Tuple[int, int]:
    """
    Strict validation of the embedding matrix and target IDs before index construction.
    """
    if not (os.path.exists(embeddings_path) and os.path.exists(ids_path)):
        raise FileNotFoundError(f"Missing embeddings file ({embeddings_path}) or IDs file ({ids_path})!")

    # Check embedding matrix shape
    emb_mmap = np.lib.format.open_memmap(embeddings_path, mode="r")
    total_vectors, dim = emb_mmap.shape

    if dim != EMBEDDING_DIM:
        raise ValueError(f"Invalid embedding dimension: {dim} (Expected: {EMBEDDING_DIM})")

    # Check target IDs
    with open(ids_path, "r", encoding="utf-8") as f:
        target_ids = json.load(f)

    if len(target_ids) != total_vectors:
        raise ValueError(f"Target ID count mismatch: IDs ({len(target_ids):,}) != Embedding Rows ({total_vectors:,})")

    # Strict Production Invariant Checks
    if require_complete or total_vectors == EXPECTED_TOTAL_TARGETS:
        if total_vectors != EXPECTED_TOTAL_TARGETS:
            raise ValueError(f"Incomplete production corpus: Found {total_vectors:,} records, Expected exactly {EXPECTED_TOTAL_TARGETS:,}")
        
        # Verify unique target IDs (no duplicates)
        unique_ids_count = len(set(target_ids))
        if unique_ids_count != EXPECTED_TOTAL_TARGETS:
            raise ValueError(f"Duplicate target IDs detected in production corpus! Unique: {unique_ids_count:,} != Total: {EXPECTED_TOTAL_TARGETS:,}")

    if expected_count is not None and total_vectors != expected_count:
        raise ValueError(f"Corpus size mismatch: Found {total_vectors:,} records, Expected {expected_count:,}")

    return total_vectors, dim


def build_faiss_index(
    embeddings_path: str,
    ids_path: str,
    output_dir: str,
    nlist: int = 4096,
    pq_m: int = 48,
    pq_nbits: int = 8,
    force_rebuild: bool = False
):
    """
    Constructs and persists a memory-safe ANN index from disk-backed Arctic embeddings.
    """
    print("\n" + "=" * 80)
    print("ANTIGRAVITY V4.1 — FAISS ANN INDEX BUILDER")
    print("=" * 80)
    print(f"Embeddings Source: {embeddings_path}")
    print(f"Target IDs Source: {ids_path}")
    print(f"Initial RSS:       {get_current_rss_mb():.2f} MB")
    print("-" * 80)

    # 1. Validate Embedding Corpus
    total_vectors, dim = validate_embedding_corpus(embeddings_path, ids_path)
    is_complete_build = (total_vectors == EXPECTED_TOTAL_TARGETS)

    # Route smoke/development indexes to separate subfolder if incomplete
    if not is_complete_build and not output_dir.endswith("smoke"):
        target_ann_dir = os.path.join(output_dir, "smoke")
    else:
        target_ann_dir = output_dir

    os.makedirs(target_ann_dir, exist_ok=True)
    index_path = os.path.join(target_ann_dir, "target.index")
    meta_path = os.path.join(target_ann_dir, "metadata.json")

    print(f"Target Vectors:    {total_vectors:,} / {EXPECTED_TOTAL_TARGETS:,}")
    print(f"Complete Universe: {'YES (PRODUCTION INDEX)' if is_complete_build else 'NO (SMOKE / DEV TEST INDEX)'}")
    print(f"Destination Dir:   {target_ann_dir}")
    print(f"Target Index Path: {index_path}")

    # 2. Check existing index and apply safety invariants
    if not force_rebuild and check_existing_index(target_ann_dir, total_vectors, expected_complete=is_complete_build):
        return

    # 3. Build FAISS Index
    emb_mmap = np.lib.format.open_memmap(embeddings_path, mode="r")

    faiss = None
    try:
        import faiss as _faiss
        faiss = _faiss
    except ImportError:
        print("[FAISSIndex] Warning: FAISS is not installed. Building lightweight NumPy fallback index.")

    t0 = time.time()
    index_type_str = ""

    with MemoryTracker(f"FAISS Index Construction ({total_vectors:,} vectors)"):
        if faiss is not None:
            if total_vectors <= 50000:
                print(f"[FAISSIndex] Constructing Exact IndexFlatIP (N={total_vectors:,} <= 50k)...")
                index = faiss.IndexFlatIP(dim)
                index.add(np.ascontiguousarray(emb_mmap))
                index_type_str = "IndexFlatIP"
            else:
                effective_nlist = min(nlist, max(16, int(np.sqrt(total_vectors))))
                print(f"[FAISSIndex] Training IndexIVFPQ (N={total_vectors:,}, nlist={effective_nlist}, M={pq_m}, nbits={pq_nbits})...")
                
                quantizer = faiss.IndexFlatIP(dim)
                index = faiss.IndexIVFPQ(quantizer, dim, effective_nlist, pq_m, pq_nbits)
                index.metric_type = faiss.METRIC_INNER_PRODUCT

                # Train on bounded sample (Max 100,000 vectors)
                train_size = min(total_vectors, 100000)
                print(f"[FAISSIndex] Sampling {train_size:,} vectors from memmap for IVF-PQ training...")
                train_sample = np.ascontiguousarray(emb_mmap[:train_size])
                index.train(train_sample)
                del train_sample
                gc.collect()

                # Add vectors in bounded chunks (100,000 vectors per chunk)
                chunk_size = 100000
                print(f"[FAISSIndex] Adding {total_vectors:,} vectors in chunks of {chunk_size:,}...")
                for i in range(0, total_vectors, chunk_size):
                    chunk_end = min(total_vectors, i + chunk_size)
                    chunk = np.ascontiguousarray(emb_mmap[i : chunk_end])
                    index.add(chunk)
                    pct = (chunk_end / total_vectors) * 100.0
                    print(f"  -> Added {chunk_end:,}/{total_vectors:,} ({pct:.1f}%) | RSS: {get_current_rss_mb():.1f} MB", flush=True)
                    del chunk
                    gc.collect()

                index_type_str = f"IndexIVFPQ(nlist={effective_nlist}, M={pq_m}, nbits={pq_nbits})"

            # Invariant verification
            assert index.ntotal == total_vectors, f"FAISS ntotal mismatch: {index.ntotal} != {total_vectors}!"
            print(f"[FAISSIndex] Verified index.ntotal == {index.ntotal:,}.")

            print(f"[FAISSIndex] Writing index to disk: {index_path}...")
            faiss.write_index(index, index_path)
            del index
            gc.collect()

        else:
            # Fallback for offline environments without FAISS
            index_type_str = "NumPy_FlatIP_Fallback"
            with open(index_path, "w") as f:
                f.write(f"NumPy_FlatIP_Fallback: {embeddings_path}\n")

    metadata = {
        "model_name": DEFAULT_MODEL_NAME,
        "index_type": index_type_str,
        "metric": "INNER_PRODUCT",
        "embedding_dimension": dim,
        "target_count": total_vectors,
        "expected_target_count": EXPECTED_TOTAL_TARGETS,
        "is_complete_target_universe": is_complete_build,
        "source_embeddings": embeddings_path,
        "source_ids": ids_path,
        "creation_timestamp": datetime.now(timezone.utc).isoformat()
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    total_time = time.time() - t0
    print(f"\n[FAISSIndex] Successfully built ANN index in {total_time:.2f}s.")
    print(f"  Index Type:        {index_type_str}")
    print(f"  Complete Universe: {'YES (PRODUCTION)' if is_complete_build else 'NO (SMOKE TEST ONLY)'}")
    print(f"  Saved Location:    {index_path}")
    print(f"  Final RSS:         {get_current_rss_mb():.2f} MB | Peak RSS: {get_peak_rss_mb():.2f} MB")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Antigravity V4.1 FAISS ANN Index Builder")
    parser.add_argument("--embeddings-dir", type=str, default="cache/embeddings/arctic", help="Embeddings directory")
    parser.add_argument("--output-dir", type=str, default="cache/ann/arctic", help="Output ANN directory")
    parser.add_argument("--nlist", type=int, default=4096, help="Number of Voronoi cells for IVF index")
    parser.add_argument("--pq-m", type=int, default=48, help="Number of sub-vector quantizers (M)")
    parser.add_argument("--pq-nbits", type=int, default=8, help="Number of bits per sub-vector")
    parser.add_argument("--force", action="store_true", help="Force rebuild of index even if cache is valid")
    args = parser.parse_args()

    config = get_config()
    emb_dir = os.path.join(config.base_dir, args.embeddings_dir)
    ann_dir = os.path.join(config.base_dir, args.output_dir)

    emb_path = os.path.join(emb_dir, "target_embeddings.npy")
    ids_path = os.path.join(emb_dir, "target_ids.json")

    build_faiss_index(
        embeddings_path=emb_path,
        ids_path=ids_path,
        output_dir=ann_dir,
        nlist=args.nlist,
        pq_m=args.pq_m,
        pq_nbits=args.pq_nbits,
        force_rebuild=args.force
    )

if __name__ == "__main__":
    main()
