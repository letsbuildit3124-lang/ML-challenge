"""
Antigravity V4.1 Persistent FAISS ANN Index Builder.
Builds memory-bounded, scalable Inner Product / Cosine Similarity index over Arctic target embeddings.

Features:
- Sub-2GB RAM Footprint via Streaming Memmap & IVF-PQ Quantization
- Zero-copy memmap streaming (np.load(mmap_mode='r'))
- Bounded sample training (100k vectors) for IVF-PQ
- Strict index completeness verification (assert index.ntotal == expected)
- Fallback NumPy IP index for offline/test environments
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
from src.arctic_embeddings import EMBEDDING_DIM

EXPECTED_TOTAL_TARGETS = 10320219

def check_existing_index(
    ann_dir: str,
    target_count: int,
    metric: str = "INNER_PRODUCT"
) -> bool:
    """Checks whether valid persisted index matches metadata."""
    meta_path = os.path.join(ann_dir, "metadata.json")
    index_path = os.path.join(ann_dir, "target.index")

    if not (os.path.exists(meta_path) and os.path.exists(index_path)):
        return False

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if meta.get("target_count") == target_count and meta.get("metric") == metric:
            is_complete = meta.get("is_complete_target_universe", False)
            print(f"[FAISSIndex] Existing valid ANN index found ({target_count:,} vectors | Complete: {is_complete}). Reusing index!")
            return True
    except Exception as e:
        print(f"[FAISSIndex] Error reading index metadata: {e}")

    return False


def build_faiss_index(
    embeddings_path: str,
    ids_path: str,
    output_dir: str,
    nlist: int = 4096,
    pq_m: int = 48,
    pq_nbits: int = 8
):
    """
    Constructs and persists a memory-safe ANN index from disk-backed Arctic embeddings.
    """
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "target.index")
    meta_path = os.path.join(output_dir, "metadata.json")

    print("\n" + "=" * 80)
    print("ANTIGRAVITY V4.1 — FAISS ANN INDEX BUILDER")
    print("=" * 80)
    print(f"Embeddings Source: {embeddings_path}")
    print(f"Target Index Path: {index_path}")
    print(f"Initial RSS:       {get_current_rss_mb():.2f} MB")
    print("-" * 80)

    if not (os.path.exists(embeddings_path) and os.path.exists(ids_path)):
        raise FileNotFoundError(f"Missing embeddings file ({embeddings_path}) or IDs file ({ids_path})!")

    # Memory-map the embeddings without loading into RAM
    emb_mmap = np.lib.format.open_memmap(embeddings_path, mode="r")
    total_vectors, dim = emb_mmap.shape
    is_complete_build = (total_vectors == EXPECTED_TOTAL_TARGETS)

    print(f"Target Vectors in MMap: {total_vectors:,} | Dimension: {dim}")
    print(f"Complete Universe Index: {'YES (Production)' if is_complete_build else 'NO (Smoke / Dev Test)'}")

    # Check if index already exists
    if check_existing_index(output_dir, total_vectors):
        return

    # Try importing FAISS
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
                # Production Memory-Bounded IVF-PQ Configuration
                # With M=48, 10.3M vectors consume only ~495MB RAM!
                effective_nlist = min(nlist, max(16, int(np.sqrt(total_vectors))))
                print(f"[FAISSIndex] Training IndexIVFPQ (N={total_vectors:,}, nlist={effective_nlist}, M={pq_m}, nbits={pq_nbits})...")
                
                quantizer = faiss.IndexFlatIP(dim)
                index = faiss.IndexIVFPQ(quantizer, dim, effective_nlist, pq_m, pq_nbits)
                index.metric_type = faiss.METRIC_INNER_PRODUCT

                # Train on bounded sample (Max 100,000 vectors) to keep train RAM under 300MB
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

            # Verify total vectors added
            assert index.ntotal == total_vectors, f"FAISS ntotal mismatch: {index.ntotal} != {total_vectors}!"
            print(f"[FAISSIndex] Verified index.ntotal == {index.ntotal:,}.")

            print(f"[FAISSIndex] Writing index to disk: {index_path}...")
            faiss.write_index(index, index_path)
            del index
            gc.collect()

        else:
            # Fallback: Save metadata referencing the mmap for direct IP dot-product search
            index_type_str = "NumPy_FlatIP_Fallback"
            with open(index_path, "w") as f:
                f.write(f"NumPy_FlatIP_Fallback: {embeddings_path}\n")

    metadata = {
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
    print(f"  Complete Universe: {'YES' if is_complete_build else 'NO (SMOKE TEST ONLY)'}")
    print(f"  Final RSS:         {get_current_rss_mb():.2f} MB | Peak RSS: {get_peak_rss_mb():.2f} MB")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Antigravity V4.1 FAISS ANN Index Builder")
    parser.add_argument("--embeddings-dir", type=str, default="cache/embeddings/arctic", help="Embeddings directory")
    parser.add_argument("--output-dir", type=str, default="cache/ann/arctic", help="Output ANN directory")
    parser.add_argument("--nlist", type=int, default=4096, help="Number of Voronoi cells for IVF index")
    parser.add_argument("--pq-m", type=int, default=48, help="Number of sub-vector quantizers (M)")
    parser.add_argument("--pq-nbits", type=int, default=8, help="Number of bits per sub-vector")
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
        pq_nbits=args.pq_nbits
    )

if __name__ == "__main__":
    main()
