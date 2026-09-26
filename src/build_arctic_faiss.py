"""
Antigravity V4.1 Persistent FAISS ANN Index Builder.
Builds memory-efficient Inner Product (Cosine Similarity) index over Arctic target embeddings.

Stores outputs in:
cache/ann/arctic/
  ├── target.index   (FAISS index file or memory-mapped array)
  └── metadata.json  (Validation schema & index hyperparameter registry)

Supports:
- IndexFlatIP for small/controlled subsets (N <= 50,000)
- IndexIVFFlat / IndexIVFPQ for scalable 10.3M production target universe
- Graceful NumPy fallback when FAISS binary is not installed
- Process-level RSS tracking
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

def check_existing_index(
    ann_dir: str,
    target_count: int,
    index_type: str = "IVFFlat",
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
            print(f"[FAISSIndex] Existing valid ANN index found ({target_count:,} vectors). Reusing index!")
            return True
    except Exception as e:
        print(f"[FAISSIndex] Error reading index metadata: {e}")

    return False


def build_faiss_index(
    embeddings_path: str,
    ids_path: str,
    output_dir: str,
    nlist: int = 1024,
    use_pq: bool = False
):
    """
    Constructs and persists an ANN index from disk-backed Arctic embeddings.
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

    # Memory-map the embeddings (Zero RAM copy)
    emb_mmap = np.lib.format.open_memmap(embeddings_path, mode="r")
    total_vectors, dim = emb_mmap.shape
    print(f"Target Vectors:    {total_vectors:,} | Dimension: {dim}")

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
                effective_nlist = min(nlist, max(4, int(np.sqrt(total_vectors))))
                print(f"[FAISSIndex] Training IndexIVFFlat (N={total_vectors:,}, nlist={effective_nlist})...")
                quantizer = faiss.IndexFlatIP(dim)
                if use_pq:
                    # Compressed Product Quantization for strictly bounded RAM
                    index = faiss.IndexIVFPQ(quantizer, dim, effective_nlist, 32, 8)
                    index_type_str = f"IndexIVFPQ(nlist={effective_nlist}, m=32, nbits=8)"
                else:
                    index = faiss.IndexIVFFlat(quantizer, dim, effective_nlist, faiss.METRIC_INNER_PRODUCT)
                    index_type_str = f"IndexIVFFlat(nlist={effective_nlist})"

                # Train on sample or full mmap
                train_size = min(total_vectors, 100000)
                train_sample = np.ascontiguousarray(emb_mmap[:train_size])
                index.train(train_sample)
                del train_sample
                gc.collect()

                # Add vectors in chunks
                chunk_add = 50000
                for i in range(0, total_vectors, chunk_add):
                    chunk = np.ascontiguousarray(emb_mmap[i : i + chunk_add])
                    index.add(chunk)
                    del chunk

            print(f"[FAISSIndex] Writing index to disk: {index_path}...")
            faiss.write_index(index, index_path)
            del index
            gc.collect()

        else:
            # Fallback: Save metadata referencing the mmap for direct IP dot-product search
            index_type_str = "NumPy_FlatIP_Fallback"
            # Create a marker file for target.index
            with open(index_path, "w") as f:
                f.write(f"NumPy_FlatIP_Fallback: {embeddings_path}\n")

    metadata = {
        "index_type": index_type_str,
        "metric": "INNER_PRODUCT",
        "embedding_dimension": dim,
        "target_count": total_vectors,
        "source_embeddings": embeddings_path,
        "source_ids": ids_path,
        "creation_timestamp": datetime.now(timezone.utc).isoformat()
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    total_time = time.time() - t0
    print(f"\n[FAISSIndex] Successfully built ANN index in {total_time:.2f}s.")
    print(f"  Index Type: {index_type_str}")
    print(f"  Final RSS:  {get_current_rss_mb():.2f} MB | Peak RSS: {get_peak_rss_mb():.2f} MB")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Antigravity V4.1 FAISS ANN Index Builder")
    parser.add_argument("--embeddings-dir", type=str, default="cache/embeddings/arctic", help="Embeddings directory")
    parser.add_argument("--output-dir", type=str, default="cache/ann/arctic", help="Output ANN directory")
    parser.add_argument("--nlist", type=int, default=1024, help="Number of Voronoi cells for IVF index")
    parser.add_argument("--use-pq", action="store_true", help="Use Product Quantization compression")
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
        use_pq=args.use_pq
    )

if __name__ == "__main__":
    main()
