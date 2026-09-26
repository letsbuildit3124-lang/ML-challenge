"""
Antigravity V4.1 Persistent Arctic Target Embedding Builder.
Generates chunked, disk-backed L2-normalized 384-d embeddings for target records (S2 + S3).

Stores outputs in:
cache/embeddings/arctic/
  ├── target_embeddings.npy  (or .mmap, shape (N, 384), float32)
  ├── target_ids.json        (List of entity_ids corresponding to row index)
  └── metadata.json          (Validation schema for cache reuse & integrity)

Guarantees:
- Strict bounded RAM via disk-backed np.memmap and chunked DuckDB stream
- Accurate process-level RSS instrumentation
- Safe metadata validation (Refuses silent reuse on incompatible configs)
"""

import os
import sys
import gc
import json
import time
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import duckdb
import polars as pl
import numpy as np

from src.config import get_config
from src.resource_tracker import get_current_rss_mb, get_peak_rss_mb, MemoryTracker
from src.arctic_embeddings import ArcticEmbedder, format_entity_text, EMBEDDING_DIM, DEFAULT_MODEL_NAME

METADATA_VERSION = "v1_name_addr_ctry"
TARGET_CACHE_VERSION = "v4_duckdb"

def check_existing_cache(
    cache_dir: str,
    target_count: int,
    model_name: str = DEFAULT_MODEL_NAME,
    embedding_dim: int = EMBEDDING_DIM,
    dtype: str = "float32"
) -> bool:
    """
    Validates existing embedding cache against metadata.
    Returns True if valid for reuse, False otherwise.
    """
    meta_path = os.path.join(cache_dir, "metadata.json")
    emb_path = os.path.join(cache_dir, "target_embeddings.npy")
    ids_path = os.path.join(cache_dir, "target_ids.json")

    if not (os.path.exists(meta_path) and os.path.exists(emb_path) and os.path.exists(ids_path)):
        return False

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        diffs = []
        if meta.get("model_name") != model_name:
            diffs.append(f"model_name: {meta.get('model_name')} != {model_name}")
        if meta.get("embedding_dimension") != embedding_dim:
            diffs.append(f"embedding_dimension: {meta.get('embedding_dimension')} != {embedding_dim}")
        if meta.get("dtype") != dtype:
            diffs.append(f"dtype: {meta.get('dtype')} != {dtype}")
        if meta.get("normalization") != "L2":
            diffs.append(f"normalization: {meta.get('normalization')} != L2")
        if meta.get("target_count") != target_count:
            diffs.append(f"target_count: {meta.get('target_count')} != {target_count}")
        if meta.get("text_representation_version") != METADATA_VERSION:
            diffs.append(f"text_rep_version: {meta.get('text_representation_version')} != {METADATA_VERSION}")

        if diffs:
            print(f"[ArcticEmbeddingCache] Cache metadata mismatch detected:")
            for d in diffs:
                print(f"  - {d}")
            print("[ArcticEmbeddingCache] Refusing silent reuse. Cache must be regenerated.")
            return False

        print(f"[ArcticEmbeddingCache] Existing persistent embeddings verified ({target_count:,} records). Reusing cache!")
        return True
    except Exception as e:
        print(f"[ArcticEmbeddingCache] Error reading metadata: {e}")
        return False


def build_arctic_embeddings_from_duckdb(
    db_path: str,
    output_dir: str,
    embedder: ArcticEmbedder,
    batch_size: int = 128,
    chunk_rows: int = 50000,
    limit: Optional[int] = None
):
    """
    Streams target records from persistent DuckDB cache in chunks,
    generates L2-normalized Arctic embeddings, and writes directly to disk memmap.
    """
    os.makedirs(output_dir, exist_ok=True)
    mmap_path = os.path.join(output_dir, "target_embeddings.npy")
    ids_path = os.path.join(output_dir, "target_ids.json")
    meta_path = os.path.join(output_dir, "metadata.json")

    print("\n" + "=" * 80)
    print("ANTIGRAVITY V4.1 — PERSISTENT ARCTIC TARGET EMBEDDING BUILDER")
    print("=" * 80)
    print(f"Source DuckDB:    {db_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Batch Size:       {batch_size} | Chunk Rows: {chunk_rows:,}")
    print(f"Limit:            {limit if limit else 'ALL (Full targets universe)'}")
    print(f"Initial RSS:      {get_current_rss_mb():.2f} MB")
    print("-" * 80)

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Target DuckDB database not found at {db_path}!")

    conn = duckdb.connect(db_path, read_only=True)

    # Determine total target rows
    total_in_db = conn.execute("SELECT COUNT(*) FROM targets;").fetchone()[0]
    total_rows = min(total_in_db, limit) if limit else total_in_db

    print(f"Total Targets in DB: {total_in_db:,} | Target rows to process: {total_rows:,}")

    # Check existing valid cache
    if check_existing_cache(output_dir, total_rows, embedder.model_name_or_path, EMBEDDING_DIM):
        conn.close()
        return

    # Pre-allocate numpy memmap on disk
    print(f"[ArcticBuilder] Allocating disk-backed memmap: {mmap_path} (Shape: ({total_rows:,}, {EMBEDDING_DIM}))")
    fp = np.lib.format.open_memmap(
        mmap_path,
        mode="w+",
        dtype="float32",
        shape=(total_rows, EMBEDDING_DIM)
    )

    ids_list: List[str] = []
    processed_count = 0
    t0 = time.time()

    with MemoryTracker(f"Embedding Generation ({total_rows:,} records)"):
        offset = 0
        while offset < total_rows:
            chunk_n = min(chunk_rows, total_rows - offset)
            query = f"""
                SELECT eid, name, norm_name, addr, norm_addr, country 
                FROM targets 
                ORDER BY target_row_id 
                LIMIT {chunk_n} OFFSET {offset};
            """
            rows = conn.execute(query).fetchall()

            chunk_ids = []
            chunk_texts = []
            for r in rows:
                eid, raw_name, norm_name, raw_addr, norm_addr, ctry = r
                b_name = norm_name or raw_name or ""
                b_addr = norm_addr or raw_addr or ""
                b_ctry = ctry or ""
                chunk_ids.append(eid)
                chunk_texts.append(format_entity_text(b_name, b_addr, b_ctry))

            # Encode chunk with Arctic embedder
            emb_chunk = embedder.encode(
                chunk_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True
            )

            # Assign directly into memmap
            fp[offset : offset + len(chunk_ids)] = emb_chunk
            ids_list.extend(chunk_ids)

            offset += len(chunk_ids)
            processed_count += len(chunk_ids)

            elapsed = time.time() - t0
            speed = processed_count / elapsed if elapsed > 0 else 0
            pct = (processed_count / total_rows) * 100.0
            print(f"  -> Encoded {processed_count:,}/{total_rows:,} ({pct:.1f}%) | Speed: {speed:.1f} ent/s | RSS: {get_current_rss_mb():.1f} MB", flush=True)

            del rows, chunk_ids, chunk_texts, emb_chunk
            gc.collect()

        # Flush memmap
        del fp
        gc.collect()

    conn.close()

    # Save IDs mapping
    print(f"[ArcticBuilder] Saving target ID index to {ids_path}...")
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(ids_list, f)

    # Save metadata JSON
    metadata = {
        "model_name": embedder.model_name_or_path,
        "model_revision": "main",
        "embedding_dimension": EMBEDDING_DIM,
        "dtype": "float32",
        "normalization": "L2",
        "target_count": total_rows,
        "target_cache_version": TARGET_CACHE_VERSION,
        "text_representation_version": METADATA_VERSION,
        "creation_timestamp": datetime.now(timezone.utc).isoformat()
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    total_time = time.time() - t0
    print(f"\n[ArcticBuilder] Successfully built persistent Arctic target embeddings in {total_time:.2f}s.")
    print(f"  Embeddings: {mmap_path} ({os.path.getsize(mmap_path) / (1024*1024):.2f} MB)")
    print(f"  Metadata:   {meta_path}")
    print(f"  Final RSS:  {get_current_rss_mb():.2f} MB | Peak RSS: {get_peak_rss_mb():.2f} MB")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Antigravity V4.1 Arctic Embedding Builder")
    parser.add_argument("--limit", type=int, default=1000, help="Row limit for target embedding generation (default: 1000 for smoke testing)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for Arctic encoding")
    parser.add_argument("--chunk-rows", type=int, default=10000, help="Rows per DuckDB stream chunk")
    parser.add_argument("--output-dir", type=str, default="cache/embeddings/arctic", help="Output directory")
    parser.add_argument("--full", action="store_true", help="Explicit flag required to run full 10.3M embedding build")
    args = parser.parse_args()

    config = get_config()
    db_path = os.path.join(config.base_dir, "cache", "entity_resolution.duckdb")
    out_dir = os.path.join(config.base_dir, args.output_dir)

    if args.limit is None and not args.full:
        print("\n[SAFETY GUARD ERROR]: Full 10.3M target embedding generation requires explicit '--full' flag.")
        print("For controlled testing, use e.g. '--limit 1000' or '--limit 10000'. Exiting.\n")
        sys.exit(1)

    effective_limit = args.limit if not args.full else None

    embedder = ArcticEmbedder(
        model_name_or_path=DEFAULT_MODEL_NAME,
        batch_size=args.batch_size
    )

    build_arctic_embeddings_from_duckdb(
        db_path=db_path,
        output_dir=out_dir,
        embedder=embedder,
        batch_size=args.batch_size,
        chunk_rows=args.chunk_rows,
        limit=effective_limit
    )

if __name__ == "__main__":
    main()
