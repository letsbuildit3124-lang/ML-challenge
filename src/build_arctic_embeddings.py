"""
Antigravity V3 Disk-Backed Arctic Embedding Builder.
Generates memory-mapped 384-d normalized embeddings for S1, S2, and S3 entities.

Stores outputs in:
cache/arctic/
  ├── {split}_{source}_embeddings.npy  (or .mmap)
  └── {split}_{source}_ids.json

Designed for low memory footprint (< 500MB RAM) on EC2 2-vCPU / 8GB RAM instances.
"""

import os
import sys
import gc
import json
import time
import argparse
from typing import List, Dict, Any, Optional
import polars as pl
import numpy as np

from src.config import get_config
from src.arctic_embeddings import ArcticEmbedder, format_entity_text, EMBEDDING_DIM

def build_embeddings_for_file(
    input_file: str,
    output_prefix: str,
    embedder: ArcticEmbedder,
    expected_prefix: str = "S1-",
    batch_size: int = 128,
    limit: Optional[int] = None,
    chunk_rows: int = 50000
):
    """
    Reads a TSV file in chunks, computes normalized Arctic embeddings,
    and saves them incrementally to disk.
    """
    print(f"\n" + "=" * 70)
    print(f"[ArcticBuilder] Processing file: {input_file}")
    print(f"Target Output: {output_prefix}_embeddings.npy & {output_prefix}_ids.json")
    print(f"=" * 70)

    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist!")
        return

    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    t0 = time.time()

    # Determine total rows using polars scan
    lf = pl.scan_csv(input_file, separator="\t", infer_schema_length=1000, ignore_errors=True)
    total_rows = lf.select(pl.len()).collect().item()
    if limit and limit < total_rows:
        total_rows = limit

    print(f"[ArcticBuilder] Total entities to encode: {total_rows:,} (Chunk size: {chunk_rows:,})")

    ids_list = []
    # Pre-allocate numpy memmap on disk for zero RAM spike
    mmap_path = f"{output_prefix}_embeddings.npy"
    # Using np.lib.format.open_memmap for standard .npy header
    fp = np.lib.format.open_memmap(
        mmap_path,
        mode='w+',
        dtype='float32',
        shape=(total_rows, EMBEDDING_DIM)
    )

    processed_count = 0
    start_time = time.time()

    # Read in chunks using Polars
    offset = 0
    while offset < total_rows:
        current_chunk_size = min(chunk_rows, total_rows - offset)
        df_chunk = (
            pl.read_csv(
                input_file,
                separator="\t",
                infer_schema_length=1000,
                ignore_errors=True,
                skip_rows=offset,
                n_rows=current_chunk_size
            )
        )

        # Standardize column names
        cols = {c.lower(): c for c in df_chunk.columns}
        id_col = cols.get("entity_id") or cols.get("record_id") or cols.get("id") or df_chunk.columns[0]
        name_col = cols.get("business_name") or cols.get("name") or df_chunk.columns[1]
        addr_col = cols.get("business_address") or cols.get("address") or (df_chunk.columns[2] if len(df_chunk.columns) > 2 else None)
        ctry_col = cols.get("country") if "country" in cols else None

        chunk_ids = [str(x) for x in df_chunk[id_col].to_list()]
        chunk_names = [str(x) if x is not None else "" for x in df_chunk[name_col].to_list()]
        chunk_addrs = [str(x) if x is not None else "" for x in df_chunk[addr_col].to_list()] if addr_col else [""] * len(chunk_ids)
        chunk_ctrys = [str(x) if x is not None else "" for x in df_chunk[ctry_col].to_list()] if ctry_col else [""] * len(chunk_ids)

        # Format texts
        texts = [
            format_entity_text(n, a, c)
            for n, a, c in zip(chunk_names, chunk_addrs, chunk_ctrys)
        ]

        # Compute embeddings in mini-batches
        emb_chunk = embedder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )

        # Write directly to memmap
        fp[offset : offset + current_chunk_size] = emb_chunk
        ids_list.extend(chunk_ids)

        offset += current_chunk_size
        processed_count += current_chunk_size
        elapsed = time.time() - start_time
        speed = processed_count / elapsed if elapsed > 0 else 0
        pct = (processed_count / total_rows) * 100.0
        print(f"  -> Encoded {processed_count:,}/{total_rows:,} ({pct:.1f}%) | Speed: {speed:.1f} entities/sec | Elapsed: {elapsed:.1f}s", flush=True)

        del df_chunk, texts, emb_chunk
        gc.collect()

    # Flush memmap to disk
    del fp
    gc.collect()

    # Save ID list mapping
    ids_json_path = f"{output_prefix}_ids.json"
    with open(ids_json_path, "w", encoding="utf-8") as f:
        json.dump(ids_list, f)

    total_time = time.time() - t0
    print(f"[ArcticBuilder] Completed {output_prefix} in {total_time:.2f}s ({processed_count / total_time:.1f} ent/sec).")
    print(f"  Artifacts: {mmap_path} ({os.path.getsize(mmap_path) / (1024*1024):.2f} MB), {ids_json_path}")


def main():
    parser = argparse.ArgumentParser(description="Antigravity V3 Arctic Embedding Cache Builder")
    parser.add_argument("--dataset", choices=["train", "test", "all"], default="train", help="Dataset split")
    parser.add_argument("--source", choices=["s1", "s2", "s3", "targets", "all"], default="all", help="Source files")
    parser.add_argument("--batch-size", type=int, default=128, help="Inference batch size")
    parser.add_argument("--limit", type=int, default=None, help="Optional max row limit per file")
    parser.add_argument("--output-dir", type=str, default="cache/arctic", help="Output directory for embeddings")
    parser.add_argument("--model", type=str, default=None, help="Model name or local path")
    args = parser.parse_args()

    config = get_config()
    out_dir = os.path.join(config.base_dir, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("ANTIGRAVITY V3 — ARCTIC EMBEDDING CACHE GENERATOR")
    print("=" * 80)
    print(f"Target Directory: {out_dir}")
    print(f"Batch Size: {args.batch_size} | Limit: {args.limit}")

    embedder = ArcticEmbedder(
        model_name_or_path=args.model or "themelder/arctic-embed-xs-entity-resolution",
        batch_size=args.batch_size
    )

    # Train Sources
    if args.dataset in ["train", "all"]:
        if args.source in ["s1", "all"]:
            build_embeddings_for_file(
                config.train_s1_path,
                os.path.join(out_dir, "train_s1"),
                embedder,
                expected_prefix="S1-",
                batch_size=args.batch_size,
                limit=args.limit
            )
        if args.source in ["s2", "targets", "all"]:
            build_embeddings_for_file(
                config.train_s2_path,
                os.path.join(out_dir, "train_s2"),
                embedder,
                expected_prefix="S2-",
                batch_size=args.batch_size,
                limit=args.limit
            )
        if args.source in ["s3", "targets", "all"]:
            build_embeddings_for_file(
                config.train_s3_path,
                os.path.join(out_dir, "train_s3"),
                embedder,
                expected_prefix="S3-",
                batch_size=args.batch_size,
                limit=args.limit
            )

    # Test Sources
    if args.dataset in ["test", "all"]:
        if args.source in ["s1", "all"]:
            build_embeddings_for_file(
                config.test_s1_path,
                os.path.join(out_dir, "test_s1"),
                embedder,
                expected_prefix="S1-",
                batch_size=args.batch_size,
                limit=args.limit
            )
        if args.source in ["s2", "targets", "all"]:
            build_embeddings_for_file(
                config.test_s2_path,
                os.path.join(out_dir, "test_s2"),
                embedder,
                expected_prefix="S2-",
                batch_size=args.batch_size,
                limit=args.limit
            )
        if args.source in ["s3", "targets", "all"]:
            build_embeddings_for_file(
                config.test_s3_path,
                os.path.join(out_dir, "test_s3"),
                embedder,
                expected_prefix="S3-",
                batch_size=args.batch_size,
                limit=args.limit
            )

    print("\n[ArcticBuilder] Embedding generation completed successfully.")

if __name__ == "__main__":
    main()
