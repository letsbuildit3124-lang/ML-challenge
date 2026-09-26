"""
Antigravity V4.1 Persistent Arctic Target Embedding Builder.
Generates chunked, disk-backed L2-normalized 384-d embeddings for target records (S2 + S3).

Features:
- Resumable Embedding Generation with Atomic Checkpoints
- Strict Positional Target ID Alignment (ORDER BY target_row_id)
- Pre-flight Disk Space Validation
- Bounded RAM Footprint via Disk-Backed Memmap & Streaming DuckDB Chunks
- Strict Index Completeness Metadata Tracking
"""

import os
import sys
import gc
import json
import time
import shutil
import argparse
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
import duckdb
import numpy as np

from src.config import get_config
from src.resource_tracker import get_current_rss_mb, get_peak_rss_mb, MemoryTracker
from src.arctic_embeddings import ArcticEmbedder, format_entity_text, EMBEDDING_DIM, DEFAULT_MODEL_NAME

EXPECTED_TOTAL_TARGETS = 10320219
METADATA_VERSION = "v1_name_addr_ctry"
TARGET_CACHE_VERSION = "v4_duckdb"

def check_disk_space(output_dir: str, required_gb: float = 20.0) -> bool:
    """Verifies that sufficient free disk space exists before starting."""
    try:
        total, used, free = shutil.disk_usage(output_dir if os.path.exists(output_dir) else ".")
        free_gb = free / (1024.0 ** 3)
        print(f"[DiskCheck] Available Disk Space: {free_gb:.2f} GB (Required: {required_gb:.2f} GB)")
        if free_gb < required_gb:
            print(f"[ERROR] Insufficient disk space! Shortfall: {required_gb - free_gb:.2f} GB. Aborting.")
            return False
        return True
    except Exception as e:
        print(f"[DiskCheck] Warning: Could not verify disk space ({e}). Proceeding.")
        return True


def check_existing_cache(
    cache_dir: str,
    target_count: int,
    model_name: str = DEFAULT_MODEL_NAME,
    embedding_dim: int = EMBEDDING_DIM,
    dtype: str = "float32"
) -> bool:
    """Validates existing embedding cache against metadata."""
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

        is_complete = meta.get("is_complete_target_universe", False)
        print(f"[ArcticEmbeddingCache] Existing persistent embeddings verified ({target_count:,} records | Complete: {is_complete}). Reusing cache!")
        return True
    except Exception as e:
        print(f"[ArcticEmbeddingCache] Error reading metadata: {e}")
        return False


def build_arctic_embeddings_from_duckdb(
    db_path: str,
    output_dir: str,
    embedder: ArcticEmbedder,
    batch_size: int = 128,
    chunk_rows: int = 10000,
    limit: Optional[int] = None
):
    """
    Streams target records from DuckDB, computes L2-normalized Arctic embeddings,
    and writes directly to a disk-backed np.memmap with checkpoint resumability.
    """
    os.makedirs(output_dir, exist_ok=True)
    mmap_path = os.path.join(output_dir, "target_embeddings.npy")
    ids_path = os.path.join(output_dir, "target_ids.json")
    meta_path = os.path.join(output_dir, "metadata.json")
    checkpoint_path = os.path.join(output_dir, "checkpoint.json")

    print("\n" + "=" * 80)
    print("ANTIGRAVITY V4.1 — PERSISTENT ARCTIC TARGET EMBEDDING BUILDER")
    print("=" * 80)
    print(f"Source DuckDB:    {db_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Batch Size:       {batch_size} | Chunk Rows: {chunk_rows:,}")
    print(f"Initial RSS:      {get_current_rss_mb():.2f} MB")
    print("-" * 80)

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Target DuckDB database not found at {db_path}!")

    conn = duckdb.connect(db_path, read_only=True)
    total_in_db = conn.execute("SELECT COUNT(*) FROM targets;").fetchone()[0]
    total_rows = min(total_in_db, limit) if limit else total_in_db
    is_complete_build = (total_rows == EXPECTED_TOTAL_TARGETS)

    # Estimate disk requirements
    required_bytes = total_rows * EMBEDDING_DIM * 4 # float32
    required_gb = (required_bytes / (1024.0 ** 3)) + 4.0 # with 4GB safety buffer
    print(f"Target Records to Process: {total_rows:,} / {total_in_db:,} (Expected Total: {EXPECTED_TOTAL_TARGETS:,})")
    print(f"Complete Universe Build:   {'YES (Production)' if is_complete_build else 'NO (Smoke / Dev Test)'}")
    print(f"Estimated Memmap Footprint: {required_bytes / (1024.0 ** 3):.2f} GB on Disk (NOT RAM)")

    if not check_disk_space(output_dir, required_gb=required_gb):
        conn.close()
        sys.exit(1)

    # Check existing valid cache
    if check_existing_cache(output_dir, total_rows, embedder.model_name_or_path, EMBEDDING_DIM):
        conn.close()
        return

    # Check for resumable checkpoint
    offset = 0
    ids_list: List[str] = []
    if os.path.exists(checkpoint_path) and os.path.exists(mmap_path) and os.path.exists(ids_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
            if ckpt.get("target_count") == total_rows and ckpt.get("model_name") == embedder.model_name_or_path:
                offset = ckpt.get("processed_rows", 0)
                with open(ids_path, "r", encoding="utf-8") as f:
                    ids_list = json.load(f)
                if len(ids_list) == offset:
                    print(f"[ArcticBuilder] >>> Resuming interrupted embedding build from row {offset:,}/{total_rows:,} <<<")
                else:
                    print("[ArcticBuilder] Checkpoint ID count mismatch. Restarting from 0.")
                    offset = 0
                    ids_list = []
        except Exception as e:
            print(f"[ArcticBuilder] Checkpoint load failed ({e}). Starting fresh.")
            offset = 0
            ids_list = []

    # Open or create memmap
    mode = "r+" if (offset > 0 and os.path.exists(mmap_path)) else "w+"
    print(f"[ArcticBuilder] Opening disk-backed memmap (mode={mode}): {mmap_path}")
    fp = np.lib.format.open_memmap(
        mmap_path,
        mode=mode,
        dtype="float32",
        shape=(total_rows, EMBEDDING_DIM)
    )

    processed_count = offset
    t0 = time.time()
    last_save_time = time.time()

    with MemoryTracker(f"Embedding Generation ({total_rows:,} records)"):
        while offset < total_rows:
            chunk_n = min(chunk_rows, total_rows - offset)
            # Strictly ordered by target_row_id for 100% deterministic positional alignment
            query = f"""
                SELECT eid, norm_name, norm_addr, country 
                FROM targets 
                ORDER BY target_row_id 
                LIMIT {chunk_n} OFFSET {offset};
            """
            rows = conn.execute(query).fetchall()

            chunk_ids = []
            chunk_texts = []
            for r in rows:
                eid, norm_name, norm_addr, ctry = r
                b_name = norm_name or ""
                b_addr = norm_addr or ""
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
            speed = (processed_count - (ckpt.get("processed_rows", 0) if "ckpt" in locals() else 0)) / elapsed if elapsed > 0 else 0
            pct = (processed_count / total_rows) * 100.0
            print(f"  -> Encoded {processed_count:,}/{total_rows:,} ({pct:.1f}%) | Speed: {speed:.1f} ent/s | RSS: {get_current_rss_mb():.1f} MB", flush=True)

            # Periodic checkpoint save (every 50,000 rows or 2 minutes)
            if (offset % 50000 == 0) or (time.time() - last_save_time > 120.0):
                fp.flush()
                with open(ids_path, "w", encoding="utf-8") as f:
                    json.dump(ids_list, f)
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "processed_rows": offset,
                        "target_count": total_rows,
                        "model_name": embedder.model_name_or_path,
                        "last_updated": datetime.now(timezone.utc).isoformat()
                    }, f)
                last_save_time = time.time()

            del rows, chunk_ids, chunk_texts, emb_chunk
            gc.collect()

        # Final flush
        fp.flush()
        del fp
        gc.collect()

    conn.close()

    # Save final ID list
    print(f"[ArcticBuilder] Saving final target ID index ({len(ids_list):,} entries) to {ids_path}...")
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
        "expected_target_count": EXPECTED_TOTAL_TARGETS,
        "is_complete_target_universe": is_complete_build,
        "target_cache_version": TARGET_CACHE_VERSION,
        "text_representation_version": METADATA_VERSION,
        "creation_timestamp": datetime.now(timezone.utc).isoformat()
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Remove checkpoint on successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    total_time = time.time() - t0
    print(f"\n[ArcticBuilder] Successfully built persistent Arctic target embeddings in {total_time:.2f}s.")
    print(f"  Complete Universe: {'YES' if is_complete_build else 'NO (SMOKE TEST ONLY)'}")
    print(f"  Embeddings:        {mmap_path} ({os.path.getsize(mmap_path) / (1024*1024):.2f} MB)")
    print(f"  Metadata:          {meta_path}")
    print(f"  Final RSS:         {get_current_rss_mb():.2f} MB | Peak RSS: {get_peak_rss_mb():.2f} MB")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Antigravity V4.1 Arctic Embedding Builder")
    parser.add_argument("--limit", type=int, default=1000, help="Row limit for target embedding generation (default: 1000 for smoke testing)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for Arctic encoding")
    parser.add_argument("--chunk-rows", type=int, default=10000, help="Rows per DuckDB stream chunk")
    parser.add_argument("--output-dir", type=str, default="cache/embeddings/arctic", help="Output directory")
    parser.add_argument("--full", action="store_true", help="Explicit flag required to run full 10,320,219 target embedding build")
    args = parser.parse_args()

    config = get_config()
    db_path = os.path.join(config.base_dir, "cache", "entity_resolution.duckdb")
    out_dir = os.path.join(config.base_dir, args.output_dir)

    if args.full:
        effective_limit = None
        print("\n>>> FULL PRODUCTION TARGET UNIVERSE BUILD REQUESTED (10,320,219 targets) <<<\n")
    else:
        effective_limit = args.limit
        print(f"\n>>> DEVELOPMENT SMOKE TEST BUILD (Limit: {effective_limit:,} targets) <<<\n")

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
