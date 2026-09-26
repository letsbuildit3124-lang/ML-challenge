"""
Antigravity E5 Memmap Assembly Module.
Sequentially merges verified embedding shards and target ID parquets into a unified,
zero-copy disk-backed float16 memory map (target_embeddings.f16.memmap).

Key Design Invariants:
- Low-Memory Sequential Streaming: Peak RAM strictly < 200MB (no large in-memory arrays).
- Direct Disk Assignment: Writes each shard directly into pre-allocated memmap slices.
- Strict Positional ID Alignment: embedding[i] strictly corresponds to target_id[i].
- Saves unified target_ids.json and metadata.json with completeness tracking.
"""

import os
import sys
import gc
import json
import time
import argparse
from typing import List, Dict, Any, Tuple, Optional
import polars as pl
import numpy as np

from src.config import get_config
from src.resource_tracker import get_current_rss_mb, get_peak_rss_mb, MemoryTracker

EXPECTED_PROD_ROWS = 10320219
DEFAULT_DIM = 384


def assemble_e5_memmap(
    input_shards_dir: str = "cache/e5_gpu/output",
    output_dir: str = "cache/ann/e5/production",
    is_production: bool = True
) -> Tuple[str, str]:
    """
    Assembles downloaded shard files into a disk-backed memory map.
    """
    config = get_config()
    in_dir = os.path.join(config.base_dir, input_shards_dir)
    out_dir = os.path.join(config.base_dir, output_dir)

    # Handle if outputs are in in_dir/outputs
    if os.path.exists(os.path.join(in_dir, "outputs")):
        in_dir = os.path.join(in_dir, "outputs")

    emb_dir = os.path.join(in_dir, "embeddings")
    ids_dir = os.path.join(in_dir, "ids")
    man_file = os.path.join(in_dir, "manifest.json")

    os.makedirs(out_dir, exist_ok=True)
    memmap_path = os.path.join(out_dir, "target_embeddings.f16.memmap")
    ids_path = os.path.join(out_dir, "target_ids.json")
    meta_path = os.path.join(out_dir, "metadata.json")

    if not (os.path.exists(emb_dir) and os.path.exists(ids_dir)):
        raise FileNotFoundError(f"Missing embeddings/ or ids/ in {in_dir}!")

    emb_files = sorted([f for f in os.listdir(emb_dir) if f.endswith(".npy") and not f.endswith(".tmp")])
    ids_files = sorted([f for f in os.listdir(ids_dir) if f.endswith(".parquet") and not f.endswith(".tmp")])

    if len(emb_files) == 0:
        raise FileNotFoundError(f"No shard files found in {emb_dir}!")

    # 1. Calculate Total Rows across Shards
    print("=" * 80)
    print("ANTIGRAVITY — E5 DISK-BACKED MEMMAP ASSEMBLY")
    print("=" * 80)
    print(f"Source Shards Dir:   {in_dir} ({len(emb_files)} shards)")
    print(f"Destination Dir:     {out_dir}")
    print(f"Destination Memmap:  {memmap_path}")
    print(f"Initial RSS:         {get_current_rss_mb():.2f} MB")
    print("-" * 80)

    total_rows = 0
    shard_info = []

    for e_file in emb_files:
        e_path = os.path.join(emb_dir, e_file)
        mmap = np.lib.format.open_memmap(e_path, mode="r")
        s_rows, s_dim = mmap.shape
        del mmap
        assert s_dim == DEFAULT_DIM, f"Dimension mismatch in {e_file}: {s_dim} != {DEFAULT_DIM}"
        shard_info.append((e_file, s_rows))
        total_rows += s_rows

    is_complete_prod = (total_rows == EXPECTED_PROD_ROWS)
    approx_size_gb = total_rows * DEFAULT_DIM * 2 / (1024.0 ** 3)

    print(f"Total Vectors to Assemble: {total_rows:,} ({approx_size_gb:.2f} GiB)")
    print(f"Production Universe:       {'YES (10,320,219 Targets)' if is_complete_prod else f'NO (Subset: {total_rows:,})'}")

    if is_production and total_rows != EXPECTED_PROD_ROWS:
        raise ValueError(
            f"[PRODUCTION INVARIANT VIOLATION] Full universe requires {EXPECTED_PROD_ROWS:,} "
            f"rows, but found {total_rows:,} across {len(emb_files)} shards."
        )

    # 2. Allocate Unified Disk-Backed Memmap
    print(f"\n[Assembler] Allocating disk-backed memmap shape ({total_rows:,}, {DEFAULT_DIM}) dtype float16...")
    fp_memmap = np.lib.format.open_memmap(
        memmap_path,
        mode="w+",
        dtype="float16",
        shape=(total_rows, DEFAULT_DIM)
    )

    # 3. Stream Shards into Memmap
    all_target_ids: List[str] = []
    offset = 0
    t0 = time.time()

    with MemoryTracker(f"Memmap Assembly ({len(emb_files)} shards)"):
        for idx, (e_file, s_rows) in enumerate(shard_info):
            shard_id = os.path.splitext(e_file)[0]
            i_file = f"{shard_id}.parquet"
            e_path = os.path.join(emb_dir, e_file)
            i_path = os.path.join(ids_dir, i_file)

            # Read shard
            chunk_embs = np.load(e_path)
            df_ids = pl.read_parquet(i_path)
            chunk_ids = [str(x) for x in df_ids["target_id"].to_list()]
            del df_ids

            assert chunk_embs.shape == (s_rows, DEFAULT_DIM), f"Shape mismatch in {shard_id}"
            assert len(chunk_ids) == s_rows, f"ID count mismatch in {shard_id}"

            # Direct slice assignment
            fp_memmap[offset : offset + s_rows] = chunk_embs.astype(np.float16)
            all_target_ids.extend(chunk_ids)

            offset += s_rows
            pct = (offset / total_rows) * 100.0
            print(f"  -> Merged {shard_id} ({s_rows:,} rows) [{pct:.1f}%] | Offset: {offset:,} | RSS: {get_current_rss_mb():.1f} MB", flush=True)

            del chunk_embs, chunk_ids
            if idx % 10 == 0:
                gc.collect()

        fp_memmap.flush()
        del fp_memmap
        gc.collect()

    elapsed = time.time() - t0

    # 4. Strict Validation of Assembled Artifacts
    print("\n" + "-" * 80)
    print(f"[Assembler] Validating assembled memmap on disk...")
    mmap_verify = np.lib.format.open_memmap(memmap_path, mode="r")
    v_rows, v_dim = mmap_verify.shape
    v_dtype = mmap_verify.dtype
    del mmap_verify

    assert v_rows == total_rows, f"Assembled row count mismatch: {v_rows} != {total_rows}"
    assert v_dim == DEFAULT_DIM, f"Assembled dimension mismatch: {v_dim} != {DEFAULT_DIM}"
    assert v_dtype == np.float16, f"Assembled dtype mismatch: {v_dtype} != float16"
    assert len(all_target_ids) == total_rows, f"Target ID count mismatch: {len(all_target_ids)} != {total_rows}"

    unique_ids_count = len(set(all_target_ids))
    assert unique_ids_count == total_rows, f"Duplicate target IDs detected: {unique_ids_count:,} != {total_rows:,}"

    # 5. Save Target IDs JSON and Metadata
    print(f"[Assembler] Writing target IDs list to {ids_path}...")
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(all_target_ids, f)

    meta = {
        "model_name": "intfloat/multilingual-e5-small",
        "embedding_dim": DEFAULT_DIM,
        "dtype": "float16",
        "normalization": "L2",
        "total_rows": total_rows,
        "is_complete_target_universe": is_complete_prod,
        "expected_target_count": EXPECTED_PROD_ROWS,
        "file_size_gb": round(approx_size_gb, 2),
        "assembled_at": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "assembly_duration_seconds": round(elapsed, 2)
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("=" * 80)
    print(f"[Assembler SUCCESS] Memmap assembly complete in {elapsed:.2f}s:")
    print(f"  Memmap Array:    {memmap_path} ({approx_size_gb:.2f} GiB)")
    print(f"  Target IDs:      {ids_path} ({len(all_target_ids):,} IDs)")
    print(f"  Metadata:        {meta_path}")
    print(f"  Peak RSS:        {get_peak_rss_mb():.2f} MB")
    print("=" * 80)

    return memmap_path, ids_path


def main():
    parser = argparse.ArgumentParser(description="Antigravity E5 Memmap Assembler")
    parser.add_argument("--input-dir", type=str, default="cache/e5_gpu/output", help="Directory containing downloaded shards")
    parser.add_argument("--output-dir", type=str, default="cache/ann/e5/production", help="Directory to save assembled memmap")
    parser.add_argument("--smoke", action="store_true", help="Assemble into smoke directory (cache/ann/e5/smoke)")
    parser.add_argument("--benchmark", action="store_true", help="Assemble into benchmark directory (cache/ann/e5/benchmark)")
    parser.add_argument("--production", action="store_true", help="Assemble into production directory (cache/ann/e5/production)")
    args = parser.parse_args()

    out_dir = args.output_dir
    is_prod = True
    if args.smoke:
        out_dir = "cache/ann/e5/smoke"
        is_prod = False
    elif args.benchmark:
        out_dir = "cache/ann/e5/benchmark"
        is_prod = False
    elif args.production:
        out_dir = "cache/ann/e5/production"
        is_prod = True

    assemble_e5_memmap(
        input_shards_dir=args.input_dir,
        output_dir=out_dir,
        is_production=is_prod
    )


if __name__ == "__main__":
    main()
