"""
Antigravity One-Shot Kaggle GPU Multilingual E5 Embedding Worker.
Runs the entire 10,320,219 target embedding workload in ONE single Kaggle GPU notebook/kernel.

Key Architectural Invariants:
1. One-Shot Lifecycle: Model & Tokenizer loaded ONCE to CUDA.
2. High-Throughput Sharding: Processes target data in ~100,000-row shards sequentially.
3. Adaptive Batch Size: Dynamically tunes GPU batch size (2048 -> 1024 -> 512 -> 256) on OOM.
4. FP16 Storage: Serializes embeddings in float16 (384-d, L2 normalized, ~7.38 GB total).
5. Exact ID Alignment: embedding[i] strictly corresponds to target_id[i].
6. Atomic Resumability: Writes temporary files and renames upon validation; skips completed shards.
7. Structured Manifest: Generates root manifest.json with full telemetry and per-shard checksums.
"""

import os
import sys
import gc
import json
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
import polars as pl
import numpy as np

MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384
DEFAULT_SHARD_SIZE = 100000
DEFAULT_MAX_LENGTH = 128
BATCH_CANDIDATES = [2048, 1024, 512, 256]


def compute_sha256(file_path: str) -> str:
    """Computes SHA256 checksum in 64KB blocks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_input_parquet() -> str:
    """Locates input targets.parquet inside mounted Kaggle dataset directories."""
    search_dirs = ["/kaggle/input"] if os.path.exists("/kaggle/input") else [".", "/kaggle/working"]
    found_files = []

    for d in search_dirs:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                for f in sorted(files):
                    if f.endswith(".parquet") and not f.endswith("_ids.parquet"):
                        found_files.append(os.path.join(root, f))

    if not found_files:
        print(f"[FATAL ERROR]: No input Parquet file found in {search_dirs}!")
        print("Ensure the Kaggle Dataset containing targets.parquet is mounted in dataset_sources.")
        sys.exit(1)

    print(f"[Worker] Found input file: {found_files[0]}")
    return found_files[0]


def build_passage_texts(df: pl.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Vectorized construction of compact entity strings with E5 'passage: ' prefix.
    Order: passage: business_name | business_address | country
    """
    cols = {c.lower(): c for c in df.columns}
    id_col = cols.get("target_id") or cols.get("eid") or cols.get("entity_id") or df.columns[0]
    name_col = cols.get("business_name") or cols.get("norm_name") or cols.get("name")
    addr_col = cols.get("business_address") or cols.get("norm_addr") or cols.get("address")
    ctry_col = cols.get("country")

    ids_list = [str(x) for x in df[id_col].to_list()]

    def clean_col(c_name: Optional[str]) -> pl.Expr:
        if c_name and c_name in df.columns:
            return (
                pl.col(c_name)
                .cast(pl.Utf8, strict=False)
                .fill_null("")
                .str.strip_chars()
            )
        return pl.lit("")

    e_name = clean_col(name_col)
    e_addr = clean_col(addr_col)
    e_ctry = clean_col(ctry_col)

    df_text = df.select(
        pl.concat_str(
            [e_name, e_addr, e_ctry],
            separator=" | ",
            ignore_nulls=True
        ).alias("raw_text")
    )

    prefix = "passage: "
    texts = [
        f"{prefix}{t.strip(' | ')}" if t.strip(" | ") else f"{prefix}unknown"
        for t in df_text["raw_text"].to_list()
    ]

    return ids_list, texts


def tune_gpu_batch_size(
    model,
    tokenizer,
    sample_texts: List[str],
    device: str,
    max_length: int = 128
) -> int:
    """Dynamic auto-tuning to find the largest safe GPU batch size without CUDA OOM."""
    import torch
    if device != "cuda":
        return 256

    print("[Worker] Auto-tuning GPU batch size...")
    for bs in BATCH_CANDIDATES:
        torch.cuda.empty_cache()
        gc.collect()
        try:
            test_batch = sample_texts[:bs] if len(sample_texts) >= bs else (sample_texts * (bs // len(sample_texts) + 1))[:bs]
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    encoded = tokenizer(
                        test_batch,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt"
                    ).to(device, non_blocking=True)

                    out = model(**encoded)
                    tok_emb = out.last_hidden_state
                    mask = encoded["attention_mask"].unsqueeze(-1).expand(tok_emb.size()).float()
                    sum_emb = torch.sum(tok_emb * mask, 1)
                    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                    pooled = sum_emb / sum_mask
                    _ = torch.nn.functional.normalize(pooled, p=2, dim=1)

            torch.cuda.synchronize()
            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(f"  -> Batch size {bs} PASSED (Peak GPU Mem: {peak_mb:.1f} MB).")
            return bs

        except torch.cuda.OutOfMemoryError:
            print(f"  -> Batch size {bs} OOM! Stepping down...")
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"  -> Batch size {bs} error: {e}. Stepping down...")
            torch.cuda.empty_cache()
            gc.collect()

    return 256


def run_one_shot_pipeline():
    print("=" * 80)
    print("ANTIGRAVITY — ONE-SHOT MULTILINGUAL E5 KAGGLE GPU WORKER")
    print("=" * 80)
    t_start_all = time.time()

    # 1. Device & CUDA Setup
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    if device == "cuda":
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    print(f"Device:              {device} ({gpu_name})")
    print(f"PyTorch Version:     v{torch.__version__}")
    print(f"Model:               {MODEL_NAME}")
    print(f"Target Dimension:    {EMBEDDING_DIM} (FP16)")
    print("-" * 80)

    # 2. Output Directories Setup
    working_dir = os.path.abspath("/kaggle/working/outputs" if os.path.exists("/kaggle/working") else "./outputs")
    emb_dir = os.path.join(working_dir, "embeddings")
    ids_dir = os.path.join(working_dir, "ids")
    os.makedirs(emb_dir, exist_ok=True)
    os.makedirs(ids_dir, exist_ok=True)

    # 3. Locate Input Parquet File & Count Total Rows
    input_parquet = find_input_parquet()
    df_scan = pl.scan_parquet(input_parquet)
    total_rows = df_scan.select(pl.len()).collect().item()
    shard_size = DEFAULT_SHARD_SIZE
    total_shards = (total_rows + shard_size - 1) // shard_size

    print(f"Total Target Rows:   {total_rows:,}")
    print(f"Shard Size:          {shard_size:,} rows")
    print(f"Total Shards:        {total_shards:,} shards")
    print(f"Output Directory:    {working_dir}")
    print("-" * 80)

    # 4. Model & Tokenizer Load (ONCE FOR ENTIRE RUN)
    t0_load = time.time()
    print(f"[Worker] Loading '{MODEL_NAME}' to {device}...")
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    model_load_sec = time.time() - t0_load
    print(f"[Worker] Model loaded in {model_load_sec:.2f}s.")

    # 5. Batch Size Auto-Tuning using first slice
    df_sample = pl.read_parquet(input_parquet, n_rows=min(2048, total_rows))
    _, sample_texts = build_passage_texts(df_sample)
    del df_sample
    gc.collect()

    t0_warmup = time.time()
    selected_batch_size = tune_gpu_batch_size(model, tokenizer, sample_texts, device, DEFAULT_MAX_LENGTH)
    warmup_sec = time.time() - t0_warmup
    print(f"[Worker] Selected Batch Size: {selected_batch_size} (Warmup: {warmup_sec:.2f}s)")
    print("=" * 80)

    # 6. Sequential Shard Processing Loop
    completed_shards = []
    total_processed_rows = 0
    total_encoding_sec = 0.0

    for shard_idx in range(total_shards):
        shard_id_str = f"part-{shard_idx:05d}"
        emb_file = os.path.join(emb_dir, f"{shard_id_str}.npy")
        ids_file = os.path.join(ids_dir, f"{shard_id_str}.parquet")

        offset = shard_idx * shard_size
        n_shard_rows = min(shard_size, total_rows - offset)

        # Resumability Check: If already validly generated, skip
        if os.path.exists(emb_file) and os.path.exists(ids_file):
            try:
                mmap_check = np.lib.format.open_memmap(emb_file, mode="r")
                rows_in_file, dim_in_file = mmap_check.shape
                del mmap_check
                if rows_in_file == n_shard_rows and dim_in_file == EMBEDDING_DIM:
                    print(f"[Worker] Shard {shard_id_str} ({shard_idx + 1}/{total_shards}) already completed. Skipping.")
                    completed_shards.append({
                        "shard_id": shard_id_str,
                        "row_count": n_shard_rows,
                        "sha256": compute_sha256(emb_file)
                    })
                    total_processed_rows += n_shard_rows
                    continue
            except Exception:
                pass

        print(f"\n[Worker] Processing Shard {shard_idx + 1}/{total_shards} ({shard_id_str}): rows {offset:,}..{offset + n_shard_rows:,}...")
        t_shard_0 = time.time()

        # Read shard slice
        # Using Polars slice read
        df_shard = pl.read_parquet(input_parquet).slice(offset, n_shard_rows)
        ids_list, texts = build_passage_texts(df_shard)
        del df_shard
        gc.collect()

        # Batched GPU Inference
        shard_embs = []
        t_enc_0 = time.time()

        with torch.inference_mode():
            for i in range(0, n_shard_rows, selected_batch_size):
                b_texts = texts[i : i + selected_batch_size]

                encoded = tokenizer(
                    b_texts,
                    padding=True,
                    truncation=True,
                    max_length=DEFAULT_MAX_LENGTH,
                    return_tensors="pt"
                ).to(device, non_blocking=True)

                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                    out = model(**encoded)
                    tok_emb = out.last_hidden_state
                    mask = encoded["attention_mask"].unsqueeze(-1).expand(tok_emb.size()).float()
                    sum_emb = torch.sum(tok_emb * mask, 1)
                    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                    pooled = sum_emb / sum_mask
                    normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)

                if device == "cuda":
                    torch.cuda.synchronize()

                # Store as float16 numpy array
                shard_embs.append(normalized.cpu().half().numpy())

        shard_embs_np = np.vstack(shard_embs)
        enc_duration = time.time() - t_enc_0
        total_encoding_sec += enc_duration

        # Invariant Assertions
        assert shard_embs_np.shape == (n_shard_rows, EMBEDDING_DIM), f"Shape mismatch in {shard_id_str}"
        assert len(ids_list) == n_shard_rows, f"ID count mismatch in {shard_id_str}"
        assert np.isfinite(shard_embs_np).all(), f"NaN or Inf detected in {shard_id_str}!"

        norms = np.linalg.norm(shard_embs_np.astype(np.float32), axis=1)
        assert np.all(np.isclose(norms, 1.0, atol=2e-3)), f"Embeddings not normalized in {shard_id_str}!"

        # Atomic Serialization (.tmp -> final)
        tmp_emb = f"{emb_file}.tmp"
        tmp_ids = f"{ids_file}.tmp"

        np.save(tmp_emb, shard_embs_np)
        pl.DataFrame({"target_id": ids_list}).write_parquet(tmp_ids, compression="zstd")

        if os.path.exists(emb_file):
            os.remove(emb_file)
        if os.path.exists(ids_file):
            os.remove(ids_file)

        os.replace(tmp_emb, emb_file)
        os.replace(tmp_ids, ids_file)

        emb_sha = compute_sha256(emb_file)
        shard_wall = time.time() - t_shard_0
        rows_sec = n_shard_rows / enc_duration if enc_duration > 0 else 0

        completed_shards.append({
            "shard_id": shard_id_str,
            "row_count": n_shard_rows,
            "first_target_id": ids_list[0] if ids_list else "",
            "last_target_id": ids_list[-1] if ids_list else "",
            "embedding_shape": [n_shard_rows, EMBEDDING_DIM],
            "sha256": emb_sha,
            "encoding_seconds": round(enc_duration, 2),
            "throughput_rows_sec": round(rows_sec, 1)
        })

        total_processed_rows += n_shard_rows
        print(f"  -> {shard_id_str} COMPLETE: {n_shard_rows:,} rows in {shard_wall:.2f}s ({rows_sec:.1f} enc rows/s) | SHA: {emb_sha[:12]}...")

        del shard_embs, shard_embs_np, ids_list, texts
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # 7. Root manifest.json Generation
    total_wall_sec = time.time() - t_start_all
    effective_rows_sec = total_processed_rows / total_wall_sec if total_wall_sec > 0 else 0

    manifest = {
        "pipeline_version": "5.0_one_shot",
        "model_name": MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "dtype": "float16",
        "normalization": "L2",
        "total_expected_rows": total_rows,
        "total_processed_rows": total_processed_rows,
        "shard_size": shard_size,
        "number_of_shards": total_shards,
        "completed_shards_count": len(completed_shards),
        "selected_batch_size": selected_batch_size,
        "device": device,
        "gpu_name": gpu_name,
        "telemetry": {
            "model_load_seconds": round(model_load_sec, 2),
            "warmup_seconds": round(warmup_sec, 2),
            "total_encoding_seconds": round(total_encoding_sec, 2),
            "total_wall_seconds": round(total_wall_sec, 2),
            "effective_rows_per_second": round(effective_rows_sec, 1)
        },
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "shards": completed_shards
    }

    manifest_path = os.path.join(working_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("ANTIGRAVITY — ONE-SHOT KAGGLE GPU WORKER RUN COMPLETE")
    print("=" * 80)
    print(f"Total Rows Encoded:  {total_processed_rows:,} / {total_rows:,}")
    print(f"Total Shards:        {len(completed_shards)} / {total_shards}")
    print(f"Effective Speed:     {effective_rows_sec:.1f} rows/sec (Total Wall: {total_wall_sec:.2f}s)")
    print(f"Manifest File:       {manifest_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_one_shot_pipeline()
