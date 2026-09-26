"""
Antigravity V5 Multilingual E5 Kaggle GPU Grouped-Job Worker.
Runs inside a Kaggle Kernel / Notebook environment to encode multi-chunk jobs (~1,000,000 rows).

Key Architecture:
- Tokenizer & Model loaded ONCE per Kaggle job (zero reload overhead across chunks).
- CUDA TF32 & Float32 matmul precision optimizations enabled.
- Automatic GPU batch-size tuning with CUDA OOM recovery.
- Vectorized text construction with asymmetric E5 prefix ("passage: ...").
- Mean pooling, L2 normalization, FP16 storage dtype.
- Strict positional ID alignment verification & NaN/Inf safety assertions.
- Reads explicit job_manifest.json from mounted dataset.
- Outputs individual chunk folders + root job_result.json with comprehensive telemetry.
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

DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384
BATCH_CANDIDATES = [2048, 1536, 1024, 768, 512]


def compute_sha256(file_path: str) -> str:
    """Computes SHA256 hash in streaming 64KB blocks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_job_manifest() -> Tuple[str, Dict[str, Any]]:
    """
    Locates the explicit job_manifest.json inside mounted Kaggle dataset directories.
    """
    search_dirs = ["/kaggle/input"] if os.path.exists("/kaggle/input") else [".", "/kaggle/working"]
    found_manifests = []

    for d in search_dirs:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                if "job_manifest.json" in files:
                    found_manifests.append(os.path.join(root, "job_manifest.json"))

    if not found_manifests:
        print(f"[FATAL ERROR]: No 'job_manifest.json' found in search paths {search_dirs}!")
        print("Ensure the Kaggle Dataset containing job_manifest.json is mounted in dataset_sources.")
        sys.exit(1)

    manifest_path = found_manifests[0]
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    return manifest_path, manifest


def build_passage_texts(df: pl.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Vectorized construction of compact entity strings with the E5 'passage: ' prefix.
    Order: passage: business_name | business_address | country
    """
    cols = {c.lower(): c for c in df.columns}
    id_col = cols.get("target_id") or cols.get("eid") or cols.get("entity_id") or df.columns[0]
    name_col = cols.get("business_name") or cols.get("norm_name") or cols.get("name")
    addr_col = cols.get("business_address") or cols.get("norm_addr") or cols.get("address")
    ctry_col = cols.get("country")

    ids_list = [str(x) for x in df[id_col].to_list()]

    def clean_expr(c_name: Optional[str]) -> pl.Expr:
        if c_name and c_name in df.columns:
            return (
                pl.col(c_name)
                .cast(pl.Utf8, strict=False)
                .fill_null("")
                .str.strip_chars()
            )
        return pl.lit("")

    e_name = clean_expr(name_col)
    e_addr = clean_expr(addr_col)
    e_ctry = clean_expr(ctry_col)

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


def tune_batch_size(
    model,
    tokenizer,
    sample_texts: List[str],
    device: str,
    max_length: int = 128,
    use_fp16: bool = True
) -> int:
    """
    Performs dynamic auto-tuning on GPU to select the optimal batch size without OOM.
    """
    import torch
    if device != "cuda":
        return 256

    print("[Worker] Auto-tuning GPU batch size...")
    for candidate in BATCH_CANDIDATES:
        torch.cuda.empty_cache()
        gc.collect()
        try:
            test_batch = sample_texts[:candidate] if len(sample_texts) >= candidate else (sample_texts * (candidate // len(sample_texts) + 1))[:candidate]
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
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
            print(f"  -> Batch size {candidate} PASSED (Peak GPU Mem: {peak_mb:.1f} MB).")
            return candidate

        except torch.cuda.OutOfMemoryError:
            print(f"  -> Batch size {candidate} OOM! Stepping down...")
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"  -> Batch size {candidate} error: {e}. Stepping down...")
            torch.cuda.empty_cache()
            gc.collect()

    return 256


def run_grouped_worker():
    print("=" * 80)
    print("ANTIGRAVITY V5 — MULTILINGUAL E5 GPU GROUPED-JOB WORKER")
    print("=" * 80)
    t_start_wall = time.time()

    # 1. Locate Job Manifest
    manifest_file, job_manifest = find_job_manifest()
    dataset_dir = os.path.dirname(manifest_file)
    job_id = job_manifest.get("job_id", "job_unknown")
    model_name = job_manifest.get("model_name", DEFAULT_MODEL_NAME)
    expected_dim = int(job_manifest.get("embedding_dimension", EMBEDDING_DIM))
    max_length = int(job_manifest.get("max_length", 128))
    want_precision = job_manifest.get("precision", "fp16").lower()
    input_files = job_manifest.get("input_filenames", [])
    expected_total_rows = int(job_manifest.get("expected_row_count", 0))

    print(f"Job ID:              {job_id}")
    print(f"Dataset Dir:         {dataset_dir}")
    print(f"Model:               {model_name}")
    print(f"Dimension:           {expected_dim}")
    print(f"Target Precision:    {want_precision}")
    print(f"Expected Chunks:     {len(input_files)}")
    print(f"Expected Rows:       {expected_total_rows:,}")
    print("-" * 80)

    # 2. PyTorch & CUDA Initialization (ONCE)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    use_fp16 = (want_precision == "fp16") and (device == "cuda")

    if device == "cuda":
        # Enable CUDA TF32 and high precision matmul
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    print(f"Device:              {device} ({gpu_name})")
    print(f"Active Precision:    {'FP16 (Autocast)' if use_fp16 else 'FP32'}")
    print(f"PyTorch Version:     v{torch.__version__}")

    # 3. Model & Tokenizer Load (ONCE FOR ENTIRE JOB)
    t_model_load_0 = time.time()
    print(f"[Worker] Loading model & tokenizer '{model_name}' to {device}...")

    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    model_load_sec = time.time() - t_model_load_0
    print(f"[Worker] Model loaded in {model_load_sec:.2f}s.")

    # 4. Read First Chunk & Auto-Tune Batch Size
    first_chunk_path = os.path.join(dataset_dir, input_files[0])
    if not os.path.exists(first_chunk_path):
        print(f"[FATAL ERROR]: Input file not found: {first_chunk_path}")
        sys.exit(1)

    df_first = pl.read_parquet(first_chunk_path)
    _, sample_texts = build_passage_texts(df_first)
    del df_first
    gc.collect()

    t_warmup_0 = time.time()
    selected_batch_size = tune_batch_size(
        model=model,
        tokenizer=tokenizer,
        sample_texts=sample_texts,
        device=device,
        max_length=max_length,
        use_fp16=use_fp16
    )
    warmup_sec = time.time() - t_warmup_0
    print(f"[Worker] Batch size locked: {selected_batch_size} (Warmup duration: {warmup_sec:.2f}s).")
    print("-" * 80)

    # 5. Process All Chunks Sequentially with Model Persistence
    total_processed_rows = 0
    completed_chunks_meta = []
    total_tokenization_sec = 0.0
    total_inference_sec = 0.0
    total_serialization_sec = 0.0
    total_encoding_sec = 0.0

    working_dir = os.path.abspath("/kaggle/working" if os.path.exists("/kaggle/working") else ".")

    for chunk_idx, in_filename in enumerate(input_files):
        chunk_path = os.path.join(dataset_dir, in_filename)
        chunk_prefix = os.path.splitext(in_filename)[0]
        chunk_out_dir = os.path.join(working_dir, chunk_prefix)
        os.makedirs(chunk_out_dir, exist_ok=True)

        print(f"\n[Worker] Processing Chunk {chunk_idx + 1}/{len(input_files)}: {in_filename}...")
        t_chunk_0 = time.time()

        # Load Parquet
        df = pl.read_parquet(chunk_path)
        n_rows = len(df)
        ids_list, texts = build_passage_texts(df)
        del df
        gc.collect()

        # Batched GPU Inference
        chunk_embs = []
        t_enc_chunk = time.time()
        
        with torch.inference_mode():
            for i in range(0, n_rows, selected_batch_size):
                b_texts = texts[i : i + selected_batch_size]

                t_tok_0 = time.time()
                encoded = tokenizer(
                    b_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                ).to(device, non_blocking=True)
                total_tokenization_sec += (time.time() - t_tok_0)

                t_inf_0 = time.time()
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
                    out = model(**encoded)
                    tok_emb = out.last_hidden_state
                    mask = encoded["attention_mask"].unsqueeze(-1).expand(tok_emb.size()).float()
                    sum_emb = torch.sum(tok_emb * mask, 1)
                    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                    pooled = sum_emb / sum_mask
                    normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)

                if device == "cuda":
                    torch.cuda.synchronize()
                total_inference_sec += (time.time() - t_inf_0)

                # Store as float16 numpy array
                chunk_embs.append(normalized.cpu().half().numpy())

        chunk_embs_np = np.vstack(chunk_embs)
        chunk_enc_dur = time.time() - t_enc_chunk
        total_encoding_sec += chunk_enc_dur

        # 6. Verification & Invariant Checks
        assert chunk_embs_np.shape == (n_rows, expected_dim), f"Shape mismatch: {chunk_embs_np.shape} != ({n_rows}, {expected_dim})"
        assert len(ids_list) == n_rows, f"Target ID count mismatch: {len(ids_list)} != {n_rows}"

        # Assert no NaN or Inf
        assert np.isfinite(chunk_embs_np).all(), f"NaN or Inf detected in embeddings for chunk {chunk_prefix}!"

        # Assert L2 Normalization
        norms = np.linalg.norm(chunk_embs_np.astype(np.float32), axis=1)
        assert np.all(np.isclose(norms, 1.0, atol=2e-3)), f"Embeddings not L2 normalized in chunk {chunk_prefix} (Norm min: {norms.min():.4f}, max: {norms.max():.4f})!"

        # 7. Serialization
        t_ser_0 = time.time()
        out_emb_path = os.path.join(chunk_out_dir, "embeddings.npy")
        out_ids_path = os.path.join(chunk_out_dir, "target_ids.parquet")
        out_meta_path = os.path.join(chunk_out_dir, "metadata.json")

        np.save(out_emb_path, chunk_embs_np)
        pl.DataFrame({"target_id": ids_list}).write_parquet(out_ids_path, compression="zstd")

        emb_sha256 = compute_sha256(out_emb_path)
        ids_sha256 = compute_sha256(out_ids_path)

        chunk_meta = {
            "job_id": job_id,
            "chunk_id": chunk_prefix,
            "input_filename": in_filename,
            "row_count": n_rows,
            "model_name": model_name,
            "embedding_dim": expected_dim,
            "dtype": "float16",
            "precision": want_precision,
            "batch_size": selected_batch_size,
            "max_length": max_length,
            "first_target_id": ids_list[0] if ids_list else "",
            "last_target_id": ids_list[-1] if ids_list else "",
            "output_sha256": emb_sha256,
            "ids_sha256": ids_sha256,
            "encoding_seconds": chunk_enc_dur,
            "throughput_rows_per_sec": n_rows / chunk_enc_dur if chunk_enc_dur > 0 else 0,
            "status": "completed",
            "complete": True
        }

        with open(out_meta_path, "w", encoding="utf-8") as f:
            json.dump(chunk_meta, f, indent=2)

        total_serialization_sec += (time.time() - t_ser_0)
        total_processed_rows += n_rows
        completed_chunks_meta.append(chunk_meta)

        chunk_wall = time.time() - t_chunk_0
        print(f"  -> Chunk {chunk_prefix} COMPLETE: {n_rows:,} rows in {chunk_wall:.2f}s ({n_rows / chunk_enc_dur:.1f} enc rows/s).")

        del chunk_embs, chunk_embs_np, ids_list, texts
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # 8. Write Root job_result.json
    total_wall_sec = time.time() - t_start_wall
    effective_rows_per_sec = total_processed_rows / total_wall_sec if total_wall_sec > 0 else 0
    gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device == "cuda" else 0

    job_result = {
        "job_id": job_id,
        "status": "completed" if total_processed_rows == expected_total_rows else "failed",
        "total_input_rows": expected_total_rows,
        "total_output_rows": total_processed_rows,
        "completed_chunks": len(completed_chunks_meta),
        "total_chunks": len(input_files),
        "failed_chunks": 0 if total_processed_rows == expected_total_rows else len(input_files) - len(completed_chunks_meta),
        "model": model_name,
        "embedding_dim": expected_dim,
        "dtype": "float16",
        "precision": want_precision,
        "selected_batch_size": selected_batch_size,
        "max_length": max_length,
        "gpu_name": gpu_name,
        "gpu_memory_peak_mb": gpu_peak_mb,
        "telemetry": {
            "model_load_seconds": model_load_sec,
            "warmup_seconds": warmup_sec,
            "tokenization_seconds": total_tokenization_sec,
            "gpu_inference_seconds": total_inference_sec,
            "serialization_seconds": total_serialization_sec,
            "total_encoding_seconds": total_encoding_sec,
            "total_wall_seconds": total_wall_sec
        },
        "effective_rows_per_second": effective_rows_per_sec,
        "pure_encoding_rows_per_second": total_processed_rows / total_encoding_sec if total_encoding_sec > 0 else 0
    }

    job_result_path = os.path.join(working_dir, "job_result.json")
    with open(job_result_path, "w", encoding="utf-8") as f:
        json.dump(job_result, f, indent=2)

    print("\n" + "=" * 80)
    print(f"ANTIGRAVITY V5 GROUPED JOB {job_id} COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Total Rows Processed:   {total_processed_rows:,} / {expected_total_rows:,}")
    print(f"Total Chunks Encoded:   {len(completed_chunks_meta)} / {len(input_files)}")
    print(f"Effective Throughput:   {effective_rows_per_sec:.1f} rows/s (Total Wall: {total_wall_sec:.2f}s)")
    print(f"Pure GPU Throughput:    {job_result['pure_encoding_rows_per_second']:.1f} rows/s")
    print(f"Peak GPU Memory:        {gpu_peak_mb:.1f} MB")
    print(f"Job Result Summary:     {job_result_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_grouped_worker()
