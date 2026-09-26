"""
Kaggle GPU Worker for Arctic Entity Resolution Embeddings.
Runs inside a Kaggle Kernel / Notebook environment to encode target text chunks on GPU.

Compliance:
- Zero external data lookups (No web scraping, no external APIs).
- Uses only challenge-provided data & approved Arctic ER model.
- High-throughput batched GPU inference with FP16 / FP32 precision and L2 normalization.
"""

import os
import sys
import gc
import json
import time
import hashlib
from typing import List, Optional
import polars as pl
import numpy as np

MODEL_NAME = "themelder/arctic-embed-xs-entity-resolution"
EMBEDDING_DIM = 384
BATCH_SIZE = 256

def compute_sha256(file_path: str) -> str:
    """Computes SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def format_entity_text(name: Optional[str], address: Optional[str] = "", country: Optional[str] = "") -> str:
    """Formats entity record into canonical template."""
    bname = str(name).strip() if name and str(name).strip() else "unknown"
    baddr = str(address).strip() if address and str(address).strip() else "unknown"
    bctry = str(country).strip() if country and str(country).strip() else "unknown"
    return f"Business name: {bname}\nAddress: {baddr}\nCountry: {bctry}"

def run_worker():
    print("=" * 80)
    print("KAGGLE GPU ARCTIC EMBEDDING WORKER")
    print("=" * 80)
    t0 = time.time()

    # 1. GPU Detection
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device:           {device} ({gpu_name})")
    print(f"PyTorch Version:  v{torch.__version__}")
    print(f"Target Model:     {MODEL_NAME}")
    print(f"Embedding Dim:    {EMBEDDING_DIM}")

    # 2. Locate Input Parquet Chunk
    # Searches current directory or /kaggle/working or /kaggle/input
    search_dirs = [".", "/kaggle/working", "/kaggle/input"]
    input_file = None
    for d in search_dirs:
        if os.path.exists(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".parquet") and not f.endswith("_ids.parquet"):
                    input_file = os.path.join(d, f)
                    break
        if input_file:
            break

    if not input_file:
        print("[ERROR] No input Parquet chunk found in workspace! Exiting.")
        sys.exit(1)

    print(f"Input Parquet:    {input_file}")
    df = pl.read_parquet(input_file)
    n_rows = len(df)
    print(f"Total Records:    {n_rows:,}")

    # Standardize columns
    cols = {c.lower(): c for c in df.columns}
    id_col = cols.get("target_id") or cols.get("eid") or cols.get("entity_id") or df.columns[0]
    name_col = cols.get("business_name") or cols.get("norm_name") or cols.get("name")
    addr_col = cols.get("business_address") or cols.get("norm_addr") or cols.get("address")
    ctry_col = cols.get("country")

    ids_list = [str(x) for x in df[id_col].to_list()]
    names_list = [str(x) if x is not None else "" for x in df[name_col].to_list()] if name_col else [""] * n_rows
    addrs_list = [str(x) if x is not None else "" for x in df[addr_col].to_list()] if addr_col else [""] * n_rows
    ctrys_list = [str(x) if x is not None else "" for x in df[ctry_col].to_list()] if ctry_col else [""] * n_rows

    print(f"Target IDs:       {len(ids_list):,} (First: {ids_list[0]}, Last: {ids_list[-1]})")

    # 3. Format Texts
    texts = [
        format_entity_text(n, a, c)
        for n, a, c in zip(names_list, addrs_list, ctrys_list)
    ]
    del df, names_list, addrs_list, ctrys_list
    gc.collect()

    # 4. Load Model
    print(f"[Worker] Loading model {MODEL_NAME} on {device}...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME, device=device)
        use_st = True
    except Exception as e:
        print(f"[Worker] SentenceTransformers load failed ({e}). Using Transformers...")
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME).to(device)
        model.eval()
        use_st = False

    # 5. Batched Inference
    print(f"[Worker] Encoding {n_rows:,} records in batches of {BATCH_SIZE}...")
    t_encode = time.time()
    
    if use_st:
        with torch.inference_mode():
            # encode with FP16 if on CUDA
            embeddings = model.encode(
                texts,
                batch_size=BATCH_SIZE,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True
            ).astype(np.float32)
    else:
        all_embs = []
        with torch.inference_mode():
            for i in range(0, n_rows, BATCH_SIZE):
                b_texts = texts[i : i + BATCH_SIZE]
                encoded = tokenizer(b_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
                out = model(**encoded)
                tok_emb = out.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand(tok_emb.size()).float()
                sum_emb = torch.sum(tok_emb * mask, 1)
                sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                pooled = sum_emb / sum_mask
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                all_embs.append(pooled.cpu().numpy().astype(np.float32))
                if i % 10000 == 0:
                    torch.cuda.empty_cache()
        embeddings = np.vstack(all_embs)

    encode_duration = time.time() - t_encode
    throughput = n_rows / encode_duration if encode_duration > 0 else 0
    print(f"[Worker] Encoding completed in {encode_duration:.2f}s ({throughput:.1f} rows/s).")

    # 6. Verify Dimensionality & L2 Normalization
    assert embeddings.shape == (n_rows, EMBEDDING_DIM), f"Shape mismatch: {embeddings.shape} != ({n_rows}, {EMBEDDING_DIM})"
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.all(np.isclose(norms, 1.0, atol=1e-3)), "Embeddings not L2 normalized!"

    # 7. Save Outputs
    out_prefix = os.path.splitext(os.path.basename(input_file))[0]
    out_emb_path = f"{out_prefix}_embeddings.npy"
    out_ids_path = f"{out_prefix}_ids.parquet"
    out_meta_path = f"{out_prefix}_meta.json"

    print(f"[Worker] Saving embeddings: {out_emb_path}...")
    np.save(out_emb_path, embeddings)

    print(f"[Worker] Saving target IDs: {out_ids_path}...")
    pl.DataFrame({"target_id": ids_list}).write_parquet(out_ids_path, compression="zstd")

    emb_sha256 = compute_sha256(out_emb_path)
    ids_sha256 = compute_sha256(out_ids_path)

    metadata = {
        "chunk_id": out_prefix,
        "model_name": MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIM,
        "dtype": "float32",
        "row_count": n_rows,
        "first_target_id": ids_list[0] if ids_list else "",
        "last_target_id": ids_list[-1] if ids_list else "",
        "output_sha256": emb_sha256,
        "ids_sha256": ids_sha256,
        "device": device,
        "gpu_name": gpu_name,
        "encode_duration_sec": encode_duration,
        "throughput_rows_per_sec": throughput,
        "status": "completed"
    }

    with open(out_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    total_time = time.time() - t0
    print("\n" + "=" * 80)
    print(f"[Worker] SUCCESS: Chunk {out_prefix} completed in {total_time:.2f}s.")
    print(f"  Embeddings: {out_emb_path} ({os.path.getsize(out_emb_path) / (1024*1024):.2f} MB | SHA256: {emb_sha256[:12]}...)")
    print(f"  Target IDs: {out_ids_path} ({os.path.getsize(out_ids_path) / (1024*1024):.2f} MB | SHA256: {ids_sha256[:12]}...)")
    print(f"  Metadata:   {out_meta_path}")
    print("=" * 80)

if __name__ == "__main__":
    run_worker()
