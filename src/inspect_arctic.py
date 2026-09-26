"""
Antigravity V4.1 Model & Dependency Inspection Utility.
Inspects PyTorch, SentenceTransformers, FAISS, and Arctic ER Model artifacts.
Checks local caching, embedding dimension, normalization, and inference backend.
"""

import os
import sys
import time
import numpy as np

from src.resource_tracker import get_current_rss_mb, get_peak_rss_mb, MemoryTracker

MODEL_NAME = "themelder/arctic-embed-xs-entity-resolution"
EMBEDDING_DIM = 384

def inspect_environment():
    print("=" * 80)
    print("ANTIGRAVITY V4.1 — ARCTIC EMBEDDING & ANN DEPENDENCY INSPECTION")
    print("=" * 80)
    print(f"Python Version:   {sys.version.split()[0]}")
    print(f"Platform:         {sys.platform}")
    print(f"Current RSS:      {get_current_rss_mb():.2f} MB")
    print("-" * 80)

    # 1. Inspect PyTorch
    torch_available = False
    cuda_available = False
    try:
        import torch
        torch_available = True
        cuda_available = torch.cuda.is_available()
        print(f"[OK] PyTorch:             v{torch.__version__} (CUDA: {cuda_available})")
    except ImportError:
        print("[WARN] PyTorch:           NOT installed in current environment")

    # 2. Inspect SentenceTransformers
    st_available = False
    try:
        import sentence_transformers
        st_available = True
        print(f"[OK] SentenceTransformers: v{sentence_transformers.__version__}")
    except ImportError:
        print("[INFO] SentenceTransformers: NOT installed")

    # 3. Inspect Transformers
    hf_available = False
    try:
        import transformers
        hf_available = True
        print(f"[OK] HF Transformers:     v{transformers.__version__}")
    except ImportError:
        print("[INFO] HF Transformers:    NOT installed")

    # 4. Inspect FAISS
    faiss_available = False
    try:
        import faiss
        faiss_available = True
        print(f"[OK] FAISS:                v{getattr(faiss, '__version__', 'Installed')}")
    except ImportError:
        print("[WARN] FAISS:              NOT installed (Will use fallback/NumPy IP search if offline)")

    print("-" * 80)

    # 5. Inspect Model Cache
    hf_hub_cache = os.path.expanduser("~/.cache/huggingface/hub")
    local_model_path = os.path.join(hf_hub_cache, "models--themelder--arctic-embed-xs-entity-resolution")
    is_cached = os.path.exists(local_model_path)
    print(f"Target Model:     {MODEL_NAME}")
    print(f"Model Dimension:  {EMBEDDING_DIM}")
    print(f"HF Cache Path:    {local_model_path}")
    print(f"Locally Cached:   {'YES' if is_cached else 'NO (Will download on first run if connected)'}")

    # 6. Initialize Embedder and verify L2 normalization
    print("-" * 80)
    print("[ArcticEmbedder] Initializing and validating L2 normalization on sample text...")
    from src.arctic_embeddings import ArcticEmbedder, format_entity_text

    with MemoryTracker("Model Initialization & Sample Encoding"):
        embedder = ArcticEmbedder(model_name_or_path=MODEL_NAME, batch_size=4)
        print(f"Selected Backend: {embedder.backend}")

        test_records = [
            {"business_name": "Acme Industrial Supplies Ltd", "business_address": "123 Main St, Springfield", "country": "US"},
            {"business_name": "State Bank of India", "business_address": "Nariman Point, Mumbai", "country": "IN"}
        ]
        test_texts = [
            format_entity_text(r["business_name"], r["business_address"], r["country"])
            for r in test_records
        ]
        print(f"Formatted Text 0:\n  '{test_texts[0]}'")

        emb = embedder.encode(test_texts, normalize_embeddings=True)
        norms = np.linalg.norm(emb, axis=1)

        print(f"Encoded Shape:    {emb.shape}")
        print(f"Embedding Dtype:  {emb.dtype}")
        print(f"L2 Norms:         {norms} (Expected: ~1.000000)")
        
        norm_valid = bool(np.all(np.isclose(norms, 1.0, atol=1e-4)))
        print(f"Normalization OK: {norm_valid}")
        assert norm_valid, "L2 Normalization verification failed!"

    print("=" * 80)
    print("Inspection Completed Successfully.")
    print("=" * 80)

if __name__ == "__main__":
    inspect_environment()
