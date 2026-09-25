"""
Antigravity V3 Arctic Embedding Module for Business Entity Resolution.
Model: themelder/arctic-embed-xs-entity-resolution (384-dimensional dense representations)

Features:
- Standardized entity text formatting for entity resolution: business_name | business_address | country
- SentenceTransformer & HuggingFace Transformers dual backend support with mean pooling & L2 normalization
- Low-memory chunked batch encoding optimized for 2-vCPU / 8GB RAM instances
- Fallback & mock mode for lightweight testing environments without PyTorch
"""

import os
import time
from typing import List, Dict, Any, Optional, Union
import numpy as np

DEFAULT_MODEL_NAME = "themelder/arctic-embed-xs-entity-resolution"
EMBEDDING_DIM = 384

def format_entity_text(name: Optional[str], address: Optional[str], country: Optional[str] = "") -> str:
    """
    Formats business entity fields into canonical string for Arctic ER model.
    Format: '<name> | <address> | <country>'
    """
    parts = []
    if name and str(name).strip():
        parts.append(str(name).strip())
    if address and str(address).strip():
        parts.append(str(address).strip())
    if country and str(country).strip():
        parts.append(str(country).strip())
    
    if not parts:
        return "unknown entity"
    return " | ".join(parts)


class ArcticEmbedder:
    """
    High-throughput, memory-bounded Arctic entity embedder.
    Produces L2-normalized 384-d embeddings where cosine similarity is simple dot product.
    """
    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL_NAME,
        batch_size: int = 128,
        device: str = "cpu",
        num_threads: int = 2
    ):
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.device = device
        self.num_threads = num_threads
        self.model = None
        self.tokenizer = None
        self.backend = None # 'sentence_transformers', 'transformers', or 'mock'
        self.embedding_dim = EMBEDDING_DIM
        self._init_model()

    def _init_model(self):
        """Initializes model using available backends."""
        try:
            import torch
            torch.set_num_threads(self.num_threads)
            
            # Try sentence-transformers first
            try:
                from sentence_transformers import SentenceTransformer
                print(f"[ArcticEmbedder] Loading SentenceTransformer: {self.model_name_or_path}...")
                self.model = SentenceTransformer(self.model_name_or_path, device=self.device)
                self.backend = "sentence_transformers"
                print("[ArcticEmbedder] Successfully loaded SentenceTransformer backend.")
                return
            except ImportError:
                pass

            # Fallback to huggingface transformers
            try:
                from transformers import AutoTokenizer, AutoModel
                print(f"[ArcticEmbedder] Loading HF Transformers: {self.model_name_or_path}...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
                self.model = AutoModel.from_pretrained(self.model_name_or_path)
                self.model.eval()
                self.model.to(self.device)
                self.backend = "transformers"
                print("[ArcticEmbedder] Successfully loaded Transformers backend.")
                return
            except Exception as e:
                print(f"[ArcticEmbedder] Transformers load failed: {e}")

        except ImportError:
            print("[ArcticEmbedder] PyTorch not installed. Initializing Mock/Fallback embedder for testing.")
            self.backend = "mock"

    def encode(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        Encodes a list of entity text strings into a (N, 384) numpy float32 matrix.
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        bs = batch_size or self.batch_size

        if self.backend == "sentence_transformers":
            emb = self.model.encode(
                texts,
                batch_size=bs,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=True
            )
            return emb.astype(np.float32)

        elif self.backend == "transformers":
            import torch
            all_embeddings = []
            
            for i in range(0, len(texts), bs):
                batch_texts = texts[i : i + bs]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(self.device)

                with torch.inference_mode():
                    model_output = self.model(**encoded)
                    # Mean pooling with attention mask
                    token_embeddings = model_output.last_hidden_state
                    input_mask_expanded = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    pooled = sum_embeddings / sum_mask

                    if normalize_embeddings:
                        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

                    all_embeddings.append(pooled.cpu().numpy().astype(np.float32))

            return np.vstack(all_embeddings)

        else:
            # Deterministic mock pseudo-embeddings for environments without PyTorch / offline development
            # Uses character frequency and hash projection to generate normalized 384-d vectors
            print(f"[ArcticEmbedder] Mock backend encoding {len(texts)} texts...")
            np.random.seed(42)
            proj_matrix = np.random.randn(256, self.embedding_dim).astype(np.float32)
            
            vecs = []
            for t in texts:
                char_counts = np.zeros(256, dtype=np.float32)
                for c in t[:256]:
                    char_counts[ord(c) % 256] += 1.0
                vec = np.dot(char_counts, proj_matrix)
                if normalize_embeddings:
                    norm = np.linalg.norm(vec)
                    if norm > 1e-9:
                        vec = vec / norm
                vecs.append(vec.astype(np.float32))
            return np.array(vecs, dtype=np.float32)

    def encode_records(
        self,
        records: List[Dict[str, Any]],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Encodes records directly by formatting their name, address, and country."""
        texts = [
            format_entity_text(
                r.get("norm_name") or r.get("name") or r.get("business_name"),
                r.get("norm_addr") or r.get("address") or r.get("business_address"),
                r.get("country", "")
            )
            for r in records
        ]
        return self.encode(texts, batch_size=batch_size)
