"""
Antigravity V5 Multilingual E5 Embedding Module.
Model: intfloat/multilingual-e5-small (384-dimensional dense representations)

Features:
- SentenceTransformer & HuggingFace Transformers dual backend support with mean pooling & L2 normalization
- Strict asymmetric prefix handling via src.dense.text_builder (query: vs passage:)
- Low-memory chunked batch encoding optimized for CPU/GPU instances
- Deterministic mock mode for offline testing without PyTorch / model weights
"""

import os
import time
from typing import List, Dict, Any, Optional, Union
import numpy as np

from src.dense.text_builder import (
    build_entity_text,
    format_e5_text,
    vectorized_build_e5_texts,
    E5_QUERY_PREFIX,
    E5_PASSAGE_PREFIX,
    DEFAULT_MAX_LENGTH
)

DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384


class DenseEmbedder:
    """
    High-throughput, memory-bounded Multilingual E5 entity embedder.
    Produces L2-normalized 384-d embeddings where cosine similarity is simple dot product.
    """
    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL_NAME,
        batch_size: int = 128,
        device: str = "cpu",
        num_threads: int = 2,
        precision: str = "fp32",
        max_length: int = DEFAULT_MAX_LENGTH
    ):
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.device = device
        self.num_threads = num_threads
        self.precision = precision
        self.max_length = max_length
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
            
            # SentenceTransformer first
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name_or_path, device=self.device)
                if self.precision == "fp16" and self.device == "cuda":
                    self.model = self.model.half()
                self.backend = "sentence_transformers"
                return
            except ImportError:
                pass

            # Fallback to HuggingFace Transformers
            try:
                from transformers import AutoTokenizer, AutoModel
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
                self.model = AutoModel.from_pretrained(self.model_name_or_path)
                if self.precision == "fp16" and self.device == "cuda":
                    self.model = self.model.half()
                self.model.eval()
                self.model.to(self.device)
                self.backend = "transformers"
                return
            except Exception as e:
                pass

        except ImportError:
            pass

        self.backend = "mock"

    def encode(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        Encodes a list of formatted text strings into an (N, 384) numpy float32 matrix.
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
                    max_length=self.max_length,
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

    def encode_queries(
        self,
        records_or_df: Union[List[Dict[str, Any]], Any],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Encodes S1 queries with the required 'query: ' prefix."""
        if isinstance(records_or_df, list):
            texts = [
                format_e5_text(
                    build_entity_text(
                        r.get("norm_name") or r.get("name") or r.get("business_name"),
                        r.get("norm_addr") or r.get("address") or r.get("business_address"),
                        r.get("country", "")
                    ),
                    is_query=True
                )
                for r in records_or_df
            ]
        else:
            texts = vectorized_build_e5_texts(records_or_df, is_query=True)
        return self.encode(texts, batch_size=batch_size)

    def encode_passages(
        self,
        records_or_df: Union[List[Dict[str, Any]], Any],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Encodes target entities with the required 'passage: ' prefix."""
        if isinstance(records_or_df, list):
            texts = [
                format_e5_text(
                    build_entity_text(
                        r.get("norm_name") or r.get("name") or r.get("business_name"),
                        r.get("norm_addr") or r.get("address") or r.get("business_address"),
                        r.get("country", "")
                    ),
                    is_query=False
                )
                for r in records_or_df
            ]
        else:
            texts = vectorized_build_e5_texts(records_or_df, is_query=False)
        return self.encode(texts, batch_size=batch_size)
