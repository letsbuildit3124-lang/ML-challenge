"""
Antigravity V5 Multilingual E5 Dense ANN Candidate Retriever.
Loads disk-backed E5 target embeddings & FAISS index to retrieve semantic nearest-neighbors for S1 queries.

Features:
- Encodes queries with asymmetric 'query: ' prefix.
- Sub-50ms batch ANN search via FAISS / NumPy chunked fallback.
- Provenance bitmask tracking (PROV_DENSE = 16).
- Cross-country retrieval (no country pruning during candidate generation).
- Memory-bounded zero-copy memory-mapped embedding access.
"""

import os
import sys
import gc
import json
import time
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
import polars as pl
import numpy as np

from src.config import get_config
from src.resource_tracker import get_current_rss_mb, MemoryTracker
from src.dense_embeddings import DenseEmbedder, DEFAULT_MODEL_NAME, EMBEDDING_DIM
from src.dense.text_builder import vectorized_build_e5_texts

PROV_DET = 1
PROV_SPARSE_NAME = 2
PROV_SPARSE_ADDR = 4
PROV_SPARSE_TRANSLIT = 8
PROV_DENSE = 16


class DenseRetriever:
    """
    Antigravity V5 Multilingual E5 Dense Retriever.
    """
    def __init__(
        self,
        embeddings_dir: str = "cache/embeddings/multilingual_e5/production",
        ann_dir: str = "cache/ann/multilingual_e5/production",
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = 128
    ):
        config = get_config()
        self.embeddings_dir = os.path.join(config.base_dir, embeddings_dir)
        self.ann_dir = os.path.join(config.base_dir, ann_dir)
        self.model_name = model_name
        self.batch_size = batch_size

        self.embedder: Optional[DenseEmbedder] = None
        self.faiss_index = None
        self.target_ids: Optional[List[str]] = None
        self.target_embeddings = None # mmap
        self.is_loaded = False

    def load(self):
        """Loads embedder model, target ID list, and ANN index."""
        if self.is_loaded:
            return

        with MemoryTracker("V5 Dense Retriever Load"):
            # 1. Initialize E5 embedder
            print(f"[DenseRetriever] Initializing Multilingual E5 embedder ({self.model_name})...")
            self.embedder = DenseEmbedder(
                model_name_or_path=self.model_name,
                batch_size=self.batch_size
            )

            # 2. Load target IDs mapping
            ids_path = os.path.join(self.embeddings_dir, "target_ids.json")
            if not os.path.exists(ids_path):
                raise FileNotFoundError(f"Target IDs file not found at {ids_path}. Run dense_pipeline first!")

            print(f"[DenseRetriever] Loading target IDs from {ids_path}...")
            with open(ids_path, "r", encoding="utf-8") as f:
                self.target_ids = json.load(f)
            print(f"[DenseRetriever] Loaded {len(self.target_ids):,} target IDs.")

            # 3. Load ANN Index / MMap Embeddings
            index_path = os.path.join(self.ann_dir, "target.index")
            embeddings_path = os.path.join(self.embeddings_dir, "target_embeddings.npy")

            faiss_loaded = False
            try:
                import faiss
                if os.path.exists(index_path) and os.path.getsize(index_path) > 1024:
                    print(f"[DenseRetriever] Loading FAISS index from {index_path}...")
                    self.faiss_index = faiss.read_index(index_path)
                    faiss_loaded = True
                    print(f"[DenseRetriever] FAISS index loaded (ntotal={self.faiss_index.ntotal:,}).")
            except Exception as e:
                print(f"[DenseRetriever] Note: FAISS load skipped/fallback ({e}).")

            if not faiss_loaded:
                if not os.path.exists(embeddings_path):
                    raise FileNotFoundError(f"Target embeddings mmap not found at {embeddings_path}!")
                print(f"[DenseRetriever] Opening disk-backed memmap: {embeddings_path}...")
                self.target_embeddings = np.lib.format.open_memmap(embeddings_path, mode="r")

            self.is_loaded = True

    def retrieve_dense_candidates(
        self,
        s1_df: pl.DataFrame,
        top_k: int = 50
    ) -> Dict[str, Dict[str, Tuple[int, float]]]:
        """
        Computes S1 query embeddings with 'query: ' prefix and retrieves top-K nearest targets.
        Returns:
            Dict[s1_id, Dict[target_id, (PROV_DENSE, cosine_sim_score)]]
        """
        if len(s1_df) == 0:
            return {}

        self.load()

        cols = {c.lower(): c for c in s1_df.columns}
        id_col = cols.get("eid") or cols.get("entity_id") or cols.get("record_id") or s1_df.columns[0]
        s1_ids = [str(x) for x in s1_df[id_col].to_list()]
        n_queries = len(s1_ids)

        # 1. Format S1 text strings with 'query: ' prefix
        s1_texts = vectorized_build_e5_texts(s1_df, is_query=True)

        # 2. Encode S1 queries
        t0 = time.time()
        s1_embs = self.embedder.encode(
            s1_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        encode_time = time.time() - t0

        # 3. ANN Nearest Neighbor Search
        t1 = time.time()
        results: Dict[str, Dict[str, Tuple[int, float]]] = defaultdict(dict)

        if self.faiss_index is not None:
            distances, indices = self.faiss_index.search(s1_embs.astype(np.float32), top_k)
            for q_idx in range(n_queries):
                s1_id = s1_ids[q_idx]
                for rank in range(top_k):
                    t_idx = indices[q_idx][rank]
                    if 0 <= t_idx < len(self.target_ids):
                        target_id = self.target_ids[t_idx]
                        score = float(distances[q_idx][rank])
                        results[s1_id][target_id] = (PROV_DENSE, score)

        else:
            # Memory-efficient chunked NumPy Inner Product search
            n_targets = self.target_embeddings.shape[0]
            chunk_size = 50000

            topk_scores = np.full((n_queries, top_k), -np.inf, dtype=np.float32)
            topk_indices = np.full((n_queries, top_k), -1, dtype=np.int64)

            for c_start in range(0, n_targets, chunk_size):
                c_end = min(n_targets, c_start + chunk_size)
                target_chunk = self.target_embeddings[c_start:c_end].astype(np.float32)
                sims = np.dot(s1_embs, target_chunk.T)

                for q_idx in range(n_queries):
                    q_sims = sims[q_idx]
                    if len(q_sims) <= top_k:
                        loc_top = np.arange(len(q_sims))
                    else:
                        loc_top = np.argpartition(q_sims, -top_k)[-top_k:]

                    cand_scores = np.concatenate([topk_scores[q_idx], q_sims[loc_top]])
                    cand_indices = np.concatenate([topk_indices[q_idx], c_start + loc_top])

                    best_k_idx = np.argsort(cand_scores)[::-1][:top_k]
                    topk_scores[q_idx] = cand_scores[best_k_idx]
                    topk_indices[q_idx] = cand_indices[best_k_idx]

                del target_chunk, sims

            for q_idx in range(n_queries):
                s1_id = s1_ids[q_idx]
                for rank in range(top_k):
                    t_idx = topk_indices[q_idx][rank]
                    if 0 <= t_idx < len(self.target_ids):
                        target_id = self.target_ids[t_idx]
                        score = float(topk_scores[q_idx][rank])
                        results[s1_id][target_id] = (PROV_DENSE, score)

        search_time = time.time() - t1
        print(f"[DenseRetriever] Queried {n_queries:,} S1 entities (Top-{top_k}): Encode {encode_time:.2f}s | Search {search_time:.2f}s | RSS: {get_current_rss_mb():.1f} MB")
        return results

    def close(self):
        """Releases open memory maps and resources."""
        if hasattr(self, "target_embeddings") and self.target_embeddings is not None:
            if hasattr(self.target_embeddings, "_mmap") and self.target_embeddings._mmap is not None:
                self.target_embeddings._mmap.close()
            self.target_embeddings = None
        self.faiss_index = None
        self.target_ids = None
        self.embedder = None
        self.is_loaded = False
        gc.collect()
