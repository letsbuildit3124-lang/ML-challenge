"""
Antigravity V3 Arctic Cosine Similarity Engine.
Provides high-speed vector similarity lookup and feature integration for candidate pairs.
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.arctic_embeddings import EMBEDDING_DIM

class ArcticSimilarityScorer:
    """
    High-speed pairwise similarity engine backed by disk memory-mapped embeddings.
    """
    def __init__(self, cache_dir: Optional[str] = None, split: str = "train"):
        self.cache_dir = cache_dir
        self.split = split
        self.s1_embeddings: Optional[np.ndarray] = None
        self.s1_id_to_idx: Dict[str, int] = {}
        self.target_embeddings: Optional[np.ndarray] = None
        self.target_id_to_idx: Dict[str, int] = {}
        self.is_loaded = False

        if cache_dir and os.path.exists(cache_dir):
            self.load_cache(cache_dir, split=split)

    def load_cache(self, cache_dir: str, split: str = "train"):
        """Loads memory-mapped embeddings and ID indexes from disk cache."""
        self.cache_dir = cache_dir
        self.split = split
        print(f"[ArcticSimilarity] Loading embedding cache from: {cache_dir} (Split: {split})...")

        # 1. Load S1
        s1_npy = os.path.join(cache_dir, f"{split}_s1_embeddings.npy")
        s1_json = os.path.join(cache_dir, f"{split}_s1_ids.json")

        if os.path.exists(s1_npy) and os.path.exists(s1_json):
            self.s1_embeddings = np.load(s1_npy, mmap_mode="r")
            with open(s1_json, "r", encoding="utf-8") as f:
                s1_ids = json.load(f)
            self.s1_id_to_idx = {sid: idx for idx, sid in enumerate(s1_ids)}
            print(f"  Loaded S1 embeddings: {self.s1_embeddings.shape} ({len(self.s1_id_to_idx):,} IDs)")
        else:
            print(f"  Warning: S1 embedding files not found in {cache_dir}")

        # 2. Load S2 and S3 targets
        target_embs_list = []
        target_ids_list = []

        for src in ["s2", "s3"]:
            t_npy = os.path.join(cache_dir, f"{split}_{src}_embeddings.npy")
            t_json = os.path.join(cache_dir, f"{split}_{src}_ids.json")
            if os.path.exists(t_npy) and os.path.exists(t_json):
                emb = np.load(t_npy, mmap_mode="r")
                with open(t_json, "r", encoding="utf-8") as f:
                    t_ids = json.load(f)
                target_embs_list.append(emb)
                target_ids_list.extend(t_ids)
                print(f"  Loaded {src.upper()} embeddings: {emb.shape} ({len(t_ids):,} IDs)")

        if target_embs_list:
            self.target_embeddings = np.vstack(target_embs_list) if len(target_embs_list) > 1 else target_embs_list[0]
            self.target_id_to_idx = {tid: idx for idx, tid in enumerate(target_ids_list)}
            print(f"  Total Indexed Targets: {len(self.target_id_to_idx):,}")

        self.is_loaded = (self.s1_embeddings is not None and self.target_embeddings is not None)

    def compute_similarity(self, s1_id: str, target_id: str) -> float:
        """
        Computes pairwise cosine similarity between S1 entity and Target entity.
        Since embeddings are L2 normalized, cosine similarity is the dot product.
        """
        if not self.is_loaded:
            return 0.0

        idx_s1 = self.s1_id_to_idx.get(s1_id)
        idx_t = self.target_id_to_idx.get(target_id)

        if idx_s1 is None or idx_t is None:
            return 0.0

        v1 = self.s1_embeddings[idx_s1]
        v2 = self.target_embeddings[idx_t]
        sim = float(np.dot(v1, v2))
        return float(np.clip(sim, -1.0, 1.0))

    def compute_batch_similarities(self, s1_id: str, candidate_ids: List[str]) -> np.ndarray:
        """
        High-speed batch matrix-vector dot product for all candidate targets of a single S1.
        Returns float32 array of similarities.
        """
        if not self.is_loaded or not candidate_ids:
            return np.zeros(len(candidate_ids), dtype=np.float32)

        idx_s1 = self.s1_id_to_idx.get(s1_id)
        if idx_s1 is None:
            return np.zeros(len(candidate_ids), dtype=np.float32)

        v_s1 = self.s1_embeddings[idx_s1]

        # Gather target vector indices
        valid_indices = []
        pos_mapping = []
        for i, tid in enumerate(candidate_ids):
            t_idx = self.target_id_to_idx.get(tid)
            if t_idx is not None:
                valid_indices.append(t_idx)
                pos_mapping.append(i)

        sims = np.zeros(len(candidate_ids), dtype=np.float32)
        if valid_indices:
            target_sub_matrix = self.target_embeddings[valid_indices]
            computed_sims = np.dot(target_sub_matrix, v_s1)
            for pos, sim_val in zip(pos_mapping, computed_sims):
                sims[pos] = np.clip(sim_val, -1.0, 1.0)

        return sims
