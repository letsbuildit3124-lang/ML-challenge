"""
Antigravity V3 Arctic Semantic Candidate Generation Module.
Provides bounded Top-K semantic candidate retrieval and merging with multi-pass deterministic blocking.
"""

import time
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np

from src.arctic_similarity import ArcticSimilarityScorer
from src.arctic_embeddings import ArcticEmbedder

def retrieve_topk_semantic_candidates(
    s1_ids: List[str],
    s1_embeddings: np.ndarray,
    target_ids: List[str],
    target_embeddings: np.ndarray,
    top_k: int = 10,
    min_sim_threshold: float = 0.55,
    chunk_size: int = 500
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Computes bounded Top-K semantic nearest neighbors for S1 entities against Target pool.
    Uses chunked matrix multiplication for bounded memory and CPU parallelism.
    
    Returns:
        Dict[s1_id, List[(target_id, cosine_sim)]]
    """
    semantic_cands: Dict[str, List[Tuple[str, float]]] = {}
    num_s1 = len(s1_ids)
    num_tgt = len(target_ids)

    if num_s1 == 0 or num_tgt == 0:
        return semantic_cands

    k = min(top_k, num_tgt)
    t0 = time.time()

    for i in range(0, num_s1, chunk_size):
        chunk_s1_ids = s1_ids[i : i + chunk_size]
        chunk_s1_embs = s1_embeddings[i : i + chunk_size] # (B, 384)

        # Matrix multiply: (B, 384) @ (384, N_tgt) -> (B, N_tgt)
        sim_matrix = np.dot(chunk_s1_embs, target_embeddings.T)

        for row_idx, s1_id in enumerate(chunk_s1_ids):
            row_sims = sim_matrix[row_idx]
            
            # Find top-K indices using argpartition
            if num_tgt > k:
                top_indices = np.argpartition(row_sims, -k)[-k:]
                top_indices = top_indices[np.argsort(-row_sims[top_indices])]
            else:
                top_indices = np.argsort(-row_sims)

            cands_for_s1 = []
            for t_idx in top_indices:
                sim_val = float(row_sims[t_idx])
                if sim_val >= min_sim_threshold:
                    cands_for_s1.append((target_ids[t_idx], sim_val))

            if cands_for_s1:
                semantic_cands[s1_id] = cands_for_s1

    elapsed = time.time() - t0
    total_retrieved = sum(len(v) for v in semantic_cands.values())
    avg_per_s1 = total_retrieved / max(num_s1, 1)
    print(f"[ArcticSemanticGen] Retrieved {total_retrieved:,} semantic candidates for {num_s1:,} S1 entities in {elapsed:.2f}s (Avg: {avg_per_s1:.1f}/S1).")
    return semantic_cands


def merge_candidate_dictionaries(
    deterministic_cands: Dict[str, List[str]],
    semantic_cands: Dict[str, List[Tuple[str, float]]],
    max_total_cands: int = 50
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], int]]:
    """
    Merges deterministic blocker candidates and semantic candidates.
    Updates candidate lists and constructs provenance bitmasks.
    
    Bit 0-4: Deterministic blockers (Exact name, compact name, soundex, postal, translit)
    Bit 5 (32): Semantic Arctic Blocker
    """
    merged_cands: Dict[str, List[str]] = {}
    provenance_map: Dict[Tuple[str, str], int] = {}

    all_s1_ids = set(deterministic_cands.keys()) | set(semantic_cands.keys())

    for s1_id in all_s1_ids:
        det_list = deterministic_cands.get(s1_id, [])
        sem_list = semantic_cands.get(s1_id, [])

        combined_set: Set[str] = set()
        final_list: List[str] = []

        # 1. Add deterministic candidates first (priority)
        for tid in det_list:
            if tid not in combined_set:
                combined_set.add(tid)
                final_list.append(tid)
                # Mark with base deterministic provenance (default 1)
                provenance_map[(s1_id, tid)] = provenance_map.get((s1_id, tid), 1)

        # 2. Add semantic candidates
        for tid, sim_score in sem_list:
            mask = provenance_map.get((s1_id, tid), 0)
            mask |= (1 << 5) # Set bit 5 for semantic retrieval
            provenance_map[(s1_id, tid)] = mask

            if tid not in combined_set and len(final_list) < max_total_cands:
                combined_set.add(tid)
                final_list.append(tid)

        merged_cands[s1_id] = final_list

    return merged_cands, provenance_map
