"""
Antigravity V5 Hybrid Candidate Generation Pipeline (V4 Sparse + E5 Dense).
Combines:
- Branch A: V4 Deterministic High-Precision Blockers (Bitmask: 1)
- Branch B: V4 Sparse Lexical TF-IDF (Name: 2, Address: 4, Transliteration: 8)
- Branch C: One-Shot E5 Dense ANN Retrieval (Bitmask: 16)
- Candidate Union, Deduplication, Multi-Source Agreement Ranking & Provenance Tagging

Does NOT replace V4 or XGBoost; supplies an enriched candidate pool to the existing matcher.
"""

import os
import gc
import json
import time
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
import polars as pl
import numpy as np

from src.config import get_config
from src.duckdb_indexer import DuckDBTargetIndexer
from src.v4_sparse_retriever import V4SparseRetriever
from src.e5_dense_retriever import E5DenseRetriever, PROV_E5_DENSE

PROV_DET = 1
PROV_SPARSE_NAME = 2
PROV_SPARSE_ADDR = 4
PROV_SPARSE_TRANSLIT = 8
PROV_V4_MASK = PROV_DET | PROV_SPARSE_NAME | PROV_SPARSE_ADDR | PROV_SPARSE_TRANSLIT


def get_provenance_label(mask: int) -> Tuple[bool, bool, bool, str]:
    """
    Returns (came_from_v4, came_from_e5, came_from_both, provenance_string).
    """
    has_v4 = bool(mask & PROV_V4_MASK)
    has_e5 = bool(mask & PROV_E5_DENSE)
    has_both = has_v4 and has_e5
    
    if has_both:
        label = "FROM_BOTH"
    elif has_e5:
        label = "FROM_E5"
    else:
        label = "FROM_V4"

    return has_v4, has_e5, has_both, label


class V5E5HybridPipeline:
    """
    Antigravity V5 Production Candidate Pipeline with V4 + E5 Dense Hybridization.
    """
    def __init__(
        self,
        memory_limit: str = "2GB",
        threads: int = 2,
        e5_ann_dir: str = "cache/ann/e5/production",
        enable_e5: bool = True
    ):
        self.indexer = DuckDBTargetIndexer(memory_limit=memory_limit, threads=threads)
        self.sparse_retriever = V4SparseRetriever(db_path=self.indexer.db_path, memory_limit=memory_limit, threads=threads)
        self.enable_e5 = enable_e5
        self.e5_retriever: Optional[E5DenseRetriever] = None

        if self.enable_e5:
            try:
                self.e5_retriever = E5DenseRetriever(ann_dir=e5_ann_dir)
            except Exception as e:
                print(f"[V5HybridPipeline] Warning: E5 retriever initialization skipped ({e}).")

    def generate_candidates(
        self,
        s1_df: pl.DataFrame,
        enable_deterministic: bool = True,
        enable_sparse_name: bool = True,
        enable_sparse_addr: bool = True,
        enable_sparse_translit: bool = True,
        enable_e5: Optional[bool] = None,
        top_k_deterministic: int = 100,
        top_k_sparse_name: int = 50,
        top_k_sparse_addr: int = 50,
        top_k_sparse_translit: int = 25,
        top_k_e5: int = 50,
        max_candidates_per_s1: int = 250
    ) -> Dict[str, List[Tuple[str, int, Dict[str, Any]]]]:
        """
        Generates candidates across Deterministic, Sparse, and Dense branches.
        Returns:
            Dict[s1_id, List[(target_id, provenance_mask, candidate_metadata_dict)]]
        """
        if len(s1_df) == 0:
            return {}

        run_e5 = self.enable_e5 if enable_e5 is None else (enable_e5 and self.e5_retriever is not None)
        candidates_map: Dict[str, Dict[str, Tuple[int, Dict[str, Any]]]] = defaultdict(dict)

        # 1. Branch A: V4 Deterministic High-Precision Blockers
        if enable_deterministic:
            det_results = self.indexer.query_candidates_for_s1(s1_df, max_cands_per_s1=top_k_deterministic)
            for s1_id, cand_tuples in det_results.items():
                for tid, mask, rec in cand_tuples:
                    rec["e5_similarity"] = 0.0
                    candidates_map[s1_id][tid] = (PROV_DET, rec)

        # 2. Branch B: V4 Sparse Lexical Retrieval
        if enable_sparse_name or enable_sparse_addr or enable_sparse_translit:
            sparse_results = self.sparse_retriever.retrieve_sparse_candidates(
                s1_df,
                top_k_name=top_k_sparse_name if enable_sparse_name else 0,
                top_k_addr=top_k_sparse_addr if enable_sparse_addr else 0,
                top_k_translit=top_k_sparse_translit if enable_sparse_translit else 0
            )

            for s1_id, t_dict in sparse_results.items():
                for tid, (sparse_mask, score_dict) in t_dict.items():
                    if tid in candidates_map[s1_id]:
                        prev_mask, rec = candidates_map[s1_id][tid]
                        candidates_map[s1_id][tid] = (prev_mask | sparse_mask, rec)
                    else:
                        rec = {
                            "eid": tid,
                            "norm_name": score_dict.get("name", ""),
                            "compact_name": score_dict.get("compact_name", ""),
                            "norm_addr": score_dict.get("addr", ""),
                            "country": score_dict.get("country", ""),
                            "name_tokens": score_dict.get("name", "").split(),
                            "addr_tokens": score_dict.get("addr", "").split(),
                            "numeric_tokens": [tok for tok in score_dict.get("addr", "").split() if any(c.isdigit() for c in tok)],
                            "e5_similarity": 0.0
                        }
                        candidates_map[s1_id][tid] = (sparse_mask, rec)

        # 3. Branch C: E5 Dense ANN Candidate Retrieval
        if run_e5:
            try:
                dense_results = self.e5_retriever.retrieve_dense_candidates(s1_df, top_k=top_k_e5)
                for s1_id, t_dict in dense_results.items():
                    for tid, (d_mask, sim_score) in t_dict.items():
                        if tid in candidates_map[s1_id]:
                            prev_mask, rec = candidates_map[s1_id][tid]
                            rec["e5_similarity"] = max(rec.get("e5_similarity", 0.0), sim_score)
                            candidates_map[s1_id][tid] = (prev_mask | d_mask, rec)
                        else:
                            rec = {
                                "eid": tid,
                                "norm_name": "",
                                "compact_name": "",
                                "norm_addr": "",
                                "country": "",
                                "name_tokens": [],
                                "addr_tokens": [],
                                "numeric_tokens": [],
                                "e5_similarity": sim_score
                            }
                            candidates_map[s1_id][tid] = (d_mask, rec)
            except Exception as e:
                print(f"[V5HybridPipeline] Note: E5 dense retrieval error ({e}). Continuing with V4 candidates.")

        # 4. Candidate Budgeting, Multi-Source Agreement Ranking & Provenance
        final_results: Dict[str, List[Tuple[str, int, Dict[str, Any]]]] = {}
        for s1_id, cand_dict in candidates_map.items():
            cand_items = list(cand_dict.items())

            for tid, (mask, rec) in cand_items:
                has_v4, has_e5, has_both, label = get_provenance_label(mask)
                rec["came_from_v4"] = has_v4
                rec["came_from_e5"] = has_e5
                rec["came_from_both"] = has_both
                rec["provenance_source"] = label
                rec["provenance_mask"] = mask

            # Rank by multi-source agreement and dense similarity
            cand_items.sort(
                key=lambda x: (bin(x[1][0]).count("1") * 1000 + x[1][1].get("e5_similarity", 0.0)),
                reverse=True
            )

            if len(cand_items) > max_candidates_per_s1:
                cand_items = cand_items[:max_candidates_per_s1]

            final_results[s1_id] = [(tid, mask, rec) for tid, (mask, rec) in cand_items]

        return final_results

    def close(self):
        """Closes all underlying resources."""
        self.indexer.close()
        self.sparse_retriever.close()
        if self.e5_retriever is not None:
            self.e5_retriever.close()
