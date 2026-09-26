"""
Antigravity V5 Master Hybrid Candidate Generation Pipeline.
Integrates:
- Branch A: V4 Deterministic High-Precision Blockers (Bitmask: 1)
- Branch B: V4 Sparse Lexical Retrieval (Name: 2, Address: 4, Transliteration: 8)
- Branch C: V5 Multilingual E5 Dense ANN Retrieval (Bitmask: 16)
- Unified Candidate Budgeting, Multi-Source Agreement Ranking & Provenance Tagging
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
from src.dense_retriever import DenseRetriever, PROV_DET, PROV_SPARSE_NAME, PROV_SPARSE_ADDR, PROV_SPARSE_TRANSLIT, PROV_DENSE

PROV_V4_MASK = PROV_DET | PROV_SPARSE_NAME | PROV_SPARSE_ADDR | PROV_SPARSE_TRANSLIT


def get_provenance_label(mask: int) -> str:
    """Categorizes provenance into FROM_V4, FROM_DENSE, or FROM_BOTH."""
    has_v4 = bool(mask & PROV_V4_MASK)
    has_dense = bool(mask & PROV_DENSE)
    if has_v4 and has_dense:
        return "FROM_BOTH"
    elif has_dense:
        return "FROM_DENSE"
    else:
        return "FROM_V4"


class V5HybridPipeline:
    """
    Antigravity V5 Production Hybrid Candidate Pipeline.
    Combines Deterministic, Sparse Lexical, and Dense ANN candidates into unified schema.
    """
    def __init__(
        self,
        memory_limit: str = "2GB",
        threads: int = 2,
        dense_embeddings_dir: str = "cache/embeddings/multilingual_e5/production",
        dense_ann_dir: str = "cache/ann/multilingual_e5/production",
        enable_dense: bool = True
    ):
        self.indexer = DuckDBTargetIndexer(memory_limit=memory_limit, threads=threads)
        self.sparse_retriever = V4SparseRetriever(db_path=self.indexer.db_path, memory_limit=memory_limit, threads=threads)
        self.enable_dense = enable_dense
        self.dense_retriever = None
        if self.enable_dense:
            self.dense_retriever = DenseRetriever(
                embeddings_dir=dense_embeddings_dir,
                ann_dir=dense_ann_dir
            )

    def generate_candidates(
        self,
        s1_df: pl.DataFrame,
        enable_deterministic: bool = True,
        enable_sparse_name: bool = True,
        enable_sparse_addr: bool = True,
        enable_sparse_translit: bool = True,
        enable_dense: Optional[bool] = None,
        top_k_deterministic: int = 100,
        top_k_sparse_name: int = 50,
        top_k_sparse_addr: int = 50,
        top_k_sparse_translit: int = 25,
        top_k_dense: int = 50,
        max_candidates_per_s1: int = 250
    ) -> Dict[str, List[Tuple[str, int, Dict[str, Any]]]]:
        """
        Executes unified candidate generation across Deterministic, Sparse, and Dense branches.
        Returns:
            Dict[s1_id, List[(target_id, provenance_mask, candidate_metadata_dict)]]
        """
        if len(s1_df) == 0:
            return {}

        run_dense = self.enable_dense if enable_dense is None else (enable_dense and self.dense_retriever is not None)
        candidates_map: Dict[str, Dict[str, Tuple[int, Dict[str, Any]]]] = defaultdict(dict)

        # 1. Branch A: Deterministic Blocker Retrieval
        if enable_deterministic:
            det_results = self.indexer.query_candidates_for_s1(s1_df, max_cands_per_s1=top_k_deterministic)
            for s1_id, cand_tuples in det_results.items():
                for tid, mask, rec in cand_tuples:
                    rec["dense_similarity"] = 0.0
                    candidates_map[s1_id][tid] = (PROV_DET, rec)

        # 2. Branch B: Sparse Lexical Retrieval
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
                            "dense_similarity": 0.0
                        }
                        candidates_map[s1_id][tid] = (sparse_mask, rec)

        # 3. Branch C: Dense ANN Retrieval
        if run_dense:
            try:
                dense_results = self.dense_retriever.retrieve_dense_candidates(
                    s1_df,
                    top_k=top_k_dense
                )
                for s1_id, t_dict in dense_results.items():
                    for tid, (d_mask, sim_score) in t_dict.items():
                        if tid in candidates_map[s1_id]:
                            prev_mask, rec = candidates_map[s1_id][tid]
                            rec["dense_similarity"] = max(rec.get("dense_similarity", 0.0), sim_score)
                            candidates_map[s1_id][tid] = (prev_mask | d_mask, rec)
                        else:
                            # Target details will be hydrated by indexer if needed
                            rec = {
                                "eid": tid,
                                "norm_name": "",
                                "compact_name": "",
                                "norm_addr": "",
                                "country": "",
                                "name_tokens": [],
                                "addr_tokens": [],
                                "numeric_tokens": [],
                                "dense_similarity": sim_score
                            }
                            candidates_map[s1_id][tid] = (d_mask, rec)
            except Exception as e:
                print(f"[V5HybridPipeline] Dense retrieval warning: {e}. Continuing with V4 candidates.")

        # 4. Candidate Budgeting, Multi-Source Agreement Ranking & Provenance Labeling
        final_results: Dict[str, List[Tuple[str, int, Dict[str, Any]]]] = {}
        for s1_id, cand_dict in candidates_map.items():
            cand_items = list(cand_dict.items())

            for tid, (mask, rec) in cand_items:
                rec["provenance_source"] = get_provenance_label(mask)
                rec["provenance_mask"] = mask

            # Multi-source agreement ranking: agreement score * 1000 + dense_sim
            cand_items.sort(
                key=lambda x: (bin(x[1][0]).count("1") * 1000 + x[1][1].get("dense_similarity", 0.0)),
                reverse=True
            )

            if len(cand_items) > max_candidates_per_s1:
                cand_items = cand_items[:max_candidates_per_s1]

            final_results[s1_id] = [(tid, mask, rec) for tid, (mask, rec) in cand_items]

        return final_results

    def close(self):
        """Closes all retriever submodules."""
        self.indexer.close()
        self.sparse_retriever.close()
        if self.dense_retriever is not None:
            self.dense_retriever.close()
