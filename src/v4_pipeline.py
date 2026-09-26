"""
Antigravity V4 Hybrid Candidate Generation Pipeline.
Integrates:
- Branch A: Deterministic High-Precision Blockers (Bitmask: 1)
- Branch B: Sparse Lexical Retrieval (Name: 2, Address: 4, Transliteration: 8)
- Candidate Union & Provenance Tracking
- Bounded Candidate Budgeting & Ranking
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
from src.v4_sparse_retriever import V4SparseRetriever, PROV_DET, PROV_NAME_TFIDF, PROV_ADDR_TFIDF, PROV_TRANSLIT_TFIDF

class V4HybridPipeline:
    """
    Production-Grade Hybrid Candidate Generation Pipeline for Antigravity V4.
    """
    def __init__(
        self,
        memory_limit: str = "2GB",
        threads: int = 2
    ):
        self.indexer = DuckDBTargetIndexer(memory_limit=memory_limit, threads=threads)
        self.sparse_retriever = V4SparseRetriever(db_path=self.indexer.db_path, memory_limit=memory_limit, threads=threads)

    def generate_candidates(
        self,
        s1_df: pl.DataFrame,
        enable_deterministic: bool = True,
        enable_sparse_name: bool = True,
        enable_sparse_addr: bool = True,
        enable_sparse_translit: bool = True,
        top_k_deterministic: int = 100,
        top_k_sparse_name: int = 50,
        top_k_sparse_addr: int = 50,
        top_k_sparse_translit: int = 25,
        max_candidates_per_s1: int = 250
    ) -> Dict[str, List[Tuple[str, int, Dict[str, Any]]]]:
        """
        Executes hybrid candidate generation with union & provenance tracking.
        Returns:
            Dict[s1_id, List[(target_id, provenance_mask, target_records_dict)]]
        """
        if len(s1_df) == 0:
            return {}

        candidates_map: Dict[str, Dict[str, Tuple[int, Dict[str, Any]]]] = defaultdict(dict)

        # 1. Branch A: Deterministic Blocker Retrieval
        if enable_deterministic:
            det_results = self.indexer.query_candidates_for_s1(s1_df, max_cands_per_s1=top_k_deterministic)
            for s1_id, cand_tuples in det_results.items():
                for tid, mask, rec in cand_tuples:
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
                            "numeric_tokens": [tok for tok in score_dict.get("addr", "").split() if any(c.isdigit() for c in tok)]
                        }
                        candidates_map[s1_id][tid] = (sparse_mask, rec)

        # 3. Candidate Budgeting and Ranking
        final_results: Dict[str, List[Tuple[str, int, Dict[str, Any]]]] = {}
        for s1_id, cand_dict in candidates_map.items():
            cand_items = list(cand_dict.items())
            # Rank by provenance bitmask priority (multi-source agreement first)
            cand_items.sort(key=lambda x: bin(x[1][0]).count("1") * 1000 + x[1][0], reverse=True)
            if len(cand_items) > max_candidates_per_s1:
                cand_items = cand_items[:max_candidates_per_s1]

            final_results[s1_id] = [(tid, mask, rec) for tid, (mask, rec) in cand_items]

        return final_results

    def close(self):
        self.indexer.close()
        self.sparse_retriever.close()
