"""
Antigravity V5 Unified CPU-Only High-Recall Retrieval Engine.
Orchestrates multi-pass candidate retrieval across deterministic, character n-gram,
rare token, structural address, DuckDB FTS, and RapidFuzz C++ reranking layers.

Features:
- Full CPU optimization utilizing 8 vCPUs & 32 GB RAM
- Compact internal integer ID manipulation
- Comprehensive provenance bitmask tracking (11 channels)
- Bounded memory profile (< 20 GB peak RSS)
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
from src.v5_types import (
    PROV_DET, PROV_NAME_NGRAM_3, PROV_NAME_NGRAM_4, PROV_NAME_NGRAM_5,
    PROV_ADDR_NGRAM, PROV_TOKEN, PROV_RARE_TOKEN, PROV_ADDR_NUM,
    PROV_TRANSLIT, PROV_FTS, PROV_FUZZY, decode_provenance
)
from src.v5_ngram_indexer import V5NgramRetriever
from src.v5_token_indexer import V5TokenRetriever
from src.v5_address_indexer import V5AddressRetriever
from src.v5_fts_retriever import V5FTSRetriever
from src.v5_fuzzy_reranker import V5FuzzyReranker
from src.resource_tracker import get_current_rss_mb, get_peak_rss_mb, MemoryTracker

class V5RetrievalEngine:
    """
    Antigravity V5 High-Recall Multi-Pass Retrieval Engine.
    """
    def __init__(
        self,
        memory_limit: str = "8GB",
        threads: int = 8,
        workers: int = 8
    ):
        config = get_config()
        self.db_path = os.path.join(config.base_dir, "cache", "entity_resolution.duckdb")
        self.memory_limit = memory_limit
        self.threads = threads
        self.workers = workers

        self.indexer = DuckDBTargetIndexer(memory_limit=memory_limit, threads=threads)
        self.ngram_retriever = V5NgramRetriever(db_path=self.db_path, memory_limit=memory_limit, threads=threads)
        self.token_retriever = V5TokenRetriever(db_path=self.db_path, memory_limit=memory_limit, threads=threads)
        self.address_retriever = V5AddressRetriever(db_path=self.db_path, memory_limit=memory_limit, threads=threads)
        self.fts_retriever = V5FTSRetriever(db_path=self.db_path, memory_limit=memory_limit, threads=threads)
        self.fuzzy_reranker = V5FuzzyReranker(workers=workers)

    def generate_candidates(
        self,
        s1_df: pl.DataFrame,
        enable_deterministic: bool = True,
        enable_ngram: bool = True,
        enable_token: bool = True,
        enable_address: bool = True,
        enable_fts: bool = False,
        enable_fuzzy_rerank: bool = True,
        top_k_deterministic: int = 150,
        top_k_ngram: int = 100,
        top_k_token: int = 100,
        top_k_address: int = 100,
        top_k_fts: int = 50,
        max_candidates_per_s1: int = 250,
        enable_early_prune: bool = False
    ) -> Dict[str, List[Tuple[str, int, float, Dict[str, Any]]]]:
        """
        Executes multi-pass candidate retrieval and consolidates results per S1 entity.
        Returns:
            Dict[s1_id, List[(target_id, provenance_bitmask, fuzzy_score, record_dict)]]
        """
        if len(s1_df) == 0:
            return {}

        candidates_map: Dict[str, Dict[str, Tuple[int, Dict[str, Any]]]] = defaultdict(dict)

        # Prepare lookup dictionary for S1 entity names/addresses
        cols = {c.lower(): c for c in s1_df.columns}
        id_col = cols.get("eid") or cols.get("entity_id") or cols.get("record_id") or s1_df.columns[0]
        name_col = cols.get("norm_name") or cols.get("business_name") or cols.get("name") or s1_df.columns[1]
        addr_col = cols.get("norm_addr") or cols.get("business_address") or cols.get("address")
        ctry_col = cols.get("country")

        s1_lookup = {}
        for row in s1_df.iter_rows(named=True):
            eid = str(row[id_col])
            s1_lookup[eid] = {
                "s1_norm_name": str(row.get(name_col) or ""),
                "s1_norm_addr": str(row.get(addr_col) or ""),
                "s1_country": str(row.get(ctry_col) or "")
            }

        # 1. Deterministic Blocker Passes
        if enable_deterministic:
            det_results = self.indexer.query_candidates_for_s1(s1_df, max_cands_per_s1=top_k_deterministic)
            for s1_id, cand_tuples in det_results.items():
                s1_info = s1_lookup.get(s1_id, {})
                for tid, mask, rec in cand_tuples:
                    rec_copy = dict(rec)
                    rec_copy.update(s1_info)
                    candidates_map[s1_id][tid] = (PROV_DET, rec_copy)

        # 2. Character N-Gram Inverted Retrieval Passes
        if enable_ngram:
            ngram_results = self.ngram_retriever.retrieve_ngram_candidates(
                s1_df,
                top_k=top_k_ngram,
                enable_char3=True,
                enable_char4=True,
                enable_char5=True,
                enable_addr_ngram=True
            )
            for s1_id, t_dict in ngram_results.items():
                s1_info = s1_lookup.get(s1_id, {})
                for tid, (ng_mask, score, t_rec) in t_dict.items():
                    if tid in candidates_map[s1_id]:
                        prev_mask, rec = candidates_map[s1_id][tid]
                        candidates_map[s1_id][tid] = (prev_mask | ng_mask, rec)
                    else:
                        rec = dict(t_rec)
                        rec.update(s1_info)
                        candidates_map[s1_id][tid] = (ng_mask, rec)

        # 3. Token-Level Inverted Retrieval Passes
        if enable_token:
            token_results = self.token_retriever.retrieve_token_candidates(s1_df, top_k=top_k_token)
            for s1_id, t_dict in token_results.items():
                s1_info = s1_lookup.get(s1_id, {})
                for tid, (tok_mask, score, t_rec) in t_dict.items():
                    if tid in candidates_map[s1_id]:
                        prev_mask, rec = candidates_map[s1_id][tid]
                        candidates_map[s1_id][tid] = (prev_mask | tok_mask, rec)
                    else:
                        rec = dict(t_rec)
                        rec.update(s1_info)
                        candidates_map[s1_id][tid] = (tok_mask, rec)

        # 4. Dedicated Address Structural Retrieval Passes
        if enable_address:
            addr_results = self.address_retriever.retrieve_address_candidates(s1_df, top_k=top_k_address)
            for s1_id, t_dict in addr_results.items():
                s1_info = s1_lookup.get(s1_id, {})
                for tid, (addr_mask, score, t_rec) in t_dict.items():
                    if tid in candidates_map[s1_id]:
                        prev_mask, rec = candidates_map[s1_id][tid]
                        candidates_map[s1_id][tid] = (prev_mask | addr_mask, rec)
                    else:
                        rec = dict(t_rec)
                        rec.update(s1_info)
                        candidates_map[s1_id][tid] = (addr_mask, rec)

        # 5. DuckDB FTS / BM25 Adapter Passes (Optional)
        if enable_fts:
            fts_results = self.fts_retriever.retrieve_fts_candidates(s1_df, top_k=top_k_fts)
            for s1_id, t_dict in fts_results.items():
                s1_info = s1_lookup.get(s1_id, {})
                for tid, (fts_mask, score) in t_dict.items():
                    if tid in candidates_map[s1_id]:
                        prev_mask, rec = candidates_map[s1_id][tid]
                        candidates_map[s1_id][tid] = (prev_mask | fts_mask, rec)
                    else:
                        rec = {"eid": tid, "norm_name": "", "norm_addr": ""}
                        rec.update(s1_info)
                        candidates_map[s1_id][tid] = (fts_mask, rec)

        # 6. RapidFuzz C++ Multi-Worker Reranker & Budgeting
        if enable_fuzzy_rerank:
            final_results = self.fuzzy_reranker.rerank_and_prune_candidates(
                candidates_map,
                top_k=max_candidates_per_s1,
                enable_gate=enable_early_prune
            )
        else:
            final_results = {}
            for s1_id, cand_dict in candidates_map.items():
                items = [(tid, mask, 0.0, rec) for tid, (mask, rec) in cand_dict.items()]
                items.sort(key=lambda x: bin(x[1]).count("1"), reverse=True)
                final_results[s1_id] = items[:max_candidates_per_s1]

        return final_results

    def close(self):
        self.indexer.close()
        self.ngram_retriever.close()
        self.token_retriever.close()
        self.address_retriever.close()
        self.fts_retriever.close()
