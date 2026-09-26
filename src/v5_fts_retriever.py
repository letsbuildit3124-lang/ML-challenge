"""
Antigravity V5 DuckDB Full-Text Search (FTS) & BM25 Retrieval Adapter.
Leverages DuckDB native FTS index extension for multi-field BM25 text retrieval.

Features:
- Native C++ BM25 ranking inside DuckDB engine
- Multi-field scoring across normalized business names & addresses
- Provenance bitmask tracking (FTS = 512)
"""

import os
import gc
import json
import time
import re
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
import duckdb
import polars as pl

from src.config import get_config
from src.v5_types import PROV_FTS

class V5FTSRetriever:
    """
    DuckDB Full-Text Search (BM25) Candidate Retriever.
    """
    def __init__(
        self,
        db_path: Optional[str] = None,
        memory_limit: str = "8GB",
        threads: int = 8
    ):
        config = get_config()
        self.db_path = db_path or os.path.join(config.base_dir, "cache", "entity_resolution.duckdb")
        self.memory_limit = memory_limit
        self.threads = threads
        self.conn: Optional[duckdb.DuckDBPyConnection] = None

    def _init_connection(self):
        if self.conn is None:
            self.conn = duckdb.connect(self.db_path)
            self.conn.execute(f"SET memory_limit='{self.memory_limit}';")
            self.conn.execute(f"SET threads={self.threads};")
            self.conn.execute("SET preserve_insertion_order=false;")

    def retrieve_fts_candidates(
        self,
        s1_df: pl.DataFrame,
        top_k: int = 50
    ) -> Dict[str, Dict[str, Tuple[int, float]]]:
        """
        Retrieves candidates using DuckDB FTS / BM25 search.
        If FTS index does not exist in DB, falls back gracefully without errors.
        """
        if len(s1_df) == 0:
            return {}

        self._init_connection()

        cols = {c.lower(): c for c in s1_df.columns}
        id_col = cols.get("eid") or cols.get("entity_id") or cols.get("record_id") or s1_df.columns[0]
        name_col = cols.get("norm_name") or cols.get("business_name") or cols.get("name") or s1_df.columns[1]

        # Check if FTS index exists
        try:
            test_res = self.conn.execute("SELECT fts_main_targets.match_bm25(1, 'test');").fetchone()
        except Exception:
            # FTS index not created on this DB, return empty dict without crashing
            return {}

        results: Dict[str, Dict[str, Tuple[int, float]]] = defaultdict(dict)
        for row in s1_df.iter_rows(named=True):
            s1_id = str(row[id_col])
            raw_name = str(row.get(name_col) or "")
            cleaned_query = re.sub(r"[^a-zA-Z0-9\s]", " ", raw_name).strip()
            if not cleaned_query:
                continue

            query_sql = f"""
                SELECT eid, score FROM (
                    SELECT eid, fts_main_targets.match_bm25(target_row_id, '{cleaned_query}') AS score 
                    FROM targets
                ) WHERE score IS NOT NULL 
                ORDER BY score DESC 
                LIMIT {top_k};
            """
            try:
                hits = self.conn.execute(query_sql).fetchall()
                for target_id, score in hits:
                    results[s1_id][target_id] = (PROV_FTS, float(score))
            except Exception:
                continue

        return results

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None
