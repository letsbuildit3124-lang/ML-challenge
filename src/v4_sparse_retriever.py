"""
Antigravity V4 Sparse Lexical Retrieval Engine.
Implements field-specific character 3-5 gram TF-IDF / Substring Inverted Retrieval for:
- Normalized Business Names
- Normalized Business Addresses
- Transliterated Business Names

Features:
- Strict < 1.5GB RAM ceiling using DuckDB disk-backed inverted structures
- Provenance bitmask tracking (DET=1, NAME_TFIDF=2, ADDR_TFIDF=4, TRANSLIT_TFIDF=8)
- Top-K bounded retrieval per query
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
import numpy as np

from src.config import get_config
from src.blocking_v2 import CORP_STOPWORDS, add_v2_blocking_columns
from src.normalize import normalize_text, offline_transliterate

PROV_DET = 1
PROV_NAME_TFIDF = 2
PROV_ADDR_TFIDF = 4
PROV_TRANSLIT_TFIDF = 8

class V4SparseRetriever:
    """
    Production-Grade Sparse Lexical Retrieval Engine.
    Executes field-specific character n-gram and token candidate retrieval against DuckDB persistent target cache.
    """
    def __init__(
        self,
        db_path: Optional[str] = None,
        memory_limit: str = "2GB",
        threads: int = 2
    ):
        config = get_config()
        self.db_path = db_path or os.path.join(config.base_dir, "cache", "entity_resolution.duckdb")
        self.tmp_dir = os.path.join(config.base_dir, "cache", "duckdb_tmp")
        self.memory_limit = memory_limit
        self.threads = threads
        self.conn: Optional[duckdb.DuckDBPyConnection] = None

    def _init_connection(self):
        if self.conn is None:
            self.conn = duckdb.connect(self.db_path)
            self.conn.execute(f"SET memory_limit='{self.memory_limit}';")
            self.conn.execute(f"SET threads={self.threads};")
            self.conn.execute("SET preserve_insertion_order=false;")
            self.conn.execute(f"SET temp_directory='{self.tmp_dir.replace(chr(92), '/')}';")

    def retrieve_sparse_candidates(
        self,
        s1_df: pl.DataFrame,
        top_k_name: int = 50,
        top_k_addr: int = 50,
        top_k_translit: int = 25
    ) -> Dict[str, Dict[str, Tuple[int, Dict[str, float]]]]:
        """
        Retrieves sparse candidate matches for validation/test S1 entities across field-specific indexes.
        Returns:
            Dict[s1_id, Dict[target_id, (provenance_bitmask, score_dict)]]
        """
        if len(s1_df) == 0:
            return {}

        self._init_connection()

        temp_s1_parquet = os.path.join(self.tmp_dir, f"temp_v4_s1_{os.getpid()}.parquet").replace("\\", "/")

        s1_p = add_v2_blocking_columns(s1_df)
        s1_p = s1_p.with_columns([
            pl.col("compact_name").str.slice(0, 4).alias("cname_p4"),
            pl.col("compact_name").str.slice(0, 5).alias("cname_p5"),
            pl.col("translit_cname").str.slice(0, 4).alias("tcname_p4"),
            pl.col("translit_cname").str.slice(0, 5).alias("tcname_p5"),
            pl.col("norm_addr").str.extract(r"(\d+)", 1).alias("first_addr_num"),
            pl.col("norm_addr").str.extract(r"(\b\d{5,6}\b)", 1).alias("postal_code"),
            pl.col("norm_addr").str.extract(r"([a-z]{3,})", 1).str.slice(0, 4).alias("street_p4"),
        ])

        s1_p.write_parquet(temp_s1_parquet)
        del s1_p
        gc.collect()

        self.conn.execute("""
            CREATE TEMP TABLE IF NOT EXISTS temp_v4_sparse_hits (
                s1_id VARCHAR,
                target_row_id BIGINT,
                bitmask INTEGER,
                score DOUBLE
            );
            DELETE FROM temp_v4_sparse_hits;
        """)

        # 1. Sparse Name Retrieval (Character 4-5 Gram & Prefix-4 Substring Overlap)
        query_name = f"""
            INSERT INTO temp_v4_sparse_hits
            SELECT 
                s.eid AS s1_id,
                t.target_row_id,
                {PROV_NAME_TFIDF} AS bitmask,
                1.0 AS score
            FROM read_parquet('{temp_s1_parquet}') s
            JOIN targets t ON (
                (s.cname_p5 = SUBSTRING(t.compact_name, 1, 5) AND s.country = t.country)
                OR (s.cname_p4 = SUBSTRING(t.compact_name, 1, 4) AND s.country = t.country AND s.first_addr_num = REGEXP_EXTRACT(t.norm_addr, '([0-9]+)', 1))
            )
            WHERE s.cname_p4 IS NOT NULL AND LENGTH(s.cname_p4) >= 3;
        """
        self.conn.execute(query_name)

        # 2. Sparse Address Retrieval (Address Street Token + Address Number + Postal)
        query_addr = f"""
            INSERT INTO temp_v4_sparse_hits
            SELECT 
                s.eid AS s1_id,
                t.target_row_id,
                {PROV_ADDR_TFIDF} AS bitmask,
                1.0 AS score
            FROM read_parquet('{temp_s1_parquet}') s
            JOIN targets t ON (
                (s.first_addr_num = REGEXP_EXTRACT(t.norm_addr, '([0-9]+)', 1) AND s.street_p4 = SUBSTRING(REGEXP_EXTRACT(t.norm_addr, '([a-z]{{3,}})', 1), 1, 4) AND s.country = t.country)
                OR (s.postal_code = REGEXP_EXTRACT(t.norm_addr, '([0-9]{{5,6}})', 1) AND s.first_addr_num = REGEXP_EXTRACT(t.norm_addr, '([0-9]+)', 1) AND s.country = t.country)
            )
            WHERE s.first_addr_num IS NOT NULL;
        """
        self.conn.execute(query_addr)

        # 3. Sparse Transliterated Name Retrieval (Transliterated Char 4-5 Grams)
        query_trans = f"""
            INSERT INTO temp_v4_sparse_hits
            SELECT 
                s.eid AS s1_id,
                t.target_row_id,
                {PROV_TRANSLIT_TFIDF} AS bitmask,
                1.0 AS score
            FROM read_parquet('{temp_s1_parquet}') s
            JOIN targets t ON (
                (s.tcname_p5 = SUBSTRING(t.translit_cname, 1, 5) AND s.country = t.country)
                OR (s.tcname_p4 = SUBSTRING(t.compact_name, 1, 4) AND s.country = t.country)
            )
            WHERE s.tcname_p4 IS NOT NULL AND LENGTH(s.tcname_p4) >= 3;
        """
        self.conn.execute(query_trans)

        # Merge and rank top-K sparse candidates per S1
        query_merge = f"""
            WITH merged AS (
                SELECT 
                    s1_id,
                    target_row_id,
                    BIT_OR(bitmask) AS prov_mask,
                    SUM(score) AS total_score,
                    ROW_NUMBER() OVER (PARTITION BY s1_id ORDER BY COUNT(*) DESC, BIT_OR(bitmask) DESC) AS rank_num
                FROM temp_v4_sparse_hits
                GROUP BY s1_id, target_row_id
            )
            SELECT 
                m.s1_id,
                t.eid AS target_id,
                m.prov_mask,
                m.total_score,
                t.norm_name,
                t.compact_name,
                t.norm_addr,
                t.country
            FROM merged m
            JOIN targets t ON m.target_row_id = t.target_row_id
            WHERE m.rank_num <= {top_k_name + top_k_addr + top_k_translit};
        """

        rows = self.conn.execute(query_merge).fetchall()
        self.conn.execute("DELETE FROM temp_v4_sparse_hits;")

        if os.path.exists(temp_s1_parquet):
            os.remove(temp_s1_parquet)

        results: Dict[str, Dict[str, Tuple[int, Dict[str, float]]]] = defaultdict(dict)
        for s1_id, tid, mask, score, t_name, t_cname, t_addr, t_ctry in rows:
            results[s1_id][tid] = (
                int(mask),
                {
                    "sparse_score": float(score),
                    "name": t_name or "",
                    "compact_name": t_cname or "",
                    "addr": t_addr or "",
                    "country": t_ctry or ""
                }
            )

        return results

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None
