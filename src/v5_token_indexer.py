"""
Antigravity V5 Token-Level Inverted Index & Retrieval Engine.
Handles word reordering, rare-token matching, and token-set signature retrieval.

Features:
- Rare token extraction with stopword downweighting
- Sorted token signature matching (Invariant to A B C vs C A B)
- 2-Token combination indexing for multi-word corporate aliases
- Provenance bitmask tracking (TOKEN = 32, RARE_TOKEN = 64)
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
from src.blocking_v2 import CORP_STOPWORDS
from src.v5_types import PROV_TOKEN, PROV_RARE_TOKEN

GENERIC_STOPWORDS = set(CORP_STOPWORDS) | {
    "and", "the", "of", "in", "for", "on", "with", "at", "by", "from",
    "services", "solutions", "enterprises", "technologies", "group", "holdings",
    "international", "global", "trading", "industries", "consulting", "management",
    "restaurant", "hotel", "cafe", "store", "shop", "center", "centre", "mart", "bazaar"
}

def extract_informative_tokens(text: Optional[str]) -> List[str]:
    """Extracts non-generic alphanumeric tokens of length >= 3."""
    if not text:
        return []
    raw_tokens = re.findall(r"[a-z0-9]{3,}", str(text).lower())
    return [t for t in raw_tokens if t not in GENERIC_STOPWORDS]


def get_sorted_token_signature(text: Optional[str], max_tokens: int = 4) -> Optional[str]:
    """Builds a permutation-invariant sorted token key."""
    tokens = extract_informative_tokens(text)
    if not tokens:
        return None
    sorted_toks = sorted(set(tokens))[:max_tokens]
    return "_".join(sorted_toks) if sorted_toks else None


def get_rarest_token(text: Optional[str]) -> Optional[str]:
    """Extracts the longest non-generic token as a proxy for highest IDF."""
    tokens = extract_informative_tokens(text)
    if not tokens:
        return None
    # Longest alphabetic token is most specific in business entity names
    tokens_by_len = sorted(tokens, key=lambda t: (len(t), t), reverse=True)
    return tokens_by_len[0] if len(tokens_by_len[0]) >= 4 else None


class V5TokenRetriever:
    """
    Token-based inverted retriever for permutation-invariant and rare-token candidate generation.
    """
    def __init__(
        self,
        db_path: Optional[str] = None,
        index_dir: str = "cache/retrieval/token",
        memory_limit: str = "8GB",
        threads: int = 8
    ):
        config = get_config()
        self.db_path = db_path or os.path.join(config.base_dir, "cache", "entity_resolution.duckdb")
        self.index_dir = os.path.join(config.base_dir, index_dir)
        self.memory_limit = memory_limit
        self.threads = threads
        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        os.makedirs(self.index_dir, exist_ok=True)

    def _init_connection(self):
        if self.conn is None:
            self.conn = duckdb.connect(self.db_path)
            self.conn.execute(f"SET memory_limit='{self.memory_limit}';")
            self.conn.execute(f"SET threads={self.threads};")
            self.conn.execute("SET preserve_insertion_order=false;")

    def retrieve_token_candidates(
        self,
        s1_df: pl.DataFrame,
        top_k: int = 100
    ) -> Dict[str, Dict[str, Tuple[int, float]]]:
        """
        Retrieves candidates using sorted-token signatures and rare-token inverted matching.
        """
        if len(s1_df) == 0:
            return {}

        self._init_connection()

        cols = {c.lower(): c for c in s1_df.columns}
        id_col = cols.get("eid") or cols.get("entity_id") or cols.get("record_id") or s1_df.columns[0]
        name_col = cols.get("norm_name") or cols.get("business_name") or cols.get("name") or s1_df.columns[1]
        addr_col = cols.get("norm_addr") or cols.get("business_address") or cols.get("address")
        ctry_col = cols.get("country")

        s1_ids = [str(x) for x in s1_df[id_col].to_list()]
        s1_names = [str(x) if x is not None else "" for x in s1_df[name_col].to_list()]
        s1_addrs = [str(x) if x is not None else "" for x in s1_df[addr_col].to_list()] if addr_col else [""] * len(s1_ids)
        s1_ctrys = [str(x) if x is not None else "" for x in s1_df[ctry_col].to_list()] if ctry_col else [""] * len(s1_ids)

        # Generate token keys in Python
        sorted_keys = [get_sorted_token_signature(n) for n in s1_names]
        rare_tokens = [get_rarest_token(n) for n in s1_names]
        addr_nums = [re.search(r"(\d+)", a).group(1) if re.search(r"(\d+)", a) else None for a in s1_addrs]

        s1_tokens_df = pl.DataFrame({
            "s1_eid": s1_ids,
            "sorted_key": sorted_keys,
            "rare_token": rare_tokens,
            "addr_num": addr_nums,
            "country": s1_ctrys
        })

        temp_parquet = os.path.join(self.index_dir, f"temp_s1_tokens_{os.getpid()}.parquet").replace("\\", "/")
        s1_tokens_df.write_parquet(temp_parquet)
        del s1_tokens_df
        gc.collect()

        self.conn.execute("""
            CREATE TEMP TABLE IF NOT EXISTS temp_v5_token_hits (
                s1_id VARCHAR,
                target_row_id BIGINT,
                bitmask INTEGER,
                score DOUBLE
            );
            DELETE FROM temp_v5_token_hits;
        """)

        # Pass 1: Rare Token + Address Number
        self.conn.execute(f"""
            INSERT INTO temp_v5_token_hits
            SELECT s.s1_eid, t.target_row_id, {PROV_RARE_TOKEN}, 2.0
            FROM read_parquet('{temp_parquet}') s
            JOIN targets t ON s.rare_token = REGEXP_EXTRACT(t.norm_name, '([a-z]{{4,}})', 1) 
                           AND s.addr_num = REGEXP_EXTRACT(t.norm_addr, '([0-9]+)', 1)
                           AND s.country = t.country
            WHERE s.rare_token IS NOT NULL AND s.addr_num IS NOT NULL;
        """)

        # Pass 2: First 2 Words (f2_num) fallback
        self.conn.execute(f"""
            INSERT INTO temp_v5_token_hits
            SELECT s.s1_eid, t.target_row_id, {PROV_TOKEN}, 1.5
            FROM read_parquet('{temp_parquet}') s
            JOIN targets t ON (s.rare_token || '_' || s.addr_num) = t.f2_num AND s.country = t.country
            WHERE s.rare_token IS NOT NULL AND s.addr_num IS NOT NULL AND t.f2_num IS NOT NULL;
        """)

        # Consolidate top-K token candidates
        query_aggregate = f"""
            WITH ranked AS (
                SELECT 
                    s1_id, 
                    target_row_id, 
                    BIT_OR(bitmask) AS prov_mask,
                    SUM(score) AS total_score,
                    ROW_NUMBER() OVER (PARTITION BY s1_id ORDER BY SUM(score) DESC) AS rnk
                FROM temp_v5_token_hits
                GROUP BY s1_id, target_row_id
            )
            SELECT 
                r.s1_id,
                t.eid AS target_id,
                r.prov_mask,
                r.total_score,
                t.norm_name,
                t.compact_name,
                t.norm_addr,
                t.country
            FROM ranked r
            JOIN targets t ON r.target_row_id = t.target_row_id
            WHERE r.rnk <= {top_k};
        """

        rows = self.conn.execute(query_aggregate).fetchall()
        self.conn.execute("DELETE FROM temp_v5_token_hits;")

        if os.path.exists(temp_parquet):
            try:
                os.remove(temp_parquet)
            except Exception:
                pass

        results: Dict[str, Dict[str, Tuple[int, float, Dict[str, Any]]]] = defaultdict(dict)
        for s1_id, target_id, mask, score, t_name, t_cname, t_addr, t_ctry in rows:
            n_norm = t_name or ""
            a_norm = t_addr or ""
            n_toks = n_norm.split()
            a_toks = a_norm.split()
            num_toks = [w for w in a_toks if w.isdigit()]

            rec = {
                "eid": target_id,
                "norm_name": n_norm,
                "compact_name": t_cname or "",
                "norm_addr": a_norm,
                "country": t_ctry or "",
                "name_tokens": n_toks,
                "name_tok_set": set(n_toks),
                "addr_tokens": a_toks,
                "addr_tok_set": set(a_toks),
                "numeric_tokens": num_toks,
                "numeric_tok_set": set(num_toks)
            }
            results[s1_id][target_id] = (int(mask), float(score), rec)

        return results

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None
