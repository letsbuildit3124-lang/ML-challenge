"""
Antigravity V5 Dedicated Address Retrieval Engine.
Implements structural address matching resilient to missing postal codes and format variations.

Features:
- Multi-representation address number parsing (12A, 12-A, 12/1 -> 12)
- Collocation indexing: Address Number + Street Token
- Postal PIN fallback & country-specific street indexing
- Provenance bitmask tracking (ADDR_NUM = 128)
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
from src.v5_types import PROV_ADDR_NUM

def parse_address_components(address: Optional[str]) -> Dict[str, Optional[str]]:
    """
    Extracts structural address components without destructive normalization.
    """
    if not address:
        return {"num": None, "street_token": None, "postal": None}
    
    addr_str = str(address).lower()
    
    # 1. Extract postal code (5-6 digits)
    postal_match = re.search(r"\b(\d{5,6})\b", addr_str)
    postal = postal_match.group(1) if postal_match else None

    # Remove postal code so it is not confused with house number
    addr_no_postal = re.sub(r"\b\d{5,6}\b", " ", addr_str)

    # 2. Extract leading or primary building/unit number (e.g. 12A -> 12, 12-A -> 12, 12/1 -> 12, #45 -> 45)
    num_match = re.search(r"(?:^|[#\s,])(\d+)(?:[a-z\-/]+)?(?:\s|$|,)", addr_no_postal)
    num = num_match.group(1) if num_match else None
    if not num:
        fallback_num = re.search(r"(\d+)", addr_no_postal)
        num = fallback_num.group(1) if fallback_num else None

    # 3. Extract primary street token (first alphabetic word of length >= 3 not in stopwords)
    words = re.findall(r"[a-z]{3,}", addr_no_postal)
    street_stopwords = {"street", "road", "ave", "avenue", "near", "opp", "opposite", "floor", "building", "lane", "nagar", "block", "plot", "shop", "flat"}
    street_token = None
    for w in words:
        if w not in street_stopwords:
            street_token = w[:4] # 4-char prefix
            break

    return {
        "num": num,
        "street_token": street_token,
        "postal": postal
    }


class V5AddressRetriever:
    """
    Dedicated structural address retrieval engine for entity collocation.
    """
    def __init__(
        self,
        db_path: Optional[str] = None,
        index_dir: str = "cache/retrieval/address",
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

    def retrieve_address_candidates(
        self,
        s1_df: pl.DataFrame,
        top_k: int = 100
    ) -> Dict[str, Dict[str, Tuple[int, float]]]:
        """
        Retrieves candidates matching physical address structure.
        """
        if len(s1_df) == 0:
            return {}

        self._init_connection()

        cols = {c.lower(): c for c in s1_df.columns}
        id_col = cols.get("eid") or cols.get("entity_id") or cols.get("record_id") or s1_df.columns[0]
        addr_col = cols.get("norm_addr") or cols.get("business_address") or cols.get("address")
        name_col = cols.get("compact_name") or cols.get("norm_name") or cols.get("name")
        ctry_col = cols.get("country")

        s1_ids = [str(x) for x in s1_df[id_col].to_list()]
        s1_addrs = [str(x) if x is not None else "" for x in s1_df[addr_col].to_list()] if addr_col else [""] * len(s1_ids)
        s1_names = [str(x) if x is not None else "" for x in s1_df[name_col].to_list()] if name_col else [""] * len(s1_ids)
        s1_ctrys = [str(x) if x is not None else "" for x in s1_df[ctry_col].to_list()] if ctry_col else [""] * len(s1_ids)

        parsed_list = [parse_address_components(a) for a in s1_addrs]
        cname4_list = [re.sub(r"[^a-zA-Z0-9]", "", n).lower()[:4] for n in s1_names]

        df_addr = pl.DataFrame({
            "s1_eid": s1_ids,
            "addr_num": [p["num"] for p in parsed_list],
            "street_p4": [p["street_token"] for p in parsed_list],
            "postal": [p["postal"] for p in parsed_list],
            "cname4": cname4_list,
            "country": s1_ctrys
        })

        temp_parquet = os.path.join(self.index_dir, f"temp_s1_addr_{os.getpid()}.parquet").replace("\\", "/")
        df_addr.write_parquet(temp_parquet)
        del df_addr, parsed_list
        gc.collect()

        self.conn.execute("""
            CREATE TEMP TABLE IF NOT EXISTS temp_v5_addr_hits (
                s1_id VARCHAR,
                target_row_id BIGINT,
                bitmask INTEGER,
                score DOUBLE
            );
            DELETE FROM temp_v5_addr_hits;
        """)

        # Pass 1: Postal PIN + Name Prefix-4 (High precision geographic collocation)
        self.conn.execute(f"""
            INSERT INTO temp_v5_addr_hits
            SELECT s.s1_eid, t.target_row_id, {PROV_ADDR_NUM}, 2.5
            FROM read_parquet('{temp_parquet}') s
            JOIN targets t ON (s.postal || '_' || s.cname4) = t.pin_cname4
            WHERE s.postal IS NOT NULL AND s.cname4 IS NOT NULL AND LENGTH(s.cname4) >= 4;
        """)

        # Pass 2: Address Number + Street Prefix-4
        self.conn.execute(f"""
            INSERT INTO temp_v5_addr_hits
            SELECT s.s1_eid, t.target_row_id, {PROV_ADDR_NUM}, 1.5
            FROM read_parquet('{temp_parquet}') s
            JOIN targets t ON (s.addr_num || '_' || s.street_p4) = (REGEXP_EXTRACT(t.norm_addr, '([0-9]+)', 1) || '_' || SUBSTRING(REGEXP_EXTRACT(t.norm_addr, '([a-z]{{3,}})', 1), 1, 4))
                           AND s.country = t.country
            WHERE s.addr_num IS NOT NULL AND s.street_p4 IS NOT NULL;
        """)

        # Consolidate top-K address candidates
        query_aggregate = f"""
            WITH ranked AS (
                SELECT 
                    s1_id, 
                    target_row_id, 
                    BIT_OR(bitmask) AS prov_mask,
                    SUM(score) AS total_score,
                    ROW_NUMBER() OVER (PARTITION BY s1_id ORDER BY SUM(score) DESC) AS rnk
                FROM temp_v5_addr_hits
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
        self.conn.execute("DELETE FROM temp_v5_addr_hits;")

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
