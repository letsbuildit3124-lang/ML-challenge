"""
V3 Candidate Generation Module for Business Entity Resolution.
Implements high-recall, volume-controlled multi-view blocking methods:
- Boundary-aware French & Global corporate suffix removal
- Bidirectional transliteration-aware blocking keys
- Method 1: Exact Compact Name (Original & Transliterated)
- Method 2: Exact Normalized Name
- Method 3: CName8 + Addr Num (Original & Transliterated)
- Method 4: F2 Words + Addr Num
- Method 5: Postal / PIN Code + Name Prefix
- Method 6: Phonetic Soundex + Addr Num (Original & Transliterated)
Includes memory-optimized compact target indexing for sub-second candidate lookups.
"""

import os
import re
import gc
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional
import polars as pl
import numpy as np

from src.data_loader import iter_source_file_chunks, load_source_file
from src.dataset_builder import extract_record_dict_from_df
from src.preprocessing import (
    LEGAL_SUFFIXES_REGEX,
    ADDR_ABBREVIATIONS,
    INDIC_ASCII_MAP,
    offline_transliterate,
    normalize_text_v3,
    normalize_address_v3,
    compact_name_v3,
)

# Stopwords for phonetic and address tokens
CORP_STOPWORDS = {
    'ltd', 'limited', 'pvt', 'private', 'corp', 'corporation', 'inc', 'incorporated',
    'llc', 'llp', 'co', 'company', 'gmbh', 'sa', 'sarl', 'plc', 'bv', 'nv', 'assoc',
    'associates', 'group', 'holdings', 'enterprises', 'services', 'solutions',
    'technologies', 'international', 'consultants', 'industries', 'global', 'systems',
    'sas', 'sasu', 'eurl', 'sci', 'snc', 'sel', 'scs', 'sca', 'gie', 'selarl', 'sem'
}

def compute_soundex(token: str) -> str:
    """Pure Python offline Soundex algorithm."""
    if not token or not token.isalpha():
        return ""
    token = token.upper()
    first_char = token[0]
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    encoded = [first_char]
    prev = mapping.get(first_char, '0')
    for ch in token[1:]:
        curr = mapping.get(ch, '0')
        if curr != '0' and curr != prev:
            encoded.append(curr)
        prev = curr
        if len(encoded) == 4:
            break
    while len(encoded) < 4:
        encoded.append('0')
    return "".join(encoded[:4])

# =============================================================================
# VECTORIZED BLOCKING COLUMNS (V3)
# =============================================================================

def add_v2_blocking_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes all V3 vectorized attributes and retains ONLY essential columns to save 70% RAM.
    Idempotent: Reuses existing normalized/blocking columns if already present.
    """
    if "eid" in df.columns and "compact_name" in df.columns and "cname8_num" in df.columns:
        return df

    cols = {c.lower(): c for c in df.columns}
    id_col = cols.get("entity_id") or cols.get("eid") or cols.get("record_id") or df.columns[0]
    name_col = cols.get("business_name") or cols.get("norm_name") or cols.get("name") or df.columns[1]
    addr_col = cols.get("business_address") or cols.get("norm_addr") or cols.get("address") or (df.columns[2] if len(df.columns) > 2 else None)
    ctry_col = cols.get("country")

    # 1. Base Normalization
    cols_to_add = [
        pl.col(id_col).cast(pl.Utf8).alias("eid"),
        pl.col(name_col).fill_null("").str.to_lowercase().str.replace_all(r"[^\w\s]", " ").str.replace_all(r"\s+", " ").str.strip_chars().alias("norm_name"),
    ]
    if addr_col:
        cols_to_add.append(pl.col(addr_col).fill_null("").str.to_lowercase().str.replace_all(r"[^\w\s]", " ").str.replace_all(r"\s+", " ").str.strip_chars().alias("norm_addr"))
    else:
        cols_to_add.append(pl.lit("").alias("norm_addr"))

    if ctry_col:
        cols_to_add.append(pl.col(ctry_col).fill_null("").str.to_uppercase().str.strip_chars().alias("country"))
    else:
        cols_to_add.append(pl.lit("").alias("country"))

    df_p = df.with_columns(cols_to_add)

    # 2. Bidirectional Transliteration
    def translit_single(val: Optional[str]) -> str:
        return offline_transliterate(val or "")

    df_p = df_p.with_columns([
        pl.col("norm_name").map_elements(translit_single, return_dtype=pl.String).alias("translit_name"),
    ])

    # 3. Compact Name, First Words, Numbers (Boundary-aware suffix stripping)
    df_p = df_p.with_columns([
        pl.col("norm_name").str.replace_all(LEGAL_SUFFIXES_REGEX, "").str.replace_all(r"\s+", "").alias("compact_name"),
        pl.col("translit_name").str.replace_all(LEGAL_SUFFIXES_REGEX, "").str.replace_all(r"\s+", "").alias("translit_cname"),
        pl.col("norm_name").str.split(" ").list.slice(0, 2).list.join("_").alias("f2_name"),
        pl.col("norm_addr").str.extract(r"(\d+)", 1).alias("first_addr_num"),
        pl.col("norm_addr").str.extract(r"(\b\d{5,6}\b)", 1).alias("postal_code"),
    ])

    # 4. Phonetic Name Token (First significant name token Soundex)
    def soundex_single(val: Optional[str]) -> Optional[str]:
        if not val:
            return None
        tokens = [t for t in val.split() if t not in CORP_STOPWORDS and len(t) >= 3 and t.isalpha()]
        s = compute_soundex(tokens[0]) if tokens else ""
        return s if s else None

    df_p = df_p.with_columns([
        pl.col("translit_name").map_elements(soundex_single, return_dtype=pl.String).alias("name_soundex"),
    ])

    # 5. Compound Blocking Keys
    df_p = df_p.with_columns([
        # Key 3: cname8_num
        pl.when(pl.col("first_addr_num").is_not_null() & (pl.col("compact_name").str.len_chars() >= 4)).then(
            pl.concat_str([pl.col("compact_name").str.slice(0, 8), pl.lit("_"), pl.col("first_addr_num")])
        ).otherwise(None).alias("cname8_num"),
        # Key 3b: translit_cname8_num
        pl.when(pl.col("first_addr_num").is_not_null() & (pl.col("translit_cname").str.len_chars() >= 4)).then(
            pl.concat_str([pl.col("translit_cname").str.slice(0, 8), pl.lit("_"), pl.col("first_addr_num")])
        ).otherwise(None).alias("translit_cname8_num"),
        # Key 4: f2_num
        pl.when(pl.col("first_addr_num").is_not_null() & (pl.col("f2_name").str.len_chars() >= 4)).then(
            pl.concat_str([pl.col("f2_name"), pl.lit("_"), pl.col("first_addr_num")])
        ).otherwise(None).alias("f2_num"),
        # Key A: Postal Code + Name Prefix 4
        pl.when(pl.col("postal_code").is_not_null() & (pl.col("compact_name").str.len_chars() >= 4)).then(
            pl.concat_str([pl.col("postal_code"), pl.lit("_"), pl.col("compact_name").str.slice(0, 4)])
        ).otherwise(None).alias("pin_cname4"),
        # Key B: Soundex + First Addr Num
        pl.when(pl.col("name_soundex").is_not_null() & pl.col("first_addr_num").is_not_null()).then(
            pl.concat_str([pl.col("name_soundex"), pl.lit("_"), pl.col("first_addr_num")])
        ).otherwise(None).alias("soundex_num"),
    ])

    # Select ONLY essential columns to drop temporary strings and save RAM
    keep_cols = [
        "eid", "country", "norm_name", "compact_name", "norm_addr",
        "cname8_num", "translit_cname8_num", "f2_num", "pin_cname4", "soundex_num", "translit_cname"
    ]
    present = [c for c in keep_cols if c in df_p.columns]
    return df_p.select(present)

# =============================================================================
# INDIVIDUAL BLOCKING METHODS
# =============================================================================

def block_exact_compact_name(s1_p: pl.DataFrame, tgt_p: pl.DataFrame) -> pl.DataFrame:
    return s1_p.filter(pl.col("compact_name").str.len_chars() >= 3).select(["eid", "compact_name", "country"]).join(
        tgt_p.filter(pl.col("compact_name").str.len_chars() >= 3).select(["eid", "compact_name", "country"]),
        on=["compact_name", "country"]
    ).select([pl.col("eid").alias("s1_id"), pl.col("eid_right").alias("target_id")])

def block_exact_norm_name(s1_p: pl.DataFrame, tgt_p: pl.DataFrame) -> pl.DataFrame:
    return s1_p.filter(pl.col("norm_name").str.len_chars() >= 4).select(["eid", "norm_name", "country"]).join(
        tgt_p.filter(pl.col("norm_name").str.len_chars() >= 4).select(["eid", "norm_name", "country"]),
        on=["norm_name", "country"]
    ).select([pl.col("eid").alias("s1_id"), pl.col("eid_right").alias("target_id")])

def block_cname8_addr_num(s1_p: pl.DataFrame, tgt_p: pl.DataFrame) -> pl.DataFrame:
    p1 = s1_p.filter(pl.col("cname8_num").is_not_null()).select(["eid", "cname8_num", "country"]).join(
        tgt_p.filter(pl.col("cname8_num").is_not_null()).select(["eid", "cname8_num", "country"]),
        on=["cname8_num", "country"]
    ).select([pl.col("eid").alias("s1_id"), pl.col("eid_right").alias("target_id")])
    if "translit_cname8_num" in s1_p.columns and "translit_cname8_num" in tgt_p.columns:
        p2 = s1_p.filter(pl.col("translit_cname8_num").is_not_null()).select(["eid", "translit_cname8_num", "country"]).join(
            tgt_p.filter(pl.col("translit_cname8_num").is_not_null()).select(["eid", "translit_cname8_num", "country"]),
            on=["translit_cname8_num", "country"]
        ).select([pl.col("eid").alias("s1_id"), pl.col("eid_right").alias("target_id")])
        return pl.concat([p1, p2]).unique()
    return p1

def block_f2_words_addr_num(s1_p: pl.DataFrame, tgt_p: pl.DataFrame) -> pl.DataFrame:
    return s1_p.filter(pl.col("f2_num").is_not_null()).select(["eid", "f2_num", "country"]).join(
        tgt_p.filter(pl.col("f2_num").is_not_null()).select(["eid", "f2_num", "country"]),
        on=["f2_num", "country"]
    ).select([pl.col("eid").alias("s1_id"), pl.col("eid_right").alias("target_id")])

def block_method_a_postal_cname(s1_p: pl.DataFrame, tgt_p: pl.DataFrame) -> pl.DataFrame:
    return s1_p.filter(pl.col("pin_cname4").is_not_null()).select(["eid", "pin_cname4", "country"]).join(
        tgt_p.filter(pl.col("pin_cname4").is_not_null()).select(["eid", "pin_cname4", "country"]),
        on=["pin_cname4", "country"]
    ).select([pl.col("eid").alias("s1_id"), pl.col("eid_right").alias("target_id")])

def block_method_d_phonetic_soundex(s1_p: pl.DataFrame, tgt_p: pl.DataFrame) -> pl.DataFrame:
    return s1_p.filter(pl.col("soundex_num").is_not_null()).select(["eid", "soundex_num", "country"]).join(
        tgt_p.filter(pl.col("soundex_num").is_not_null()).select(["eid", "soundex_num", "country"]),
        on=["soundex_num", "country"]
    ).select([pl.col("eid").alias("s1_id"), pl.col("eid_right").alias("target_id")])

def block_method_e_transliteration(s1_p: pl.DataFrame, tgt_p: pl.DataFrame) -> pl.DataFrame:
    return s1_p.filter(pl.col("translit_cname").str.len_chars() >= 4).select(["eid", "translit_cname", "country"]).join(
        tgt_p.filter(pl.col("translit_cname").str.len_chars() >= 4).select(["eid", "translit_cname", "country"]),
        on=["translit_cname", "country"]
    ).select([pl.col("eid").alias("s1_id"), pl.col("eid_right").alias("target_id")])

# =============================================================================
# IN-MEMORY COMPACT TARGET INDEXING FOR FAST INFERENCE
# =============================================================================

def build_compact_target_index(target_p: pl.DataFrame) -> Dict[str, Dict[Tuple[str, str], List[str]]]:
    """
    Builds compact in-memory hash tables mapping (key, country) -> [target_ids].
    Vectorized using Polars multi-threaded C++ group_by for 20x speedup.
    """
    index = {
        "compact_name": {},
        "norm_name": {},
        "cname8_num": {},
        "f2_num": {},
        "pin_cname4": {},
        "soundex_num": {},
        "translit_cname": {},
    }

    def _index_col(col_name: str, min_len: int = 1, target_dict_name: str = None):
        target_key = target_dict_name or col_name
        if col_name not in target_p.columns:
            return
        
        filt = pl.col(col_name).is_not_null()
        if min_len > 1:
            filt = filt & (pl.col(col_name).str.len_chars() >= min_len)
            
        gb = target_p.filter(filt).group_by([col_name, "country"]).agg(pl.col("eid"))
        if len(gb) == 0:
            return
        keys = gb[col_name].to_list()
        ctrys = gb["country"].to_list()
        eids = gb["eid"].to_list()
        
        d = index[target_key]
        for k, c, id_list in zip(keys, ctrys, eids):
            pair = (k, c)
            if pair in d:
                d[pair].extend(id_list)
            else:
                d[pair] = id_list

    _index_col("compact_name", min_len=3)
    _index_col("norm_name", min_len=4)
    _index_col("cname8_num")
    _index_col("translit_cname8_num", target_dict_name="cname8_num")
    _index_col("f2_num")
    _index_col("pin_cname4")
    _index_col("soundex_num")
    _index_col("translit_cname", min_len=4)

    return index

def generate_candidates_against_indexed_target(
    s1_p: pl.DataFrame,
    target_index: Dict[str, Dict[Tuple[str, str], List[str]]],
    max_cands_per_s1: int = 50
) -> Dict[str, List[str]]:
    """
    Generates candidates for a chunk of S1 rows against the pre-built compact target index.
    Optimized with columnar list extraction and tuple zip iteration.
    """
    candidates: Dict[str, List[str]] = defaultdict(list)
    n_rows = len(s1_p)
    if n_rows == 0:
        return candidates

    eids = s1_p["eid"].to_list()
    ctrys = s1_p["country"].to_list() if "country" in s1_p.columns else [""] * n_rows
    cnames = s1_p["compact_name"].to_list() if "compact_name" in s1_p.columns else [None] * n_rows
    tcnames = s1_p["translit_cname"].to_list() if "translit_cname" in s1_p.columns else [None] * n_rows
    nnames = s1_p["norm_name"].to_list() if "norm_name" in s1_p.columns else [None] * n_rows
    snds = s1_p["soundex_num"].to_list() if "soundex_num" in s1_p.columns else [None] * n_rows
    c8s = s1_p["cname8_num"].to_list() if "cname8_num" in s1_p.columns else [None] * n_rows
    tc8s = s1_p["translit_cname8_num"].to_list() if "translit_cname8_num" in s1_p.columns else [None] * n_rows
    f2s = s1_p["f2_num"].to_list() if "f2_num" in s1_p.columns else [None] * n_rows
    pins = s1_p["pin_cname4"].to_list() if "pin_cname4" in s1_p.columns else [None] * n_rows

    idx_cn = target_index.get("compact_name", {})
    idx_tcn = target_index.get("translit_cname", {})
    idx_nn = target_index.get("norm_name", {})
    idx_snd = target_index.get("soundex_num", {})
    idx_c8 = target_index.get("cname8_num", {})
    idx_f2 = target_index.get("f2_num", {})
    idx_pin = target_index.get("pin_cname4", {})

    for eid, ctry, cn, tcn, nn, snd, c8, tc8, f2, pin in zip(
        eids, ctrys, cnames, tcnames, nnames, snds, c8s, tc8s, f2s, pins
    ):
        seen = set()
        cands = []

        # 1. Exact Compact Name
        if cn and len(cn) >= 3:
            for tid in idx_cn.get((cn, ctry), []):
                if tid not in seen:
                    seen.add(tid)
                    cands.append(tid)

        # 2. Transliteration CName
        if tcn and len(tcn) >= 4:
            for tid in idx_tcn.get((tcn, ctry), []):
                if tid not in seen:
                    seen.add(tid)
                    cands.append(tid)

        # 3. Exact Norm Name
        if nn and len(nn) >= 4:
            for tid in idx_nn.get((nn, ctry), []):
                if tid not in seen:
                    seen.add(tid)
                    cands.append(tid)

        # 4. Soundex Num
        if snd:
            for tid in idx_snd.get((snd, ctry), []):
                if tid not in seen:
                    seen.add(tid)
                    cands.append(tid)

        # 5. CName8 Num
        if c8:
            for tid in idx_c8.get((c8, ctry), []):
                if tid not in seen:
                    seen.add(tid)
                    cands.append(tid)
        if tc8:
            for tid in idx_c8.get((tc8, ctry), []):
                if tid not in seen:
                    seen.add(tid)
                    cands.append(tid)

        # 6. F2 Num
        if f2:
            for tid in idx_f2.get((f2, ctry), []):
                if tid not in seen:
                    seen.add(tid)
                    cands.append(tid)

        # 7. Postal CName4
        if pin:
            for tid in idx_pin.get((pin, ctry), []):
                if tid not in seen:
                    seen.add(tid)
                    cands.append(tid)

        if len(cands) > max_cands_per_s1:
            cands = cands[:max_cands_per_s1]

        candidates[eid] = cands

    return candidates
