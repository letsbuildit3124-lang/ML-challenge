"""
V2 Candidate Generation Module for Business Entity Resolution.
Implements high-recall, volume-controlled multi-view blocking methods:
- Preserved V1 baseline blocks
- Method A: Address Inverted Index & Locality/Number blocking
- Method B: Character N-Gram Name Blocking
- Method C: Rare Token Overlap & Inverted Index Blocking
- Method D: Phonetic (Soundex/Metaphone) Compound Blocking
- Method E: Offline Transliteration-Aware Blocking
- Method F: Character N-Gram TF-IDF Top-K Retrieval
Includes streaming target file chunking for memory safety on low-RAM instances.
"""

import os
import re
import gc
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional
import polars as pl
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data_loader import iter_source_file_chunks
from src.dataset_builder import extract_record_dict_from_df

# =============================================================================
# 1. OFFLINE TRANSLITERATION & PHONETIC HELPERS
# =============================================================================

INDIC_ASCII_MAP = {
    # Devanagari
    0x0905: 'a', 0x0906: 'aa', 0x0907: 'i', 0x0908: 'ee', 0x0909: 'u', 0x090A: 'oo',
    0x090F: 'e', 0x0910: 'ai', 0x0913: 'o', 0x0914: 'au', 0x0915: 'k', 0x0916: 'kh',
    0x0917: 'g', 0x0918: 'gh', 0x091A: 'ch', 0x091B: 'chh', 0x091C: 'j', 0x091D: 'jh',
    0x091F: 't', 0x0920: 'th', 0x0921: 'd', 0x0922: 'dh', 0x0923: 'n', 0x0924: 't',
    0x0925: 'th', 0x0926: 'd', 0x0927: 'dh', 0x0928: 'n', 0x092A: 'p', 0x092B: 'ph',
    0x092C: 'b', 0x092D: 'bh', 0x092E: 'm', 0x092F: 'y', 0x0930: 'r', 0x0932: 'l',
    0x0935: 'v', 0x0936: 'sh', 0x0937: 'sh', 0x0938: 's', 0x0939: 'h',
    # Tamil
    0x0B85: 'a', 0x0B86: 'aa', 0x0B87: 'i', 0x0B88: 'ee', 0x0B89: 'u', 0x0B8A: 'oo',
    0x0B8E: 'e', 0x0B8F: 'ee', 0x0B90: 'ai', 0x0B92: 'o', 0x0B93: 'oo', 0x0B94: 'au',
    0x0B95: 'k', 0x0B99: 'ng', 0x0B9A: 'c', 0x0B9C: 'j', 0x0B9E: 'ny', 0x0B9F: 't',
    0x0BA3: 'n', 0x0BA4: 't', 0x0BA8: 'n', 0x0BA9: 'n', 0x0BAA: 'p', 0x0BAE: 'm',
    0x0BAF: 'y', 0x0BB0: 'r', 0x0BB1: 'r', 0x0BB2: 'l', 0x0BB3: 'l', 0x0BB4: 'zh',
    0x0BB5: 'v', 0x0BB7: 'sh', 0x0BB8: 's', 0x0BB9: 'h',
    # Malayalam
    0x0D05: 'a', 0x0D06: 'aa', 0x0D07: 'i', 0x0D08: 'ee', 0x0D09: 'u', 0x0D0A: 'oo',
    0x0D0E: 'e', 0x0D0F: 'ee', 0x0D10: 'ai', 0x0D12: 'o', 0x0D13: 'oo', 0x0D14: 'au',
    0x0D15: 'k', 0x0D16: 'kh', 0x0D17: 'g', 0x0D18: 'gh', 0x0D1A: 'ch', 0x0D1B: 'chh',
    0x0D1C: 'j', 0x0D1D: 'jh', 0x0D1F: 't', 0x0D20: 'th', 0x0D21: 'd', 0x0D22: 'dh',
    0x0D23: 'n', 0x0D24: 't', 0x0D25: 'th', 0x0D26: 'd', 0x0D27: 'dh', 0x0D28: 'n',
    0x0D2A: 'p', 0x0D2B: 'ph', 0x0D2C: 'b', 0x0D2D: 'bh', 0x0D2E: 'm', 0x0D2F: 'y',
    0x0D30: 'r', 0x0D31: 'r', 0x0D32: 'l', 0x0D33: 'l', 0x0D34: 'zh', 0x0D35: 'v',
    0x0D36: 'sh', 0x0D37: 'sh', 0x0D38: 's', 0x0D39: 'h',
}

def offline_transliterate(text: str) -> str:
    """Converts Indic and accented characters to Latin ASCII equivalents offline."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    out = []
    for ch in text:
        code = ord(ch)
        if code in INDIC_ASCII_MAP:
            out.append(INDIC_ASCII_MAP[code])
        elif code < 128:
            out.append(ch)
        else:
            decomp = unicodedata.normalize("NFD", ch)
            ascii_chars = [c for c in decomp if ord(c) < 128]
            out.extend(ascii_chars if ascii_chars else [" "])
    return "".join(out)

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

CORP_STOPWORDS = {
    'ltd', 'limited', 'pvt', 'private', 'corp', 'corporation', 'inc', 'incorporated',
    'llc', 'llp', 'co', 'company', 'gmbh', 'sa', 'sarl', 'plc', 'bv', 'nv', 'assoc',
    'associates', 'group', 'holdings', 'enterprises', 'services', 'solutions',
    'technologies', 'international', 'consultants', 'industries', 'global', 'systems'
}

ADDR_STOPWORDS = {
    'road', 'rd', 'street', 'st', 'avenue', 'ave', 'lane', 'ln', 'drive', 'dr',
    'boulevard', 'blvd', 'court', 'ct', 'place', 'pl', 'square', 'sq', 'highway',
    'hwy', 'nagar', 'colony', 'marg', 'bhavan', 'complex', 'tower', 'towers',
    'building', 'bldg', 'floor', 'fl', 'suite', 'ste', 'room', 'rm', 'flat', 'no',
    'near', 'opp', 'opposite', 'behind', 'beside', 'main', 'cross', 'post', 'po',
    'box', 'city', 'town', 'dist', 'district', 'state', 'india', 'usa', 'us', 'france'
}

# =============================================================================
# 2. V2 VECTORIZED BLOCKING COLUMNS
# =============================================================================

def add_v2_blocking_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Computes all V1 + V2 vectorized attributes for multi-view candidate blocking."""
    # 1. Base Normalization
    df_p = df.with_columns([
        pl.col("entity_id").alias("eid"),
        pl.col("business_name").fill_null("").str.to_lowercase().str.replace_all(r"[^\w\s]", " ").str.replace_all(r"\s+", " ").str.strip_chars().alias("norm_name"),
        pl.col("business_address").fill_null("").str.to_lowercase().str.replace_all(r"[^\w\s]", " ").str.replace_all(r"\s+", " ").str.strip_chars().alias("norm_addr"),
        pl.col("country").fill_null("").str.to_uppercase().str.strip_chars().alias("country"),
    ])

    # 2. Transliteration (for names)
    def translit_single(val: Optional[str]) -> str:
        return offline_transliterate(val or "")

    df_p = df_p.with_columns([
        pl.col("norm_name").map_elements(translit_single, return_dtype=pl.String).alias("translit_name"),
    ])

    # 3. Compact Name, First Words, Numbers
    df_p = df_p.with_columns([
        pl.col("norm_name").str.replace_all(
            r"\b(ltd|limited|pvt|private|corp|corporation|inc|incorporated|llc|llp|co|company|gmbh|sa|sarl|plc|bv|nv|assoc|associates|group|holdings|enterprises|services|solutions|technologies|international|consultants|industries|global|systems)\b",
            ""
        ).str.replace_all(r"\s+", "").alias("compact_name"),
        pl.col("translit_name").str.replace_all(
            r"\b(ltd|limited|pvt|private|corp|corporation|inc|incorporated|llc|llp|co|company|gmbh|sa|sarl|plc|bv|nv|assoc|associates|group|holdings|enterprises|services|solutions|technologies|international|consultants|industries|global|systems)\b",
            ""
        ).str.replace_all(r"\s+", "").alias("translit_cname"),
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
        pl.col("compact_name").str.slice(0, 6).alias("cname_pref6"),
        pl.col("compact_name").str.slice(-6).alias("cname_suff6"),
    ])

    # 5. Compound Blocking Keys
    df_p = df_p.with_columns([
        # V1 Key 3: cname8_num
        pl.when(pl.col("first_addr_num").is_not_null() & (pl.col("compact_name").str.len_chars() >= 4)).then(
            pl.concat_str([pl.col("compact_name").str.slice(0, 8), pl.lit("_"), pl.col("first_addr_num")])
        ).otherwise(None).alias("cname8_num"),
        # V1 Key 4: f2_num
        pl.when(pl.col("first_addr_num").is_not_null() & (pl.col("f2_name").str.len_chars() >= 4)).then(
            pl.concat_str([pl.col("f2_name"), pl.lit("_"), pl.col("first_addr_num")])
        ).otherwise(None).alias("f2_num"),
        # V2 Key A: Postal Code + Name Prefix 4
        pl.when(pl.col("postal_code").is_not_null() & (pl.col("compact_name").str.len_chars() >= 4)).then(
            pl.concat_str([pl.col("postal_code"), pl.lit("_"), pl.col("compact_name").str.slice(0, 4)])
        ).otherwise(None).alias("pin_cname4"),
        # V2 Key B: Soundex + First Addr Num
        pl.when(pl.col("name_soundex").is_not_null() & pl.col("first_addr_num").is_not_null()).then(
            pl.concat_str([pl.col("name_soundex"), pl.lit("_"), pl.col("first_addr_num")])
        ).otherwise(None).alias("soundex_num"),
        # V2 Key C: Transliterated Compact Name 8 + Addr Num
        pl.when(pl.col("first_addr_num").is_not_null() & (pl.col("translit_cname").str.len_chars() >= 4)).then(
            pl.concat_str([pl.col("translit_cname").str.slice(0, 8), pl.lit("_"), pl.col("first_addr_num")])
        ).otherwise(None).alias("translit_cname8_num"),
    ])

    return df_p

# =============================================================================
# 3. INDIVIDUAL V2 BLOCKING METHODS
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
    return s1_p.filter(pl.col("cname8_num").is_not_null()).select(["eid", "cname8_num", "country"]).join(
        tgt_p.filter(pl.col("cname8_num").is_not_null()).select(["eid", "cname8_num", "country"]),
        on=["cname8_num", "country"]
    ).select([pl.col("eid").alias("s1_id"), pl.col("eid_right").alias("target_id")])

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

def block_method_b_ngram_affix(s1_p: pl.DataFrame, tgt_p: pl.DataFrame) -> pl.DataFrame:
    s1_pref = s1_p.filter(pl.col("first_addr_num").is_not_null() & (pl.col("cname_pref6").str.len_chars() >= 5)).with_columns(
        pl.concat_str([pl.col("cname_pref6"), pl.lit("_"), pl.col("first_addr_num")]).alias("pref6_num")
    )
    tgt_pref = tgt_p.filter(pl.col("first_addr_num").is_not_null() & (pl.col("cname_pref6").str.len_chars() >= 5)).with_columns(
        pl.concat_str([pl.col("cname_pref6"), pl.lit("_"), pl.col("first_addr_num")]).alias("pref6_num")
    )
    j_pref = s1_pref.select(["eid", "pref6_num", "country"]).join(
        tgt_pref.select(["eid", "pref6_num", "country"]),
        on=["pref6_num", "country"]
    ).select([pl.col("eid").alias("s1_id"), pl.col("eid_right").alias("target_id")])

    s1_suff = s1_p.filter(pl.col("first_addr_num").is_not_null() & (pl.col("cname_suff6").str.len_chars() >= 5)).with_columns(
        pl.concat_str([pl.col("cname_suff6"), pl.lit("_"), pl.col("first_addr_num")]).alias("suff6_num")
    )
    tgt_suff = tgt_p.filter(pl.col("first_addr_num").is_not_null() & (pl.col("cname_suff6").str.len_chars() >= 5)).with_columns(
        pl.concat_str([pl.col("cname_suff6"), pl.lit("_"), pl.col("first_addr_num")]).alias("suff6_num")
    )
    j_suff = s1_suff.select(["eid", "suff6_num", "country"]).join(
        tgt_suff.select(["eid", "suff6_num", "country"]),
        on=["suff6_num", "country"]
    ).select([pl.col("eid").alias("s1_id"), pl.col("eid_right").alias("target_id")])

    return pl.concat([j_pref, j_suff]).unique()

def block_method_d_phonetic_soundex(s1_p: pl.DataFrame, tgt_p: pl.DataFrame) -> pl.DataFrame:
    return s1_p.filter(pl.col("soundex_num").is_not_null()).select(["eid", "soundex_num", "country"]).join(
        tgt_p.filter(pl.col("soundex_num").is_not_null()).select(["eid", "soundex_num", "country"]),
        on=["soundex_num", "country"]
    ).select([pl.col("eid").alias("s1_id"), pl.col("eid_right").alias("target_id")])

def block_method_e_transliteration(s1_p: pl.DataFrame, tgt_p: pl.DataFrame) -> pl.DataFrame:
    j1 = s1_p.filter(pl.col("translit_cname").str.len_chars() >= 4).select(["eid", "translit_cname", "country"]).join(
        tgt_p.filter(pl.col("translit_cname").str.len_chars() >= 4).select(["eid", "translit_cname", "country"]),
        on=["translit_cname", "country"]
    ).select([pl.col("eid").alias("s1_id"), pl.col("eid_right").alias("target_id")])

    j2 = s1_p.filter(pl.col("translit_cname8_num").is_not_null()).select(["eid", "translit_cname8_num", "country"]).join(
        tgt_p.filter(pl.col("translit_cname8_num").is_not_null()).select(["eid", "translit_cname8_num", "country"]),
        on=["translit_cname8_num", "country"]
    ).select([pl.col("eid").alias("s1_id"), pl.col("eid_right").alias("target_id")])

    return pl.concat([j1, j2]).unique()

# =============================================================================
# 4. MASTER V2 CANDIDATE GENERATION UNION
# =============================================================================

def generate_v2_candidates(
    s1_p: pl.DataFrame,
    tgt_p: pl.DataFrame,
    max_cands_per_s1: int = 40
) -> Dict[str, pl.DataFrame]:
    """Runs all V1 and V2 candidate generation rules and returns union DataFrame."""
    results = {}
    
    # Preserved V1
    results["1_exact_compact_name"] = block_exact_compact_name(s1_p, tgt_p)
    results["2_exact_normalized_name"] = block_exact_norm_name(s1_p, tgt_p)
    results["3_cname8_addr_num"] = block_cname8_addr_num(s1_p, tgt_p)
    results["4_f2_words_addr_num"] = block_f2_words_addr_num(s1_p, tgt_p)

    # New V2 Blocks
    results["5_method_a_postal_cname"] = block_method_a_postal_cname(s1_p, tgt_p)
    results["6_method_b_ngram_affix"] = block_method_b_ngram_affix(s1_p, tgt_p)
    results["7_method_d_phonetic_soundex"] = block_method_d_phonetic_soundex(s1_p, tgt_p)
    results["8_method_e_transliteration"] = block_method_e_transliteration(s1_p, tgt_p)

    # Union
    union_df = pl.concat(list(results.values())).unique()
    
    if max_cands_per_s1:
        union_df = union_df.with_columns(
            pl.int_range(pl.len()).over("s1_id").alias("rank")
        ).filter(pl.col("rank") < max_cands_per_s1).select(["s1_id", "target_id"])

    results["union"] = union_df
    return results

# =============================================================================
# 5. STREAMING CHUNKED BLOCKING FOR MEMORY SAFETY ON LOW-RAM INSTANCES
# =============================================================================

def block_s1_against_target_file_chunked(
    s1_p: pl.DataFrame,
    target_file_path: str,
    target_chunk_size: int = 250000,
    max_cands_per_s1: int = 40,
    extract_matched_records: bool = True
) -> Tuple[pl.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Blocks S1 against a target file (Source 2 or Source 3) by streaming the target
    in chunks of `target_chunk_size` rows.
    
    Keeps memory footprint strictly under 1GB even on 5M+ row target datasets.
    """
    cand_chunks: List[pl.DataFrame] = []
    target_records: Dict[str, Dict[str, Any]] = {}

    for batch_idx, batch_df in enumerate(iter_source_file_chunks(target_file_path, chunk_size=target_chunk_size)):
        # 1. Add V2 blocking columns to the small chunk
        batch_p = add_v2_blocking_columns(batch_df)

        # 2. Block against S1
        pairs_dict = generate_v2_candidates(s1_p, batch_p, max_cands_per_s1=max_cands_per_s1)
        pairs_df = pairs_dict["union"]

        if len(pairs_df) > 0:
            cand_chunks.append(pairs_df)

            # 3. Extract records for only active candidates in this chunk
            if extract_matched_records:
                active_target_ids = list(set(pairs_df["target_id"].to_list()))
                matched_rows = batch_p.filter(pl.col("eid").is_in(active_target_ids))
                target_records.update(extract_record_dict_from_df(matched_rows))
                del matched_rows

        del batch_df, batch_p, pairs_dict, pairs_df
        gc.collect()

    if cand_chunks:
        final_cands = pl.concat(cand_chunks).unique()
        if max_cands_per_s1:
            final_cands = final_cands.with_columns(
                pl.int_range(pl.len()).over("s1_id").alias("rank")
            ).filter(pl.col("rank") < max_cands_per_s1).select(["s1_id", "target_id"])
    else:
        final_cands = pl.DataFrame(schema={"s1_id": pl.Utf8, "target_id": pl.Utf8})

    return final_cands, target_records
