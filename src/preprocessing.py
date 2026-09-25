"""
Preprocessing and normalization module.
Provides multi-view text normalization, tokenization, and Polars-vectorized processing.
"""

import re
import unicodedata
from typing import Dict, List, Set, Any, Optional
import polars as pl

# Conservative legal business suffixes
LEGAL_SUFFIXES: Set[str] = {
    'ltd', 'limited', 'pvt', 'private', 'corp', 'corporation', 'inc', 'incorporated',
    'llc', 'llp', 'co', 'company', 'gmbh', 'sa', 'sarl', 'plc', 'bv', 'nv', 'assoc',
    'associates', 'group', 'holdings', 'enterprises', 'services', 'solutions', 'technologies',
    'international', 'consultants', 'industries', 'global', 'systems', 'cie', 'ets'
}

ADDR_ABBREVIATIONS: Dict[str, str] = {
    'st': 'street', 'ave': 'avenue', 'av': 'avenue', 'rd': 'road', 'dr': 'drive',
    'blvd': 'boulevard', 'ln': 'lane', 'pkwy': 'parkway', 'hwy': 'highway',
    'ste': 'suite', 'apt': 'apartment', 'fl': 'floor', 'bldg': 'building',
    'ct': 'court', 'pl': 'place', 'sq': 'square', 'terr': 'terrace', 'tr': 'terrace',
    'cir': 'circle', 'dept': 'department', 'no': 'number', 'opp': 'opposite', 'nr': 'near'
}

def normalize_text(text: Optional[str]) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFKD', text).lower().replace('&', ' and ')
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def normalize_address(text: Optional[str]) -> str:
    norm = normalize_text(text)
    if not norm:
        return ""
    words = norm.split()
    return " ".join([ADDR_ABBREVIATIONS.get(w, w) for w in words])

def compact_name(text: Optional[str]) -> str:
    norm = normalize_text(text)
    if not norm:
        return ""
    tokens = [w for w in norm.split() if w not in LEGAL_SUFFIXES]
    return "".join(tokens)

def get_name_tokens(text: Optional[str]) -> List[str]:
    norm = normalize_text(text)
    if not norm:
        return []
    return [w for w in norm.split() if len(w) >= 3 and w not in LEGAL_SUFFIXES]

def get_address_tokens(text: Optional[str]) -> List[str]:
    norm = normalize_address(text)
    if not norm:
        return []
    return [w for w in norm.split() if len(w) >= 2]

def get_numeric_tokens(text: Optional[str]) -> List[str]:
    if not text or not isinstance(text, str):
        return []
    return re.findall(r'\b\d+\b', text)

def preprocess_record(name: str, addr: str, country: str) -> Dict[str, Any]:
    norm_n = normalize_text(name)
    comp_n = compact_name(name)
    n_tokens = get_name_tokens(name)
    norm_a = normalize_address(addr)
    a_tokens = get_address_tokens(addr)
    num_tokens = get_numeric_tokens(addr)
    c_norm = (country or "").strip().upper()

    return {
        "orig_name": name or "",
        "norm_name": norm_n,
        "compact_name": comp_n,
        "name_tokens": n_tokens,
        "orig_addr": addr or "",
        "norm_addr": norm_a,
        "addr_tokens": a_tokens,
        "numeric_tokens": num_tokens,
        "country": c_norm
    }

def preprocess_dataframe(df: pl.DataFrame) -> Dict[str, Dict[str, Any]]:
    records = {}
    for row in df.iter_rows():
        eid = str(row[0])
        name = str(row[1]) if row[1] is not None else ""
        addr = str(row[2]) if row[2] is not None else ""
        country = str(row[3]) if row[3] is not None else ""
        records[eid] = preprocess_record(name, addr, country)
    return records
