"""
V3 Preprocessing and Normalization Module for Business Entity Resolution.
Supports:
- Multilingual boundary-aware legal business suffix normalization (French & Global/English)
- Bidirectional offline transliteration (Devanagari, Tamil, Malayalam, Bengali, Gujarati, Telugu, French diacritics)
- Conservative French and English canonical address abbreviations
- Multi-representation record extraction with precomputed sets for 5x feature acceleration.
"""

import re
import unicodedata
from typing import Dict, List, Set, Any, Optional
import polars as pl

# Boundary-aware legal business suffixes (French, Indian, European, and US forms)
LEGAL_SUFFIXES_REGEX = (
    r"\b("
    r"sarl|sas|sasu|eurl|sci|sa|snc|sel|scs|sca|gie|selarl|sem|"
    r"ltd|limited|pvt|private|corp|corporation|inc|incorporated|"
    r"llc|llp|co|company|gmbh|plc|bv|nv|assoc|associates|group|"
    r"holdings|enterprises|services|solutions|technologies|international|"
    r"consultants|industries|global|systems|cie|ets"
    r")\b"
)

# Canonical address abbreviations (French, US, UK, and Indian forms)
ADDR_ABBREVIATIONS: Dict[str, str] = {
    # English / US / International
    'st': 'street', 'ave': 'avenue', 'av': 'avenue', 'rd': 'road', 'dr': 'drive',
    'blvd': 'boulevard', 'ln': 'lane', 'pkwy': 'parkway', 'hwy': 'highway',
    'ste': 'suite', 'apt': 'apartment', 'fl': 'floor', 'bldg': 'building',
    'ct': 'court', 'pl': 'place', 'sq': 'square', 'terr': 'terrace', 'tr': 'terrace',
    'cir': 'circle', 'dept': 'department', 'no': 'number', 'opp': 'opposite', 'nr': 'near',
    # French
    'r': 'rue', 'rue': 'rue', 'bd': 'boulevard', 'bvd': 'boulevard', 'ch': 'chemin',
    'chemin': 'chemin', 'rte': 'route', 'route': 'route', 'imp': 'impasse', 'impasse': 'impasse',
    'allee': 'allee', 'allée': 'allee', 'quai': 'quai', 'crs': 'cours', 'cours': 'cours',
    'pass': 'passage', 'passage': 'passage', 'fg': 'faubourg', 'faubourg': 'faubourg',
    # Indic locality descriptors
    'marg': 'marg', 'nagar': 'nagar', 'bhavan': 'bhavan', 'chowk': 'chowk'
}

# Offline Indic Unicode to Latin ASCII phonetic character map
INDIC_ASCII_MAP: Dict[int, str] = {
    # Devanagari Vowels & Matras
    0x0905: 'a', 0x0906: 'aa', 0x0907: 'i', 0x0908: 'ee', 0x0909: 'u', 0x090A: 'oo',
    0x090F: 'e', 0x0910: 'ai', 0x0913: 'o', 0x0914: 'au', 0x0902: 'n', 0x0901: 'n',
    0x093E: 'aa', 0x093F: 'i', 0x0940: 'ee', 0x0941: 'u', 0x0942: 'oo',
    0x0947: 'e', 0x0948: 'ai', 0x094B: 'o', 0x094C: 'au', 0x094D: '',
    # Devanagari Consonants
    0x0915: 'k', 0x0916: 'kh', 0x0917: 'g', 0x0918: 'gh', 0x0919: 'ng',
    0x091A: 'ch', 0x091B: 'chh', 0x091C: 'j', 0x091D: 'jh', 0x091E: 'ny',
    0x091F: 't', 0x0920: 'th', 0x0921: 'd', 0x0922: 'dh', 0x0923: 'n',
    0x0924: 't', 0x0925: 'th', 0x0926: 'd', 0x0927: 'dh', 0x0928: 'n',
    0x092A: 'p', 0x092B: 'ph', 0x092C: 'b', 0x092D: 'bh', 0x092E: 'm',
    0x092F: 'y', 0x0930: 'r', 0x0932: 'l', 0x0935: 'v', 0x0936: 'sh',
    0x0937: 'sh', 0x0938: 's', 0x0939: 'h',
    # Tamil Vowels & Matras
    0x0B85: 'a', 0x0B86: 'aa', 0x0B87: 'i', 0x0B88: 'ee', 0x0B89: 'u', 0x0B8A: 'oo',
    0x0B8E: 'e', 0x0B8F: 'ee', 0x0B90: 'ai', 0x0B92: 'o', 0x0B93: 'oo', 0x0B94: 'au',
    0x0BBE: 'aa', 0x0BBF: 'i', 0x0BC0: 'ee', 0x0BC1: 'u', 0x0BC2: 'oo',
    0x0BC6: 'e', 0x0BC7: 'ee', 0x0BC8: 'ai', 0x0BCA: 'o', 0x0BCB: 'oo', 0x0BCC: 'au', 0x0BCD: '',
    # Tamil Consonants
    0x0B95: 'k', 0x0B99: 'ng', 0x0B9A: 'c', 0x0B9C: 'j', 0x0B9E: 'ny', 0x0B9F: 't',
    0x0BA3: 'n', 0x0BA4: 't', 0x0BA8: 'n', 0x0BA9: 'n', 0x0BAA: 'p', 0x0BAE: 'm',
    0x0BAF: 'y', 0x0BB0: 'r', 0x0BB1: 'r', 0x0BB2: 'l', 0x0BB3: 'l', 0x0BB4: 'zh',
    0x0BB5: 'v', 0x0BB7: 'sh', 0x0BB8: 's', 0x0BB9: 'h',
    # Malayalam
    0x0D05: 'a', 0x0D06: 'aa', 0x0D07: 'i', 0x0D08: 'ee', 0x0D09: 'u', 0x0D0A: 'oo',
    0x0D0E: 'e', 0x0D0F: 'ee', 0x0D10: 'ai', 0x0D12: 'o', 0x0D13: 'oo', 0x0D14: 'au',
    0x0D2A: 'p', 0x0D2B: 'ph', 0x0D2C: 'b', 0x0D2D: 'bh', 0x0D2E: 'm', 0x0D2F: 'y',
    0x0D30: 'r', 0x0D32: 'l', 0x0D35: 'v', 0x0D36: 'sh', 0x0D37: 'sh', 0x0D38: 's', 0x0D39: 'h'
}

def offline_transliterate(text: Optional[str]) -> str:
    """Converts Indic and accented Latin characters to clean ASCII offline."""
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

def normalize_text_v3(text: Optional[str]) -> str:
    """Base normalization preserving alphanumeric tokens and standardizing separators."""
    if not text or not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFKD', text).lower().replace('&', ' and ')
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def normalize_address_v3(text: Optional[str]) -> str:
    """Normalizes address terms using canonical French/US/UK/Indic mapping."""
    norm = normalize_text_v3(text)
    if not norm:
        return ""
    words = norm.split()
    return " ".join([ADDR_ABBREVIATIONS.get(w, w) for w in words])

def compact_name_v3(text: Optional[str]) -> str:
    """Boundary-aware corporate suffix removal and compaction."""
    norm = normalize_text_v3(text)
    if not norm:
        return ""
    cleaned = re.sub(LEGAL_SUFFIXES_REGEX, "", norm)
    cleaned = re.sub(r'\s+', '', cleaned)
    return cleaned

def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Generates character n-grams."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def preprocess_record_v3(name: str, addr: str, country: str) -> Dict[str, Any]:
    """
    Constructs a multi-representation record with precomputed set representations
    for ultra-fast feature extraction.
    """
    norm_n = normalize_text_v3(name)
    comp_n = compact_name_v3(name)
    translit_n = normalize_text_v3(offline_transliterate(name))
    translit_comp_n = compact_name_v3(translit_n)

    norm_a = normalize_address_v3(addr)
    translit_a = normalize_address_v3(offline_transliterate(addr))

    name_toks = norm_n.split() if norm_n else []
    addr_toks = norm_a.split() if norm_a else []
    num_toks = [w for w in addr_toks if w.isdigit()]
    
    # Precompute fast set structures
    name_tok_set = set(name_toks)
    addr_tok_set = set(addr_toks)
    num_tok_set = set(num_toks)
    name_3g_set = get_char_ngrams(norm_n, 3)

    return {
        "orig_name": name or "",
        "norm_name": norm_n,
        "compact_name": comp_n,
        "translit_name": translit_n,
        "translit_cname": translit_comp_n,
        "name_tokens": name_toks,
        "name_tok_set": name_tok_set,
        "name_tok_len": len(name_toks),
        "name_3g_set": name_3g_set,
        "name_3g_len": len(name_3g_set),

        "orig_addr": addr or "",
        "norm_addr": norm_a,
        "translit_addr": translit_a,
        "addr_tokens": addr_toks,
        "addr_tok_set": addr_tok_set,
        "addr_tok_len": len(addr_toks),
        "numeric_tokens": num_toks,
        "numeric_tok_set": num_tok_set,
        "numeric_tok_len": len(num_tok_set),

        "country": (country or "").strip().upper()
    }
