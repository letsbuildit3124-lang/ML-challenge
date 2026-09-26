"""
ER-X Multi-View Normalization Engine.
Provides:
- Bidirectional offline multilingual transliteration (Indic + Accented French/European Latin)
- Boundary-aware international legal corporate suffix normalization
- Canonical address abbreviation expansion
- Phonetic signatures (Soundex & consonant-skeleton representations)
- Numeric signatures and address component parsing
"""

import re
import unicodedata
from typing import Dict, List, Set, Tuple, Optional, Any
from src.erx.types import MultiViewRecord

# Multilingual legal corporate suffixes (French, Indian, European, and US forms)
LEGAL_SUFFIXES_REGEX = (
    r"\b("
    r"sarl|sas|sasu|eurl|sci|sa|snc|sel|scs|sca|gie|selarl|sem|"
    r"ltd|limited|pvt|private|corp|corporation|inc|incorporated|"
    r"llc|llp|co|company|gmbh|plc|bv|nv|assoc|associates|group|"
    r"holdings|enterprises|services|solutions|technologies|international|"
    r"consultants|industries|global|systems|cie|ets|spa|srl|ag"
    r")\b"
)

# Comprehensive address abbreviations (French, US, UK, Indic)
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
    'marg': 'marg', 'nagar': 'nagar', 'bhavan': 'bhavan', 'chowk': 'chowk', 'colony': 'colony'
}

# Indic Unicode Character Mappings
INDIC_ASCII_MAP: Dict[int, str] = {
    # Devanagari Vowels & Matras
    0x0905: 'a', 0x0906: 'aa', 0x0907: 'i', 0x0908: 'ee', 0x0909: 'u', 0x090A: 'oo',
    0x090D: 'e', 0x090E: 'e', 0x090F: 'e', 0x0910: 'ai', 0x0911: 'o', 0x0912: 'o', 0x0913: 'o', 0x0914: 'au',
    0x0902: 'n', 0x0901: 'n', 0x0903: 'h',
    0x093E: 'aa', 0x093F: 'i', 0x0940: 'ee', 0x0941: 'u', 0x0942: 'oo',
    0x0945: 'e', 0x0946: 'e', 0x0947: 'e', 0x0948: 'ai', 0x0949: 'o', 0x094A: 'o', 0x094B: 'o', 0x094C: 'au',
    0x094D: '', 0x093C: '',
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
    # Telugu Vowels & Matras
    0x0C05: 'a', 0x0C06: 'aa', 0x0C07: 'i', 0x0C08: 'ee', 0x0C09: 'u', 0x0C0A: 'oo',
    0x0C0E: 'e', 0x0C0F: 'ee', 0x0C10: 'ai', 0x0C12: 'o', 0x0C13: 'oo', 0x0C14: 'au',
    0x0C3E: 'aa', 0x0C3F: 'i', 0x0C40: 'ee', 0x0C41: 'u', 0x0C42: 'oo',
    0x0C46: 'e', 0x0C47: 'ee', 0x0C48: 'ai', 0x0C4A: 'o', 0x0C4B: 'oo', 0x0C4C: 'au', 0x0C4D: '',
    # Telugu Consonants
    0x0C15: 'k', 0x0C16: 'kh', 0x0C17: 'g', 0x0C18: 'gh', 0x0C19: 'ng',
    0x0C1A: 'ch', 0x0C1B: 'chh', 0x0C1C: 'j', 0x0C1D: 'jh', 0x0C1E: 'ny',
    0x0C1F: 't', 0x0C20: 'th', 0x0C21: 'd', 0x0C22: 'dh', 0x0C23: 'n',
    0x0C24: 't', 0x0C25: 'th', 0x0C26: 'd', 0x0C27: 'dh', 0x0C28: 'n',
    0x0C2A: 'p', 0x0C2B: 'ph', 0x0C2C: 'b', 0x0C2D: 'bh', 0x0C2E: 'm',
    0x0C2F: 'y', 0x0C30: 'r', 0x0C32: 'l', 0x0C35: 'v', 0x0C36: 'sh',
    0x0C37: 'sh', 0x0C38: 's', 0x0C39: 'h',
}


def offline_transliterate(text: Optional[str]) -> str:
    """Converts non-Latin Indic scripts and accented European Latin characters to clean ASCII."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    out: List[str] = []
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


def normalize_text(text: Optional[str]) -> str:
    """Standardizes casing, ampersands, punctuation, and whitespace."""
    if not text or not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFKD', text).lower().replace('&', ' and ')
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def normalize_address(text: Optional[str]) -> str:
    """Standardizes address terms with canonical abbreviations."""
    norm = normalize_text(text)
    if not norm:
        return ""
    words = norm.split()
    return " ".join([ADDR_ABBREVIATIONS.get(w, w) for w in words])


def compact_name(text: Optional[str]) -> str:
    """Strips boundary-aware corporate legal suffixes and whitespace."""
    norm = normalize_text(text)
    if not norm:
        return ""
    cleaned = re.sub(LEGAL_SUFFIXES_REGEX, "", norm)
    return re.sub(r'\s+', '', cleaned)


def compute_soundex(token: str) -> str:
    """Computes standard American Soundex for a token."""
    if not token or not token.isalpha():
        return ""
    token = token.upper()
    soundex_map = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    first_char = token[0]
    encoded = [first_char]
    prev = soundex_map.get(first_char, '0')
    for ch in token[1:]:
        code = soundex_map.get(ch, '0')
        if code != '0' and code != prev:
            encoded.append(code)
            if len(encoded) == 4:
                break
        prev = code
    return "".join(encoded).ljust(4, '0')


def compute_phonetic_signature(name_norm: str) -> str:
    """Produces token-level phonetic signature for indexing."""
    tokens = [w for w in name_norm.split() if w.isalpha() and len(w) > 2]
    if not tokens:
        return ""
    soundexes = [compute_soundex(t) for t in tokens[:3]]
    return "_".join(soundexes)


def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Generates character n-grams from text."""
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[i:i+n] for i in range(len(text) - n + 1)}


class ERXNormalizer:
    """High-performance multi-view record normalizer."""

    def __init__(self, learned_aliases: Optional[Dict[str, str]] = None):
        self.learned_aliases = learned_aliases or {}

    def apply_learned_aliases(self, norm_name: str) -> str:
        """Expands/replaces learned aliases on normalized name tokens."""
        if not self.learned_aliases or not norm_name:
            return norm_name
        toks = norm_name.split()
        transformed = [self.learned_aliases.get(t, t) for t in toks]
        return " ".join(transformed)

    def normalize_record(
        self,
        internal_id: int,
        entity_id: str,
        name: Optional[str],
        addr: Optional[str],
        country: Optional[str]
    ) -> MultiViewRecord:
        raw_n = str(name or "").strip()
        raw_a = str(addr or "").strip()
        raw_c = str(country or "").strip().upper()

        is_name_missing = (not raw_n)
        is_addr_missing = (not raw_a)
        is_country_missing = (not raw_c)

        # Name normalization views
        norm_n = normalize_text(raw_n)
        is_ascii = norm_n.isascii()
        translit_n = normalize_text(offline_transliterate(raw_n)) if not is_ascii else norm_n
        learned_n = self.apply_learned_aliases(norm_n)
        comp_n = compact_name(learned_n)
        translit_comp_n = compact_name(translit_n)

        name_toks = norm_n.split() if norm_n else []
        name_tok_set = set(name_toks)
        translit_toks = translit_n.split() if translit_n else []
        translit_tok_set = set(translit_toks)
        sorted_token_name = " ".join(sorted(name_toks))

        # Precomputed n-gram sets (union of native and transliterated if non-ASCII)
        c3_set = get_char_ngrams(norm_n, 3) | (get_char_ngrams(translit_n, 3) if not is_ascii else set())
        c4_set = get_char_ngrams(norm_n, 4) | (get_char_ngrams(translit_n, 4) if not is_ascii else set())
        c5_set = get_char_ngrams(norm_n, 5) | (get_char_ngrams(translit_n, 5) if not is_ascii else set())

        phonetic_sig = compute_phonetic_signature(translit_n if not is_ascii else norm_n)

        # Address views
        norm_a = normalize_address(raw_a)
        translit_a = normalize_address(offline_transliterate(raw_a)) if not norm_a.isascii() else norm_a
        addr_toks = norm_a.split() if norm_a else []
        addr_tok_set = set(addr_toks)

        # Numeric components and house numbers
        num_toks = [w for w in addr_toks if w.isdigit()]
        house_numbers = set(num_toks[:2]) if num_toks else set()
        postal_codes = {w for w in num_toks if len(w) in (5, 6)}
        numeric_sig = "-".join(sorted(num_toks)) if num_toks else ""

        is_s2 = entity_id.startswith("S2-")
        is_s3 = entity_id.startswith("S3-")

        return MultiViewRecord(
            internal_id=internal_id,
            entity_id=entity_id,
            country=raw_c,
            raw_name=raw_n,
            norm_name=norm_n,
            compact_name=comp_n,
            translit_name=translit_n,
            translit_comp_name=translit_comp_n,
            learned_name=learned_n,
            sorted_token_name=sorted_token_name,
            name_tokens=name_toks,
            name_tok_set=name_tok_set,
            translit_tokens=translit_toks,
            translit_tok_set=translit_tok_set,
            name_char3_set=c3_set,
            name_char4_set=c4_set,
            name_char5_set=c5_set,
            name_phonetic_sig=phonetic_sig,
            raw_addr=raw_a,
            norm_addr=norm_a,
            translit_addr=translit_a,
            addr_tokens=addr_toks,
            addr_tok_set=addr_tok_set,
            house_numbers=house_numbers,
            postal_codes=postal_codes,
            city_tokens=set(),
            numeric_signature=numeric_sig,
            is_s2=is_s2,
            is_s3=is_s3,
            is_name_missing=is_name_missing,
            is_addr_missing=is_addr_missing,
            is_country_missing=is_country_missing,
        )
