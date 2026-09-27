"""
ER-X Ultimate: High-Throughput Normalization Engine
"""

from __future__ import annotations
import re
import unicodedata
from typing import List, Tuple, Set

# Precompiled Regexes
RE_WHITESPACE = re.compile(r"\s+")
RE_NON_ALPHANUM = re.compile(r"[^a-z0-9\s]")
RE_NUMBERS = re.compile(r"\b\d+\b")
RE_PHONE_CHARS = re.compile(r"[^\d]")
RE_URL_PREFIX = re.compile(r"^https?://(www\.)?")

# Business entity abbreviations and noise words
LEGAL_SUFFIXES = {
    "llc", "inc", "ltd", "corp", "corporation", "co", "company", "gmbh", 
    "sa", "bv", "pvt", "limited", "incorporated", "services", "group", "holdings"
}

ADDRESS_ABBREVIATIONS = {
    "st": "street", "rd": "road", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "pl": "place",
    "sq": "square", "ste": "suite", "apt": "apartment", "pkwy": "parkway",
    "hwy": "highway", "fl": "floor", "bldg": "building", "dept": "department",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest"
}


def normalize_text(text: str) -> str:
    """Normalize arbitrary text: unicode decompose, lowercase, strip non-alphanumeric, compact whitespace."""
    if not text:
        return ""
    # Unicode decomposition
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = text.lower()
    text = RE_NON_ALPHANUM.sub(" ", text)
    text = RE_WHITESPACE.sub(" ", text).strip()
    return text


def normalize_business_name(name: str) -> str:
    """Normalize business name, expanding common tokens and stripping noise."""
    text = normalize_text(name)
    if not text:
        return ""
    tokens = text.split()
    cleaned = [t for t in tokens if t not in LEGAL_SUFFIXES]
    return " ".join(cleaned) if cleaned else text


def normalize_address(address: str) -> str:
    """Normalize address string, expanding directional and street abbreviations."""
    text = normalize_text(address)
    if not text:
        return ""
    tokens = text.split()
    expanded = [ADDRESS_ABBREVIATIONS.get(t, t) for t in tokens]
    return " ".join(expanded)


def normalize_phone(phone: str) -> str:
    """Normalize phone numbers to digits only, trimming leading country codes if standard."""
    if not phone:
        return ""
    digits = RE_PHONE_CHARS.sub("", phone)
    if digits.startswith("1") and len(digits) == 11:
        digits = digits[1:]
    return digits


def normalize_website(url: str) -> str:
    """Normalize URLs: strip protocol, www, trailing slashes, path queries."""
    if not url:
        return ""
    u = url.strip().lower()
    u = RE_URL_PREFIX.sub("", u)
    u = u.split("/")[0].split("?")[0].split("#")[0].strip()
    return u


def compute_soundex(token: str) -> str:
    """Compute American Soundex code for phonetic matching."""
    if not token or not token.isalpha():
        return ""
    token = token.upper()
    first_letter = token[0]
    
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    
    encoded = [first_letter]
    prev = mapping.get(first_letter, '0')
    
    for char in token[1:]:
        curr = mapping.get(char, '0')
        if curr != '0' and curr != prev:
            encoded.append(curr)
        prev = curr
        if len(encoded) == 4:
            break
            
    while len(encoded) < 4:
        encoded.append('0')
        
    return "".join(encoded[:4])


def extract_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from text."""
    if len(text) < n:
        return {text} if text else set()
    padded = f"#{text}#"
    return {padded[i : i + n] for i in range(len(padded) - n + 1)}
