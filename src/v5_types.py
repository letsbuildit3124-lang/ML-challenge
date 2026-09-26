"""
Antigravity V5 Constants, Provenance Bitmasks, and Type Definitions.
High-recall CPU-Only Business Entity Resolution Engine.
"""

from typing import Dict, List, Set, Tuple, Any, Optional

# Provenance Bitmask Flags (Powers of 2 for lossless bitwise OR composition)
PROV_DET           = 1     # 0x0001: Deterministic Blocker Keys (10+ equi-join blockers)
PROV_NAME_NGRAM_3  = 2     # 0x0002: Character 3-Gram Inverted Retrieval (Name)
PROV_NAME_NGRAM_4  = 4     # 0x0004: Character 4-Gram Inverted Retrieval (Name)
PROV_NAME_NGRAM_5  = 8     # 0x0008: Character 5-Gram Inverted Retrieval (Name)
PROV_ADDR_NGRAM    = 16    # 0x0010: Address Character N-Gram Retrieval
PROV_TOKEN         = 32    # 0x0020: Token-Set / Sorted-Token Signature Retrieval
PROV_RARE_TOKEN    = 64    # 0x0040: Rare-Token & Token Pair Inverted Retrieval
PROV_ADDR_NUM      = 128   # 0x0080: Structural Address Number & Street Collocation
PROV_TRANSLIT      = 256   # 0x0100: Transliterated Multi-Pass Retrieval
PROV_FTS           = 512   # 0x0200: DuckDB Native FTS / BM25 Retrieval
PROV_FUZZY         = 1024  # 0x0400: RapidFuzz C++ High-Similarity Candidate Retrieval

PROVENANCE_NAMES = {
    PROV_DET: "Deterministic Blocker",
    PROV_NAME_NGRAM_3: "Name 3-Gram",
    PROV_NAME_NGRAM_4: "Name 4-Gram",
    PROV_NAME_NGRAM_5: "Name 5-Gram",
    PROV_ADDR_NGRAM: "Address N-Gram",
    PROV_TOKEN: "Token Signature",
    PROV_RARE_TOKEN: "Rare Token",
    PROV_ADDR_NUM: "Address Number & Street",
    PROV_TRANSLIT: "Transliteration",
    PROV_FTS: "DuckDB FTS BM25",
    PROV_FUZZY: "RapidFuzz C++ Match",
}

def decode_provenance(bitmask: int) -> List[str]:
    """Decodes a provenance integer bitmask into human-readable channel names."""
    channels = []
    for flag, name in PROVENANCE_NAMES.items():
        if bitmask & flag:
            channels.append(name)
    return channels
