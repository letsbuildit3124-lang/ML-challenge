"""
Antigravity V5 Multilingual Dense Retrieval Module.
"""
from src.dense.text_builder import (
    build_entity_text,
    format_e5_text,
    vectorized_build_e5_texts,
    E5_QUERY_PREFIX,
    E5_PASSAGE_PREFIX,
)

__all__ = [
    "build_entity_text",
    "format_e5_text",
    "vectorized_build_e5_texts",
    "E5_QUERY_PREFIX",
    "E5_PASSAGE_PREFIX",
]
