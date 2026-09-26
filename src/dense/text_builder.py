"""
Antigravity V5 Multilingual E5 Text Builder.
Handles asymmetric retrieval prefixes and vectorized multilingual entity text construction.

Prefix Specification:
- Target Entity (Passage): "passage: <business_name | business_address | country>"
- S1 Query (Query):        "query: <business_name | business_address | country>"

Design Invariants:
- Preserves full Unicode / native scripts (Devanagari, Chinese, Arabic, Cyrillic, etc.).
- Safely collapses nulls and redundant delimiters.
- Fully vectorized for high throughput (Arrow / Polars / NumPy / Pandas).
- Zero external data lookups or hallucinated metadata.
"""

from typing import Optional, List, Union
import polars as pl

E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "
DEFAULT_MAX_LENGTH = 128


def clean_field(val: Optional[Union[str, int, float]]) -> str:
    """Safely sanitizes a field string while preserving all Unicode scripts."""
    if val is None:
        return ""
    s = str(val).strip()
    if not s or s.lower() in ("none", "nan", "null", "unknown", "<na>"):
        return ""
    # Collapse internal excessive whitespace while preserving unicode characters
    return " ".join(s.split())


def build_entity_text(
    name: Optional[str],
    address: Optional[str] = "",
    country: Optional[str] = ""
) -> str:
    """
    Constructs compact canonical entity text in the strict order:
    business_name | business_address | country
    """
    n = clean_field(name)
    a = clean_field(address)
    c = clean_field(country)

    parts = [p for p in (n, a, c) if p]
    if not parts:
        return "unknown"
    return " | ".join(parts)


def format_e5_text(
    entity_text: str,
    is_query: bool = False
) -> str:
    """
    Applies asymmetric E5 retrieval prefixes.
    - S1 query: "query: <entity text>"
    - Target:   "passage: <entity text>"
    """
    prefix = E5_QUERY_PREFIX if is_query else E5_PASSAGE_PREFIX
    return f"{prefix}{entity_text.strip()}"


def vectorized_build_e5_texts(
    df: Union[pl.DataFrame, Any],
    is_query: bool = False,
    name_col: Optional[str] = None,
    addr_col: Optional[str] = None,
    ctry_col: Optional[str] = None
) -> List[str]:
    """
    High-speed vectorized text construction directly from a Polars or Pandas DataFrame.
    Avoids slow python dictionary iteration.
    """
    # Normalize column names
    col_map = {str(col).lower(): str(col) for col in df.columns}
    
    n_col = name_col or col_map.get("business_name") or col_map.get("norm_name") or col_map.get("name")
    a_col = addr_col or col_map.get("business_address") or col_map.get("norm_addr") or col_map.get("address")
    c_col = ctry_col or col_map.get("country") or col_map.get("norm_country")

    if isinstance(df, pl.DataFrame):
        # High performance Polars expression
        def clean_expr(c_name: Optional[str]) -> pl.Expr:
            if c_name and c_name in df.columns:
                return (
                    pl.col(c_name)
                    .cast(pl.Utf8, strict=False)
                    .fill_null("")
                    .str.strip_chars()
                )
            return pl.lit("")

        e_name = clean_expr(n_col)
        e_addr = clean_expr(a_col)
        e_ctry = clean_expr(c_col)

        # Vectorized string concatenation with conditional pipe delimiters
        # Construct compact text: name | addr | ctry
        df_text = df.select(
            pl.concat_str(
                [e_name, e_addr, e_ctry],
                separator=" | ",
                ignore_nulls=True
            ).alias("raw_text")
        )

        prefix = E5_QUERY_PREFIX if is_query else E5_PASSAGE_PREFIX
        # Fast prefix prepend & empty fallback
        texts = [
            f"{prefix}{t.strip(' | ')}" if t.strip(" | ") else f"{prefix}unknown"
            for t in df_text["raw_text"].to_list()
        ]
        return texts

    else:
        # Pandas fallback
        import pandas as pd
        if isinstance(df, pd.DataFrame):
            n_vals = df[n_col].fillna("").astype(str).tolist() if n_col and n_col in df.columns else [""] * len(df)
            a_vals = df[a_col].fillna("").astype(str).tolist() if a_col and a_col in df.columns else [""] * len(df)
            c_vals = df[c_col].fillna("").astype(str).tolist() if ctry_col and ctry_col in df.columns else [""] * len(df)
            
            prefix = E5_QUERY_PREFIX if is_query else E5_PASSAGE_PREFIX
            return [
                f"{prefix}{build_entity_text(n, a, c)}"
                for n, a, c in zip(n_vals, a_vals, c_vals)
            ]
        else:
            raise TypeError(f"Unsupported DataFrame type: {type(df)}")
