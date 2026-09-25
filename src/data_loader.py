"""
Data loading and validation module with memory-safe chunked readers.
"""

import os
import sys
import gc
import polars as pl
from typing import Dict, List, Any, Tuple, Optional, Generator
from src.config import Config, get_config

def load_source_file(
    file_path: str,
    expected_prefix: Optional[str] = None,
    n_rows: Optional[int] = None,
    columns: Optional[List[str]] = None
) -> pl.DataFrame:
    """
    Loads a source TSV file using Polars with validation of columns, dtypes, and prefixes.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file not found at: {file_path}")

    df = pl.read_csv(
        file_path,
        separator="\t",
        n_rows=n_rows,
        columns=columns,
        truncate_ragged_lines=True,
        null_values=["", "NULL", "null", "None", "NaN"]
    )

    if columns is None:
        expected_cols = ["entity_id", "business_name", "business_address", "country"]
        for col in expected_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' in {file_path}. Found: {df.columns}")

    # Validate entity_id prefix
    if expected_prefix:
        sample_ids = df["entity_id"].head(100).to_list()
        invalid_ids = [eid for eid in sample_ids if not str(eid).startswith(expected_prefix)]
        if invalid_ids:
            raise ValueError(f"Found invalid entity_id prefix in {file_path}. Expected '{expected_prefix}', got examples: {invalid_ids[:5]}")

    return df

def iter_source_file_chunks(
    file_path: str,
    chunk_size: int = 100000,
    expected_prefix: Optional[str] = None
) -> Generator[pl.DataFrame, None, None]:
    """
    Streams a TSV file in sequential memory-safe chunks of `chunk_size` rows.
    Uses pure file streaming to strictly avoid large memory mappings or RAM spikes.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline()
        if not header_line:
            return
        headers = [h.strip() for h in header_line.split("\t")]
        
        batch_rows = []
        for line in f:
            line_str = line.strip("\r\n")
            if not line_str:
                continue
            parts = line_str.split("\t")
            # Pad or truncate to match header length
            if len(parts) < len(headers):
                parts.extend([""] * (len(headers) - len(parts)))
            elif len(parts) > len(headers):
                parts = parts[:len(headers)]
            batch_rows.append(parts)

            if len(batch_rows) >= chunk_size:
                # Convert to Polars DataFrame
                col_data = {headers[i]: [r[i] for r in batch_rows] for i in range(len(headers))}
                chunk_df = pl.DataFrame(col_data)
                del batch_rows, col_data
                yield chunk_df
                batch_rows = []
                gc.collect()

        if batch_rows:
            col_data = {headers[i]: [r[i] for r in batch_rows] for i in range(len(headers))}
            chunk_df = pl.DataFrame(col_data)
            del batch_rows, col_data
            yield chunk_df
            gc.collect()

def load_ground_truth(file_path: str, n_rows: Optional[int] = None) -> pl.DataFrame:
    """
    Loads train_ground_truth.tsv and validates columns.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ground truth file not found at: {file_path}")

    df = pl.read_csv(
        file_path,
        separator="\t",
        n_rows=n_rows,
        truncate_ragged_lines=True,
        null_values=["NULL", "null", "None", "NaN"]
    ).with_columns(
        pl.col("matched_entity_ids").fill_null("")
    )

    expected_cols = ["source1_entity_id", "matched_entity_ids"]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {file_path}. Found: {df.columns}")

    return df
