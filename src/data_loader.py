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
    chunk_size: int = 250000,
    expected_prefix: Optional[str] = None
) -> Generator[pl.DataFrame, None, None]:
    """
    Streams a source TSV file in memory-safe chunks using Polars batched reader.
    Keeps memory footprint strictly low on resource-constrained servers.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file not found at: {file_path}")

    reader = pl.read_csv_batched(
        file_path,
        separator="\t",
        batch_size=chunk_size,
        truncate_ragged_lines=True,
        null_values=["", "NULL", "null", "None", "NaN"]
    )

    while True:
        batches = reader.next_batches(1)
        if not batches:
            break
        chunk = batches[0]
        if len(chunk) == 0:
            break
        yield chunk

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
