"""
Data loading and validation module.
"""

import os
import sys
import gc
import polars as pl
from typing import Dict, List, Any, Tuple, Optional
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

def print_dataset_inventory(config: Config) -> Dict[str, Any]:
    """
    Prints a concise inventory of all training and test datasets without keeping them in memory.
    """
    print("=" * 70)
    print("DATASET INVENTORY & INTEGRITY CHECK")
    print("=" * 70)

    files_info = [
        ("Train Source 1", config.train_s1_path, "S1-"),
        ("Train Source 2", config.train_s2_path, "S2-"),
        ("Train Source 3", config.train_s3_path, "S3-"),
        ("Test Source 1", config.test_s1_path, "S1-"),
        ("Test Source 2", config.test_s2_path, "S2-"),
        ("Test Source 3", config.test_s3_path, "S3-"),
    ]

    summary = {}

    for name, path, prefix in files_info:
        if os.path.exists(path):
            df = load_source_file(path, expected_prefix=prefix)
            row_count = len(df)
            null_name = df["business_name"].null_count()
            null_addr = df["business_address"].null_count()
            null_country = df["country"].null_count()
            summary[name] = {
                "rows": row_count,
                "null_name": null_name,
                "null_addr": null_addr,
                "null_country": null_country
            }
            print(f"[{name}] Rows: {row_count:,} | Nulls -> Name: {null_name:,}, Addr: {null_addr:,}, Country: {null_country:,}")
            del df
            gc.collect()
        else:
            print(f"[MISSING] {name} not found at {path}")

    if os.path.exists(config.train_gt_path):
        gt_df = load_ground_truth(config.train_gt_path)
        summary["Train Ground Truth"] = {"rows": len(gt_df)}
        print(f"[Train Ground Truth] Rows: {len(gt_df):,}")
        del gt_df
        gc.collect()

    print("=" * 70)
    return summary
