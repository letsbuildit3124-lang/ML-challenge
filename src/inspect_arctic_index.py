"""
Antigravity V4.1 Arctic Index Completeness & Alignment Inspection Utility.
Verifies synchronization across:
1. DuckDB Target Database (targets table)
2. Disk-Backed Arctic Embeddings (target_embeddings.npy)
3. Target ID Index (target_ids.json)
4. FAISS ANN Index (target.index)
"""

import os
import sys
import json
import duckdb
import numpy as np

from src.config import get_config
from src.resource_tracker import get_current_rss_mb
from src.arctic_embeddings import EMBEDDING_DIM, DEFAULT_MODEL_NAME

EXPECTED_TARGET_COUNT = 10320219

def inspect_arctic_index():
    config = get_config()
    db_path = os.path.join(config.base_dir, "cache", "entity_resolution.duckdb")
    emb_dir = os.path.join(config.base_dir, "cache", "embeddings", "arctic")
    ann_dir = os.path.join(config.base_dir, "cache", "ann", "arctic")

    emb_path = os.path.join(emb_dir, "target_embeddings.npy")
    ids_path = os.path.join(emb_dir, "target_ids.json")
    emb_meta_path = os.path.join(emb_dir, "metadata.json")

    index_path = os.path.join(ann_dir, "target.index")
    ann_meta_path = os.path.join(ann_dir, "metadata.json")

    print("=" * 80)
    print("ANTIGRAVITY V4.1 — ARCTIC TARGET INDEX VALIDATION AUDIT")
    print("=" * 80)
    print(f"Expected Production Targets: {EXPECTED_TARGET_COUNT:,}")
    print(f"Current RSS:                 {get_current_rss_mb():.2f} MB")
    print("-" * 80)

    # 1. DuckDB Targets Count
    duckdb_count = 0
    if os.path.exists(db_path):
        try:
            conn = duckdb.connect(db_path, read_only=True)
            duckdb_count = conn.execute("SELECT COUNT(*) FROM targets;").fetchone()[0]
            conn.close()
            print(f"DuckDB Targets in DB:        {duckdb_count:,}")
        except Exception as e:
            print(f"DuckDB Targets in DB:        ERROR ({e})")
    else:
        print(f"DuckDB Targets in DB:        NOT FOUND ({db_path})")

    # 2. Embedding MMap Count & Shape
    emb_rows = 0
    emb_dim = 0
    if os.path.exists(emb_path):
        try:
            mmap = np.lib.format.open_memmap(emb_path, mode="r")
            emb_rows, emb_dim = mmap.shape
            print(f"Disk-Backed Embedding Rows:  {emb_rows:,} (Dimension: {emb_dim})")
            del mmap
        except Exception as e:
            print(f"Disk-Backed Embedding Rows:  ERROR ({e})")
    else:
        print(f"Disk-Backed Embedding Rows:  NOT FOUND ({emb_path})")

    # 3. Target IDs Count
    ids_count = 0
    if os.path.exists(ids_path):
        try:
            with open(ids_path, "r", encoding="utf-8") as f:
                ids_data = json.load(f)
                ids_count = len(ids_data)
            print(f"Target IDs Registered:       {ids_count:,}")
        except Exception as e:
            print(f"Target IDs Registered:       ERROR ({e})")
    else:
        print(f"Target IDs Registered:       NOT FOUND ({ids_path})")

    # 4. FAISS Index ntotal
    faiss_ntotal = 0
    index_type = "None"
    if os.path.exists(index_path):
        try:
            import faiss
            if os.path.getsize(index_path) > 1024:
                idx = faiss.read_index(index_path)
                faiss_ntotal = idx.ntotal
                index_type = type(idx).__name__
                del idx
                print(f"FAISS Index Total Vectors:   {faiss_ntotal:,} ({index_type})")
            else:
                print(f"FAISS Index Total Vectors:   Fallback Marker ({os.path.getsize(index_path)} bytes)")
        except ImportError:
            print(f"FAISS Index Total Vectors:   NumPy Fallback (FAISS not installed)")
            faiss_ntotal = emb_rows
        except Exception as e:
            print(f"FAISS Index Total Vectors:   ERROR ({e})")
    else:
        print(f"FAISS Index Total Vectors:   NOT FOUND ({index_path})")

    # 5. Metadata Check
    is_complete_emb = False
    if os.path.exists(emb_meta_path):
        with open(emb_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            is_complete_emb = meta.get("is_complete_target_universe", False)

    is_complete_ann = False
    if os.path.exists(ann_meta_path):
        with open(ann_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            is_complete_ann = meta.get("is_complete_target_universe", False)

    print("-" * 80)
    print(f"Embedding Universe Complete: {'YES' if is_complete_emb else 'NO (Smoke / Dev Test)'}")
    print(f"ANN Index Universe Complete: {'YES' if is_complete_ann else 'NO (Smoke / Dev Test)'}")
    print(f"Embedding Model:             {DEFAULT_MODEL_NAME}")
    print(f"Metric:                      Inner Product (Cosine Similarity)")
    print("-" * 80)

    # 6. Evaluation Verdict
    all_matched = (
        duckdb_count == EXPECTED_TARGET_COUNT and
        emb_rows == EXPECTED_TARGET_COUNT and
        ids_count == EXPECTED_TARGET_COUNT and
        (faiss_ntotal == EXPECTED_TARGET_COUNT or faiss_ntotal == 0) and
        is_complete_emb and
        is_complete_ann
    )

    if all_matched:
        print("STATUS: VALID (COMPLETE 10.32M PRODUCTION INDEX)")
        print("Ready for full V4.1 dense recall benchmarking.")
    elif emb_rows > 0 and emb_rows == ids_count:
        print(f"STATUS: SMOKE TEST / INCOMPLETE ({emb_rows:,} / {EXPECTED_TARGET_COUNT:,} targets)")
        print("WARNING: This index is a partial development subset. It cannot be used for production recall benchmarks.")
    else:
        print("STATUS: INVALID / DESYNCHRONIZED ARTIFACTS")
        print("Error: Target rows, IDs, and index counts do not match. Re-run build_arctic_embeddings.")

    print("=" * 80)

if __name__ == "__main__":
    inspect_arctic_index()
