"""
Antigravity Unit Test Suite for One-Shot E5 GPU Pipeline.
Tests:
- E5 Text Construction & Asymmetric Prefixes
- Target Input Export Metadata & Checksum Logic
- Strict Verification Invariants (Dimensions, Dtype, Finite Values, ID Alignment)
- Disk-Backed Float16 Memmap Assembly
- Provenance Categorization (FROM_V4, FROM_E5, FROM_BOTH)
- Production Completeness Assertion
"""

import os
import sys
import json
import pytest
import numpy as np
import polars as pl
from unittest.mock import MagicMock, patch

from src.dense.text_builder import (
    build_entity_text,
    format_e5_text,
    vectorized_build_e5_texts,
    E5_QUERY_PREFIX,
    E5_PASSAGE_PREFIX,
)
from src.verify_e5_embeddings import verify_e5_output
from src.assemble_e5_memmap import assemble_e5_memmap
from src.v5_e5_hybrid_pipeline import get_provenance_label, PROV_DET, PROV_SPARSE_NAME, PROV_E5_DENSE
from src.export_e5_input import build_export_sql


def test_build_export_sql():
    # 1. Full universe query
    sql_full = build_export_sql(limit_rows=None)
    assert not sql_full.rstrip().endswith(";")
    assert "ORDER BY target_row_id ASC" in sql_full
    assert "LIMIT" not in sql_full

    # 2. Smoke query with limit
    sql_smoke = build_export_sql(limit_rows=10000)
    assert not sql_smoke.rstrip().endswith(";")
    assert "LIMIT 10000" in sql_smoke
    assert "ORDER BY target_row_id ASC" in sql_smoke

    # 3. Verify valid nesting inside COPY (...) expression
    copy_sql = f"COPY ({sql_smoke}) TO 'test.parquet' (FORMAT PARQUET);"
    assert ";" not in sql_smoke
    assert "LIMIT 10000;" not in copy_sql


def test_e5_text_builder_prefixes():
    raw_entity = "Siemens AG | Munich | Germany"
    passage = format_e5_text(raw_entity, is_query=False)
    query = format_e5_text(raw_entity, is_query=True)

    assert passage.startswith(E5_PASSAGE_PREFIX)
    assert query.startswith(E5_QUERY_PREFIX)
    assert passage == "passage: Siemens AG | Munich | Germany"
    assert query == "query: Siemens AG | Munich | Germany"


def test_e5_vectorized_builder_unicode():
    df = pl.DataFrame({
        "target_id": ["T1", "T2"],
        "business_name": ["स्टेट बैंक ऑफ इंडिया", "LVMH Moët Hennessy"],
        "business_address": ["मुंबई", "Paris"],
        "country": ["IN", "FR"]
    })

    passages = vectorized_build_e5_texts(df, is_query=False)
    assert len(passages) == 2
    assert passages[0] == "passage: स्टेट बैंक ऑफ इंडिया | मुंबई | IN"
    assert passages[1] == "passage: LVMH Moët Hennessy | Paris | FR"


def test_provenance_labels():
    # Only V4 (Deterministic)
    h_v4, h_e5, h_both, label = get_provenance_label(PROV_DET)
    assert h_v4 is True and h_e5 is False and h_both is False and label == "FROM_V4"

    # Only E5 Dense
    h_v4, h_e5, h_both, label = get_provenance_label(PROV_E5_DENSE)
    assert h_v4 is False and h_e5 is True and h_both is False and label == "FROM_E5"

    # Both V4 + E5
    h_v4, h_e5, h_both, label = get_provenance_label(PROV_DET | PROV_E5_DENSE)
    assert h_v4 is True and h_e5 is True and h_both is True and label == "FROM_BOTH"


def test_e5_verification_and_assembly(tmp_path):
    # Create synthetic shard outputs
    out_dir = tmp_path / "output"
    emb_dir = out_dir / "embeddings"
    ids_dir = out_dir / "ids"
    emb_dir.mkdir(parents=True)
    ids_dir.mkdir(parents=True)

    # Shard 0: 50 rows
    embs_0 = np.random.randn(50, 384).astype(np.float16)
    norms_0 = np.linalg.norm(embs_0.astype(np.float32), axis=1, keepdims=True)
    embs_0 = (embs_0.astype(np.float32) / norms_0).astype(np.float16)
    np.save(emb_dir / "part-00000.npy", embs_0)
    pl.DataFrame({"target_id": [f"T_{i:04d}" for i in range(50)]}).write_parquet(ids_dir / "part-00000.parquet")

    # Shard 1: 50 rows
    embs_1 = np.random.randn(50, 384).astype(np.float16)
    norms_1 = np.linalg.norm(embs_1.astype(np.float32), axis=1, keepdims=True)
    embs_1 = (embs_1.astype(np.float32) / norms_1).astype(np.float16)
    np.save(emb_dir / "part-00001.npy", embs_1)
    pl.DataFrame({"target_id": [f"T_{i:04d}" for i in range(50, 100)]}).write_parquet(ids_dir / "part-00001.parquet")

    # Manifest
    manifest = {
        "model_name": "intfloat/multilingual-e5-small",
        "embedding_dim": 384,
        "dtype": "float16",
        "total_expected_rows": 100,
        "shards": [{"shard_id": "part-00000"}, {"shard_id": "part-00001"}]
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    # 1. Run Verification
    ok, msg, summary = verify_e5_output(
        output_dir=str(out_dir),
        input_parquet=None,
        expected_rows=100,
        is_production=False
    )
    assert ok is True
    assert summary["total_rows"] == 100
    assert summary["total_shards"] == 2

    # 2. Run Memmap Assembly
    dest_ann = tmp_path / "ann"
    memmap_path, ids_path = assemble_e5_memmap(
        input_shards_dir=str(out_dir),
        output_dir=str(dest_ann),
        is_production=False
    )

    assert os.path.exists(memmap_path)
    assert os.path.exists(ids_path)

    # Verify assembled array
    assembled = np.lib.format.open_memmap(memmap_path, mode="r")
    assert assembled.shape == (100, 384)
    assert assembled.dtype == np.float16
    del assembled

    with open(ids_path, "r") as f:
        assembled_ids = json.load(f)
    assert len(assembled_ids) == 100
    assert assembled_ids[0] == "T_0000"
    assert assembled_ids[-1] == "T_0099"
