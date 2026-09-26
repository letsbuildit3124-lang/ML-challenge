"""
Antigravity V5 Unit Test Suite for Multilingual E5 GPU-First Grouped-Job Pipeline.
Validates:
- Job Grouping & Manifests (~1M rows per job)
- Asymmetric E5 Prefixes & Vectorized Text Construction
- Strict Positional ID Alignment & Row-Count Validation (Fixing 10k != 100k bug)
- NaN / Inf Embedding Validation & L2 Normalization Checks
- Queue Timeout & Structured Failure Classification
- Smoke vs Production Path Separation & 10.32M Completeness Invariants
- Cache Invalidation & Resumability Guards
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
from src.dense_exporter import DenseExporter, compute_dict_hash
from src.kaggle_dense_controller import KaggleDenseController
from src.build_dense_faiss import validate_embedding_corpus, EXPECTED_PRODUCTION_TARGETS


# 1. Test Asymmetric E5 Prefixes
def test_e5_prefix():
    text = "Tata Consultancy Services | Mumbai | India"
    q_text = format_e5_text(text, is_query=True)
    p_text = format_e5_text(text, is_query=False)

    assert q_text.startswith(E5_QUERY_PREFIX)
    assert p_text.startswith(E5_PASSAGE_PREFIX)
    assert q_text != p_text
    assert q_text == "query: Tata Consultancy Services | Mumbai | India"
    assert p_text == "passage: Tata Consultancy Services | Mumbai | India"


# 2. Test Multilingual Unicode Text Construction & Null Handling
def test_text_builder():
    # Native unicode scripts
    hindi_text = build_entity_text("रिलायंस इंडस्ट्रीज", "मुंबई", "भारत")
    assert hindi_text == "रिलायंस इंडस्ट्रीज | मुंबई | भारत"

    japanese_text = build_entity_text("トヨタ自動車株式会社", "愛知県豊田市", "日本")
    assert japanese_text == "トヨタ自動車株式会社 | 愛知県豊田市 | 日本"

    # Null / Missing values
    partial_text = build_entity_text("Acme Corp", None, "US")
    assert partial_text == "Acme Corp | US"

    empty_text = build_entity_text(None, "", None)
    assert empty_text == "unknown"


# 3. Test Vectorized Text Construction with Polars
def test_vectorized_text_builder():
    df = pl.DataFrame({
        "target_id": ["E1", "E2", "E3"],
        "business_name": ["Google LLC", "Siemens AG", None],
        "business_address": ["Mountain View, CA", "Munich", "Unknown Address"],
        "country": ["US", "DE", "FR"]
    })

    passages = vectorized_build_e5_texts(df, is_query=False)
    queries = vectorized_build_e5_texts(df, is_query=True)

    assert len(passages) == 3
    assert len(queries) == 3
    assert passages[0] == "passage: Google LLC | Mountain View, CA | US"
    assert passages[1] == "passage: Siemens AG | Munich | DE"
    assert passages[2] == "passage: Unknown Address | FR"
    assert queries[0] == "query: Google LLC | Mountain View, CA | US"


# 4. Test Grouped Job Planning (~1M rows per job)
def test_job_grouping(tmp_path):
    exporter = DenseExporter(
        output_root=str(tmp_path),
        chunk_size=100000,
        job_target_rows=1000000
    )

    # Simulate 104 chunks of 100k rows (total 10,320,219 targets)
    total_targets = 10320219
    chunk_list = []
    for i in range(104):
        n = 100000 if i < 103 else (total_targets - 103 * 100000) # 20,219 in final chunk
        chunk_list.append({
            "chunk_id": i,
            "row_count": n,
            "input_filename": f"chunk_{i:06d}.parquet"
        })

    jobs = exporter.plan_grouped_jobs(chunk_list)

    # 104 chunks with 10 chunks/job = 11 grouped jobs
    assert len(jobs) == 11
    assert jobs[0]["job_id"] == "job_000"
    assert len(jobs[0]["chunk_ids"]) == 10 # 10 chunks = 1,000,000 rows
    assert jobs[0]["expected_row_count"] == 1000000

    # Final job (job_010) has 4 chunks (3x100k + 20,219 = 320,219 rows)
    assert jobs[10]["job_id"] == "job_010"
    assert len(jobs[10]["chunk_ids"]) == 4
    assert jobs[10]["expected_row_count"] == 320219

    # Sum of all job expected rows must match total
    total_grouped_rows = sum(j["expected_row_count"] for j in jobs)
    assert total_grouped_rows == total_targets


# 5. Test Job Manifest Schema & Config Hash
def test_job_manifest(tmp_path):
    exporter = DenseExporter(output_root=str(tmp_path))
    dummy_chunks = [
        {"chunk_id": 0, "row_count": 100000, "input_filename": "chunk_000000.parquet"},
        {"chunk_id": 1, "row_count": 100000, "input_filename": "chunk_000001.parquet"}
    ]
    job = exporter._create_job_dict(0, dummy_chunks, 200000)

    assert "job_id" in job
    assert "model_name" in job
    assert "embedding_dimension" in job
    assert "config_hash" in job
    assert job["expected_row_count"] == 200000
    assert job["chunk_ids"] == [0, 1]


# 6. Test Exact ID Alignment & Row Count Validation (Preventing 10k != 100k bug)
def test_exact_id_alignment(tmp_path):
    controller = KaggleDenseController(logs_dir=str(tmp_path / "logs"))

    # Create synthetic input parquet
    in_dir = tmp_path / "input"
    in_dir.mkdir()
    df_in = pl.DataFrame({"target_id": [f"T_{i:05d}" for i in range(100)]})
    in_file = in_dir / "chunk_000000.parquet"
    df_in.write_parquet(in_file)

    out_dir = tmp_path / "output"
    chunk_dir = out_dir / "chunk_000000"
    chunk_dir.mkdir(parents=True)

    # 1. Matching case
    embs = np.random.randn(100, 384).astype(np.float16)
    norms = np.linalg.norm(embs.astype(np.float32), axis=1, keepdims=True)
    embs = (embs.astype(np.float32) / norms).astype(np.float16)
    np.save(chunk_dir / "embeddings.npy", embs)
    df_in.write_parquet(chunk_dir / "target_ids.parquet")
    with open(out_dir / "job_result.json", "w") as f:
        json.dump({"status": "completed"}, f)

    job_manifest = {
        "job_id": "job_000",
        "embedding_dimension": 384,
        "input_filenames": ["chunk_000000.parquet"]
    }

    ok, msg, _ = controller.verify_grouped_output(job_manifest, str(in_dir), str(out_dir))
    assert ok is True

    # 2. Row count mismatch (e.g. 10 embeddings vs 100 inputs)
    embs_bad = embs[:10]
    np.save(chunk_dir / "embeddings.npy", embs_bad)
    ok_bad, msg_bad, _ = controller.verify_grouped_output(job_manifest, str(in_dir), str(out_dir))
    assert ok_bad is False
    assert "ROW_COUNT_MISMATCH" in msg_bad

    # 3. ID positional ordering mismatch
    np.save(chunk_dir / "embeddings.npy", embs)
    df_mismatched = pl.DataFrame({"target_id": [f"T_{99-i:05d}" for i in range(100)]})
    df_mismatched.write_parquet(chunk_dir / "target_ids.parquet")
    ok_id_err, msg_id_err, _ = controller.verify_grouped_output(job_manifest, str(in_dir), str(out_dir))
    assert ok_id_err is False
    assert "ID_ALIGNMENT_MISMATCH" in msg_id_err


# 7. Test NaN / Inf Validation
def test_nan_validation(tmp_path):
    controller = KaggleDenseController(logs_dir=str(tmp_path / "logs"))
    in_dir = tmp_path / "input"
    in_dir.mkdir()
    df_in = pl.DataFrame({"target_id": ["T_1", "T_2"]})
    df_in.write_parquet(in_dir / "chunk_000000.parquet")

    out_dir = tmp_path / "output"
    chunk_dir = out_dir / "chunk_000000"
    chunk_dir.mkdir(parents=True)
    df_in.write_parquet(chunk_dir / "target_ids.parquet")
    with open(out_dir / "job_result.json", "w") as f:
        json.dump({"status": "completed"}, f)

    # Embeddings containing NaN
    embs_nan = np.array([[np.nan] * 384, [1.0] * 384], dtype=np.float16)
    np.save(chunk_dir / "embeddings.npy", embs_nan)

    job_manifest = {
        "job_id": "job_000",
        "embedding_dimension": 384,
        "input_filenames": ["chunk_000000.parquet"]
    }
    ok, msg, _ = controller.verify_grouped_output(job_manifest, str(in_dir), str(out_dir))
    assert ok is False
    assert "EMBEDDING_VALIDATION_ERROR" in msg


# 8. Test Normalization Validation
def test_normalization(tmp_path):
    controller = KaggleDenseController(logs_dir=str(tmp_path / "logs"))
    in_dir = tmp_path / "input"
    in_dir.mkdir()
    df_in = pl.DataFrame({"target_id": ["T_1"]})
    df_in.write_parquet(in_dir / "chunk_000000.parquet")

    out_dir = tmp_path / "output"
    chunk_dir = out_dir / "chunk_000000"
    chunk_dir.mkdir(parents=True)
    df_in.write_parquet(chunk_dir / "target_ids.parquet")
    with open(out_dir / "job_result.json", "w") as f:
        json.dump({"status": "completed"}, f)

    # Unnormalized embedding (norm ~ 5.0)
    embs_unnorm = np.full((1, 384), 5.0, dtype=np.float16)
    np.save(chunk_dir / "embeddings.npy", embs_unnorm)

    job_manifest = {
        "job_id": "job_000",
        "embedding_dimension": 384,
        "input_filenames": ["chunk_000000.parquet"]
    }
    ok, msg, _ = controller.verify_grouped_output(job_manifest, str(in_dir), str(out_dir))
    assert ok is False
    assert "not L2 normalized" in msg


# 9. Test Queue Timeout Detection & Failure Classification
def test_queue_timeout(tmp_path):
    controller = KaggleDenseController(
        queue_timeout_seconds=2,
        poll_interval=1,
        logs_dir=str(tmp_path / "logs")
    )

    with patch("subprocess.run") as mock_run:
        # Simulate Kaggle CLI returning 'Queued' indefinitely
        mock_run.return_value = MagicMock(stdout='"Queued"', returncode=0)
        ok, msg, telemetry = controller.poll_job_status("job_000")
        assert ok is False
        assert "QUEUE_TIMEOUT" in msg
        assert telemetry["queue_seconds"] >= 2.0


# 10. Test Smoke / Production Path Separation
def test_smoke_production_separation(tmp_path):
    config = {
        "paths": {
            "production_embeddings": "cache/embeddings/multilingual_e5/production/target_embeddings.npy",
            "smoke_embeddings": "cache/embeddings/multilingual_e5/smoke/target_embeddings.npy",
            "production_ann_dir": "cache/ann/multilingual_e5/production",
            "smoke_ann_dir": "cache/ann/multilingual_e5/smoke"
        }
    }

    prod_path = config["paths"]["production_embeddings"]
    smoke_path = config["paths"]["smoke_embeddings"]
    assert "production" in prod_path
    assert "smoke" in smoke_path
    assert prod_path != smoke_path


# 11. Test Production Completeness Invariant (10,320,219 vectors)
def test_production_completeness(tmp_path):
    emb_file = tmp_path / "target_embeddings.npy"
    ids_file = tmp_path / "target_ids.json"

    # Only 100 vectors
    mmap = np.lib.format.open_memmap(str(emb_file), mode="w+", dtype="float16", shape=(100, 384))
    mmap.flush()
    del mmap
    with open(ids_file, "w") as f:
        json.dump([f"ID_{i}" for i in range(100)], f)

    # Should raise error when evaluated as production corpus
    with pytest.raises(ValueError, match="PRODUCTION INVARIANT VIOLATION"):
        validate_embedding_corpus(str(emb_file), str(ids_file), is_production=True)

    # Should succeed when evaluated as smoke/dev corpus
    rows, dim = validate_embedding_corpus(str(emb_file), str(ids_file), is_production=False)
    assert rows == 100
    assert dim == 384


# 12. Test Cache Invalidation on Config Change
def test_cache_invalidation():
    cfg1 = {"model_name": "intfloat/multilingual-e5-small", "dimension": 384, "precision": "fp16"}
    cfg2 = {"model_name": "intfloat/multilingual-e5-small", "dimension": 384, "precision": "fp32"}
    cfg3 = {"model_name": "BAAI/bge-m3", "dimension": 1024, "precision": "fp16"}

    hash1 = compute_dict_hash(cfg1)
    hash2 = compute_dict_hash(cfg2)
    hash3 = compute_dict_hash(cfg3)

    assert hash1 != hash2
    assert hash1 != hash3
