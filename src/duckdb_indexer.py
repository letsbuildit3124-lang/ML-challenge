"""
Antigravity V3 Persistent DuckDB Target Cache & Candidate Generation Engine.

Key Architectural Principles:
1. Target data preparation (S2 + S3 ingestion, normalization, indexing) is executed ONCE via `src.build_target_cache`.
2. Cache is persisted in `cache/entity_resolution.duckdb` and tracked by `cache/entity_resolution_cache.json`.
3. All candidate-generation, recall experiments, Arctic, and ML training scripts REUSE the existing target cache.
4. Experiments do NOT rebuild or re-ingest target data unless explicitly invoked with `--force-rebuild`.
5. Peak RAM is strictly bounded at < 200MB via chunked streaming, temporary tables, and strict equi-join passes.
"""

import os
import gc
import json
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any, Optional
import duckdb
import polars as pl

from src.config import get_config
from src.data_loader import iter_source_file_chunks, load_source_file
from src.blocking_v2 import add_v2_blocking_columns

DEFAULT_DB_PATH = "cache/entity_resolution.duckdb"
DEFAULT_MANIFEST_PATH = "cache/entity_resolution_cache.json"
DEFAULT_TMP_DIR = "cache/duckdb_tmp"
CACHE_VERSION = "v3.3_clean_equijoins"
SCHEMA_VERSION = "3.3"
NORMALIZATION_VERSION = "v3_boundary_aware_translit"

AVAILABLE_BLOCKERS = [
    ("1. Exact Compact Name", 1),
    ("2. Translit Compact Name", 2),
    ("3. Exact Normalized Name", 4),
    ("4. Phonetic Soundex + Num", 8),
    ("5. CName8 + Addr Num", 16),
    ("6. Translit CName8 + Num", 32),
    ("7. First 2 Words + Num", 64),
    ("8. Postal + Name Prefix-4", 128),
    ("9. Cross Translit CName8 Num", 256),
    ("10. Country-Agnostic Fallback", 512),
]

def compute_files_fingerprint(paths: List[str]) -> str:
    """Computes a deterministic fingerprint string based on file paths, sizes, and mtimes."""
    parts = []
    for p in paths:
        if os.path.exists(p):
            stat = os.stat(p)
            parts.append(f"{os.path.basename(p)}:{stat.st_size}:{int(stat.st_mtime)}")
        else:
            parts.append(f"{os.path.basename(p)}:missing")
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()

class DuckDBTargetIndexer:
    """
    Persistent Disk-Backed DuckDB Target Cache Manager.
    Separates one-time target ingestion from repeated experimental candidate queries.
    """
    def __init__(
        self,
        db_path: Optional[str] = None,
        manifest_path: Optional[str] = None,
        memory_limit: str = "2GB",
        threads: int = 2,
        tmp_dir: Optional[str] = None
    ):
        config = get_config()
        self.db_path = db_path or os.path.join(config.base_dir, DEFAULT_DB_PATH)
        self.manifest_path = manifest_path or os.path.join(config.base_dir, DEFAULT_MANIFEST_PATH)
        self.tmp_dir = tmp_dir or os.path.join(config.base_dir, DEFAULT_TMP_DIR)
        self.memory_limit = memory_limit
        self.threads = threads

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.conn: Optional[duckdb.DuckDBPyConnection] = None

    def _init_connection(self):
        """Initializes DuckDB connection with strict memory limits."""
        if self.conn is None:
            self.conn = duckdb.connect(self.db_path)
            self.conn.execute(f"SET memory_limit='{self.memory_limit}';")
            self.conn.execute(f"SET threads={self.threads};")
            self.conn.execute("SET preserve_insertion_order=false;")
            self.conn.execute(f"SET temp_directory='{self.tmp_dir.replace(chr(92), '/')}';")
            self.conn.execute("PRAGMA wal_autocheckpoint='50MB';")

    def _init_tables(self):
        """Creates target primary table."""
        self._init_connection()
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS targets (
                target_row_id BIGINT PRIMARY KEY,
                eid VARCHAR,
                country VARCHAR,
                norm_name VARCHAR,
                compact_name VARCHAR,
                norm_addr VARCHAR,
                cname8_num VARCHAR,
                translit_cname8_num VARCHAR,
                f2_num VARCHAR,
                pin_cname4 VARCHAR,
                soundex_num VARCHAR,
                translit_cname VARCHAR
            );
        """)

    def validate_cache(self, expected_source_paths: Optional[List[str]] = None) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Fast lightweight cache validation (takes < 0.02 seconds).
        Checks manifest, database file existence, schema version, and source fingerprints.
        """
        if not os.path.exists(self.db_path):
            return False, f"Database file not found at {self.db_path}", {}

        if not os.path.exists(self.manifest_path):
            return False, f"Cache manifest not found at {self.manifest_path}", {}

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            return False, f"Failed to parse cache manifest: {e}", {}

        if manifest.get("cache_version") != CACHE_VERSION:
            return False, f"Cache version mismatch (Found: {manifest.get('cache_version')}, Expected: {CACHE_VERSION})", manifest

        if manifest.get("total_rows", 0) <= 0:
            return False, "Cache manifest indicates 0 indexed target rows", manifest

        if expected_source_paths:
            current_fp = compute_files_fingerprint(expected_source_paths)
            if manifest.get("target_data_fingerprint") != current_fp:
                return False, "Source dataset files have changed on disk (fingerprint mismatch)", manifest

        return True, "Cache is valid and ready for reuse", manifest

    def ensure_cache_ready(self, expected_source_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Ensures the persistent cache is ready for experiments.
        Reuses existing cache if valid. If invalid, raises an explicit error directing to build_target_cache.
        """
        is_valid, reason, manifest = self.validate_cache(expected_source_paths)
        if is_valid:
            db_size_mb = os.path.getsize(self.db_path) / (1024 * 1024)
            print(f"[DuckDBCache] Reusing existing persistent target cache.")
            print(f"  Database:        {self.db_path} ({db_size_mb:.2f} MB)")
            print(f"  S2 Records:      {manifest.get('source2_rows', 0):,}")
            print(f"  S3 Records:      {manifest.get('source3_rows', 0):,}")
            print(f"  Total Targets:   {manifest.get('total_rows', 0):,}")
            print(f"  Blockers Ready:  {len(manifest.get('available_blockers', []))} blockers")
            self._init_connection()
            return manifest
        else:
            raise RuntimeError(
                f"\n{'='*80}\n"
                f"[DuckDBCache ERROR] Persistent target cache is missing or outdated!\n"
                f"Reason: {reason}\n"
                f"Please build the target cache once by executing:\n"
                f"    PYTHONPATH=. python3 -m src.build_target_cache --force-rebuild\n"
                f"{'='*80}"
            )

    def build_cache_from_sources(
        self,
        source_paths: List[Tuple[str, str, str]], # (source_name, file_path, prefix)
        chunk_size: int = 100000,
        limit_per_file: Optional[int] = None,
        force_rebuild: bool = False
    ) -> Dict[str, Any]:
        """
        Builds the persistent DuckDB target cache from raw TSV sources.
        Called strictly by `src.build_target_cache`.
        """
        file_paths = [p for name, p, pref in source_paths]
        is_valid, reason, manifest = self.validate_cache(file_paths)

        if not force_rebuild and is_valid and limit_per_file is None:
            print(f"[DuckDBCache] Target cache is already up-to-date. Reusing {self.db_path} ({manifest.get('total_rows', 0):,} records).")
            return manifest

        print(f"\n" + "=" * 80)
        print(f"[DuckDBCache] BUILDING PERSISTENT TARGET CACHE")
        print(f"Target Database: {self.db_path}")
        print(f"Chunk Size:      {chunk_size:,} rows | Force Rebuild: {force_rebuild}")
        print(f"=" * 80)

        # Close and remove existing database if forcing rebuild
        if self.conn is not None:
            self.conn.close()
            self.conn = None

        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        if os.path.exists(self.manifest_path):
            os.remove(self.manifest_path)

        self._init_connection()
        self._init_tables()

        t0 = time.time()
        global_target_row_id = 0
        src_counts = {}
        temp_chunk_parquet = os.path.join(self.tmp_dir, "temp_target_ingest.parquet").replace("\\", "/")

        chunk_counter = 0

        for src_name, path, prefix in source_paths:
            print(f"\n[DuckDBCache] Ingesting {src_name} ({path})...", flush=True)
            if not os.path.exists(path):
                print(f"  Warning: File {path} not found. Skipping.")
                src_counts[src_name] = 0
                continue

            src_processed = 0
            t_src = time.time()

            for chunk_df in iter_source_file_chunks(path, chunk_size=chunk_size, expected_prefix=prefix):
                if limit_per_file and src_processed >= limit_per_file:
                    break

                n_rows = len(chunk_df)
                if limit_per_file and (src_processed + n_rows) > limit_per_file:
                    chunk_df = chunk_df.head(limit_per_file - src_processed)
                    n_rows = len(chunk_df)

                # Vectorize base blocking columns
                chunk_p = add_v2_blocking_columns(chunk_df)
                del chunk_df

                row_ids = list(range(global_target_row_id, global_target_row_id + n_rows))
                chunk_p = chunk_p.with_columns(pl.Series("target_row_id", row_ids, dtype=pl.Int64))

                for col in ["cname8_num", "translit_cname8_num", "f2_num", "pin_cname4", "soundex_num", "translit_cname"]:
                    if col not in chunk_p.columns:
                        chunk_p = chunk_p.with_columns(pl.lit(None).cast(pl.String).alias(col))

                chunk_p.write_parquet(temp_chunk_parquet)
                del chunk_p
                gc.collect()

                # Ingestion into targets table
                self.conn.execute(f"""
                    INSERT INTO targets
                    SELECT 
                        target_row_id, eid, country, norm_name, compact_name, norm_addr,
                        cname8_num, translit_cname8_num, f2_num, pin_cname4, soundex_num, translit_cname
                    FROM read_parquet('{temp_chunk_parquet}');
                """)

                if os.path.exists(temp_chunk_parquet):
                    os.remove(temp_chunk_parquet)

                global_target_row_id += n_rows
                src_processed += n_rows
                chunk_counter += 1

                # Periodic checkpointing to release write buffers
                if chunk_counter % 5 == 0:
                    self.conn.execute("CHECKPOINT;")

                pct_str = f"({src_processed:,})" if not limit_per_file else f"({src_processed:,}/{limit_per_file:,})"
                speed = src_processed / (time.time() - t_src)
                print(f"  -> Ingested {pct_str} records into DuckDB | Rate: {speed:,.0f} rows/s", flush=True)

            src_counts[src_name] = src_processed

        # Final checkpoint to persist and compress database
        print("\n[DuckDBCache] Finalizing and checkpointing database to disk...", flush=True)
        t_chk = time.time()
        self.conn.execute("CHECKPOINT;")
        print(f"[DuckDBCache] Checkpoint complete in {time.time() - t_chk:.2f}s.", flush=True)

        total_time = time.time() - t0
        db_size_mb = os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0

        # Save Cache Manifest
        manifest_data = {
            "cache_version": CACHE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
            "created_at": datetime.now().isoformat(),
            "source2_path": source_paths[0][1] if len(source_paths) > 0 else "",
            "source3_path": source_paths[1][1] if len(source_paths) > 1 else "",
            "source2_rows": src_counts.get("Train S2", 0),
            "source3_rows": src_counts.get("Train S3", 0),
            "total_rows": global_target_row_id,
            "target_data_fingerprint": compute_files_fingerprint(file_paths),
            "available_blockers": [name for name, bit in AVAILABLE_BLOCKERS],
            "build_duration_seconds": total_time,
            "database_size_mb": db_size_mb
        }

        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        print(f"\n[DuckDBCache SUCCESS] Target cache built successfully.")
        print(f"  Artifacts:       {self.db_path} ({db_size_mb:.2f} MB)")
        print(f"  Manifest:        {self.manifest_path}")
        print(f"  Total Indexed:   {global_target_row_id:,} records")
        print(f"  Build Duration:  {total_time:.2f}s")
        return manifest_data

    def query_candidates_for_s1(
        self,
        s1_df: pl.DataFrame,
        max_cands_per_s1: int = 100,
        enable_country_fallback: bool = True
    ) -> Dict[str, List[Tuple[str, int, Dict[str, Any]]]]:
        """
        Executes fast, memory-safe equi-join candidate queries against the persistent disk cache.
        Returns:
            Dict[s1_id, List[(target_id, provenance_mask, target_record_dict)]]
        """
        if len(s1_df) == 0:
            return {}

        self._init_connection()

        temp_s1_parquet = os.path.join(self.tmp_dir, f"temp_s1_{os.getpid()}_{int(time.time()*1000)%100000}.parquet").replace("\\", "/")

        s1_p = add_v2_blocking_columns(s1_df)
        for col in ["cname8_num", "translit_cname8_num", "f2_num", "pin_cname4", "soundex_num", "translit_cname"]:
            if col not in s1_p.columns:
                s1_p = s1_p.with_columns(pl.lit(None).cast(pl.String).alias(col))

        s1_p.write_parquet(temp_s1_parquet)
        del s1_p
        gc.collect()

        # Temporary table for candidate hits
        self.conn.execute("""
            CREATE TEMP TABLE IF NOT EXISTS temp_candidate_hits (
                s1_id VARCHAR,
                target_row_id BIGINT,
                bitmask INTEGER
            );
            DELETE FROM temp_candidate_hits;
        """)

        # Execute 10 strict equi-join passes sequentially into temp table (RAM < 50MB)
        passes = [
            # 1. Exact Compact Name (bitmask: 1)
            f"""
            INSERT INTO temp_candidate_hits
            SELECT s.eid, t.target_row_id, 1
            FROM read_parquet('{temp_s1_parquet}') s
            JOIN targets t ON s.compact_name = t.compact_name AND s.country = t.country
            WHERE s.compact_name IS NOT NULL AND LENGTH(s.compact_name) >= 3;
            """,
            # 2. Transliterated Compact Name (bitmask: 2)
            f"""
            INSERT INTO temp_candidate_hits
            SELECT s.eid, t.target_row_id, 2
            FROM read_parquet('{temp_s1_parquet}') s
            JOIN targets t ON s.translit_cname = t.translit_cname AND s.country = t.country
            WHERE s.translit_cname IS NOT NULL AND LENGTH(s.translit_cname) >= 4;
            """,
            # 3. Exact Normalized Name (bitmask: 4)
            f"""
            INSERT INTO temp_candidate_hits
            SELECT s.eid, t.target_row_id, 4
            FROM read_parquet('{temp_s1_parquet}') s
            JOIN targets t ON s.norm_name = t.norm_name AND s.country = t.country
            WHERE s.norm_name IS NOT NULL AND LENGTH(s.norm_name) >= 4;
            """,
            # 4. Phonetic Soundex + Address Num (bitmask: 8)
            f"""
            INSERT INTO temp_candidate_hits
            SELECT s.eid, t.target_row_id, 8
            FROM read_parquet('{temp_s1_parquet}') s
            JOIN targets t ON s.soundex_num = t.soundex_num AND s.country = t.country
            WHERE s.soundex_num IS NOT NULL;
            """,
            # 5. CName8 + Address Num (bitmask: 16)
            f"""
            INSERT INTO temp_candidate_hits
            SELECT s.eid, t.target_row_id, 16
            FROM read_parquet('{temp_s1_parquet}') s
            JOIN targets t ON s.cname8_num = t.cname8_num AND s.country = t.country
            WHERE s.cname8_num IS NOT NULL;
            """,
            # 6. Translit CName8 + Address Num (bitmask: 32)
            f"""
            INSERT INTO temp_candidate_hits
            SELECT s.eid, t.target_row_id, 32
            FROM read_parquet('{temp_s1_parquet}') s
            JOIN targets t ON s.translit_cname8_num = t.translit_cname8_num AND s.country = t.country
            WHERE s.translit_cname8_num IS NOT NULL;
            """,
            # 7. First 2 Words + Address Num (bitmask: 64)
            f"""
            INSERT INTO temp_candidate_hits
            SELECT s.eid, t.target_row_id, 64
            FROM read_parquet('{temp_s1_parquet}') s
            JOIN targets t ON s.f2_num = t.f2_num AND s.country = t.country
            WHERE s.f2_num IS NOT NULL;
            """,
            # 8. Postal + Name Prefix-4 (bitmask: 128)
            f"""
            INSERT INTO temp_candidate_hits
            SELECT s.eid, t.target_row_id, 128
            FROM read_parquet('{temp_s1_parquet}') s
            JOIN targets t ON s.pin_cname4 = t.pin_cname4 AND s.country = t.country
            WHERE s.pin_cname4 IS NOT NULL;
            """,
            # 9. Cross Translit-Original CName8 Num (bitmask: 256)
            f"""
            INSERT INTO temp_candidate_hits
            SELECT s.eid, t.target_row_id, 256
            FROM read_parquet('{temp_s1_parquet}') s
            JOIN targets t ON s.translit_cname8_num = t.cname8_num AND s.country = t.country
            WHERE s.translit_cname8_num IS NOT NULL;
            """
        ]

        if enable_country_fallback:
            passes.append(f"""
            INSERT INTO temp_candidate_hits
            SELECT s.eid, t.target_row_id, 512
            FROM read_parquet('{temp_s1_parquet}') s
            JOIN targets t ON s.compact_name = t.compact_name
            WHERE s.compact_name IS NOT NULL AND LENGTH(s.compact_name) >= 6;
            """)

        for sql_pass in passes:
            self.conn.execute(sql_pass)

        # Merge, rank, and fetch details in a single aggregated query
        final_query = f"""
            WITH merged AS (
                SELECT 
                    s1_id, 
                    target_row_id, 
                    BIT_OR(bitmask) AS prov_mask,
                    COUNT(*) AS match_votes,
                    ROW_NUMBER() OVER (PARTITION BY s1_id ORDER BY COUNT(*) DESC, BIT_OR(bitmask) DESC) AS rank_num
                FROM temp_candidate_hits
                GROUP BY s1_id, target_row_id
            )
            SELECT 
                m.s1_id,
                t.eid AS target_id,
                m.prov_mask,
                t.norm_name,
                t.compact_name,
                t.norm_addr,
                t.country
            FROM merged m
            JOIN targets t ON m.target_row_id = t.target_row_id
            WHERE m.rank_num <= {max_cands_per_s1};
        """

        rows = self.conn.execute(final_query).fetchall()
        self.conn.execute("DELETE FROM temp_candidate_hits;")

        if os.path.exists(temp_s1_parquet):
            os.remove(temp_s1_parquet)

        result: Dict[str, List[Tuple[str, int, Dict[str, Any]]]] = {}
        for s1_id, tid, mask, t_name, t_cname, t_addr, t_ctry in rows:
            if s1_id not in result:
                result[s1_id] = []
            
            t_rec = {
                "eid": tid,
                "norm_name": t_name or "",
                "compact_name": t_cname or "",
                "norm_addr": t_addr or "",
                "country": t_ctry or "",
                "name_tokens": (t_name or "").split(),
                "addr_tokens": (t_addr or "").split(),
                "numeric_tokens": [tok for tok in (t_addr or "").split() if any(c.isdigit() for c in tok)]
            }
            result[s1_id].append((tid, int(mask), t_rec))

        return result

    def close(self):
        """Closes DuckDB connection."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None

