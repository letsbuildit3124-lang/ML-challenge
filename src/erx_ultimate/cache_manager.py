"""
ER-X Ultimate: High-Performance Cache & Analytical Data Manager
Handles raw challenge TSV datasets and DuckDB relational caching.
"""

from __future__ import annotations
import os
import shutil
import logging
from pathlib import Path
from typing import Optional
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from src.erx_ultimate.config import CONFIG, UltimateConfig
from src.erx_ultimate.normalization import (
    normalize_text,
    normalize_business_name,
    normalize_address,
    normalize_phone,
    normalize_website,
    compute_soundex,
)

logger = logging.getLogger("erx_ultimate.cache_manager")


class CacheManager:
    """Manages persistent DuckDB database, Parquet shards, and index caches."""

    def __init__(self, config: UltimateConfig = CONFIG):
        self.config = config
        self.cache_dir = Path(config.paths.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.normalized_dir = self.cache_dir / "normalized"
        self.normalized_dir.mkdir(parents=True, exist_ok=True)
        self.indices_dir = self.cache_dir / "indices"
        self.indices_dir.mkdir(parents=True, exist_ok=True)
        self.shards_dir = self.cache_dir / "training_shards"
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.cache_dir / "entity_resolution.duckdb"

    def clean_cache(self) -> None:
        """Purge and reset all cache files for clean-start runs."""
        logger.info(f"Purging cache directory: {self.cache_dir}")
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.normalized_dir.mkdir(parents=True, exist_ok=True)
        self.indices_dir.mkdir(parents=True, exist_ok=True)
        self.shards_dir.mkdir(parents=True, exist_ok=True)

    def get_duckdb_connection(self, read_only: bool = False) -> duckdb.DuckDBPyConnection:
        """Create tuned DuckDB connection enforcing memory and thread ceilings."""
        conn = duckdb.connect(database=str(self.db_path), read_only=read_only)
        conn.execute("SET max_memory = '4GB';")
        conn.execute(f"SET threads = {min(4, self.config.system.num_threads)};")
        conn.execute("SET preserve_insertion_order = false;")
        return conn

    def ingest_raw_tsv(self, table_name: str, file_path: str, is_ground_truth: bool = False) -> None:
        """Stream raw TSV directly into DuckDB table without intermediate pandas dataframe."""
        resolved_path = self.config.paths.resolve(file_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Input TSV not found: {resolved_path}")

        logger.info(f"Ingesting raw TSV [{table_name}] from {resolved_path}...")
        conn = self.get_duckdb_connection(read_only=False)
        try:
            if is_ground_truth:
                conn.execute(f"""
                    CREATE OR REPLACE TABLE raw_gt AS 
                    SELECT * FROM read_csv_auto(
                        '{resolved_path.as_posix()}',
                        delim='\\t',
                        header=True,
                        all_varchar=True
                    );

                    CREATE OR REPLACE TABLE {table_name} AS
                    SELECT 
                        CAST(REPLACE(column0, 'S1-', '') AS BIGINT) AS source1_id,
                        CASE 
                            WHEN match_id LIKE 'S2-%' THEN CAST(REPLACE(match_id, 'S2-', '') AS BIGINT)
                            WHEN match_id LIKE 'S3-%' THEN CAST(REPLACE(match_id, 'S3-', '') AS BIGINT)
                            ELSE CAST(match_id AS BIGINT)
                        END AS target_id,
                        CASE 
                            WHEN match_id LIKE 'S2-%' THEN 2
                            WHEN match_id LIKE 'S3-%' THEN 3
                            ELSE 0
                        END AS target_source
                    FROM (
                        SELECT 
                            column0, 
                            unnest(string_split(column1, ',')) AS match_id
                        FROM raw_gt
                    )
                    WHERE match_id != '';

                    DROP TABLE IF EXISTS raw_gt;
                """)
            else:
                conn.execute(f"""
                    CREATE OR REPLACE TABLE {table_name} AS 
                    SELECT * FROM read_csv_auto(
                        '{resolved_path.as_posix()}',
                        delim='\\t',
                        header=True,
                        all_varchar=True
                    );
                """)
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            logger.info(f"Ingested {count:,} records into table [{table_name}].")
        finally:
            conn.close()

    def build_normalized_parquet(self, source_name: str, raw_table_name: str) -> Path:
        """Stream raw table from DuckDB, compute normalized columns in Python batches, write to Parquet."""
        out_parquet = self.normalized_dir / f"{source_name}.parquet"
        if out_parquet.exists():
            logger.info(f"Normalized Parquet already exists: {out_parquet}")
            return out_parquet

        logger.info(f"Building normalized Parquet for {source_name}...")
        conn = self.get_duckdb_connection(read_only=True)
        cursor = conn.cursor()
        
        cursor.execute(f"SELECT * FROM {raw_table_name}")
        schema = cursor.description
        col_names = [d[0].lower() for d in schema]
        
        id_idx = 0
        name_idx = next((i for i, c in enumerate(col_names) if "name" in c), 1)
        addr_idx = next((i for i, c in enumerate(col_names) if "address" in c or "street" in c), -1)
        city_idx = next((i for i, c in enumerate(col_names) if "city" in c), -1)
        state_idx = next((i for i, c in enumerate(col_names) if "state" in c or "province" in c), -1)
        zip_idx = next((i for i, c in enumerate(col_names) if "zip" in c or "postal" in c), -1)
        country_idx = next((i for i, c in enumerate(col_names) if "country" in c), -1)
        phone_idx = next((i for i, c in enumerate(col_names) if "phone" in c), -1)
        web_idx = next((i for i, c in enumerate(col_names) if "website" in c or "url" in c), -1)

        parquet_schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("name_raw", pa.string()),
            pa.field("name_norm", pa.string()),
            pa.field("address_raw", pa.string()),
            pa.field("address_norm", pa.string()),
            pa.field("city_norm", pa.string()),
            pa.field("state_norm", pa.string()),
            pa.field("postal_code_norm", pa.string()),
            pa.field("country_norm", pa.string()),
            pa.field("phone_norm", pa.string()),
            pa.field("website_norm", pa.string()),
        ])

        writer = pq.ParquetWriter(str(out_parquet), parquet_schema, compression="SNAPPY")

        batch_size = 50000
        total_records = 0
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            
            ids, names_raw, names_norm = [], [], []
            addrs_raw, addrs_norm, cities_norm = [], [], []
            states_norm, zips_norm, countries_norm = [], [], []
            phones_norm, webs_norm = [], []

            for r in rows:
                raw_id_str = str(r[id_idx]).replace("S1-", "").replace("S2-", "").replace("S3-", "").strip()
                rec_id = int(raw_id_str) if raw_id_str.isdigit() else total_records
                
                name_val = str(r[name_idx]) if name_idx >= 0 and r[name_idx] is not None else ""
                addr_val = str(r[addr_idx]) if addr_idx >= 0 and r[addr_idx] is not None else ""
                city_val = str(r[city_idx]) if city_idx >= 0 and r[city_idx] is not None else ""
                state_val = str(r[state_idx]) if state_idx >= 0 and r[state_idx] is not None else ""
                zip_val = str(r[zip_idx]) if zip_idx >= 0 and r[zip_idx] is not None else ""
                country_val = str(r[country_idx]) if country_idx >= 0 and r[country_idx] is not None else ""
                phone_val = str(r[phone_idx]) if phone_idx >= 0 and r[phone_idx] is not None else ""
                web_val = str(r[web_idx]) if web_idx >= 0 and r[web_idx] is not None else ""

                ids.append(rec_id)
                names_raw.append(name_val)
                names_norm.append(normalize_business_name(name_val))
                addrs_raw.append(addr_val)
                addrs_norm.append(normalize_address(addr_val))
                cities_norm.append(normalize_text(city_val))
                states_norm.append(normalize_text(state_val))
                zips_norm.append(normalize_text(zip_val))
                countries_norm.append(normalize_text(country_val))
                phones_norm.append(normalize_phone(phone_val))
                webs_norm.append(normalize_website(web_val))

            batch_table = pa.Table.from_arrays(
                [
                    pa.array(ids, type=pa.int64()),
                    pa.array(names_raw, type=pa.string()),
                    pa.array(names_norm, type=pa.string()),
                    pa.array(addrs_raw, type=pa.string()),
                    pa.array(addrs_norm, type=pa.string()),
                    pa.array(cities_norm, type=pa.string()),
                    pa.array(states_norm, type=pa.string()),
                    pa.array(zips_norm, type=pa.string()),
                    pa.array(countries_norm, type=pa.string()),
                    pa.array(phones_norm, type=pa.string()),
                    pa.array(webs_norm, type=pa.string()),
                ],
                schema=parquet_schema
            )
            writer.write_table(batch_table)
            total_records += len(rows)

        writer.close()
        conn.close()
        logger.info(f"Finished writing {total_records:,} normalized records to {out_parquet}.")
        return out_parquet


def prepare_cache(config: UltimateConfig = CONFIG, clean: bool = False) -> None:
    """Run full cache preparation: Ingestion, Normalization, CSR Inverted Indexes."""
    cache_mgr = CacheManager(config)
    if clean:
        cache_mgr.clean_cache()

    logger.info("=== STEP 1: INGESTION & PARQUET PARTITIONING ===")
    cache_mgr.ingest_raw_tsv("train_s1", config.paths.train_s1)
    cache_mgr.ingest_raw_tsv("train_s2", config.paths.train_s2)
    cache_mgr.ingest_raw_tsv("train_s3", config.paths.train_s3)
    cache_mgr.ingest_raw_tsv("train_ground_truth", config.paths.train_ground_truth, is_ground_truth=True)

    cache_mgr.build_normalized_parquet("train_s1", "train_s1")
    cache_mgr.build_normalized_parquet("train_s2", "train_s2")
    cache_mgr.build_normalized_parquet("train_s3", "train_s3")

    logger.info("Cache preparation complete.")


if __name__ == "__main__":
    prepare_cache()
