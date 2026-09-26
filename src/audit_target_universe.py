"""
Audit Target Universe and Data Schemas.
Reports record counts, missing field distributions, and country coverage.
"""

import polars as pl
from src.config import get_config

def audit_target_universe():
    config = get_config()
    print("=" * 80)
    print("AUDITING TARGET UNIVERSE & SCHEMA INTEGRITY")
    print("=" * 80)

    for name, path in [
        ("Train Source 1", config.train_s1_path),
        ("Train Source 2", config.train_s2_path),
        ("Train Source 3", config.train_s3_path),
        ("Test Source 1", config.test_s1_path),
        ("Test Source 2", config.test_s2_path),
        ("Test Source 3", config.test_s3_path)
    ]:
        print(f"\nScanning: {name} ({path})...")
        lf = pl.scan_csv(path, separator="\t", infer_schema_length=1000, ignore_errors=True)
        schema = lf.collect_schema()
        row_count = lf.select(pl.len()).collect().item()
        
        # Sample null counts
        df_sample = pl.read_csv(path, separator="\t", n_rows=100000, ignore_errors=True)
        print(f"  Total Rows: {row_count:,}")
        print(f"  Columns:    {list(schema.names())}")
        for col in df_sample.columns:
            null_pct = df_sample[col].null_count() / len(df_sample) * 100.0
            print(f"    - {col:<20}: {null_pct:.2f}% null in 100k sample")

if __name__ == "__main__":
    audit_target_universe()
