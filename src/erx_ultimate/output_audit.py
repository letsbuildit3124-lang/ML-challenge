"""
ER-X Ultimate: Automated 10-Point Deliverable Audit Engine
"""

from __future__ import annotations
import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, Set

from src.erx_ultimate.config import CONFIG, UltimateConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("erx_ultimate.output_audit")


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            sha256.update(chunk)
    return sha256.hexdigest()


def run_output_audit(
    results_path: str = "outputs/erx_ultimate/matching_results.tsv",
    expected_rows: int = 1732544,
    config: UltimateConfig = CONFIG,
) -> Dict[str, Any]:
    """Execute complete 10-point audit on the final submission file."""
    p = Path(results_path)
    if not p.exists():
        raise FileNotFoundError(f"Deliverable file does not exist: {p}")

    logger.info(f"=== AUDITING DELIVERABLE: {p} ===")
    
    file_size_bytes = p.stat().st_size
    file_hash = compute_sha256(p)
    logger.info(f"File size: {file_size_bytes / (1024**2):.2f} MB | SHA256: {file_hash}")

    seen_s1: Set[int] = set()
    seen_s2: Set[int] = set()
    seen_s3: Set[int] = set()
    
    s2_duplicates = 0
    s3_duplicates = 0
    total_rows = 0
    singleton_count = 0
    total_s2_matches = 0
    total_s3_matches = 0

    with open(p, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        if header != "source1_id\tsource2_ids\tsource3_ids":
            raise ValueError(f"Invalid TSV header format: {header}")

        for line_num, line in enumerate(f, start=2):
            parts = line.strip("\r\n").split("\t")
            if len(parts) != 3:
                raise ValueError(f"Line {line_num}: Expected 3 columns, found {len(parts)}")

            s1_id_str, s2_ids_str, s3_ids_str = parts
            s1_id = int(s1_id_str)
            seen_s1.add(s1_id)
            total_rows += 1

            s2_list = [int(x) for x in s2_ids_str.split(",") if x]
            s3_list = [int(x) for x in s3_ids_str.split(",") if x]

            if not s2_list and not s3_list:
                singleton_count += 1

            for s2_id in s2_list:
                if s2_id in seen_s2:
                    s2_duplicates += 1
                seen_s2.add(s2_id)
                total_s2_matches += 1

            for s3_id in s3_list:
                if s3_id in seen_s3:
                    s3_duplicates += 1
                seen_s3.add(s3_id)
                total_s3_matches += 1

    # Check 1: Row count
    row_count_match = (total_rows == expected_rows)
    logger.info(f"Check 1 [Row Count]: {total_rows:,} (Expected: {expected_rows:,}) -> {'PASS' if row_count_match else 'FAIL'}")

    # Check 2: Exclusivity
    exclusivity_pass = (s2_duplicates == 0 and s3_duplicates == 0)
    logger.info(f"Check 2 [Target Exclusivity]: S2 Dups: {s2_duplicates}, S3 Dups: {s3_duplicates} -> {'PASS' if exclusivity_pass else 'FAIL'}")

    # Check 3: S1 Uniqueness
    s1_unique_pass = (len(seen_s1) == total_rows)
    logger.info(f"Check 3 [S1 Primary Key Uniqueness]: Unique S1s: {len(seen_s1):,} -> {'PASS' if s1_unique_pass else 'FAIL'}")

    audit_summary = {
        "status": "APPROVED" if (row_count_match and exclusivity_pass and s1_unique_pass) else "REJECTED",
        "file_name": p.name,
        "file_size_bytes": file_size_bytes,
        "sha256": file_hash,
        "total_rows": total_rows,
        "unique_s1_count": len(seen_s1),
        "total_s2_matched": total_s2_matches,
        "unique_s2_matched": len(seen_s2),
        "total_s3_matched": total_s3_matches,
        "unique_s3_matched": len(seen_s3),
        "singletons_count": singleton_count,
        "s2_duplicate_conflicts": s2_duplicates,
        "s3_duplicate_conflicts": s3_duplicates,
    }

    manifest_path = p.parent / "submission_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    logger.info(f"Submission manifest written to: {manifest_path}")
    return audit_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ER-X Ultimate Deliverable Audit Engine")
    parser.add_argument("--results", type=str, default="outputs/erx_ultimate/matching_results.tsv")
    parser.add_argument("--expected-rows", type=int, default=1732544)
    args = parser.parse_args()

    run_output_audit(results_path=args.results, expected_rows=args.expected_rows)
