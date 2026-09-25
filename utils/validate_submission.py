"""
Submission Validation Script for Entity Resolution Competition Deliverables.
Validates matching_results.tsv and candidate_pairs.tsv against test datasets.

Usage:
    python utils/validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir dataset/test
"""

import os
import sys
import argparse
import polars as pl
from typing import Set, Dict, List


def validate_submission(matching_path: str, candidate_path: str, test_dir: str) -> bool:
    print("=" * 78)
    print("RUNNING SUBMISSION DELIVERABLES VALIDATION")
    print("=" * 78)
    errors = []
    warnings = []

    # 1. File existence
    if not os.path.exists(matching_path):
        errors.append(f"Matching results TSV not found at: {matching_path}")
    if not os.path.exists(candidate_path):
        errors.append(f"Candidate pairs TSV not found at: {candidate_path}")
    if not os.path.exists(test_dir):
        errors.append(f"Test directory not found at: {test_dir}")

    if errors:
        for e in errors:
            print(f"[FAIL] {e}")
        return False

    # 2. Load test source files
    test_s1_path = os.path.join(test_dir, "test_source1.tsv")
    test_s2_path = os.path.join(test_dir, "test_source2.tsv")
    test_s3_path = os.path.join(test_dir, "test_source3.tsv")

    print(f"Loading test source entity IDs from: {test_dir}...")
    s1_test_df = pl.read_csv(test_s1_path, separator="\t", columns=["entity_id"], truncate_ragged_lines=True)
    expected_s1_ids: Set[str] = set(s1_test_df["entity_id"].to_list())
    total_expected_s1 = len(expected_s1_ids)

    valid_target_ids: Set[str] = set()
    if os.path.exists(test_s2_path):
        s2_df = pl.read_csv(test_s2_path, separator="\t", columns=["entity_id"], truncate_ragged_lines=True)
        valid_target_ids.update(s2_df["entity_id"].to_list())
    if os.path.exists(test_s3_path):
        s3_df = pl.read_csv(test_s3_path, separator="\t", columns=["entity_id"], truncate_ragged_lines=True)
        valid_target_ids.update(s3_df["entity_id"].to_list())

    print(f"Expected Test S1 Count: {total_expected_s1:,}")
    print(f"Valid Test Targets (S2 + S3): {len(valid_target_ids):,}")

    # 3. Validate matching_results.tsv
    print("\n--- Validating matching_results.tsv ---")
    matching_df = pl.read_csv(
        matching_path,
        separator="\t",
        null_values=["", "NULL", "null", "None", "NaN"],
        truncate_ragged_lines=True
    ).with_columns(pl.col("matched_entity_ids").fill_null(""))

    match_cols = matching_df.columns
    if "source1_entity_id" not in match_cols or "matched_entity_ids" not in match_cols:
        errors.append(f"matching_results.tsv has invalid columns {match_cols}. Expected ['source1_entity_id', 'matched_entity_ids'].")

    matching_s1_list = matching_df["source1_entity_id"].to_list()
    matching_s1_set = set(matching_s1_list)

    if len(matching_s1_list) != len(matching_s1_set):
        errors.append(f"matching_results.tsv contains {len(matching_s1_list) - len(matching_s1_set):,} DUPLICATE Source-1 rows.")

    missing_s1_matching = expected_s1_ids - matching_s1_set
    if missing_s1_matching:
        errors.append(f"matching_results.tsv is MISSING {len(missing_s1_matching):,} expected test Source-1 entities.")

    extra_s1_matching = matching_s1_set - expected_s1_ids
    if extra_s1_matching:
        errors.append(f"matching_results.tsv contains {len(extra_s1_matching):,} UNKNOWN Source-1 entities not in test_source1.tsv.")

    # 4. Validate candidate_pairs.tsv
    print("\n--- Validating candidate_pairs.tsv ---")
    candidate_df = pl.read_csv(
        candidate_path,
        separator="\t",
        null_values=["", "NULL", "null", "None", "NaN"],
        truncate_ragged_lines=True
    ).with_columns(pl.col("candidate_entity_ids").fill_null(""))

    cand_cols = candidate_df.columns
    if "source1_entity_id" not in cand_cols or "candidate_entity_ids" not in cand_cols:
        errors.append(f"candidate_pairs.tsv has invalid columns {cand_cols}. Expected ['source1_entity_id', 'candidate_entity_ids'].")

    candidate_s1_list = candidate_df["source1_entity_id"].to_list()
    candidate_s1_set = set(candidate_s1_list)

    if len(candidate_s1_list) != len(candidate_s1_set):
        errors.append(f"candidate_pairs.tsv contains {len(candidate_s1_list) - len(candidate_s1_set):,} DUPLICATE Source-1 rows.")

    missing_s1_cand = expected_s1_ids - candidate_s1_set
    if missing_s1_cand:
        errors.append(f"candidate_pairs.tsv is MISSING {len(missing_s1_cand):,} expected test Source-1 entities.")

    # 5. Validate Candidate Inclusion Invariant & IDs
    print("\n--- Validating Pair Integrity & Candidate Subset Guarantee ---")
    match_map: Dict[str, List[str]] = {}
    for row in matching_df.iter_rows():
        s1 = str(row[0])
        m_str = str(row[1]) if row[1] else ""
        m_list = [x.strip() for x in m_str.split(",") if x.strip()]
        match_map[s1] = m_list

    cand_map: Dict[str, List[str]] = {}
    for row in candidate_df.iter_rows():
        s1 = str(row[0])
        c_str = str(row[1]) if row[1] else ""
        c_list = [x.strip() for x in c_str.split(",") if x.strip()]
        cand_map[s1] = c_list

    invalid_target_matches = 0
    invalid_s1_in_matches = 0
    duplicate_matches_count = 0
    candidate_subset_violations = 0
    singletons_count = 0
    total_matches_count = 0
    total_cands_count = 0

    for s1, m_list in match_map.items():
        c_list = cand_map.get(s1, [])
        c_set = set(c_list)
        total_cands_count += len(c_list)

        if not m_list:
            singletons_count += 1
        else:
            total_matches_count += len(m_list)

        if len(m_list) != len(set(m_list)):
            duplicate_matches_count += 1

        for m_id in m_list:
            # Must not be S1-
            if m_id.startswith("S1-"):
                invalid_s1_in_matches += 1
            # Must be valid S2 or S3
            if m_id not in valid_target_ids:
                invalid_target_matches += 1
            # MUST be in candidate list
            if m_id not in c_set:
                candidate_subset_violations += 1

    if invalid_s1_in_matches > 0:
        errors.append(f"Found {invalid_s1_in_matches} matches referencing S1- IDs instead of S2-/S3- targets.")

    if invalid_target_matches > 0:
        errors.append(f"Found {invalid_target_matches} matched IDs that DO NOT exist in test_source2.tsv or test_source3.tsv.")

    if duplicate_matches_count > 0:
        errors.append(f"Found {duplicate_matches_count} entities with duplicate matched IDs in their match list.")

    if candidate_subset_violations > 0:
        errors.append(f"CRITICAL: Found {candidate_subset_violations} final matched entities that ARE NOT in candidate_pairs.tsv!")

    # Summary
    print("-" * 78)
    print("SUBMISSION PROFILE SUMMARY:")
    print(f"Total Test S1 Entities:        {len(matching_df):,} (Expected: {total_expected_s1:,})")
    print(f"Total Matches Assigned:        {total_matches_count:,} (Avg {total_matches_count/max(1, total_expected_s1):.2f} matches/S1)")
    print(f"Total Predicted Singletons:    {singletons_count:,} ({singletons_count/max(1, total_expected_s1)*100:.2f}%)")
    print(f"Total Final Candidates:        {total_cands_count:,} (Avg {total_cands_count/max(1, total_expected_s1):.1f} cands/S1)")
    print("-" * 78)

    if errors:
        print("\n" + "=" * 78)
        print(f"[FAIL] SUBMISSION VALIDATION FAILED WITH {len(errors)} ERROR(S):")
        for e in errors:
            print(f"  ❌ {e}")
        print("=" * 78 + "\n")
        return False
    else:
        print("\n" + "=" * 78)
        print("✅ ALL SUBMISSION VALIDATION CHECKS PASSED PERFECTLY!")
        print("Deliverables are 100% compliant with competition specifications.")
        print("=" * 78 + "\n")
        return True


def main():
    parser = argparse.ArgumentParser(description="Validate official competition deliverables")
    parser.add_argument("--matching", type=str, default="output/matching_results.tsv", help="Path to matching_results.tsv")
    parser.add_argument("--candidate", type=str, default="output/candidate_pairs.tsv", help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", type=str, default="dataset/test", help="Path to dataset/test directory")
    args = parser.parse_args()

    passed = validate_submission(
        matching_path=args.matching,
        candidate_path=args.candidate,
        test_dir=args.test_dir,
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
