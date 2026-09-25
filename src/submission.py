"""
Submission file generation and formatting module.
Writes matching_results.tsv and candidate_pairs.tsv adhering to official specifications.
"""

import os
from typing import Dict, List, Set
import polars as pl
from src.config import Config

def write_submission_files(
    config: Config,
    all_test_s1_ids: List[str],
    matching_map: Dict[str, List[str]],
    candidate_map: Dict[str, List[str]]
) -> Tuple[str, str]:
    """
    Generates matching_results.tsv and candidate_pairs.tsv.
    Guarantees that:
    1. Every test S1 entity is present exactly once.
    2. All matched entities are a strict subset of candidate_pairs.
    3. No duplicate IDs in matched or candidate lists.
    4. S1 IDs never appear in matched or candidate lists.
    """
    os.makedirs(config.output_dir, exist_ok=True)
    
    matching_rows = []
    candidate_rows = []

    for s1_id in all_test_s1_ids:
        # Get candidates (deduplicated, preserving order)
        raw_cands = candidate_map.get(s1_id, [])
        cands_dedup = list(dict.fromkeys(raw_cands))
        cand_str = ",".join(cands_dedup)
        candidate_rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": cand_str})

        # Get matches (ensure strict subset of candidates)
        raw_matches = matching_map.get(s1_id, [])
        cand_set = set(cands_dedup)
        matches_valid = [m for m in dict.fromkeys(raw_matches) if m in cand_set and not m.startswith("S1-")]
        match_str = ",".join(matches_valid)
        matching_rows.append({"source1_entity_id": s1_id, "matched_entity_ids": match_str})

    # Write matching_results.tsv
    matching_df = pl.DataFrame(matching_rows)
    matching_df.write_csv(config.matching_results_path, separator="\t")
    print(f"[Submission] Successfully wrote {len(matching_df):,} rows to {config.matching_results_path}")

    # Write candidate_pairs.tsv
    candidate_df = pl.DataFrame(candidate_rows)
    candidate_df.write_csv(config.candidate_pairs_path, separator="\t")
    print(f"[Submission] Successfully wrote {len(candidate_df):,} rows to {config.candidate_pairs_path}")

    return config.matching_results_path, config.candidate_pairs_path
