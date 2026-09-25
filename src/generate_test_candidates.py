"""
V3 Test Candidate Generation Module.
Streams Test Source 1 against indexed Test Target Tables (S2 + S3)
and outputs output/candidate_pairs.tsv with bounded candidate volume.
"""

import os
import sys
sys.path.insert(0, ".")
import gc
import time
from typing import Dict, List, Set, Tuple
import polars as pl

from src.config import get_config
from src.data_loader import iter_source_file_chunks
from src.blocking_v2 import (
    add_v2_blocking_columns,
    build_compact_target_index,
    generate_candidates_against_indexed_target
)

def generate_test_candidates(s1_chunk_size: int = 50000, max_cands_per_s1: int = 50):
    print("=" * 80)
    print("PHASE: V3 TEST CANDIDATE PAIR GENERATION")
    print("=" * 80)

    config = get_config()
    os.makedirs(config.output_dir, exist_ok=True)
    cand_path = config.candidate_pairs_path

    # 1. Index Test Source 2 & Source 3
    print("\n[1/2] Pre-indexing Test Source 2 & Test Source 3 into compact in-memory table...")
    t0 = time.time()
    target_dfs = []
    for s_name, path, prefix in [("Test S2", config.test_s2_path, "S2-"), ("Test S3", config.test_s3_path, "S3-")]:
        for chunk in iter_source_file_chunks(path, chunk_size=500000, expected_prefix=prefix):
            target_dfs.append(add_v2_blocking_columns(chunk))
    
    full_target_p = pl.concat(target_dfs)
    del target_dfs
    gc.collect()

    target_index = build_compact_target_index(full_target_p)
    print(f"Indexed {len(full_target_p):,} Test Targets in {time.time() - t0:.2f}s")
    del full_target_p
    gc.collect()

    # 2. Process Test S1 in streaming chunks and write candidates to disk
    print(f"\n[2/2] Generating candidates for Test Source 1 (Streaming in chunks of {s1_chunk_size:,})...")
    t0 = time.time()
    total_pairs = 0
    total_s1 = 0

    with open(cand_path, "w", encoding="utf-8") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        
        for s1_chunk in iter_source_file_chunks(config.test_s1_path, chunk_size=s1_chunk_size, expected_prefix="S1-"):
            s1_p = add_v2_blocking_columns(s1_chunk)
            cands_dict = generate_candidates_against_indexed_target(s1_p, target_index, max_cands_per_s1=max_cands_per_s1)
            
            lines = []
            for eid, c_list in cands_dict.items():
                total_s1 += 1
                total_pairs += len(c_list)
                c_str = ",".join(c_list) if c_list else ""
                lines.append(f"{eid}\t{c_str}\n")
            f_cand.writelines(lines)

    print(f"\n[Complete] Generated {total_pairs:,} total candidate pairs for {total_s1:,} Test S1 entities in {time.time() - t0:.2f}s")
    print(f"Output written to {cand_path}")

if __name__ == "__main__":
    generate_test_candidates()
