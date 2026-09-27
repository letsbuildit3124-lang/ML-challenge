# ER-X Ultimate: Output Format Invariants & Deliverable Plan

## 1. Primary Output: `matching_results.tsv`
- **Path**: `outputs/erx_ultimate/matching_results.tsv`
- **Format**: TSV (tab-delimited, `\t`), UTF-8 encoded.
- **Header**: `source1_id\tsource2_ids\tsource3_ids`
- **Row Count**: Exactly 1,732,544 rows (equal to `test_source1.tsv`).
- **Ordering**: Strict 1-to-1 row alignment with `test_source1.tsv`.
- **Value Format**: Comma-separated integer IDs (`102,504,892`) or empty string (`""`) for singletons.
- **Exclusivity**: Every S2 ID appears at most once in column 2. Every S3 ID appears at most once in column 3.

## 2. Secondary Output: `candidate_pairs.tsv`
- **Path**: `outputs/erx_ultimate/candidate_pairs.tsv`
- **Format**: TSV (tab-delimited, `\t`), UTF-8 encoded.
- **Header**: `source1_id\ttarget_id\ttarget_source\trank\tscore`
- **Contents**: Top candidate pairs evaluated per S1 cluster.

## 3. Metadata Manifest: `submission_manifest.json`
- Stores SHA256 checksums, row counts, singleton counts, execution timestamps, and peak RSS memory metrics.
