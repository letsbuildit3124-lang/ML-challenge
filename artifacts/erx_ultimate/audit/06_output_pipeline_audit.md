# ER-X Ultimate: Output Deliverable Pipeline & Invariants Audit

## 1. Challenge Format Requirements & Invariants

The Amazon ML Challenge 2026 requires two specific output artifacts:
1. `matching_results.tsv` (Final entity clusters keyed by S1 IDs)
2. `candidate_pairs.tsv` (Top candidate retrieval pairs for downstream evaluation)

### Mandatory Deliverable Invariants:

| Invariant | Requirement | Verification Guard |
| :--- | :--- | :--- |
| **Row Count** | Exactly 1,732,544 rows in `matching_results.tsv` | `assert len(df) == 1732544` |
| **Primary Key** | Column 1 is `source1_id`, exactly matches `test_source1.tsv` in exact order | Set difference is $\emptyset$, duplicate count is 0 |
| **Cluster Columns** | `source2_ids`, `source3_ids` (comma-separated lists of matched IDs) | Format regex `^[0-9]+(,[0-9]+)*$` or empty string |
| **Target Exclusivity** | Every target S2 ID occurs at most ONCE across all rows. Every target S3 ID occurs at most ONCE across all rows. | Global uniqueness set check on parsed arrays |
| **Singleton Protection** | Unmatched S1 records output empty string for S2 and S3 lists. Unmatched targets are dropped. | No artificial default IDs or dangling commas |
| **Delimiters** | Tab-separated values (`\t`), UTF-8 encoding, LF line endings, header included | TSV header check: `source1_id\tsource2_ids\tsource3_ids` |

---

## 2. Invariant Verification Engine
The `output_audit.py` module runs an automated 10-point audit suite over the generated TSV deliverables before submission:
1. File existence and non-zero byte size check.
2. Header exact matching check.
3. Row count exact equality check (1,732,544).
4. S1 ID set equivalence and index alignment check.
5. S2 ID global uniqueness check (no target S2 assigned to multiple S1s).
6. S3 ID global uniqueness check (no target S3 assigned to multiple S1s).
7. Valid target ID check (all output S2/S3 IDs exist in test input sets).
8. Formatting integrity check (no trailing commas, whitespace, or invalid quoting).
9. Non-empty match rate sanity check (validating realistic cluster density).
10. Checksum and metadata generation (`submission_manifest.json`).
