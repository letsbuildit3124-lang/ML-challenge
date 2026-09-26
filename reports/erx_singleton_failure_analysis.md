# ER-X Singleton False-Positive Failure Analysis (100-S1 Smoke Set)

## 1. Executive Summary & Failure Rates

* **Total S1 Entities Evaluated**: 100
* **True Singletons in Set**: 8 (8.00%)
* **Correctly Abstaining Singletons**: 6 (75.00% Singleton Accuracy)
* **False Positive Singletons**: 2 (25.00% False Positive Rate)

---

## 2. Detailed False-Positive Case Investigation

### Case 1: Common Generic Token / Specialty Collision
* **S1 ID**: `S1-746273151` (True Ground Truth: `[]`)
* **S1 Business Name**: `Pediatric Dental Physicians Inc`
* **S1 Business Address**: `53 Laroche Lane, Hebron, ME`
* **S1 Country**: `US`
* **False Matched Target**: `S3-161412777`
* **Target Business Name**: `pediatric dental care associates inc`
* **Target Business Address**: `492 Fairborn Ln, Round Lake, Illinois`
* **Target Country**: `US`
* **Retrieval Channels**: Channel B (Char TF-IDF) + Channel C (Rare Token: `pediatric`, `dental`)
* **Similarity Breakdown**:
  * Name Levenshtein / Jaro-Winkler: Moderate (~0.58)
  * Address Levenshtein: Near Zero (0.12)
  * House Number: Conflicting (`53` vs `492`)
  * City / State: Conflicting (`Hebron, ME` vs `Round Lake, IL`)
* **Classification**: **Common Specialty / Generic Name Collision**
* **Root Cause**: The distractor target had only 1 retrieved S1 candidate. Because `candidate_rank = 0` and `second_best_score = 0`, the margin was artificially high (`0.9237`). The model lacked a strict cross-field veto feature or agreement floor.

---

### Case 2: Address House-Number Coincidence Collision
* **S1 ID**: `S1-20267366` (True Ground Truth: `[]`)
* **S1 Business Name**: `Barbara Quinn Desert Better LLC`
* **S1 Business Address**: `57 Powers Street, Dedham, MA`
* **S1 Country**: `US`
* **False Matched Target**: `S2-298761341`
* **Target Business Name**: `Lewis, Fog1eman & Lambert LLP`
* **Target Business Address**: `57 THOMAS STREET, PORTLAND, ME`
* **Target Country**: `US`
* **Retrieval Channels**: Channel D (Address House Number `57`)
* **Similarity Breakdown**:
  * Name Levenshtein / Jaro-Winkler: Zero (~0.15)
  * Name Token Jaccard: 0.00
  * Address House Number: Equal (`57`)
  * Street / Locality: Conflicting (`Powers Street, Dedham, MA` vs `THOMAS STREET, PORTLAND, ME`)
* **Classification**: **Numeric House Number Coincidence Collision**
* **Root Cause**: The address channel retrieved an S1 sharing only the house number `57`. Because name similarity was near-zero, this should have been vetoed immediately by a compound name-address agreement rule or minimum name floor.

---

## 3. Surgical Defense Recommendations

1. **Compound Name-Address Agreement Floor**: Require that any accepted match must have either:
   - High Name Similarity ($\ge 0.70$) AND Moderate/Non-conflicting Address ($\ge 0.35$ or address missing)
   - OR Perfect Exact Name ($\text{norm\_name} == \text{norm\_name}$)
2. **Veto on Disjoint Names**: If Name Token Jaccard is 0.0 AND Name Levenshtein $< 0.40$, immediately reject candidate regardless of house number matching.
3. **Candidate-Context Feature Hardening**: Avoid letting `candidate_rank = 0` dominate when intrinsic pairwise similarity is low by including explicit interaction features (`name_lev * (1.0 - candidate_rank)` and intrinsic similarity gates).
