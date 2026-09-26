# ER-X — Final Repository Audit Report

**Date**: 2026-09-26  
**Status**: COMPLETE & VERIFIED  

---

## 1. Executive Summary & Component Categorization

A comprehensive audit was performed across all directories (`src/`, `src/erx/`, `tests/`, `utils/`, `dataset/`, `cache/`, `output/`, `models/`, `reports/`) to isolate active production code from legacy (V1–V5) artifacts and ensure zero version contamination.

| Category | Description | Paths / Artifacts Included |
| :--- | :--- | :--- |
| **ACTIVE CURRENT COMPONENT** | Production-grade ER-X components used in the final submission pipeline | `src/erx/*.py`, `tests/test_erx_modules.py`, `utils/validate_submission.py`, `dataset/train/`, `dataset/test/` |
| **STALE / UNUSED COMPONENT** | Previous iterations (V1–V5, early prototypes, legacy blockers) not called by ER-X | `src/v4_*.py`, `src/v5_*.py`, `src/blocking*.py`, `src/arctic_*.py`, `tests/test_v4*.py`, `tests/test_v5*.py` |
| **POTENTIALLY DANGEROUS ARTIFACT** | Outdated model weights, stale caches, or uncalibrated outputs | `models/lightgbm_baseline.txt` (V3 model), stale `cache/duckdb_tmp` |
| **SAFE TO IGNORE** | Exploratory reports, diagnostics, scratch investigation scripts | `reports/*.md`, `scratch/*.py`, `dataset/.DS_Store` |

---

## 2. Active Current Components Trace

The current frozen ER-X production pipeline is strictly contained in `src/erx/` and consists of:

1. **Configuration (`src/erx/config.py`)**:
   - Centralized dataclass defining all hyperparameters ($K=35$, Char 3–5 TF-IDF, rare token IDF thresholds, Isotonic calibration, Target Exclusivity, Compound Agreement Floor).
2. **Types & Data Structures (`src/erx/types.py`)**:
   - `InternalIDMapper`: Bi-directional string $\leftrightarrow$ integer ID mapping for sub-millisecond lookups.
   - `MultiViewRecord`: Canonical, compact, transliterated, phonetic, token set, house number, and numeric signatures.
   - `ProvenanceMask`: Bitmask tracking candidate discovery channel.
   - `CandidatePair`: Lightweight candidate pair representation.
3. **Normalization & Transliteration Engine (`src/erx/normalization.py`)**:
   - Comprehensive multi-script transliteration (`INDIC_ASCII_MAP` for Devanagari, Telugu, Tamil with candra/nukta vowels; NFKD accent stripping for French open-set).
   - Boundary-aware legal suffix stripper (`inc`, `llc`, `corp`, `sarl`, `sas`, `sci`, `gmbh`, `pvt ltd`).
   - Standardized address abbreviation and house number parser.
4. **Learned Rules (`src/erx/learned_rules.py`)**:
   - Leak-free training fold rule miner (542 token aliases, OCR typo rules).
5. **Multi-Channel Retrieval Engine (`src/erx/retrieval.py`)**:
   - 6 complementary retrieval channels: Channel A (Exact/Learned), Channel B (Char 3–5 TF-IDF), Channel C (Rare Token IDF), Channel D (Address/House number), Channel E (Phonetic), Channel F (Learned Typo/OCR).
   - Transliteration alignment routing: queries `translit_name` and `translit_tok_set` for non-ASCII target records.
6. **Feature Extraction (`src/erx/features.py`)**:
   - 73 RapidFuzz C++ tiered string and address similarities + candidate context features (margin, rank, count).
7. **Hard Negative Mining (`src/erx/hard_negatives.py`)**:
   - In-batch mining of challenging retrieval distractors for robust LightGBM margin training.
8. **Model & Calibration (`src/erx/model.py`)**:
   - LightGBM binary classifier + `ERXCalibrator` with Isotonic Regression on holdout validation fold.
9. **Pipeline Orchestration & Submission (`src/erx/pipeline.py`)**:
   - Streamed chunk processing for 10M target universe.
   - Target Exclusivity & Compound Agreement Floor.
   - Submission generator producing `matching_results.tsv` and `candidate_pairs.tsv`.
10. **Validation Utility (`utils/validate_submission.py`)**:
    - Official competition integrity and format validator.

---

## 3. Stale & Incompatible Artifacts Action Log

| Artifact Path | Source Version | Issue / Hazard | Action Taken |
| :--- | :--- | :--- | :--- |
| `models/lightgbm_baseline.txt` | V3 Baseline (18 features) | Incompatible with 73-feature ER-X schema | **Isolated / Not loaded**. ER-X trains its own model dynamically from training split. |
| `cache/duckdb_tmp/` | V4 / V5 experiments | Outdated schema | **Bypassed**. Fresh persistent table/cache used. |
| `src/v5_*.py` | V5 Prototype | Uses legacy FTS / token indexes | **Isolated / Not imported**. |
| `src/arctic_*.py` | Dense embedding experiments | Prohibited by CPU-only constraint | **Isolated / Disabled**. |

---

## 4. Verification Check: Transliteration Alignment

- Verified `src/erx/retrieval.py` lines 154–188, 191–205, and 265–268:
  - `target.translit_name` is correctly queried in Channel A (Exact/Compact) and Channel B (Char TF-IDF).
  - `target.translit_tok_set` is correctly unioned with `target.name_tok_set` in Channel C (Rare Token).
- Verified `src/erx/normalization.py` lines 105–170:
  - Unicode NFKD decomposition + full Indic consonant/vowel/candra maps are active and deterministic.

---

## 5. Audit Verdict

**PASSED**: The active codebase `src/erx/` is completely decoupled from all legacy V1–V5 artifacts and ready for full test production execution.
