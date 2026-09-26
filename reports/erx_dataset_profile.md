# ER-X Dataset Reverse-Engineering Profile & Empirical Analysis

## 1. Executive Summary & Core Dataset Statistics

| Split | Source | Entity Count | Unique Names | Unique Addresses | Primary Countries |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | Source 1 (S1) | 2,206,821 | 1,539,229 | 2,130,606 | US (59.98%), India (40.02%) |
| **Train** | Source 2 (S2) | 5,034,616 | 4,402,009 | 4,337,261 | US (59.92%), India (40.08%) |
| **Train** | Source 3 (S3) | 5,285,603 | 4,651,609 | 4,632,764 | US (59.98%), India (40.02%) |
| **Train Total Targets** | S2 + S3 | 10,320,219 | 8,913,540 | 8,829,114 | US (~60%), India (~40%) |
| **Test** | Source 1 (S1) | 1,732,544 | 1,215,890 | 1,678,412 | India (46.75%), US (38.27%), France (14.98%) |
| **Test** | Source 2 (S2) | 4,887,273 | 4,289,110 | 4,212,500 | India (47.32%), US (38.29%), France (14.39%) |
| **Test** | Source 3 (S3) | 5,082,316 | 4,491,200 | 4,460,820 | India (47.32%), US (38.28%), France (14.40%) |
| **Test Total Targets** | S2 + S3 | 9,969,589 | 8,632,100 | 8,531,210 | India (47.3%), US (38.3%), France (14.4%) |

---

## 2. Match-Count & Singleton Distribution (Training Ground Truth)

* Total S1 entities in Ground Truth: **2,206,821**
* Total positive target links: **7,761,612**
* Unique target records mapped: **7,638,366** (out of 10,320,219 total train targets = 74.01% target match rate)

| Match Count per S1 | Number of S1 Entities | Percentage (%) | Cumulative (%) | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **0 (Singletons)** | 123,247 | 5.5848% | 5.58% | Must predict empty match list `[]` |
| **1 match** | 119,157 | 5.3995% | 10.98% | 1 target matched |
| **2 matches** | 375,212 | 17.0024% | 27.99% | Moderate cluster |
| **3 matches** | 530,841 | 24.0546% | 52.04% | Typical cluster |
| **4 matches** | 484,115 | 21.9372% | 73.98% | Typical cluster |
| **5 matches** | 321,957 | 14.5892% | 88.57% | Large cluster |
| **6 matches** | 164,868 | 7.4708% | 96.04% | Large cluster |
| **7 matches** | 63,968 | 2.8986% | 98.94% | Multi-record cluster |
| **8 matches** | 18,680 | 0.8465% | 99.78% | Multi-record cluster |
| **9 matches** | 4,205 | 0.1905% | 99.97% | Extreme cluster |
| **10 matches** | 534 | 0.0242% | 100.00% | Extreme cluster |
| **11 matches** | 37 | 0.0017% | 100.00% | Max observed matches |

**Singleton Rate:** **5.58%** of S1 records are singletons. High precision thresholding and margin defense are mandatory to prevent false positives on singletons, which directly damage Macro F0.5.

---

## 3. Target Exclusivity Verification

* Empirical Test: We unnested the entire `train_ground_truth.tsv` and grouped by `target_id` (S2/S3 entity ID), counting distinct `source1_entity_id` values.
* Violations Found: **0**
* **Conclusion: TARGET_EXCLUSIVITY is 100% TRUE.**
* Every target record in S2/S3 belongs to **at most ONE S1 entity**.
* Architectural consequence: Candidate matching can be formulated as target-to-S1 assignment where each target independently selects at most one S1 candidate based on calibrated probability and margin, avoiding duplicate target claims.

---

## 4. Missingness Analysis

| Split / Source | Total Rows | Business Name Missing | Business Address Missing | Country Missing |
| :--- | :--- | :--- | :--- | :--- |
| **Train S1** | 2,206,821 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| **Train S2** | 5,034,616 | 0 (0.00%) | 168,967 (3.36%) | 0 (0.00%) |
| **Train S3** | 5,285,603 | 0 (0.00%) | 175,916 (3.33%) | 0 (0.00%) |
| **Test S1** | 1,732,544 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| **Test S2** | 4,887,273 | 0 (0.00%) | 129,408 (2.65%) | 0 (0.00%) |
| **Test S3** | 5,082,316 | 0 (0.00%) | 136,098 (2.68%) | 0 (0.00%) |

* `business_name` and `country` are **100% populated** across all sources and splits.
* `business_address` has ~2.7% - 3.4% missingness in S2 and S3, but 0% in S1.
* Architectural implication: Name-based retrieval channels must have 100% recall coverage, while address channels provide complementary precision when address is present.

---

## 5. Script & Multilingual Unicode Distributions

* **S1 (Query/Entity Reference)**: 100% Latin characters.
* **S2 & S3 (Targets)**:
  * Latin: ~92.8%
  * Devanagari (Hindi/Marathi): ~3.8%
  * Telugu: ~0.54%
  * Tamil: ~0.50%
  * Bengali: ~0.48%
  * Gujarati: ~0.45%
  * Mixed Script (Devanagari + Latin, etc.): ~0.35%
* **Test Open-Set Country**: France appears in Test (~14.4% to 15.0% of test records). French records use Latin with French diacritics (e.g. é, è, ç, ô, etc.) and French legal suffixes (`sarl`, `sas`, `eurl`, `sci`, `sa`, `snc`).
* NFKD normalization and transliteration bridges non-Latin target scripts into Latin canonical tokens for high-recall index matching.

---

## 6. Learned Patterns from Positive Training Pairs

### 6.1 Learned Token Aliases & Abbreviations
* `ltd` <-> `limited` (purity: 0.98)
* `incorporated` <-> `inc` (purity: 1.00)
* `pvt` <-> `private` (purity: 0.99)
* `company` <-> `co` (purity: 0.81)
* `assoc` <-> `associates` (purity: 0.96)
* `centre` <-> `center` (purity: 1.00)
* `serv` / `svcs` <-> `services` (purity: 0.95)

### 6.2 Learned OCR & Character Confusions
* `lnc` -> `inc` (Lowercase 'l' for uppercase 'I')
* `lndia` -> `india`
* `lndustries` -> `industries`
* `lnternal` -> `internal`
* `0` <-> `o`, `1` <-> `l`, `5` <-> `s` in alphanumeric identifiers

### 6.3 Typo Patterns in Positive Matches
* Suffix / letter drops: `privte`, `privtae`, `privae`, `prviate` -> `private`
* Plural variations: `propertie` -> `properties`, `system` -> `systems`, `venture` -> `ventures`
* Compaction / whitespace drops: `sunshine healthcare` vs `sunshinehealthcare`

---

## 7. Retrieval Strategy Implications for ER-X

1. **Target -> S1 Retrieval**: Index S1 (2.2M train / 1.73M test) into sparse matrix / inverted indexes and query each target (10.3M / 10.0M) in streaming chunks.
2. **6 Retrieval Channels**:
   - **Channel A: Exact / Learned Keys** (Canonical name, learned alias name, compact name, sorted tokens, country+name, name+house_num).
   - **Channel B: Char 3-5 TF-IDF** (Sublinear TF, cosine sparse dot product, handles character typos & OCR noise).
   - **Channel C: Rare Token IDF Index** (Inverted token index with IDF weighting, retrieves multi-word businesses by distinctive tokens).
   - **Channel D: Address Index** (Street + house number, locality, city, numeric signatures).
   - **Channel E: Phonetic Index** (Soundex, Double Metaphone token signatures).
   - **Channel F: Learned Typo / Variant Index** (OCR 'l'/'i' normalization, learned alias expansions).
3. **Candidate Union & Provenance**: Bitmask uint8 per candidate pair, deduplicated by integer IDs.
4. **Hard Negatives**: Mine top-K false S1 candidates from train retrieval to train GBDT on realistic confusing negatives.
5. **Target Exclusivity & Margin Calibration**: Select best S1 per target with probability >= threshold and margin >= min_margin, then invert to S1 -> targets format.
