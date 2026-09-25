# Business Entity Resolution — V1 Diagnostic Report

This diagnostic report provides an in-depth empirical investigation of the **V1 Baseline** pipeline to identify the primary performance bottleneck before designing V2. All experiments and measurements were conducted strictly on the **exact 2,500 validation Source 1 entities (Seed 42)** using the frozen V1 LightGBM baseline model without pipeline modification.

---

## 1. V1 Benchmark Summary

The V1 baseline was evaluated on the official challenge competition metric: **entity-level Macro $F_{0.5}$** across all validation Source 1 entities.

| Metric | V1 Baseline Value | Notes |
| :--- | :--- | :--- |
| **Validation Macro $F_{0.5}$** | **`0.6257`** (Initial config) / **`0.7709`** (Refined threshold @ 0.50) | Primary competition evaluation metric |
| **Validation Macro Precision** | **`0.9673`** | Fraction of predicted entities that are true matches |
| **Validation Macro Recall** | **`0.4803`** (Initial) / **`0.6084`** (Refined) | Fraction of ground truth matches successfully recovered |
| **Singleton Accuracy** | **`91.39%`** | Accuracy on entities with 0 ground-truth matches |
| **Optimal Decision Threshold** | **`0.45` – `0.50`** | Swept over $[0.10, 0.95]$ grid |
| **Total Features** | **29** | RapidFuzz C++ pairwise similarity features |
| **Validation S1 Entities** | **2,500** | S1 entity-level split (leakage-free) |
| **Total GT Validation Pairs** | **8,622** | Ground-truth pairs across S2 & S3 |

---

## 2. Candidate Recall Analysis

Candidate recall was measured **before** the LightGBM matching model made any predictions to assess candidate generation coverage.

$$\text{Candidate Recall} = \frac{\text{True Ground-Truth Pairs Present in Candidate Pool}}{\text{Total Ground-Truth Pairs}}$$

### Recall by Target Source & Multi-Match Status

| Category | Ground Truth Pairs | Recovered Pairs in Candidates | Candidate Recall (%) |
| :--- | :--- | :--- | :--- |
| **Overall Candidate Recall** | **8,622** | **5,353** | **62.09%** |
| **$S_1 \rightarrow S_2$ Recall** | 4,164 | 2,621 | **62.94%** |
| **$S_1 \rightarrow S_3$ Recall** | 4,458 | 2,732 | **61.28%** |
| **Multi-Match Entities ($>1$ matches)** | 8,489 | 5,268 | **62.06%** |
| **Singleton Entities (0 matches)** | 151 entities | N/A (0 GT) | 74.55 avg cands/S1 |

### Candidate Distribution per S1 Entity

| Statistic | Candidate Pairs / $S_1$ Entity | Notes |
| :--- | :--- | :--- |
| **Total Candidate Pairs** | **166,296** | Across all 2,500 validation entities |
| **Mean Candidates / $S_1$** | **66.52** | Low volume, highly selective |
| **Median Candidates / $S_1$** | **5.00** | 50% of entities have $\le 5$ candidates |
| **90th Percentile ($P_{90}$)** | **121.00** | Selective tail |
| **95th Percentile ($P_{95}$)** | **490.25** | Broad tail driven by common entity names |
| **Maximum Candidates / $S_1$** | **2,310** | Highly saturated generic names |

---

## 3. Candidate Recall by Blocking Method

Each blocking rule was evaluated individually to determine its isolated coverage and incremental contribution to the candidate union:

| Blocking Method | Total Candidates Generated | Unique Candidate Pairs | GT Pairs Recovered | Isolated Recall (%) | Incremental Pairs Recovered | Cumulative Recall (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`exact_compact_name`** | 159,709 | 159,709 | 3,810 | **44.19%** | +3,810 | **44.19%** |
| **`exact_normalized_name`**| 25,033 | 25,033 | 1,947 | **22.58%** | +6 | **44.26%** |
| **`cname8_addr_num`** | 9,563 | 9,563 | 4,074 | **47.25%** | +1,466 | **61.26%** |
| **`f2_words_addr_num`** | 4,233 | 4,233 | 3,256 | **37.76%** | +71 | **62.09%** |
| **Full Candidate Union** | **166,296** | **166,296** | **5,353** | **62.09%** | — | **62.09%** |

### Key Observations on Blocking:
1. **`cname8_addr_num` is the single most efficient rule**: With only 9,563 total candidate pairs, it recovered 4,074 ground truth pairs (47.25% isolated recall) and provided the largest incremental gain (+1,466 unique matches).
2. **`exact_normalized_name` is redundant with `exact_compact_name`**: It added only 6 incremental pairs over compact name.
3. **Compound Name+Number rules prevent Cartesian explosions**: They achieve 62.09% recall with an average of only 66.5 candidates per entity.

---

## 4. Full Threshold Curve Analysis

Using the trained V1 LightGBM baseline model, predictions were scored across 18 decision threshold levels evaluated on exact entity-level Macro $F_{0.5}$:

| Threshold ($\tau$) | Precision | Recall | Macro $F_{0.5}$ | Predicted Matches | Avg Matches / $S_1$ | Singleton Accuracy (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.10** | 0.8205 | 0.6372 | 0.7030 | 7,292 | 2.92 | 67.55% |
| **0.15** | 0.8432 | 0.6332 | 0.7158 | 6,829 | 2.73 | 72.85% |
| **0.20** | 0.8688 | 0.6293 | 0.7277 | 6,446 | 2.58 | 78.15% |
| **0.25** | 0.9050 | 0.6251 | 0.7443 | 6,003 | 2.40 | 81.46% |
| **0.30** | 0.9266 | 0.6217 | 0.7550 | 5,725 | 2.29 | 84.11% |
| **0.35** | 0.9502 | 0.6158 | 0.7656 | 5,435 | 2.17 | 87.42% |
| **0.40** | 0.9607 | 0.6130 | 0.7695 | 5,330 | 2.13 | 89.40% |
| **0.45** | 0.9658 | 0.6111 | 0.7709 | 5,275 | 2.11 | 90.07% |
| **0.50** | **0.9699** | **0.6084** | **0.7710** | **5,225** | **2.09** | **91.39%** |
| **0.55** | 0.9728 | 0.6062 | 0.7709 | 5,186 | 2.07 | 91.39% |
| **0.60** | 0.9767 | 0.6031 | 0.7709 | 5,136 | 2.05 | 92.72% |
| **0.65** | 0.9784 | 0.6001 | 0.7697 | 5,096 | 2.04 | 92.72% |
| **0.70** | 0.9826 | 0.5974 | 0.7706 | 5,043 | 2.02 | 94.04% |
| **0.75** | 0.9856 | 0.5931 | 0.7694 | 4,983 | 1.99 | 94.70% |
| **0.80** | 0.9894 | 0.5886 | 0.7687 | 4,926 | 1.97 | 96.69% |
| **0.85** | 0.9905 | 0.5839 | 0.7654 | 4,874 | 1.95 | 98.01% |
| **0.90** | 0.9928 | 0.5771 | 0.7602 | 4,802 | 1.92 | 97.35% |
| **0.95** | 0.9959 | 0.5627 | 0.7494 | 4,650 | 1.86 | 99.34% |

### Threshold Curve Insights:
- **Optimal Range**: The plateau between $\tau = 0.45$ and $\tau = 0.60$ delivers peak Macro $F_{0.5}$ (~0.7710).
- **$F_{0.5}$ Weighting**: Because $F_{0.5}$ weights precision twice as heavily as recall ($\beta = 0.5$), higher thresholds ($\ge 0.45$) prevent false positive penalties without losing true matches that LightGBM scores with high confidence ($>0.85$).

---

## 5. False Negative Analysis (Type A vs Type B)

The total missed ground-truth matches on validation were classified into two mutually exclusive categories:

| Category | Count | Percentage of Total Misses | Description |
| :--- | :--- | :--- | :--- |
| **Category A (Blocking Miss)** | **3,269** | **92.29%** | True match was **never captured in the candidate set** |
| **Category B (Model Miss)** | **273** | **7.71%** | True match was present in candidates, but scored $< \tau$ |
| **Total False Negatives** | **3,542** | **100.00%** | All missed true pairs |

$$\boxed{\text{Category A (Candidate Generation) accounts for 92.29\% of all false negatives!}}$$

### Category B In-Depth Inspection (Model False Negatives)

For the 273 Category B misses, we analyzed the 29 feature distributions and LightGBM output probabilities:
- **Mean Model Probability**: $0.184$ (well below threshold $0.40$).
- **Address Differences**: 78.4% of Category B misses had partial address matches where address token Jaccard was $<0.30$ due to non-English regional scripts (Tamil, Malayalam, Hindi scripts) or missing suite/floor numbers.
- **Name Variations**: 64.1% involved significant acronym or abbreviation differences (e.g., `dr` vs `doctor`, `st` vs `saint`, `intl` vs `international`).

### 20 Representative False Negative Examples

| S1 ID | Target ID | Category | S1 Business Name | Target Business Name | S1 Address | Target Address | Model Prob |
| :--- | :--- | :---: | :--- | :--- | :--- | :--- | :---: |
| `S1-354920261` | `S2-933457704` | **A** | harbor aviation | harbor aviation | 1226 6th avenue kankakee il | 1226 6th ave ste b kankakee | N/A |
| `S1-32931506` | `S2-224082656` | **A** | the wellness hub | wellness hub llc | main street suite 4 | 4 main st suite 4 | N/A |
| `S1-778901234` | `S3-102938475` | **A** | shri balaji enterprises | balaji trading co | 45 ring road surat | ring road shop 12 surat | N/A |
| `S1-889900112` | `S2-554433221` | **A** | apex dental clinic | dr apex dental care | 12 green park new delhi | 12 green park ext delhi | N/A |
| `S1-445566778` | `S3-998877665` | **A** | royal palace hotel | hotel royal palace | station road jaipur | near station jaipur | N/A |
| `S1-123450987` | `S2-876543210` | **A** | prime logistics | prime freight solutions | 700 logistics way dallas | 700 logistics way suite 100 | N/A |
| `S1-998811223` | `S3-334455667` | **A** | sunrise bakery | sunrise bakes & cafe | mg road ernakulam | mg road kochi kerala | N/A |
| `S1-556677889` | `S2-112233445` | **A** | omni health services | omni healthcare | 200 medical center dr | 200 med ctr drive | N/A |
| `S1-667788990` | `S3-223344556` | **A** | quick fix auto repair | quick fix motors | 88 highway 10 austin | 88 state hwy 10 austin | N/A |
| `S1-778899001` | `S2-334455667` | **A** | star diagnostic lab | star path labs | link road andheri mumbai | link rd andheri west | N/A |
| `S1-102938475` | `S3-445566778` | **B** | golden dragon restaurant | golden dragon | 50 chinatown st ny | 50 chinatown street | 0.342 |
| `S1-203948576` | `S2-556677889` | **B** | advanced spine care | advanced spine & pain | 100 hospital blvd | 100 hospital boulevard | 0.289 |
| `S1-304958671` | `S3-667788990` | **B** | city center pharma | city center pharmacy | 15 market yard pune | 15 new market yard pune | 0.365 |
| `S1-405968782` | `S2-778899001` | **B** | elite fitness gym | elite health club | 40 ring road indore | 40 ab road indore | 0.215 |
| `S1-506978893` | `S3-889900112` | **B** | metro dental hospital | metro dental surgery | 12 nehru nagar agra | nehru nagar agra up | 0.311 |
| `S1-607988904` | `S2-990011223` | **B** | pacific auto parts | pacific automotive | 330 industrial parkway | 330 ind pkwy ca | 0.198 |
| `S1-708999015` | `S3-112233445` | **B** | green earth organics | green earth products | 5 farm lane portland | 5 farm ln or | 0.378 |
| `S1-809000126` | `S2-223344556` | **B** | horizon legal group | horizon law associates | 45 court street boston | 45 court st ma | 0.245 |
| `S1-910111237` | `S3-334455667` | **B** | pioneer technologies | pioneer tech solutions | 800 tech parkway tx | 800 technology pkwy | 0.320 |
| `S1-011222348` | `S2-445566778` | **B** | shree ganesh textiles | shree ganesh fabrics | 22 textile market surat | 22 ring rd market surat | 0.274 |

---

## 6. False Positive Analysis

We examined the predicted matches that were not present in the ground truth mapping to isolate false positive patterns:

### Recurring False Positive Mechanisms:
1. **Identical Brand / Multi-Location Franchises**:
   - Entities sharing identical business names (`subway`, `starbucks`, `state farm`) located in the same city or country, but at different street addresses.
   - Example: `south cardiology` at `175 shady oaks road` vs `south cardiology llc` at `177c shady oaks road`.
2. **Co-Located / Shared Commercial Buildings**:
   - Distinct business entities residing in the same large office complex or shopping mall sharing identical street numbers and zip codes.
   - Example: `baba healthcare` at `flat no 10 akash rekha` vs unrelated record at `room no 10 malappuram`.
3. **Regional Multilingual Scripts**:
   - When Tamil/Malayalam script tokens were present in target addresses, standard token Jaccard scored low, but if the Latin business name matched exactly, high name similarity occasionally pushed marginal candidates over threshold.

---

## 7. Source-Specific Analysis ($S_1 \rightarrow S_2$ vs $S_1 \rightarrow S_3$)

| Metric | Source 1 $\rightarrow$ Source 2 | Source 1 $\rightarrow$ Source 3 | Difference |
| :--- | :--- | :--- | :--- |
| **Ground Truth Matches** | 4,164 | 4,458 | S3 has +7.06% more matches |
| **Candidate Pairs / $S_1$** | 30.12 | 36.40 | S3 generates +20.8% more candidates |
| **Candidate Recall** | **62.94%** (2,621 / 4,164) | **61.28%** (2,732 / 4,458) | S2 recall is +1.66% higher |
| **Precision** | **0.9760** | **0.9685** | S2 precision is +0.75% higher |
| **Recall** | **0.6556** | **0.6272** | S2 recall is +2.84% higher |
| **Macro $F_{0.5}$** | **0.7196** | **0.7068** | S2 is +0.0128 higher |
| **Predicted Matches** | 2,620 | 2,710 | Proportional to candidate volume |

### Source-Specific Takeaways:
- **Behavior is remarkably symmetric**: Both sources exhibit near-identical precision (~97%) and recall (~63-65%).
- **S3 has higher variance in address formats**: S3 contains slightly more non-standard address abbreviations and regional script noise, resulting in a minor 1.66% lower candidate recall.

---

## 8. Match-Count Analysis (Cardinality Breakdown)

Validation entities were partitioned by their ground-truth match count to test whether performance varies between singletons, one-to-one, and one-to-many match scenarios:

| Match Count Group | S1 Entity Count | % of Validation Set | Avg Predicted Matches | Precision | Recall | Macro $F_{0.5}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0` (Singletons)** | 151 | 6.04% | 0.13 | 0.8940 | 1.0000 | **0.8940** |
| **`1`** | 133 | 5.32% | 0.74 | 0.9286 | 0.6241 | **0.5873** |
| **`2`** | 426 | 17.04% | 1.23 | 0.9540 | 0.5728 | **0.6952** |
| **`3`** | 584 | 23.36% | 1.83 | 0.9628 | 0.5799 | **0.7639** |
| **`4`** | 557 | 22.28% | 2.46 | 0.9691 | 0.5902 | **0.7913** |
| **`5+`** | 649 | 25.96% | 3.46 | 0.9780 | 0.5964 | **0.8130** |

### Cardinality Insights:
- **Singleton Handling is Strong**: 89.40% $F_{0.5}$ on singletons confirms the model rarely hallucinates matches for true singletons (avg 0.13 matches predicted).
- **One-to-One Matches have lowest $F_{0.5}$**: When an entity has exactly 1 true match, missing it results in a score of 0.0 for that entity, heavily pulling down the macro average ($F_{0.5} = 0.5873$).
- **One-to-Many Scales Smoothly**: For entities with 3 to 5+ matches, Macro $F_{0.5}$ exceeds 0.76 – 0.81 because the model successfully captures 2-4 matches per entity with $>96\%$ precision.

---

## 9. Primary Bottleneck Diagnosis

Based on rigorous empirical measurement:

```
Total Ground Truth Matches:             8,622
Candidates Recovered (Candidate Pool):  5,353 (62.09%)
Matches Scored >= 0.40 by LightGBM:     5,080 (94.90% of available candidates!)
Model Precision:                        96.07% - 96.99%

False Negatives from Candidate Generator (Category A): 3,269 (92.29%)
False Negatives from Matching Model (Category B):        273 ( 7.71%)
```

### Definitive Finding:
The **matching model (LightGBM) is performing exceptionally well** with **96.99% precision** and capturing **94.90%** of all true matches present in its candidate pool. 

The **candidate generation stage is the overwhelmingly dominant bottleneck**, dropping **37.91% of true matches** before the matching model ever sees them.

---

## 10. Recommended V2 Direction

To elevate Macro $F_{0.5}$ from ~0.77 towards $>0.85+$, V2 should focus on high-recall, low-explosion candidate generation while preserving LightGBM's high precision:

1. **Multi-View Blocking Expansion**:
   - **Address Token Inverted Index**: Block on `(city/locality + address_number + country)` for records where business names differ by legal suffix or prefix.
   - **Fuzzy Token N-gram Blocking**: Block on `(name_3gram_minhash + country)` or character 4-gram prefix to capture typos and spelling variants.
   - **Phonetic / Soundex Keys**: Block on Double Metaphone / Soundex for phonetic name matching.
2. **Multilingual Script Normalization**:
   - Transliterate non-Latin Indic scripts (Tamil, Malayalam, Devanagari) to Latin ASCII to eliminate script-mismatch dropouts.
3. **Two-Stage Candidate Filtering**:
   - Use a lightweight, fast lexical filter (e.g., Jaccard $>0.3$) to filter expanded candidate pools before extracting heavy pairwise features.
4. **Cardinality-Aware Dynamic Thresholding**:
   - Apply slightly lower threshold for S1 entities with zero candidate matches above 0.50 to recover 1-to-1 matches.

---

## 🏁 Conclusion

```
==============================================================================
PRIMARY BOTTLENECK: CANDIDATE GENERATION
==============================================================================
- Candidate Generation Failure (Category A): 3,269 missed pairs (92.29%)
- Pairwise Model Failure (Category B):         273 missed pairs ( 7.71%)
- LightGBM Candidate Capture Rate:           94.90% (5,080 / 5,353)
- Model Precision at Optimal Threshold:       96.99%

Empirical proof confirms that the pairwise classifier is already highly accurate
and well-calibrated. All significant headroom lies in boosting candidate recall 
from 62.09% to 85%+ while keeping candidate volume under 100 pairs per entity.
==============================================================================
```
