# ER-X Retrieval Missed Positives Analysis (100-S1 Smoke Set)

**Total True Target Links**: 322
**Retrieved Target Links**: 307 (95.34%)
**Missed Positives**: 15

## 1. Missed Positives Breakdown by Category

| Category | Count | Percentage | Primary Root Cause & Fix |
| :--- | :--- | :--- | :--- |
| **transliteration** | 12 | 80.0% | Feed transliterated name views directly to TF-IDF & token indices |
| **abbreviation_or_acronym** | 2 | 13.3% | Apply learned token alias / acronym expansion |
| **name_typo_or_variation** | 1 | 6.7% | Char 3-5 TF-IDF min similarity tuning & sublinear TF |

## 2. Detailed Log of Missed Positives

| # | Target ID | True S1 ID | S1 Name | Target Name (Translit) | Category |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `S2-327309238` | `S1-564729135` | Dream Construction Limited | డ్రీమ్ కన్‌స్ట్రక్షన్ లిమిటెడ్ (dreem kn strkshn limited) | transliteration |
| 2 | `S2-23560385` | `S1-366329737` | Silver Constructions LLP | सिल्वर कंस्ट्रक्शंस एलएलपी (silvr knstrkshns elelpee) | transliteration |
| 3 | `S2-185860935` | `S1-666876590` | Ss Systems Limited | एसएस सिस्टम्स लिमिटेड (eses sistms limited) | transliteration |
| 4 | `S2-494260820` | `S1-630223771` | All International LLP | ऑल इंटरनेशनल एलएलपी (l intrneshnl elelpee) | transliteration |
| 5 | `S2-156312637` | `S1-134898809` | Corner Hypnosis | Brixlyra (brixlyra) | abbreviation_or_acronym |
| 6 | `S2-975207252` | `S1-666876590` | Ss Systems Limited | एसएस सिस्टम्स लिमिटेड (eses sistms limited) | transliteration |
| 7 | `S2-904010001` | `S1-970972284` | Gujarat Guru Agro Private Limited | गुजरात गुरु एग्रो प्राइवेट लिमिटेड (gujraat guru egro praaivet limited) | transliteration |
| 8 | `S3-355590735` | `S1-800482427` | B+ Biotechnologies LLC | B+ LLC Services (b llc services) | name_typo_or_variation |
| 9 | `S3-815538600` | `S1-630223771` | All International LLP | ऑल इंटरनेशनल एलएलपी (l intrneshnl elelpee) | transliteration |
| 10 | `S3-317748253` | `S1-216181354` | United Consultants Private Limited | यूनाइटेड कंसल्टेंट्स प्राइवेट लिमिटेड (yoonaaited knsltents praaivet limited) | transliteration |
| 11 | `S3-697578371` | `S1-630223771` | All International LLP | ऑल इंटरनेशनल एलएलपी (l intrneshnl elelpee) | transliteration |
| 12 | `S3-685337179` | `S1-216181354` | United Consultants Private Limited | यूनाइटेड कंसल्टेंट्स प्राइवेट लिमिटेड (yoonaaited knsltents praaivet limited) | transliteration |
| 13 | `S3-807299919` | `S1-970972284` | Gujarat Guru Agro Private Limited | गुजरात गुरु एग्रो प्राइवेट लिमिटेड (gujraat guru egro praaivet limited) | transliteration |
| 14 | `S3-426660831` | `S1-597762257` | Green Logistics Private Limited | ग्रीन लॉजिस्टिक्स प्राइवेट लिमिटेड (green l jistiks praaivet limited) | transliteration |
| 15 | `S3-805705209` | `S1-791867209` | WG Harbor Worldwide P.C. | Gildxyloavi (gildxyloavi) | abbreviation_or_acronym |
