# Exploratory Data Analysis (EDA) Report
## 1. Dataset Overview & Dimensions
| Dataset | Rows | Unique IDs | Null Names | Null Addr | Null Country |
|---|---|---|---|---|---|| Train Source 1 | 2,206,821 | 2,206,821 | 0 | 0 | 0 || Train Source 2 | 5,034,616 | 5,034,616 | 0 | 168,967 | 0 || Train Source 3 | 5,285,603 | 5,285,603 | 0 | 175,916 | 0 || Train Ground Truth | 2,206,821 | N/A | N/A | N/A | N/A || Test Source 1 | 1,732,544 | 1,732,544 | 0 | 0 | 0 || Test Source 2 | 4,887,273 | 4,887,273 | 0 | 129,408 | 0 || Test Source 3 | 5,082,316 | 5,082,316 | 0 | 136,098 | 0 |
## 2. Country Distribution
### Train Source 1
| Country | Count | Percentage |
|---|---|---|| India | 883,188 | 40.02% || US | 1,323,633 | 59.98% |
### Train Source 2
| Country | Count | Percentage |
|---|---|---|| India | 2,017,799 | 40.08% || US | 3,016,817 | 59.92% |
### Train Source 3
| Country | Count | Percentage |
|---|---|---|| US | 3,170,056 | 59.98% || India | 2,115,547 | 40.02% |
### Test Source 1
| Country | Count | Percentage |
|---|---|---|| India | 809,986 | 46.75% || US | 663,106 | 38.27% || France | 259,452 | 14.98% |
### Test Source 2
| Country | Count | Percentage |
|---|---|---|| France | 703,378 | 14.39% || India | 2,312,565 | 47.32% || US | 1,871,330 | 38.29% |
### Test Source 3
| Country | Count | Percentage |
|---|---|---|| US | 1,945,701 | 38.28% || India | 2,405,000 | 47.32% || France | 731,615 | 14.40% |
## 3. Ground Truth Matching Statistics
- **Total Source 1 Entities**: 2,206,821
- **Entities with 0 Matches (Singletons)**: 123,247 (5.58%)
- **Entities with Exactly 1 Match**: 119,157 (5.40%)
- **Entities with Multiple Matches (>1)**: 1,964,417 (89.02%)
- **Entities Matching S2 Only**: 143,029 (6.48%)
- **Entities Matching S3 Only**: 164,498 (7.45%)
- **Entities Matching Both S2 & S3**: 1,776,047 (80.48%)

### Distribution of Match Counts per S1 Entity
| Number of Matches | S1 Count | Percentage |
|---|---|---|| 0 | 123,247 | 5.58% || 1 | 119,157 | 5.40% || 2 | 375,212 | 17.00% || 3 | 530,841 | 24.05% || 4 | 484,115 | 21.94% || 5 | 321,957 | 14.59% || 6 | 164,868 | 7.47% || 7 | 63,968 | 2.90% || 8 | 18,680 | 0.85% || 9 | 4,205 | 0.19% || 10 | 534 | 0.02% || 11 | 37 | 0.00% |
## 4. Qualitative Match Pattern Analysis
Examples of true matching entity pairs demonstrating variations:

### Example 1
- **S1 (S1-210849781)**: Name=`Urology Partners Inc`, Address=`6207 Ocean Front Avenue, VA, Virginia Beach City`, Country=`US`
- **Match (S2-845708664)**: Name=`Urology Partners  Inc`, Address=`6207 OCEAN FRONT AVE, VIRGINIA BEACH CITY, VA`, Country=`US`

### Example 2
- **S1 (S1-523975207)**: Name=`VL Sprott`, Address=`AL, Unit 3rd Phone, Dora, 125 Mountain View Lane`, Country=`US`
- **Match (S3-804250913)**: Name=`VL SPROTT`, Address=`##125 Mountain View Lane, # 3rd Phone, Dora, Alabama`, Country=`US`

### Example 3
- **S1 (S1-990310663)**: Name=`Golden One Consultants Private Limited`, Address=`220A, Sector 11, Shri Ram Vatika Park, Shivaji Nagar, Gurgaon, Haryana`, Country=`India`
- **Match (S2-835410474)**: Name=`GOLDEN ONE CÓNSULTANTS PRIVATE LTD`, Address=`220A, SECTOR 11, SHRI RAM VATIKA PARK, SHIVAJI NAGAR, Haryana`, Country=`India`

### Example 4
- **S1 (S1-880324716)**: Name=`Siia Investments Inc`, Address=`8411 Gabrielino Court, Rancho Cucamonga, CA`, Country=`US`
- **Match (S2-992200341)**: Name=`siiainvestments.com`, Address=`8411 GABRIELINO COURT, PMB 9239, RANCHO CUCAMONGA TOWNSHIP, CA`, Country=`US`

### Example 5
- **S1 (S1-56326829)**: Name=`Ramirez, Jackson and Mason Inc.`, Address=`Cleveland, OH, 1391 51st Street`, Country=`US`
- **Match (S3-492762370)**: Name=`Ramirez, Jackson and Mason Incorporated`, Address=`1391 51st Saint, Clleveland, Ohio`, Country=`US`

### Example 6
- **S1 (S1-85659903)**: Name=`Supreme It Private Limited`, Address=`Sco No.204 Second Floor, Raja Commercials, B-Xv-79/A-1, Vishwakarma Chowk, Miller Ganj, G.T. Road, Ludhiana, Punjab`, Country=`India`
- **Match (S3-980886320)**: Name=`ਸੁਪਰੀਮ ਆਈਟੀ ਪ੍ਰਾਈਵੇਟ ਲਿਮਟਿਡ`, Address=`Sco No.b3/204 Second Floor, Ludhiana, PB`, Country=`India`

### Example 7
- **S1 (S1-317085559)**: Name=`Real It Private Limited`, Address=`C/O Manish Kumar, Gram Gejha, Meerut, Uttar Pradesh`, Country=`India`
- **Match (S2-810964968)**: Name=`Real-It Private`, Address=`C/O MANISH KUMAR, GRAM GEJHA, MEERUT, उत्तर प्रदेश`, Country=`India`

### Example 8
- **S1 (S1-187690902)**: Name=`Housing Charities`, Address=`Pineville, WV, 10542 Welch-pineville Road`, Country=`US`
- **Match (S3-188063360)**: Name=`housing charities`, Address=`PMB 7715, Pineville, West Virginia, 10542 Welch-Pineville Road`, Country=`US`

## 5. Key EDA Findings & Implications for Modeling
1. **Singletons are Prevalent**: A significant fraction of Source 1 entities have 0 matches in S2/S3. The pipeline must support empty prediction lists.
2. **Multi-Source Matches**: S1 entities frequently have matches in both S2 and S3, or multiple records in the same source. Multi-candidate classification without 1-to-1 restrictions is necessary.
3. **Name & Address Variations**: Common variations include legal entity suffixes (LLC, Inc, Pvt Ltd), address abbreviations (St, Ave, Rd, Ste, Fl), punctuation, whitespace, and case differences.
4. **Open-Set Countries**: Country distributions include US, India, France (in test), etc. Country exact matching and missing country handling must be robust across all locales.
