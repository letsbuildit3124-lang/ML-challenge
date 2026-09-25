"""
EDA Script for Business Entity Resolution Challenge.
Analyzes train and test datasets, ground truth statistics, and produces reports/eda_report.md.
"""

import os
import sys
import json
import polars as pl
import numpy as np

# Ensure UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

def run_eda(data_dir: str = "dataset", reports_dir: str = "reports"):
    os.makedirs(reports_dir, exist_ok=True)
    report_lines = ["# Exploratory Data Analysis (EDA) Report\n"]
    report_lines.append("## 1. Dataset Overview & Dimensions\n")

    files = {
        "Train Source 1": os.path.join(data_dir, "train", "train_source1.tsv"),
        "Train Source 2": os.path.join(data_dir, "train", "train_source2.tsv"),
        "Train Source 3": os.path.join(data_dir, "train", "train_source3.tsv"),
        "Train Ground Truth": os.path.join(data_dir, "train", "train_ground_truth.tsv"),
        "Test Source 1": os.path.join(data_dir, "test", "test_source1.tsv"),
        "Test Source 2": os.path.join(data_dir, "test", "test_source2.tsv"),
        "Test Source 3": os.path.join(data_dir, "test", "test_source3.tsv"),
    }

    stats = {}
    
    for name, path in files.items():
        print(f"Analyzing {name} ({path})...")
        if not os.path.exists(path):
            print(f"Warning: {path} not found.")
            continue
        
        df = pl.read_csv(path, separator="\t", truncate_ragged_lines=True)
        row_count = len(df)
        cols = df.columns
        stats[name] = {"rows": row_count, "columns": cols}
        
        if "entity_id" in cols:
            unique_ids = df["entity_id"].n_unique()
            null_name = df["business_name"].null_count() if "business_name" in cols else 0
            null_addr = df["business_address"].null_count() if "business_address" in cols else 0
            null_country = df["country"].null_count() if "country" in cols else 0
            
            # Country distribution
            country_dist = df["country"].value_counts().to_dicts() if "country" in cols else []
            
            # Length stats
            name_lens = df["business_name"].str.len_chars().drop_nulls() if "business_name" in cols else pl.Series()
            addr_lens = df["business_address"].str.len_chars().drop_nulls() if "business_address" in cols else pl.Series()
            
            stats[name].update({
                "unique_ids": unique_ids,
                "null_name": null_name,
                "null_addr": null_addr,
                "null_country": null_country,
                "country_dist": country_dist,
                "name_len_mean": float(name_lens.mean()) if len(name_lens) > 0 else 0,
                "name_len_max": int(name_lens.max()) if len(name_lens) > 0 else 0,
                "addr_len_mean": float(addr_lens.mean()) if len(addr_lens) > 0 else 0,
                "addr_len_max": int(addr_lens.max()) if len(addr_lens) > 0 else 0,
            })

    # Markdown table of dimensions
    report_lines.append("| Dataset | Rows | Unique IDs | Null Names | Null Addr | Null Country |\n|---|---|---|---|---|---|")
    for name, s in stats.items():
        if "unique_ids" in s:
            report_lines.append(f"| {name} | {s['rows']:,} | {s['unique_ids']:,} | {s['null_name']:,} | {s['null_addr']:,} | {s['null_country']:,} |")
        else:
            report_lines.append(f"| {name} | {s['rows']:,} | N/A | N/A | N/A | N/A |")
    report_lines.append("\n")

    # Country distributions
    report_lines.append("## 2. Country Distribution\n")
    for name in ["Train Source 1", "Train Source 2", "Train Source 3", "Test Source 1", "Test Source 2", "Test Source 3"]:
        if name in stats and "country_dist" in stats[name]:
            report_lines.append(f"### {name}\n")
            report_lines.append("| Country | Count | Percentage |\n|---|---|---|")
            total = stats[name]["rows"]
            for c in stats[name]["country_dist"]:
                c_val = c.get("country", "NULL")
                c_cnt = c.get("count", 0)
                report_lines.append(f"| {c_val} | {c_cnt:,} | {c_cnt/total*100:.2f}% |")
            report_lines.append("\n")

    # Ground Truth Analysis
    print("Analyzing Ground Truth...")
    gt_path = files["Train Ground Truth"]
    gt_df = pl.read_csv(gt_path, separator="\t", truncate_ragged_lines=True).with_columns(
        pl.col("matched_entity_ids").fill_null("")
    )
    
    total_gt_s1 = len(gt_df)
    
    # Analyze match counts
    match_counts = []
    has_s2_only = 0
    has_s3_only = 0
    has_both = 0
    zero_matches = 0
    one_match = 0
    multi_matches = 0
    
    for row in gt_df.iter_rows():
        m_str = row[1]
        if not m_str or not m_str.strip():
            match_counts.append(0)
            zero_matches += 1
            continue
        
        matches = [x.strip() for x in m_str.split(",") if x.strip()]
        num_m = len(matches)
        match_counts.append(num_m)
        if num_m == 1:
            one_match += 1
        else:
            multi_matches += 1
            
        s2_count = sum(1 for x in matches if x.startswith("S2-"))
        s3_count = sum(1 for x in matches if x.startswith("S3-"))
        
        if s2_count > 0 and s3_count > 0:
            has_both += 1
        elif s2_count > 0:
            has_s2_only += 1
        elif s3_count > 0:
            has_s3_only += 1

    report_lines.append("## 3. Ground Truth Matching Statistics\n")
    report_lines.append(f"- **Total Source 1 Entities**: {total_gt_s1:,}\n")
    report_lines.append(f"- **Entities with 0 Matches (Singletons)**: {zero_matches:,} ({zero_matches/total_gt_s1*100:.2f}%)\n")
    report_lines.append(f"- **Entities with Exactly 1 Match**: {one_match:,} ({one_match/total_gt_s1*100:.2f}%)\n")
    report_lines.append(f"- **Entities with Multiple Matches (>1)**: {multi_matches:,} ({multi_matches/total_gt_s1*100:.2f}%)\n")
    report_lines.append(f"- **Entities Matching S2 Only**: {has_s2_only:,} ({has_s2_only/total_gt_s1*100:.2f}%)\n")
    report_lines.append(f"- **Entities Matching S3 Only**: {has_s3_only:,} ({has_s3_only/total_gt_s1*100:.2f}%)\n")
    report_lines.append(f"- **Entities Matching Both S2 & S3**: {has_both:,} ({has_both/total_gt_s1*100:.2f}%)\n\n")

    # Match distribution table
    mc_series = pl.Series("num_matches", match_counts)
    dist = mc_series.value_counts().sort("num_matches")
    report_lines.append("### Distribution of Match Counts per S1 Entity\n")
    report_lines.append("| Number of Matches | S1 Count | Percentage |\n|---|---|---|")
    for row in dist.iter_rows():
        num_m = row[0]
        cnt = row[1]
        report_lines.append(f"| {num_m} | {cnt:,} | {cnt/total_gt_s1*100:.2f}% |")
    report_lines.append("\n")

    # Qualitative Inspection: Sample Ground Truth Matches
    print("Sampling ground truth pairs for qualitative variation inspection...")
    s1_df = pl.read_csv(files["Train Source 1"], separator="\t", truncate_ragged_lines=True)
    s2_df = pl.read_csv(files["Train Source 2"], separator="\t", truncate_ragged_lines=True)
    s3_df = pl.read_csv(files["Train Source 3"], separator="\t", truncate_ragged_lines=True)
    
    # Create sample dictionary
    s1_lookup = {r[0]: (r[1], r[2], r[3]) for r in s1_df.head(20000).iter_rows()}
    s2_lookup = {r[0]: (r[1], r[2], r[3]) for r in s2_df.head(50000).iter_rows()}
    s3_lookup = {r[0]: (r[1], r[2], r[3]) for r in s3_df.head(50000).iter_rows()}

    report_lines.append("## 4. Qualitative Match Pattern Analysis\n")
    report_lines.append("Examples of true matching entity pairs demonstrating variations:\n\n")
    
    sample_pairs = []
    for row in gt_df.iter_rows():
        s1_id = row[0]
        m_str = row[1]
        if not m_str:
            continue
        if s1_id in s1_lookup:
            matches = [x.strip() for x in m_str.split(",") if x.strip()]
            for m_id in matches:
                if m_id in s2_lookup:
                    sample_pairs.append((s1_id, s1_lookup[s1_id], m_id, s2_lookup[m_id]))
                elif m_id in s3_lookup:
                    sample_pairs.append((s1_id, s1_lookup[s1_id], m_id, s3_lookup[m_id]))
        if len(sample_pairs) >= 10:
            break

    for i, (s1_id, s1_data, target_id, target_data) in enumerate(sample_pairs[:8], 1):
        report_lines.append(f"### Example {i}\n")
        report_lines.append(f"- **S1 ({s1_id})**: Name=`{s1_data[0]}`, Address=`{s1_data[1]}`, Country=`{s1_data[2]}`\n")
        report_lines.append(f"- **Match ({target_id})**: Name=`{target_data[0]}`, Address=`{target_data[1]}`, Country=`{target_data[2]}`\n\n")

    report_lines.append("## 5. Key EDA Findings & Implications for Modeling\n")
    report_lines.append("1. **Singletons are Prevalent**: A significant fraction of Source 1 entities have 0 matches in S2/S3. The pipeline must support empty prediction lists.\n")
    report_lines.append("2. **Multi-Source Matches**: S1 entities frequently have matches in both S2 and S3, or multiple records in the same source. Multi-candidate classification without 1-to-1 restrictions is necessary.\n")
    report_lines.append("3. **Name & Address Variations**: Common variations include legal entity suffixes (LLC, Inc, Pvt Ltd), address abbreviations (St, Ave, Rd, Ste, Fl), punctuation, whitespace, and case differences.\n")
    report_lines.append("4. **Open-Set Countries**: Country distributions include US, India, France (in test), etc. Country exact matching and missing country handling must be robust across all locales.\n")

    report_text = "".join(report_lines)
    out_file = os.path.join(reports_dir, "eda_report.md")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(f"EDA Complete. Report written to {out_file}")
    return report_text

if __name__ == "__main__":
    run_eda()
