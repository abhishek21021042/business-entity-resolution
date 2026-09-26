"""
Stage 1: Exploratory Data Analysis (EDA) on training dataset.
Inspects shape, null ratios, country distributions, singleton rate, and true-match noise patterns.
"""

from pathlib import Path
import pandas as pd
from src.data_loading import load_source_df, load_ground_truth, resolve_file_path, inspect_dataset_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def main():
    print("=" * 60)
    print("STAGE 1: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    s1_path = resolve_file_path("train_source1.tsv", PROJECT_ROOT)
    s2_path = resolve_file_path("train_source2.tsv", PROJECT_ROOT)
    s3_path = resolve_file_path("train_source3.tsv", PROJECT_ROOT)
    gt_path = resolve_file_path("train_ground_truth.tsv", PROJECT_ROOT)

    print(f"Loading Source 1 from: {s1_path}")
    s1 = load_source_df(s1_path)
    print(f"Loading Source 2 from: {s2_path}")
    s2 = load_source_df(s2_path)
    print(f"Loading Source 3 from: {s3_path}")
    s3 = load_source_df(s3_path)
    print(f"Loading Ground Truth from: {gt_path}")
    gt = load_ground_truth(gt_path)

    stats = inspect_dataset_stats(s1, s2, s3, gt)

    print(f"\n--- Dimensions ---")
    print(f"Source 1: {stats['source1_shape']}")
    print(f"Source 2: {stats['source2_shape']}")
    print(f"Source 3: {stats['source3_shape']}")
    print(f"Ground Truth: {gt.shape}")

    print(f"\n--- Country Distribution ---")
    print(f"Source 1: {stats['source1_countries']}")
    print(f"Source 2: {stats['source2_countries']}")
    print(f"Source 3: {stats['source3_countries']}")

    print(f"\n--- Ground Truth Class Distribution ---")
    print(f"Total Source-1 Entities: {stats['ground_truth_total_s1']}")
    print(f"Singletons (0 matches):  {stats['singleton_count']} ({stats['singleton_ratio'] * 100:.2f}%)")
    print(f"Match count distribution:\n{pd.Series(stats['match_count_distribution'])}")

    # Check duplicate assignments in ground truth (Assumption for Stage 7)
    gt_exploded = gt.explode("match_list")
    non_empty = gt_exploded[gt_exploded["match_list"] != ""]
    dup_candidates = non_empty.duplicated("match_list").sum()
    print(f"\nDuplicate Candidate Assignments in Ground Truth: {dup_candidates}")
    print(f"(If ~0, greedy 1-to-many conflict resolution is 100% valid)")

    # Sample 5 true matches to observe noise patterns
    print("\n--- Sample True Positive Pairs ---")
    sample_gt = gt[gt["n_matches"] > 0].head(5)
    s2_dict = {row["entity_id"]: row for _, row in s2.iterrows() if row["entity_id"] in set(sample_gt.explode("match_list")["match_list"])}
    s3_dict = {row["entity_id"]: row for _, row in s3.iterrows() if row["entity_id"] in set(sample_gt.explode("match_list")["match_list"])}

    for _, row in sample_gt.iterrows():
        sid = row["source1_entity_id"]
        s1_row = s1[s1["entity_id"] == sid].iloc[0]
        print(f"\n[S1: {sid}] {s1_row['name']} | {s1_row['address']} ({s1_row['country']})")
        for cid in row["match_list"][:3]:
            cand = s2_dict.get(cid) or s3_dict.get(cid)
            if cand is not None:
                print(f"   -> [{cid}] {cand['name']} | {cand['address']} ({cand['country']})")

    print("\nEDA Completed successfully!")

if __name__ == "__main__":
    main()
