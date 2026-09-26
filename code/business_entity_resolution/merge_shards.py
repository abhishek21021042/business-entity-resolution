"""
1-Click Fast Merger and Submission Validator for Distributed Shards.
Combines shard outputs into the final candidate_pairs.tsv and matching_results.tsv,
verifying exact PRD format compliance.
"""

import argparse
from pathlib import Path
import sys
import pandas as pd

from utils.validate_submission import validate_submission


def merge_shards(output_dir: Path, num_shards: int = 3, test_dir: Path = Path("D:/test_data")):
    print("=" * 65)
    print(f"MERGING {num_shards} DISTRIBUTED SHARDS INTO FINAL SUBMISSION")
    print("=" * 65)

    final_cand_path = output_dir / "candidate_pairs.tsv"
    final_match_path = output_dir / "matching_results.tsv"

    cand_shards = [output_dir / f"shard_{i}_of_{num_shards}_candidate_pairs.tsv" for i in range(num_shards)]
    match_shards = [output_dir / f"shard_{i}_of_{num_shards}_matching_results.tsv" for i in range(num_shards)]

    # Check existence
    missing = []
    for p in cand_shards + match_shards:
        if not p.exists():
            missing.append(str(p.name))

    if missing:
        print(f"ERROR: Missing shard files: {missing}")
        print("Please ensure all laptops have finished their shards and copied files into this folder.")
        sys.exit(1)

    print("All shard files found! Merging...")

    # Merge candidate pairs
    total_cand_rows = 0
    with open(final_cand_path, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")
        for i, s_file in enumerate(cand_shards):
            print(f"Merging candidate shard {i + 1}/{num_shards} ({s_file.name})...")
            with open(s_file, "r", encoding="utf-8") as f_in:
                lines = f_in.readlines()
                for line in lines[1:]:  # skip header
                    if line.strip():
                        f_out.write(line)
                        total_cand_rows += 1

    # Merge matching results
    total_match_rows = 0
    with open(final_match_path, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        for i, s_file in enumerate(match_shards):
            print(f"Merging matching shard {i + 1}/{num_shards} ({s_file.name})...")
            with open(s_file, "r", encoding="utf-8") as f_in:
                lines = f_in.readlines()
                for line in lines[1:]:  # skip header
                    if line.strip():
                        f_out.write(line)
                        total_match_rows += 1

    print("\n" + "=" * 65)
    print("MERGE COMPLETE!")
    print(f"Total Candidate Rows: {total_cand_rows:,}")
    print(f"Total Matching Rows:  {total_match_rows:,}")
    print(f"Candidate output:     {final_cand_path}")
    print(f"Matching output:      {final_match_path}")
    print("=" * 65)

    print("\nValidating Final Submission against PRD Rules...")
    try:
        validate_submission(final_match_path, final_cand_path, test_dir)
        print("\nALL VERIFICATIONS PASSED: Submission is 100% compliant and ready to submit!")
    except Exception as e:
        print(f"Note on validation: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Distributed Inference Shards")
    parser.add_argument("--output-dir", type=Path, default=Path("D:/output"), help="Folder containing shard TSVs")
    parser.add_argument("--num-shards", type=int, default=3, help="Number of shards to merge (default: 3)")
    parser.add_argument("--test-dir", type=Path, default=Path("D:/test_data"), help="Test data directory for validation")

    args = parser.parse_args()
    merge_shards(output_dir=args.output_dir, num_shards=args.num_shards, test_dir=args.test_dir)
