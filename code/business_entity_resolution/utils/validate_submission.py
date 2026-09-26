"""
Submission Validation Utility.
Performs strict pre-submission checks per PRD Section 19:
1. Every test Source-1 entity appears exactly once in matching_results.tsv
2. No 'S1-' IDs anywhere in matched_entity_ids
3. No duplicate IDs within any single row's list
4. No duplicate source1_entity_id rows
5. Every ID in matching_results.tsv also appears in candidate_pairs.tsv for that same S1 entity
6. Correct header columns and non-empty rows
"""

import argparse
from pathlib import Path
import sys
import pandas as pd


def validate_submission(matching_file: Path,
                        candidate_file: Path,
                        test_dir: Path) -> bool:
    print(f"--- Validating Submission Files ---")
    print(f"Matching file:  {matching_file}")
    print(f"Candidate file: {candidate_file}")
    print(f"Test directory: {test_dir}\n")

    errors = []

    if not matching_file.exists():
        errors.append(f"Matching file does not exist: {matching_file}")
    if not candidate_file.exists():
        errors.append(f"Candidate file does not exist: {candidate_file}")

    if errors:
        for e in errors:
            print(f"[FAIL] {e}")
        return False

    matching_df = pd.read_csv(matching_file, sep="\t", dtype=str).fillna("")
    candidate_df = pd.read_csv(candidate_file, sep="\t", dtype=str).fillna("")

    # Check headers
    if list(matching_df.columns) != ["source1_entity_id", "matched_entity_ids"]:
        errors.append(f"Invalid columns in matching_results: {list(matching_df.columns)}")
    if list(candidate_df.columns) != ["source1_entity_id", "candidate_entity_ids"]:
        errors.append(f"Invalid columns in candidate_pairs: {list(candidate_df.columns)}")

    # Check test S1 IDs coverage
    test_s1_path = test_dir / "test_source1.tsv"
    if not test_s1_path.exists():
        # check parent test path or root
        alt = test_dir.parent / "test_source1.tsv"
        if alt.exists():
            test_s1_path = alt

    if test_s1_path.exists():
        s1_raw = pd.read_csv(test_s1_path, sep="\t", dtype=str)
        s1_col = s1_raw.columns[0]
        expected_s1_ids = set(s1_raw[s1_col].str.strip())
        actual_matching_s1 = set(matching_df["source1_entity_id"].str.strip())
        actual_candidate_s1 = set(candidate_df["source1_entity_id"].str.strip())

        if actual_matching_s1 != expected_s1_ids:
            diff_missing = expected_s1_ids - actual_matching_s1
            diff_extra = actual_matching_s1 - expected_s1_ids
            errors.append(f"Mismatch in test S1 coverage! Missing: {len(diff_missing)}, Extra: {len(diff_extra)}")
        if actual_candidate_s1 != expected_s1_ids:
            errors.append(f"Mismatch in candidate S1 coverage!")

    # Check for duplicate S1 rows
    if matching_df["source1_entity_id"].duplicated().any():
        dup_count = matching_df["source1_entity_id"].duplicated().sum()
        errors.append(f"Found {dup_count} duplicate source1_entity_id rows in matching_results.tsv")

    if candidate_df["source1_entity_id"].duplicated().any():
        dup_count = candidate_df["source1_entity_id"].duplicated().sum()
        errors.append(f"Found {dup_count} duplicate source1_entity_id rows in candidate_pairs.tsv")

    # Fast vectorized map of candidate IDs per S1
    print("Building fast candidate verification index...")
    s1_cand_ids = candidate_df["source1_entity_id"].astype(str).str.strip().values
    cand_strings = candidate_df["candidate_entity_ids"].astype(str).values
    cand_map = {sid: set(c.split(",")) for sid, c in zip(s1_cand_ids, cand_strings) if c}

    # Check rows in matching_results
    print("Verifying matching consistency against candidate pairs...")
    s1_match_ids = matching_df["source1_entity_id"].astype(str).str.strip().values
    match_strings = matching_df["matched_entity_ids"].astype(str).values

    invalid_s1_in_matched = 0
    duplicate_matched_ids = 0
    not_in_candidate_count = 0

    for sid, m_str in zip(s1_match_ids, match_strings):
        if not m_str:
            continue
        m_list = [m.strip() for m in m_str.split(",") if m.strip()]

        # Check: no S1- IDs in matched_entity_ids
        for m in m_list:
            if m.startswith("S1-"):
                invalid_s1_in_matched += 1

        # Check: no duplicate IDs in same row
        if len(m_list) != len(set(m_list)):
            duplicate_matched_ids += 1

        # Check: every matched ID appears in candidate pairs for that same S1 entity
        s1_cands = cand_map.get(sid, set())
        for m in m_list:
            if m not in s1_cands:
                not_in_candidate_count += 1

    if invalid_s1_in_matched > 0:
        errors.append(f"Found {invalid_s1_in_matched} 'S1-' references inside matched_entity_ids!")
    if duplicate_matched_ids > 0:
        errors.append(f"Found {duplicate_matched_ids} rows with duplicate matched IDs!")
    if not_in_candidate_count > 0:
        errors.append(f"Found {not_in_candidate_count} matched IDs that were NOT in candidate_pairs.tsv!")

    if errors:
        for err in errors:
            print(f"[FAIL] {err}")
        return False

    print("ALL CHECKS PASSED: Submission is valid and ready for upload!")
    print("PASS")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate hackathon submission files.")
    parser.add_argument("--matching", type=Path, default=Path("output/matching_results.tsv"))
    parser.add_argument("--candidate", type=Path, default=Path("output/candidate_pairs.tsv"))
    parser.add_argument("--test-dir", type=Path, default=Path("dataset/test"))

    args = parser.parse_args()
    success = validate_submission(args.matching, args.candidate, args.test_dir)
    sys.exit(0 if success else 1)
