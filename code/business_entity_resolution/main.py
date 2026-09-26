"""
Main End-to-End Orchestrator for Business Entity Resolution.
Supports full training with CV, threshold tuning, and inference.
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import PROJECT_ROOT, MODELS_DIR, CANDIDATE_OUTPUT_PATH, MATCHING_OUTPUT_PATH, DEFAULT_THRESHOLD
from src.data_loading import load_source_df, load_ground_truth, resolve_file_path, load_matched_sample
from src.normalization import normalize_record
from src.blocking import (
    exact_key_blocking,
    phonetic_blocking,
    tfidf_topk_blocking,
    first_word_blocking,
    address_key_blocking,
    union_candidates,
    candidate_list_to_dict
)
from src.evaluate import compute_blocking_metrics
from src.features import compute_pair_features, add_rank_and_margin_features, extract_features_parallel
from src.train_model import train_and_evaluate_cv, save_models
from src.infer import run_test_inference
from src.generate_outputs import save_submission_artifacts
from utils.validate_submission import validate_submission


def run_training_pipeline(sample_limit: int = 0):
    print("\n" + "=" * 65)
    print("STEP 1: LOADING TRAINING DATASETS")
    print("=" * 65)

    s1_path = resolve_file_path("train_source1.tsv", PROJECT_ROOT)
    s2_path = resolve_file_path("train_source2.tsv", PROJECT_ROOT)
    s3_path = resolve_file_path("train_source3.tsv", PROJECT_ROOT)
    gt_path = resolve_file_path("train_ground_truth.tsv", PROJECT_ROOT)

    if sample_limit > 0:
        print(f"Loading matched sample of {sample_limit} Source-1 entities with full true matches & distractors...")
        s1_df, s2_df, s3_df, gt_df = load_matched_sample(
            s1_path, s2_path, s3_path, gt_path,
            sample_size=sample_limit,
            n_distractors=min(25000, sample_limit * 5)
        )
    else:
        s1_df = load_source_df(s1_path)
        s2_df = load_source_df(s2_path)
        s3_df = load_source_df(s3_path)
        gt_df = load_ground_truth(gt_path)

    print(f"Source 1: {s1_df.shape} | Source 2: {s2_df.shape} | Source 3: {s3_df.shape} | GT: {gt_df.shape}")

    print("\n" + "=" * 65)
    print("STEP 2: RECORD NORMALIZATION")
    print("=" * 65)
    print("Normalizing Source 1...")
    s1_norm = pd.DataFrame([normalize_record(r) for r in s1_df.to_dict("records")])
    print("Normalizing Source 2...")
    s2_norm = pd.DataFrame([normalize_record(r) for r in s2_df.to_dict("records")])
    print("Normalizing Source 3...")
    s3_norm = pd.DataFrame([normalize_record(r) for r in s3_df.to_dict("records")])

    other_norm = pd.concat([s2_norm, s3_norm], ignore_index=True)

    print("\n" + "=" * 65)
    print("STEP 3: MULTI-PASS BLOCKING (CANDIDATE GENERATION)")
    print("=" * 65)
    print("Pass 1: Exact sorted-core + country hashing...")
    p1 = exact_key_blocking(s1_norm, other_norm)
    print(f"Pass 1 generated: {len(p1)} pairs")

    print("Pass 2: Phonetic NYSIIS hashing...")
    p2 = phonetic_blocking(s1_norm, other_norm)
    print(f"Pass 2 generated: {len(p2)} pairs")

    print("Pass 3: Character n-gram TF-IDF top-K...")
    p3 = tfidf_topk_blocking(s1_norm, other_norm, k=25)
    print(f"Pass 3 generated: {len(p3)} pairs")

    print("Pass 4: Distinctive first-word core name hashing...")
    p4 = first_word_blocking(s1_norm, other_norm)
    print(f"Pass 4 generated: {len(p4)} pairs")

    print("Pass 5: Exact house/flat number + locality hashing...")
    p5 = address_key_blocking(s1_norm, other_norm)
    print(f"Pass 5 generated: {len(p5)} pairs")

    all_pairs = union_candidates(p1, p2, p3, p4, p5)
    print(f"Total Union Candidate Pairs: {len(all_pairs)}")

    cand_dict = candidate_list_to_dict(all_pairs)
    blocking_eval = compute_blocking_metrics(cand_dict, gt_df, len(s2_df), len(s3_df))
    print(f"\n>>> BLOCKING RECALL: {blocking_eval['recall'] * 100:.2f}% | Hits: {blocking_eval['hits']}/{blocking_eval['total_true_pairs']}")
    print(f">>> REDUCTION RATIO: {blocking_eval['reduction_ratio'] * 100:.4f}%")

    print("\n" + "=" * 65)
    print("STEP 4: PAIRWISE FEATURE ENGINEERING")
    print("=" * 65)
    gt_pair_set = set()
    for _, row in gt_df.iterrows():
        for m in row["match_list"]:
            gt_pair_set.add((row["source1_entity_id"], m))

    s1_dict = {r["entity_id"]: r for r in s1_norm.to_dict("records")}
    other_dict = {r["entity_id"]: r for r in other_norm.to_dict("records")}

    print(f"Extracting features for {len(all_pairs)} pairs using all CPU cores (n_jobs=-1)...")
    cand_df = extract_features_parallel(all_pairs, s1_dict, other_dict, gt_pair_set=gt_pair_set, n_jobs=-1)


    print("\n" + "=" * 65)
    print("STEP 5: TRAINING CLASSIFIER & TUNING F0.5 THRESHOLD")
    print("=" * 65)
    models, scored_df, best_thresh, best_f05 = train_and_evaluate_cv(cand_df, gt_df, n_splits=5)
    save_models(models, best_thresh, MODELS_DIR)

    print(f"\nFinal Training Result: Out-Of-Fold Macro F0.5 = {best_f05:.4f} (Optimal Threshold = {best_thresh:.2f})")


def run_prediction_pipeline(test_dir: Path):
    print("\n" + "=" * 65)
    print("STEP 10: RUNNING INFERENCE ON TEST DATA")
    print("=" * 65)

    test_s1 = test_dir / "test_source1.tsv"
    test_s2 = test_dir / "test_source2.tsv"
    test_s3 = test_dir / "test_source3.tsv"

    for p in [test_s1, test_s2, test_s3]:
        if not p.exists():
            raise FileNotFoundError(f"Missing test file: {p}")

    cand_dict, match_dict, all_s1_ids = run_test_inference(test_s1, test_s2, test_s3)
    save_submission_artifacts(cand_dict, match_dict, all_s1_ids)

    print("\n" + "=" * 65)
    print("VALIDATING SUBMISSION TSVs")
    print("=" * 65)
    validate_submission(MATCHING_OUTPUT_PATH, CANDIDATE_OUTPUT_PATH, test_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument("--train", action="store_true", help="Run model training and CV")
    parser.add_argument("--sample", type=int, default=50000, help="Number of S1 entities to train on (default: 50000 for high precision & memory safety, 0 for full)")
    parser.add_argument("--predict", action="store_true", help="Run inference on test directory")
    parser.add_argument("--test-dir", type=Path, default=PROJECT_ROOT / "dataset" / "test", help="Test directory")

    args = parser.parse_args()
    if args.train:
        run_training_pipeline(sample_limit=args.sample)
    elif args.predict:
        run_prediction_pipeline(args.test_dir)
    else:
        parser.print_help()
