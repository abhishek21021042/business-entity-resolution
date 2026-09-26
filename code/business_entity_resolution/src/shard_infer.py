"""
Distributed Sharded Inference Engine for Business Entity Resolution.
Allows running massive test sets across multiple laptops/machines without OOM,
and safely saves partial results for seamless merging.
"""

import argparse
from pathlib import Path
import sys
import time
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import PROJECT_ROOT, MODELS_DIR, DEFAULT_THRESHOLD
from src.data_loading import load_source_df
from src.normalization import normalize_record, parallel_normalize_records
from src.blocking import (
    exact_key_blocking,
    phonetic_blocking,
    tfidf_topk_blocking,
    first_word_blocking,
    address_key_blocking,
    union_candidates,
    candidate_list_to_dict
)
from src.features import compute_pair_features, add_rank_and_margin_features, extract_features_parallel
from src.train_model import get_feature_columns
from src.infer import resolve_global_conflicts, apply_singleton_rule


def run_sharded_inference(
    test_dir: Path,
    output_dir: Path,
    num_shards: int = 3,
    shard_id: int = 0,
    chunk_size: int = 100_000,
    threshold: float = 0.95
):
    """
    Runs inference on a specific shard (1/Nth) of Source-1 entities.
    Processes S1 in streaming memory-safe chunks of `chunk_size`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_cand_file = output_dir / f"shard_{shard_id}_of_{num_shards}_candidate_pairs.tsv"
    out_match_file = output_dir / f"shard_{shard_id}_of_{num_shards}_matching_results.tsv"

    print("=" * 65)
    print(f"DISTRIBUTED INFERENCE: SHARD {shard_id + 1} OF {num_shards}")
    print("=" * 65)
    print(f"Test Directory:   {test_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Target Shard:     Part {shard_id + 1}/{num_shards}")
    print(f"Candidate Output: {out_cand_file.name}")
    print(f"Matching Output:  {out_match_file.name}")

    # 1. Load Trained Models
    model_path = MODELS_DIR / "ensemble_models.joblib"
    meta_path = MODELS_DIR / "meta.joblib"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Trained models not found in {MODELS_DIR}. Ensure models are present.")

    print("\nLoading trained ensemble models...")
    models = joblib.load(model_path)
    meta = joblib.load(meta_path)
    tuned_thresh = meta.get("threshold", threshold)
    print(f"Loaded {len(models)} models. Active Threshold: {tuned_thresh}")

    # 2. Load Source-2 and Source-3 (The Search Space)
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"
    s1_path = test_dir / "test_source1.tsv"

    for p in [s1_path, s2_path, s3_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required test file: {p}")

    print("\nLoading and normalizing Source 2 and Source 3 records...")
    t0 = time.time()
    s2_raw = load_source_df(s2_path)
    s3_raw = load_source_df(s3_path)
    other_raw = pd.concat([s2_raw, s3_raw], ignore_index=True)
    print(f"Total target records (S2 + S3): {len(other_raw):,}")

    other_norm = parallel_normalize_records(other_raw.to_dict("records"), n_jobs=-1)
    other_dict = {r["entity_id"]: r for r in other_norm.to_dict("records")}
    print(f"Normalized S2 + S3 in {time.time() - t0:.1f}s")

    # 3. Read Source-1 IDs and Slice the Shard
    print("\nReading Source 1 entity IDs...")
    s1_raw = load_source_df(s1_path)
    total_s1 = len(s1_raw)
    
    # Calculate shard boundaries
    shard_size = int(np.ceil(total_s1 / num_shards))
    start_idx = shard_id * shard_size
    end_idx = min(total_s1, start_idx + shard_size)
    shard_s1 = s1_raw.iloc[start_idx:end_idx].copy().reset_index(drop=True)
    shard_count = len(shard_s1)

    print(f"Total S1 entities in dataset: {total_s1:,}")
    print(f"Assigned to this shard: rows {start_idx:,} to {end_idx:,} ({shard_count:,} entities)")

    # Initialize output TSV headers
    with open(out_cand_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
    with open(out_match_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")

    # 4. Stream S1 in manageable chunks across 8 CPU cores
    n_chunks = int(np.ceil(shard_count / chunk_size))
    print(f"\nProcessing {shard_count:,} entities in {n_chunks} streaming chunks (chunk_size={chunk_size:,})...\n")

    total_pairs_generated = 0
    total_matches_found = 0

    for chunk_idx in range(n_chunks):
        c_start = chunk_idx * chunk_size
        c_end = min(shard_count, c_start + chunk_size)
        c_s1_raw = shard_s1.iloc[c_start:c_end]
        c_s1_ids = [str(x).strip() for x in c_s1_raw["entity_id"].tolist()]

        print(f"--- Chunk {chunk_idx + 1}/{n_chunks} ({len(c_s1_ids):,} entities) ---")
        c_t0 = time.time()

        # Normalize S1 chunk
        c_s1_norm = parallel_normalize_records(c_s1_raw.to_dict("records"), n_jobs=-1)
        c_s1_dict = {r["entity_id"]: r for r in c_s1_norm.to_dict("records")}

        # 5-Pass Blocking
        p1 = exact_key_blocking(c_s1_norm, other_norm)
        p2 = phonetic_blocking(c_s1_norm, other_norm)
        p3 = tfidf_topk_blocking(c_s1_norm, other_norm, k=15)
        p4 = first_word_blocking(c_s1_norm, other_norm)
        p5 = address_key_blocking(c_s1_norm, other_norm)

        c_pairs = union_candidates(p1, p2, p3, p4, p5)
        total_pairs_generated += len(c_pairs)
        print(f"Candidates generated: {len(c_pairs):,} pairs")

        # Candidate dict mapping
        c_cand_dict = candidate_list_to_dict(c_pairs)

        # Feature Extraction & Model Scoring
        if c_pairs:
            cand_feat_df = extract_features_parallel(c_pairs, c_s1_dict, other_dict, n_jobs=-1)
            cand_feat_df = add_rank_and_margin_features(cand_feat_df, score_col="joint_confidence")
            cols = get_feature_columns(cand_feat_df)
            X_chunk = cand_feat_df[cols]

            probs = np.zeros(len(cand_feat_df), dtype=np.float32)
            for m in models:
                probs += m.predict_proba(X_chunk)[:, 1]
            probs /= len(models)
            cand_feat_df["pred_score"] = probs

            raw_matches = resolve_global_conflicts(cand_feat_df, threshold=tuned_thresh)
            c_match_dict = apply_singleton_rule(raw_matches, c_s1_ids)
        else:
            c_match_dict = {sid: [] for sid in c_s1_ids}

        # Append to TSVs
        with open(out_cand_file, "a", encoding="utf-8") as f_cand:
            for sid in c_s1_ids:
                cands = c_cand_dict.get(sid, [])
                f_cand.write(f"{sid}\t{','.join(cands)}\n")

        with open(out_match_file, "a", encoding="utf-8") as f_match:
            for sid in c_s1_ids:
                matches = c_match_dict.get(sid, [])
                if matches:
                    total_matches_found += len(matches)
                f_match.write(f"{sid}\t{','.join(matches)}\n")

        elapsed = time.time() - c_t0
        print(f"Chunk {chunk_idx + 1} completed in {elapsed:.1f}s | Progress: {c_end:,}/{shard_count:,} S1\n")

    print("=" * 65)
    print(f"SHARD {shard_id + 1} COMPLETED SUCCESSFULLY!")
    print(f"Total S1 entities processed: {shard_count:,}")
    print(f"Total Candidate Pairs:      {total_pairs_generated:,}")
    print(f"Total Matches Found:        {total_matches_found:,}")
    print(f"Files saved:")
    print(f" - {out_cand_file}")
    print(f" - {out_match_file}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Sharded Inference")
    parser.add_argument("--test-dir", type=Path, default=Path("D:/test_data"), help="Path to test directory")
    parser.add_argument("--output-dir", type=Path, default=Path("D:/output"), help="Path to output directory")
    parser.add_argument("--num-shards", type=int, default=3, help="Total number of laptops/shards (default: 3)")
    parser.add_argument("--shard-id", type=int, default=0, help="Zero-indexed shard ID (0, 1, or 2)")
    parser.add_argument("--chunk-size", type=int, default=100000, help="Streaming batch size")
    parser.add_argument("--threshold", type=float, default=0.95, help="F0.5 threshold")

    args = parser.parse_args()
    run_sharded_inference(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        chunk_size=args.chunk_size,
        threshold=args.threshold
    )
