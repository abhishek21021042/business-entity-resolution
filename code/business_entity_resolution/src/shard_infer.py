"""
Distributed Sharded Inference Engine for Business Entity Resolution.
Allows running massive test sets across multiple laptops/machines without OOM,
and safely saves partial results for seamless merging.
"""

import argparse
from pathlib import Path
import sys
import time
import os
import sqlite3
from typing import Optional

# Ensure project root is in sys.path
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _CURRENT_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

# Force unbuffered line output for real-time web dashboard streaming
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import PROJECT_ROOT, MODELS_DIR, DEFAULT_THRESHOLD
from src.path_utils import detect_test_dir, detect_output_dir, detect_cache_dir
from src.progress_state import ProgressTracker
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


def fetch_candidate_records_sqlite(conn: sqlite3.Connection, candidate_ids: set) -> dict:
    """
    Fetches required candidate records from indexed SQLite database in fast batches.
    Ultra-low RAM (<50 MB) and instantaneous (0.1s for 20k candidates).
    """
    if not candidate_ids:
        return {}
    cids = list(candidate_ids)
    results = {}
    cur = conn.cursor()
    for i in range(0, len(cids), 999):
        batch = cids[i:i + 999]
        ph = ",".join("?" for _ in batch)
        cur.execute(
            f"SELECT entity_id, normalized_name, core_name, sorted_core, suffix, normalized_addr, postal_code, country "
            f"FROM records WHERE entity_id IN ({ph})",
            batch
        )
        for r in cur.fetchall():
            results[r[0]] = {
                "entity_id": r[0],
                "normalized_name": r[1],
                "core_name": r[2],
                "sorted_core": r[3],
                "suffix": r[4] if r[4] else None,
                "normalized_addr": r[5],
                "postal_code": r[6] if r[6] else None,
                "country": r[7],
            }
    return results


def run_sharded_inference(
    test_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    num_shards: int = 3,
    shard_id: int = 0,
    chunk_size: int = 100_000,
    threshold: float = 0.95
):
    """
    Runs inference on a specific shard (1/Nth) of Source-1 entities.
    Processes S1 in streaming memory-safe chunks of `chunk_size`.
    Auto-detects paths if not specified.
    """
    if test_dir is None:
        test_dir = detect_test_dir()
    if output_dir is None:
        output_dir = detect_output_dir()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_cand_file = output_dir / f"shard_{shard_id}_of_{num_shards}_candidate_pairs.tsv"
    out_match_file = output_dir / f"shard_{shard_id}_of_{num_shards}_matching_results.tsv"

    print("=" * 65)
    print(f"DISTRIBUTED INFERENCE: SHARD {shard_id + 1} OF {num_shards}")
    print("=" * 65)
    print(f"Test Directory:   {test_dir}  [AUTO-DETECTED]")
    print(f"Output Directory: {output_dir}  [AUTO-DETECTED]")
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

    # ── ZERO-RAM CACHE: Use SQLite for O(1) instant candidate lookups ──
    import pickle
    cache_dir = detect_cache_dir(output_dir)
    db_cache   = cache_dir / "other_norm.db"
    norm_cache = cache_dir / "other_norm.pkl"
    index_cache = cache_dir / "blocking_index.pkl"

    if db_cache.exists():
        print(f"[CACHE HIT] Connecting to ultra-low-RAM SQLite database: {db_cache.name} ...")
        db_conn = sqlite3.connect(str(db_cache))
    elif norm_cache.exists():
        print(f"Creating indexed SQLite database from {norm_cache.name} for zero-RAM overhead...")
        with open(norm_cache, "rb") as f:
            _df = pickle.load(f)
        cols_needed = ["entity_id", "normalized_name", "core_name", "sorted_core", "suffix", "normalized_addr", "postal_code", "country"]
        cols = [c for c in cols_needed if c in _df.columns]
        _df = _df[cols].fillna("")
        db_conn = sqlite3.connect(str(db_cache))
        _df.to_sql("records", db_conn, if_exists="replace", index=False, chunksize=100000)
        db_conn.cursor().execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_id ON records (entity_id)")
        db_conn.commit()
        del _df
        print("SQLite database created and connected successfully!")
    else:
        s2_raw = load_source_df(s2_path)
        s3_raw = load_source_df(s3_path)
        other_raw = pd.concat([s2_raw, s3_raw], ignore_index=True)
        print(f"Total target records (S2 + S3): {len(other_raw):,}")
        other_norm = parallel_normalize_records(other_raw.to_dict("records"), n_jobs=-1)
        db_conn = sqlite3.connect(str(db_cache))
        cols_needed = ["entity_id", "normalized_name", "core_name", "sorted_core", "suffix", "normalized_addr", "postal_code", "country"]
        cols = [c for c in cols_needed if c in other_norm.columns]
        other_norm[cols].fillna("").to_sql("records", db_conn, if_exists="replace", index=False, chunksize=100000)
        db_conn.cursor().execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_id ON records (entity_id)")
        db_conn.commit()
        del other_norm
        print("SQLite database created and connected successfully!")

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

    # Initialize live progress tracker
    tracker = ProgressTracker(output_dir, shard_id, num_shards, shard_count)
    tracker.update_stage("Cache Ready", "Loaded models & pre-normalized database")

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

    # Pre-build hash tables ONCE for the entire S2+S3 search space to avoid rebuilding on every chunk
    print("Preparing blocking hash tables for entire S2+S3 search space (Pass 1, 2, 4, 5)...")
    t_index = time.time()
    from src.phonetic import phonetic_key
    import re

    STOPWORDS = {"the", "and", "for", "with", "all", "new", "top", "pro", "best", "inc", "ltd", "pvt", "corp", "llc", "co"}

    if index_cache.exists():
        print(f"[CACHE HIT] Loading pre-built blocking index from {index_cache} ...")
        with open(index_cache, "rb") as f:
            idx_data = pickle.load(f)
        exact_buckets = idx_data["exact"]
        phonetic_buckets = idx_data["phonetic"]
        first_word_buckets = idx_data["first_word"]
        addr_buckets = idx_data["addr"]
        print(f"Loaded blocking index in {time.time() - t_index:.1f}s! All lookups will be instantaneous O(1).\n")
    else:
        # 1. Exact sorted-core buckets
        exact_buckets = {}
        other_ids = other_norm["entity_id"].astype(str).str.strip().values
        other_cores = other_norm["sorted_core"].astype(str).str.strip().values
        other_raw_cores = other_norm["core_name"].astype(str).str.strip().values
        other_countries = other_norm["country"].astype(str).str.strip().values
        other_addrs = other_norm["normalized_addr"].astype(str).values

        for cid, core, country in zip(other_ids, other_cores, other_countries):
            if len(core) >= 3:
                b = exact_buckets.setdefault((core, country), [])
                if len(b) < 30:
                    b.append(cid)

        # 2. Phonetic buckets
        phonetic_buckets = {}
        for cid, core, country in zip(other_ids, other_raw_cores, other_countries):
            if len(core) >= 3:
                pkey = phonetic_key(core)
                if pkey:
                    b = phonetic_buckets.setdefault((pkey, country), [])
                    if len(b) < 30:
                        b.append(cid)

        # 3. First-word buckets
        first_word_buckets = {}
        for cid, core, ctry in zip(other_ids, other_raw_cores, other_countries):
            toks = [t for t in core.split() if t not in {"m/s", "dr", "mr", "ms", "sri", "shri"}]
            if toks and len(toks[0]) >= 3 and toks[0] not in stopwords:
                b = first_word_buckets.setdefault((toks[0], ctry), [])
                if len(b) < 30:
                    b.append(cid)

        # 4. Address number buckets
        addr_buckets = {}
        for cid, addr, ctry in zip(other_ids, other_addrs, other_countries):
            if addr:
                raw_nums = re.findall(r"\b[a-zA-Z]?[-#]?\d+[/a-zA-Z\-_]*\d*[a-zA-Z]?\b", addr.lower())
                for n in raw_nums:
                    clean_n = n.strip("-# ").replace(" ", "")
                    if len(clean_n) >= 2 and any(c.isdigit() for c in clean_n):
                        b = addr_buckets.setdefault((clean_n, ctry), [])
                        if len(b) < 30:
                            b.append(cid)

        print(f"Saving blocking index cache to {index_cache} ...")
        with open(index_cache, "wb") as f:
            pickle.dump({
                "exact": exact_buckets,
                "phonetic": phonetic_buckets,
                "first_word": first_word_buckets,
                "addr": addr_buckets,
            }, f, protocol=5)
        print(f"Index built and cached in {time.time() - t_index:.1f}s! All lookups will be instantaneous O(1).\n")

    # Pre-normalize this shard's S1 records ONCE (cached as well for instant resume)
    shard_s1_cache = cache_dir / f"shard_{shard_id}_of_{num_shards}_s1_norm.pkl"
    if shard_s1_cache.exists():
        print(f"[CACHE HIT] Loading pre-normalized S1 shard from {shard_s1_cache.name} ...")
        with open(shard_s1_cache, "rb") as f:
            shard_s1_norm = pickle.load(f)
    else:
        print(f"Normalizing S1 records for this shard ({shard_count:,} entities) once...")
        t_s1_norm = time.time()
        shard_s1_norm = parallel_normalize_records(shard_s1.to_dict("records"), n_jobs=-1)
        with open(shard_s1_cache, "wb") as f:
            pickle.dump(shard_s1_norm, f, protocol=5)
        print(f"Shard S1 normalized in {time.time() - t_s1_norm:.1f}s!\n")

    shard_s1_dict = {r["entity_id"]: r for r in shard_s1_norm.to_dict("records")}

    for chunk_idx in range(n_chunks):
        c_start = chunk_idx * chunk_size
        c_end = min(shard_count, c_start + chunk_size)
        c_s1_norm = shard_s1_norm.iloc[c_start:c_end]
        c_s1_ids = [str(x).strip() for x in c_s1_norm["entity_id"].tolist()]
        c_s1_dict = {sid: shard_s1_dict[sid] for sid in c_s1_ids if sid in shard_s1_dict}

        print(f"--- Chunk {chunk_idx + 1}/{n_chunks} ({len(c_s1_ids):,} entities) ---")
        c_t0 = time.time()

        s1_c_ids = c_s1_norm["entity_id"].astype(str).str.strip().values
        s1_c_cores = c_s1_norm["sorted_core"].astype(str).str.strip().values
        s1_c_raw_cores = c_s1_norm["core_name"].astype(str).str.strip().values
        s1_c_countries = c_s1_norm["country"].astype(str).str.strip().values
        s1_c_addrs = c_s1_norm["normalized_addr"].astype(str).values

        # Collect candidate IDs per entity with priority cap (max 15 candidates per S1)
        s1_cand_map = {sid: [] for sid in s1_c_ids}
        s1_cand_sets = {sid: set() for sid in s1_c_ids}

        def add_cands(sid, cids_to_add, max_cap=15):
            cur_set = s1_cand_sets[sid]
            cur_list = s1_cand_map[sid]
            for cid in cids_to_add:
                if len(cur_list) >= max_cap:
                    break
                if cid not in cur_set:
                    cur_set.add(cid)
                    cur_list.append(cid)

        # Pass 1: Exact sorted core (Highest priority)
        for sid, core, country in zip(s1_c_ids, s1_c_cores, s1_c_countries):
            if len(core) >= 3:
                b = exact_buckets.get((core, country), [])
                if b:
                    add_cands(sid, b, max_cap=15)

        # Pass 2: Phonetic NYSIIS
        for sid, core, country in zip(s1_c_ids, s1_c_raw_cores, s1_c_countries):
            if len(core) >= 3:
                pkey = phonetic_key(core)
                if pkey:
                    b = phonetic_buckets.get((pkey, country), [])
                    if b:
                        add_cands(sid, b, max_cap=15)

        # Pass 3: First Word Core (only if < 15 candidates)
        for sid, core, country in zip(s1_c_ids, s1_c_raw_cores, s1_c_countries):
            if len(s1_cand_map[sid]) < 15:
                toks = [t for t in core.split() if t not in {"m/s", "dr", "mr", "ms", "sri", "shri"}]
                if toks and len(toks[0]) >= 3 and toks[0] not in STOPWORDS:
                    b = first_word_buckets.get((toks[0], country), [])
                    if b:
                        add_cands(sid, b, max_cap=15)

        # Pass 4: Address House/Flat numbers (only if < 15 candidates)
        for sid, addr, country in zip(s1_c_ids, s1_c_addrs, s1_c_countries):
            if len(s1_cand_map[sid]) < 15 and addr:
                raw_nums = re.findall(r"\b[a-zA-Z]?[-#]?\d+[/a-zA-Z\-_]*\d*[a-zA-Z]?\b", addr.lower())
                for n in raw_nums:
                    clean_n = n.strip("-# ").replace(" ", "")
                    if len(clean_n) >= 2 and any(c.isdigit() for c in clean_n):
                        b = addr_buckets.get((clean_n, country), [])
                        if b:
                            add_cands(sid, b, max_cap=15)

        c_pairs = [(sid, cid) for sid, cids in s1_cand_map.items() for cid in cids]
        n_chunk_pairs = len(c_pairs)
        total_pairs_generated += n_chunk_pairs
        print(f"Candidates generated: {n_chunk_pairs:,} pairs")
        c_cand_dict = s1_cand_map

        # Feature Extraction & Model Scoring
        if c_pairs:
            # ULTRA-FAST ZERO-RAM LOOKUP: Fetch only needed candidate rows from indexed SQLite
            needed_cids = {cid for _, cid in c_pairs}
            sub_other_dict = fetch_candidate_records_sqlite(db_conn, needed_cids)

            cand_feat_df = extract_features_parallel(c_pairs, c_s1_dict, sub_other_dict, n_jobs=-1)
            cols = get_feature_columns(cand_feat_df)
            X_chunk = cand_feat_df[cols]

            probs = np.zeros(len(cand_feat_df), dtype=np.float32)
            for m in models:
                probs += m.predict_proba(X_chunk)[:, 1]
            probs /= len(models)
            cand_feat_df["pred_score"] = probs
            cand_feat_df["score"] = probs
            raw_matches = resolve_global_conflicts(cand_feat_df, threshold=tuned_thresh)
            c_match_dict = apply_singleton_rule(raw_matches, c_s1_ids)
        else:
            c_match_dict = {sid: [] for sid in c_s1_ids}

        # Append to TSVs (open with 'w' on first chunk to clear any partial run)
        file_mode = "w" if chunk_idx == 0 else "a"
        with open(out_cand_file, file_mode, encoding="utf-8") as f_cand:
            for sid in c_s1_ids:
                cands = c_cand_dict.get(sid, [])
                f_cand.write(f"{sid}\t{','.join(cands)}\n")

        with open(out_match_file, file_mode, encoding="utf-8") as f_match:
            for sid in c_s1_ids:
                matches = c_match_dict.get(sid, [])
                if matches:
                    total_matches_found += len(matches)
                f_match.write(f"{sid}\t{','.join(matches)}\n")

        elapsed = time.time() - c_t0
        print(f"Chunk {chunk_idx + 1} completed in {elapsed:.1f}s | Progress: {c_end:,}/{shard_count:,} S1\n")
        tracker.update_chunk_progress(
            chunk_idx=chunk_idx,
            total_chunks=n_chunks,
            processed_entities=c_end,
            new_candidates=n_chunk_pairs,
            new_matches=total_matches_found,
            stage_msg=f"Finished Chunk {chunk_idx + 1}/{n_chunks} ({c_end:,}/{shard_count:,} S1)"
        )

        # Explicitly free memory after chunk completion
        if c_pairs:
            del cand_feat_df, X_chunk, sub_other_dict, c_pairs
        del c_cand_dict, c_match_dict, s1_cand_map, s1_cand_sets
        import gc
        gc.collect()

    tracker.mark_complete(out_cand_file, out_match_file)
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
    parser.add_argument("--test-dir", type=Path, default=None, help="Path to test directory (auto-detected if omitted)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Path to output directory (auto-detected if omitted)")
    parser.add_argument("--num-shards", type=int, default=3, help="Total number of laptops/shards (default: 3)")
    parser.add_argument("--shard-id", type=int, default=0, help="Zero-indexed shard ID (0, 1, or 2)")
    parser.add_argument("--chunk-size", type=int, default=100000, help="Streaming batch size")
    parser.add_argument("--threshold", type=float, default=0.65, help="F0.5 threshold")

    args = parser.parse_args()
    run_sharded_inference(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        chunk_size=args.chunk_size,
        threshold=args.threshold
    )
