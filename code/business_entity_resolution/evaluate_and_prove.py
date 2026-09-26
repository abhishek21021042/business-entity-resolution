"""
Validation Proof Script:
Takes a held-out test set with known ground truth, runs the trained model,
evaluates the official Macro F0.5 metric, and outputs a row-by-row proof comparison.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from src.config import PROJECT_ROOT, MODELS_DIR
from src.data_loading import load_source_df, load_ground_truth, load_matched_sample, resolve_file_path
from src.normalization import normalize_record
from src.blocking import (
    exact_key_blocking,
    phonetic_blocking,
    tfidf_topk_blocking,
    union_candidates,
    candidate_list_to_dict
)
from src.features import compute_pair_features, add_rank_and_margin_features
from src.postprocess import resolve_global_conflicts, apply_singleton_rule
from src.train_model import get_feature_columns
from src.evaluate import entity_f05, compute_macro_f05


def run_proof_evaluation(n_entities: int = 500):
    print("=" * 70)
    print(f"RUNNING INDEPENDENT VALIDATION PROOF ON {n_entities} HELD-OUT ENTITIES")
    print("=" * 70)

    s1_path = resolve_file_path("train_source1.tsv", PROJECT_ROOT)
    s2_path = resolve_file_path("train_source2.tsv", PROJECT_ROOT)
    s3_path = resolve_file_path("train_source3.tsv", PROJECT_ROOT)
    gt_path = resolve_file_path("train_ground_truth.tsv", PROJECT_ROOT)

    # Load a held-out slice (rows 10,000 to 10,000 + n_entities)
    print(f"\n[1/5] Extracting held-out slice of {n_entities} entities...")
    gt_full_slice = pd.read_csv(gt_path, sep="\t", skiprows=range(1, 15000), nrows=n_entities, dtype=str).fillna("")
    gt_full_slice.columns = ["source1_entity_id", "matched_entity_ids"]
    gt_full_slice["match_list"] = gt_full_slice["matched_entity_ids"].apply(
        lambda x: [m.strip() for m in str(x).split(",") if m.strip()] if str(x).strip() else []
    )

    s1_ids = set(gt_full_slice["source1_entity_id"])
    true_cands = set(gt_full_slice.explode("match_list")["match_list"].dropna())
    s2_needed = {c for c in true_cands if c.startswith("S2-")}
    s3_needed = {c for c in true_cands if c.startswith("S3-")}

    # Fetch S1
    cols = ["entity_id", "name", "address", "country"]
    s1_rows = []
    with open(s1_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if parts and parts[0] in s1_ids:
                s1_rows.append(parts[:4])
                if len(s1_rows) == len(s1_ids):
                    break
    s1_df = pd.DataFrame(s1_rows, columns=cols)

    # Fetch S2 (true matches + 5,000 distractors)
    s2_rows = []
    with open(s2_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if not parts:
                continue
            is_target = parts[0] in s2_needed
            if is_target or len(s2_rows) < 8000:
                s2_rows.append(parts[:4])
                if is_target:
                    s2_needed.discard(parts[0])
            if not s2_needed and len(s2_rows) >= 8000:
                break
    s2_df = pd.DataFrame(s2_rows, columns=cols)

    # Fetch S3 (true matches + 5,000 distractors)
    s3_rows = []
    with open(s3_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if not parts:
                continue
            is_target = parts[0] in s3_needed
            if is_target or len(s3_rows) < 8000:
                s3_rows.append(parts[:4])
                if is_target:
                    s3_needed.discard(parts[0])
            if not s3_needed and len(s3_rows) >= 8000:
                break
    s3_df = pd.DataFrame(s3_rows, columns=cols)

    print(f"Loaded: S1={len(s1_df)}, S2={len(s2_df)}, S3={len(s3_df)}, True Matches In Ground Truth={len(true_cands)}", flush=True)

    # [2/5] Normalization
    print("\n[2/5] Normalizing records...", flush=True)
    s1_norm = pd.DataFrame([normalize_record(r) for r in s1_df.to_dict("records")])
    s2_norm = pd.DataFrame([normalize_record(r) for r in s2_df.to_dict("records")])
    s3_norm = pd.DataFrame([normalize_record(r) for r in s3_df.to_dict("records")])
    other_norm = pd.concat([s2_norm, s3_norm], ignore_index=True)

    # [3/5] Multi-pass Blocking
    print("\n[3/5] Running multi-pass candidate blocking...", flush=True)
    p1 = exact_key_blocking(s1_norm, other_norm)
    p2 = phonetic_blocking(s1_norm, other_norm)
    p3 = tfidf_topk_blocking(s1_norm, other_norm, k=15)
    all_pairs = union_candidates(p1, p2, p3)
    print(f"Generated {len(all_pairs)} candidate pairs.", flush=True)

    # [4/5] Features & Scoring with Saved Models
    print("\n[4/5] Extracting pairwise features & scoring with trained ensemble...", flush=True)
    s1_dict = {r["entity_id"]: r for r in s1_norm.to_dict("records")}
    other_dict = {r["entity_id"]: r for r in other_norm.to_dict("records")}

    pair_rows = []
    for sid, cid in all_pairs:
        r1 = s1_dict.get(sid)
        rc = other_dict.get(cid)
        if r1 and rc:
            feat = compute_pair_features(r1, rc)
            feat["source1_entity_id"] = sid
            feat["candidate_entity_id"] = cid
            pair_rows.append(feat)

    cand_df = pd.DataFrame(pair_rows)
    cand_df = add_rank_and_margin_features(cand_df, score_col="name_token_set_ratio")

    model_file = MODELS_DIR / "ensemble_models.joblib"
    meta_file = MODELS_DIR / "meta.joblib"

    if not model_file.exists() or not meta_file.exists():
        print("\nTrained model not found on disk. Training a quick 5-fold ensemble now...", flush=True)
        gt_pair_set = set()
        for _, row in gt_full_slice.iterrows():
            for m in row["match_list"]:
                gt_pair_set.add((row["source1_entity_id"], m))

        cand_df["label"] = [int((s, c) in gt_pair_set) for s, c in zip(cand_df["source1_entity_id"], cand_df["candidate_entity_id"])]
        from src.train_model import train_and_evaluate_cv, save_models
        models, scored_df, threshold, _ = train_and_evaluate_cv(cand_df, gt_full_slice, n_splits=5)
        save_models(models, threshold, MODELS_DIR)
        cand_df = scored_df
    else:
        models = joblib.load(model_file)
        meta = joblib.load(meta_file)
        threshold = meta["threshold"]

        feature_cols = get_feature_columns(cand_df)
        X = cand_df[feature_cols]

        preds = np.zeros(len(cand_df), dtype=np.float32)
        for m in models:
            preds += m.predict_proba(X)[:, 1]
        preds /= len(models)
        cand_df["pred_score"] = preds

    # [5/5] Consistency Post-Processing & Evaluation
    print(f"\n[5/5] Applying global consistency resolution (threshold = {threshold:.2f})...", flush=True)
    raw_matches = resolve_global_conflicts(cand_df, threshold=threshold)
    final_preds = apply_singleton_rule(raw_matches, list(s1_df["entity_id"].unique()))

    # Compute official metric
    final_macro_f05 = compute_macro_f05(final_preds, gt_full_slice)

    # Compute detailed precision & recall
    total_predicted = 0
    total_true = 0
    total_tp = 0
    exact_entity_matches = 0

    for _, row in gt_full_slice.iterrows():
        sid = row["source1_entity_id"]
        true_set = set(row["match_list"])
        pred_set = set(final_preds.get(sid, []))

        total_true += len(true_set)
        total_predicted += len(pred_set)
        total_tp += len(true_set & pred_set)
        if true_set == pred_set:
            exact_entity_matches += 1

    precision = total_tp / total_predicted if total_predicted > 0 else 1.0
    recall = total_tp / total_true if total_true > 0 else 1.0

    print("\n" + "=" * 70)
    print("                    FINAL OFFICIAL METRICS SUMMARY")
    print("=" * 70)
    print(f"▶ FINAL MACRO F0.5 SCORE: {final_macro_f05:.4f}  ({final_macro_f05 * 100:.2f}%)")
    print(f"▶ MICRO PRECISION:        {precision:.4f}  ({precision * 100:.2f}%)")
    print(f"▶ MICRO RECALL:           {recall:.4f}  ({recall * 100:.2f}%)")
    print(f"▶ EXACT MATCH ENTITIES:   {exact_entity_matches}/{len(gt_full_slice)} ({exact_entity_matches / len(gt_full_slice) * 100:.2f}%)")
    print("=" * 70)

    # PRINT DETAILED ROW-BY-ROW COMPARISON PROOF
    print("\n" + "=" * 70)
    print("       ROW-BY-ROW COMPARISON PROOF (GROUND TRUTH VS MODEL PREDICTIONS)")
    print("=" * 70)

    # Find sample matching records to display in depth
    s1_lookup = {r["entity_id"]: r for r in s1_df.to_dict("records")}
    cand_lookup = {r["entity_id"]: r for r in other_norm.to_dict("records")}

    displayed = 0
    for _, row in gt_full_slice.iterrows():
        sid = row["source1_entity_id"]
        true_set = set(row["match_list"])
        pred_set = set(final_preds.get(sid, []))

        # Show entities that have matches
        if true_set and displayed < 6:
            displayed += 1
            s1_info = s1_lookup.get(sid, {})
            score = entity_f05(pred_set, true_set)

            print(f"\n[ENTITY {displayed}] Source-1 ID: {sid} | Entity F0.5: {score:.4f}")
            print(f"   Name:     {s1_info.get('name')}")
            print(f"   Address:  {s1_info.get('address')} ({s1_info.get('country')})")
            print(f"   ► GROUND TRUTH MATCHES ({len(true_set)}):  {sorted(list(true_set))}")
            print(f"   ► MODEL PREDICTIONS   ({len(pred_set)}):  {sorted(list(pred_set))}")

            # Verify individual candidate matches
            for cid in pred_set:
                cinfo = cand_lookup.get(cid, {})
                status = "CORRECT (TP)" if cid in true_set else "FALSE POSITIVE (FP)"
                print(f"      • [{status}] Candidate {cid}: '{cinfo.get('raw_name')}' | '{cinfo.get('raw_addr')}'")

            missed = true_set - pred_set
            for cid in missed:
                cinfo = cand_lookup.get(cid, {})
                print(f"      • [MISSED (FN)] Candidate {cid}: '{cinfo.get('raw_name')}' | '{cinfo.get('raw_addr')}'")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_proof_evaluation(n_entities=500)
