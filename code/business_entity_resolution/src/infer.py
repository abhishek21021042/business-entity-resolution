"""
Inference Pipeline for Test Data.
Loads test sources, executes normalization, multi-pass blocking,
feature computation, ensemble scoring, and consistency post-processing.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import MODELS_DIR, DEFAULT_THRESHOLD
from .data_loading import load_source_df
from .normalization import normalize_record
from .blocking import (
    exact_key_blocking,
    phonetic_blocking,
    tfidf_topk_blocking,
    first_word_blocking,
    address_key_blocking,
    union_candidates,
    candidate_list_to_dict
)
from .features import compute_pair_features, add_rank_and_margin_features
from .postprocess import resolve_global_conflicts, apply_singleton_rule
from .train_model import get_feature_columns


def run_test_inference(test_s1_path: Path,
                       test_s2_path: Path,
                       test_s3_path: Path,
                       models: Optional[List[Any]] = None,
                       threshold: Optional[float] = None) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], List[str]]:
    """
    Executes end-to-end inference on test split:
    Returns:
    - candidate_dict: {s1_id: [cand_ids]}
    - matched_dict: {s1_id: [matched_ids]}
    - all_s1_ids: sorted list of all test source 1 IDs
    """
    # Load models if not passed
    if models is None or threshold is None:
        model_path = MODELS_DIR / "ensemble_models.joblib"
        meta_path = MODELS_DIR / "meta.joblib"
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Model artifacts not found in {MODELS_DIR}. Train models first.")
        models = joblib.load(model_path)
        meta = joblib.load(meta_path)
        threshold = meta.get("threshold", DEFAULT_THRESHOLD)

    print("\n--- Ingesting and Normalizing Test Records ---")
    s1_raw = load_source_df(test_s1_path)
    s2_raw = load_source_df(test_s2_path)
    s3_raw = load_source_df(test_s3_path)

    all_s1_ids = [str(x).strip() for x in s1_raw["entity_id"].tolist()]

    s1_norm = pd.DataFrame([normalize_record(row) for row in s1_raw.to_dict("records")])
    s2_norm = pd.DataFrame([normalize_record(row) for row in s2_raw.to_dict("records")])
    s3_norm = pd.DataFrame([normalize_record(row) for row in s3_raw.to_dict("records")])

    other_norm = pd.concat([s2_norm, s3_norm], ignore_index=True)

    print("--- Executing 5-Pass Blocking on Test ---")
    pass1 = exact_key_blocking(s1_norm, other_norm)
    pass2 = phonetic_blocking(s1_norm, other_norm)
    pass3 = tfidf_topk_blocking(s1_norm, other_norm, k=25)
    pass4 = first_word_blocking(s1_norm, other_norm)
    pass5 = address_key_blocking(s1_norm, other_norm)

    all_pairs = union_candidates(pass1, pass2, pass3, pass4, pass5)
    print(f"Total candidate pairs generated: {len(all_pairs)} for {len(all_s1_ids)} S1 entities")

    candidate_dict = candidate_list_to_dict(all_pairs)

    if not all_pairs:
        # All singletons
        matched_dict = {sid: [] for sid in all_s1_ids}
        return candidate_dict, matched_dict, all_s1_ids

    print("--- Extracting Features for Candidate Pairs ---")
    s1_dict = {row["entity_id"]: row for row in s1_norm.to_dict("records")}
    other_dict = {row["entity_id"]: row for row in other_norm.to_dict("records")}

    pair_rows = []
    for sid, cid in tqdm(all_pairs, desc="Pairwise features"):
        s1_row = s1_dict.get(sid)
        cand_row = other_dict.get(cid)
        if s1_row is not None and cand_row is not None:
            feats = compute_pair_features(s1_row, cand_row)
            feats["source1_entity_id"] = sid
            feats["candidate_entity_id"] = cid
            pair_rows.append(feats)

    cand_feat_df = pd.DataFrame(pair_rows)
    cand_feat_df = add_rank_and_margin_features(cand_feat_df, score_col="joint_confidence")

    feature_cols = get_feature_columns(cand_feat_df)
    X_test = cand_feat_df[feature_cols]

    print("--- Computing Ensemble Predictions ---")
    probs = np.zeros(len(cand_feat_df), dtype=np.float32)
    for model in models:
        probs += model.predict_proba(X_test)[:, 1]
    probs /= len(models)

    cand_feat_df["pred_score"] = probs

    print(f"--- Applying Consistency Resolution (Threshold = {threshold:.2f}) ---")
    raw_matches = resolve_global_conflicts(cand_feat_df, threshold=threshold)
    matched_dict = apply_singleton_rule(raw_matches, all_s1_ids)

    return candidate_dict, matched_dict, all_s1_ids
