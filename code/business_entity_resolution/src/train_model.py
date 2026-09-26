"""
Stages 5 & 6: Matching Model Training and F0.5-Optimized Threshold Tuning.
Trains a Histogram Gradient Boosting Classifier (MIT licensed, LightGBM algorithm)
using GroupKFold cross-validation grouped by source1_entity_id.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from .config import RANDOM_SEED, THRESHOLD_GRID, MODELS_DIR
from .evaluate import compute_macro_f05
from .postprocess import resolve_global_conflicts, apply_singleton_rule


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Excludes ID, label, and text columns from training features."""
    excluded = {
        "source1_entity_id", "candidate_entity_id", "label",
        "pred_score", "score", "raw_name", "normalized_name",
        "core_name", "sorted_core", "raw_addr", "normalized_addr",
        "sorted_addr_tokens", "suffix", "postal_code", "landmark",
        "country"
    }
    return [c for c in df.columns if c not in excluded and not c.startswith("_")]


def train_and_evaluate_cv(candidates_df: pd.DataFrame,
                          ground_truth_df: pd.DataFrame,
                          n_splits: int = 8) -> Tuple[List[Any], pd.DataFrame, float, float]:
    """
    Runs GroupKFold cross-validation grouped by source1_entity_id.
    Returns:
    - Trained fold models
    - Candidates DataFrame populated with out-of-fold pred_score
    - Best threshold for macro F0.5
    - Best macro F0.5 score
    """
    feature_cols = get_feature_columns(candidates_df)
    X = candidates_df[feature_cols].copy()
    y = candidates_df["label"].values
    groups = candidates_df["source1_entity_id"].values

    gkf = GroupKFold(n_splits=n_splits)
    models = []
    oof_preds = np.zeros(len(candidates_df), dtype=np.float32)

    pos_count = int(np.sum(y == 1))
    neg_count = int(np.sum(y == 0))
    pos_weight = max(1.0, neg_count / max(pos_count, 1))

    print(f"Dataset stats: Total={len(candidates_df)}, Positives={pos_count}, Negatives={neg_count}, Class Ratio={pos_weight:.2f}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    import os
    from joblib import Parallel, delayed

    def _train_single_fold(fold_info):
        f_idx, tr_idx, v_idx = fold_info
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va = X.iloc[v_idx]
        sw = np.where(y_tr == 1, pos_weight, 1.0)
        c = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            early_stopping=True,
            n_iter_no_change=25,
            random_state=RANDOM_SEED + f_idx,
            class_weight="balanced"
        )
        c.fit(X_tr, y_tr, sample_weight=sw)
        vp = c.predict_proba(X_va)[:, 1]
        return f_idx, c, v_idx, vp

    # Downsample extreme negatives if dataset is huge to fit comfortably in RAM and avoid disk paging
    pos_mask = (y == 1)
    neg_mask = (y == 0)
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    if n_neg > 4 * n_pos and len(candidates_df) > 2_000_000:
        print(f"Sampling negatives to 4:1 ratio (keeping all {n_pos:,} positives + {int(4 * n_pos):,} negatives) for fast in-memory training...")
        rng = np.random.default_rng(RANDOM_SEED)
        neg_indices = np.where(neg_mask)[0]
        chosen_neg_indices = rng.choice(neg_indices, size=int(4 * n_pos), replace=False)
        keep_indices = np.sort(np.concatenate([np.where(pos_mask)[0], chosen_neg_indices]))
        
        candidates_df = candidates_df.iloc[keep_indices].reset_index(drop=True)
        X = candidates_df[feature_cols].copy()
        y = candidates_df["label"].values.astype(np.int32)
        groups = candidates_df["source1_entity_id"].values
        oof_preds = np.zeros(len(candidates_df), dtype=np.float32)

    fold_tasks = list(gkf.split(X, y, groups))
    print(f"Training {n_splits} folds in-memory (using multi-threaded CPU trees without disk memmapping)...")

    for f_idx, (tr_idx, va_idx) in enumerate(fold_tasks):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va = X.iloc[va_idx]
        sw = np.where(y_tr == 1, pos_weight, 1.0)
        c = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.06,
            max_iter=300,
            max_leaf_nodes=45,
            early_stopping=True,
            n_iter_no_change=25,
            random_state=RANDOM_SEED + f_idx,
            class_weight="balanced"
        )
        c.fit(X_tr, y_tr, sample_weight=sw)
        vp = c.predict_proba(X_va)[:, 1]
        models.append(c)
        oof_preds[va_idx] = vp
        print(f"Fold {f_idx + 1}/{n_splits} complete.")

    scored_df = candidates_df.copy()
    scored_df["pred_score"] = oof_preds

    # Tune threshold directly against macro F0.5 with post-processing
    all_s1_ids = list(ground_truth_df["source1_entity_id"].unique())
    best_t, best_f05 = tune_threshold_grid(scored_df, ground_truth_df, all_s1_ids)

    return models, scored_df, best_t, best_f05


def tune_threshold_grid(scored_candidates: pd.DataFrame,
                        ground_truth_df: pd.DataFrame,
                        all_s1_ids: List[str],
                        thresholds: List[float] = THRESHOLD_GRID) -> Tuple[float, float]:
    """
    Grid search for threshold maximizing macro F0.5.
    Applies global 1-to-many conflict resolution at each evaluation step.
    """
    best_t = 0.5
    best_f05 = -1.0

    print("\n--- Tuning Threshold on Out-Of-Fold Predictions ---")
    for t in thresholds:
        matches = resolve_global_conflicts(scored_candidates, threshold=t)
        complete_preds = apply_singleton_rule(matches, all_s1_ids)
        score = compute_macro_f05(complete_preds, ground_truth_df)

        if score > best_f05:
            best_f05 = score
            best_t = t

    print(f"Optimal F0.5 Threshold: {best_t:.2f} | Out-Of-Fold Macro F0.5: {best_f05:.4f}\n")
    return best_t, best_f05


def save_models(models: List[Any], threshold: float, save_dir: Path = MODELS_DIR):
    """Saves trained models and optimal threshold to disk."""
    save_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "num_models": len(models),
        "threshold": threshold,
    }
    joblib.dump(models, save_dir / "ensemble_models.joblib")
    joblib.dump(meta, save_dir / "meta.joblib")
    print(f"Saved {len(models)} models and meta to {save_dir}")
