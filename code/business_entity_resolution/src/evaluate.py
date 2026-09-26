"""
Stage 9: Local F0.5 Evaluation Harness.
Replicates the official competition scorer macro-averaged per Source-1 entity.
"""

from typing import Dict, Iterable, Set
import numpy as np
import pandas as pd


def entity_f05(pred_ids: Set[str], true_ids: Set[str]) -> float:
    """
    Computes F0.5 score for a single Source-1 entity.
    F0.5 = (1.25 * P * R) / (0.25 * P + R)
    - Singletons (true_ids is empty): 1.0 if pred_ids is empty else 0.0
    - If true_ids is non-empty and pred_ids is empty: 0.0
    """
    pred_set = {str(x).strip() for x in pred_ids if str(x).strip()}
    true_set = {str(x).strip() for x in true_ids if str(x).strip()}

    if not true_set:
        return 1.0 if not pred_set else 0.0
    if not pred_set:
        return 0.0

    tp = len(pred_set & true_set)
    precision = tp / len(pred_set)
    recall = tp / len(true_set)

    if precision == 0.0 and recall == 0.0:
        return 0.0

    denominator = 0.25 * precision + recall
    if denominator == 0.0:
        return 0.0

    return (1.25 * precision * recall) / denominator


def compute_macro_f05(pred_dict: Dict[str, Iterable[str]], ground_truth_df: pd.DataFrame) -> float:
    """
    Computes macro-averaged F0.5 across all Source-1 entities present in ground_truth_df.
    ground_truth_df must contain 'source1_entity_id' and either 'match_list' or 'matched_entity_ids'.
    """
    scores = []
    
    # Pre-parse ground truth if match_list is not already a list
    if "match_list" not in ground_truth_df.columns:
        gt_df = ground_truth_df.copy()
        gt_df["match_list"] = gt_df["matched_entity_ids"].fillna("").apply(
            lambda x: [m.strip() for m in str(x).split(",") if m.strip()] if str(x).strip() else []
        )
    else:
        gt_df = ground_truth_df

    for _, row in gt_df.iterrows():
        sid = str(row["source1_entity_id"]).strip()
        true_ids = set(row["match_list"])
        pred_ids = set(pred_dict.get(sid, []))
        score = entity_f05(pred_ids, true_ids)
        scores.append(score)

    return float(np.mean(scores)) if scores else 0.0


def compute_blocking_metrics(candidate_dict: Dict[str, Iterable[str]], ground_truth_df: pd.DataFrame,
                             total_s2_count: int = 0, total_s3_count: int = 0) -> Dict[str, float]:
    """
    Computes blocking recall and reduction ratio on ground truth.
    """
    if "match_list" not in ground_truth_df.columns:
        gt_df = ground_truth_df.copy()
        gt_df["match_list"] = gt_df["matched_entity_ids"].fillna("").apply(
            lambda x: [m.strip() for m in str(x).split(",") if m.strip()] if str(x).strip() else []
        )
    else:
        gt_df = ground_truth_df

    gt_pairs = set()
    for _, row in gt_df.iterrows():
        sid = str(row["source1_entity_id"]).strip()
        for m in row["match_list"]:
            gt_pairs.add((sid, m))

    total_candidates = 0
    hits = 0
    for sid, cands in candidate_dict.items():
        cand_set = {str(c).strip() for c in cands if str(c).strip()}
        total_candidates += len(cand_set)
        for c in cand_set:
            if (sid, c) in gt_pairs:
                hits += 1

    recall = hits / len(gt_pairs) if gt_pairs else 1.0

    total_s1 = len(gt_df)
    total_comparisons = total_s1 * (total_s2_count + total_s3_count) if (total_s2_count + total_s3_count) > 0 else 0
    reduction_ratio = 1.0 - (total_candidates / total_comparisons) if total_comparisons > 0 else 0.0

    return {
        "recall": recall,
        "hits": hits,
        "total_true_pairs": len(gt_pairs),
        "total_candidate_pairs": total_candidates,
        "reduction_ratio": reduction_ratio
    }


if __name__ == "__main__":
    # Sanity check against PRD specification:
    # entity_f05({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"}) == 0.714
    sample_val = entity_f05({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    assert abs(sample_val - 0.714) < 0.002, f"Assertion failed: expected ~0.714, got {sample_val}"
    print(f"Harness sanity check passed! Sample F0.5: {sample_val:.4f}")
