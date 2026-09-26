"""
Stages 7 & 8: Global Consistency Post-Processing and Singleton Handling.
Enforces deduplication constraints:
- Each Candidate entity (S2/S3) is assigned to at most ONE Source-1 entity (greedy score-sorted 'claim once').
- Source-1 entities with no qualified matches are cleanly preserved as singletons (empty list).
"""

from typing import Dict, List, Set, Iterable
import pandas as pd


def resolve_global_conflicts(candidates_df: pd.DataFrame,
                             threshold: float = 0.5) -> Dict[str, List[str]]:
    """
    Greedy score-sorted conflict resolution:
    1. Filter pairs with pred_score >= threshold.
    2. Sort pairs by pred_score descending.
    3. Greedily assign candidate_entity_id to at most one source1_entity_id.
    """
    if len(candidates_df) == 0:
        return {}

    score_col = "pred_score" if "pred_score" in candidates_df.columns else "score"
    if score_col not in candidates_df.columns:
        raise ValueError(f"Candidate DataFrame missing score column ({score_col})")

    above = candidates_df[candidates_df[score_col] >= threshold].copy()
    above = above.sort_values(score_col, ascending=False)

    assigned_candidates: Set[str] = set()
    final_matches: Dict[str, List[str]] = {}

    for _, row in above.iterrows():
        sid = str(row["source1_entity_id"]).strip()
        cid = str(row["candidate_entity_id"]).strip()

        # Invariant: Each S2/S3 entity can belong to at most one S1 reference entity
        if cid in assigned_candidates:
            continue

        assigned_candidates.add(cid)
        final_matches.setdefault(sid, []).append(cid)

    return final_matches


def apply_singleton_rule(final_matches: Dict[str, List[str]],
                         all_s1_ids: Iterable[str]) -> Dict[str, List[str]]:
    """
    Ensures every Source-1 ID in all_s1_ids is accounted for in output dictionary.
    Unmatched entities are explicitly mapped to an empty list [].
    """
    complete_dict = {}
    for sid in all_s1_ids:
        s_clean = str(sid).strip()
        complete_dict[s_clean] = final_matches.get(s_clean, [])
    return complete_dict
