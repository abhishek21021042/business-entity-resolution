"""
Stage 10: Output Generation and Export.
Generates output/candidate_pairs.tsv and output/matching_results.tsv
matching exact competition specifications.
"""

from pathlib import Path
from typing import Dict, List, Iterable
import pandas as pd

from .config import CANDIDATE_OUTPUT_PATH, MATCHING_OUTPUT_PATH


def export_tsv(data_dict: Dict[str, Iterable[str]],
               all_s1_ids: List[str],
               output_path: Path,
               id_col_name: str):
    """
    Exports a TSV file with columns:
    [source1_entity_id, id_col_name]
    where IDs are comma-separated strings (or empty string if singleton).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for sid in all_s1_ids:
        s_clean = str(sid).strip()
        ids = [str(x).strip() for x in data_dict.get(s_clean, []) if str(x).strip()]
        # Deduplicate while preserving order
        seen = set()
        dedup_ids = []
        for x in ids:
            if x not in seen:
                seen.add(x)
                dedup_ids.append(x)
        rows.append({
            "source1_entity_id": s_clean,
            id_col_name: ",".join(dedup_ids)
        })

    df = pd.DataFrame(rows, columns=["source1_entity_id", id_col_name])
    df.to_csv(output_path, sep="\t", index=False)
    print(f"Exported {len(df)} rows to: {output_path}")


def save_submission_artifacts(candidate_dict: Dict[str, Iterable[str]],
                              matched_dict: Dict[str, Iterable[str]],
                              all_s1_ids: List[str],
                              candidate_out: Path = CANDIDATE_OUTPUT_PATH,
                              matching_out: Path = MATCHING_OUTPUT_PATH):
    """Generates both required submission TSVs."""
    export_tsv(candidate_dict, all_s1_ids, candidate_out, "candidate_entity_ids")
    export_tsv(matched_dict, all_s1_ids, matching_out, "matched_entity_ids")
