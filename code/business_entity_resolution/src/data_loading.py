"""
Stage 1: Data Ingestion and Diagnostic Profiling.
Loads and validates Source-1, Source-2, Source-3, and Ground Truth TSVs.
Supports both standard and competition column variants (business_name / name).
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd


def resolve_file_path(filename: str, project_root: Path) -> Path:
    """Finds file in dataset/train/, dataset/test/, or directly in project_root."""
    candidates = [
        project_root / "dataset" / "train" / filename,
        project_root / "dataset" / "test" / filename,
        project_root / filename,
        project_root / "dataset" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return project_root / filename


def load_source_df(file_path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Loads source entity TSV into DataFrame with standardized column names:
    - entity_id
    - name (aliased from business_name or name)
    - address (aliased from business_address or address)
    - country
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Source file not found at: {file_path}")

    df = pd.read_csv(file_path, sep="\t", nrows=nrows, dtype=str).fillna("")

    # Standardize column naming
    rename_map = {}
    for col in df.columns:
        c_lower = col.strip().lower()
        if c_lower in ("business_name", "company_name", "businessname"):
            rename_map[col] = "name"
        elif c_lower in ("business_address", "address_text", "businessaddress"):
            rename_map[col] = "address"
        elif c_lower in ("source1_entity_id", "source_1_id", "s1_id"):
            rename_map[col] = "entity_id"

    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure required columns exist
    for req in ["entity_id", "name", "address", "country"]:
        if req not in df.columns:
            df[req] = ""

    return df


def load_ground_truth(file_path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Loads train_ground_truth.tsv and parses matched entity IDs into lists.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Ground truth file not found at: {file_path}")

    df = pd.read_csv(file_path, sep="\t", nrows=nrows, dtype=str).fillna("")

    # Standardize columns
    if "source1_entity_id" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "source1_entity_id"})

    match_col = None
    for col in df.columns:
        if "matched" in col.lower() or "candidate" in col.lower() or "target" in col.lower():
            match_col = col
            break

    if match_col and match_col != "matched_entity_ids":
        df = df.rename(columns={match_col: "matched_entity_ids"})

    if "matched_entity_ids" not in df.columns:
        df["matched_entity_ids"] = ""

    # Parse match list
    df["match_list"] = df["matched_entity_ids"].apply(
        lambda x: [m.strip() for m in str(x).split(",") if m.strip()] if str(x).strip() else []
    )
    df["n_matches"] = df["match_list"].apply(len)

    return df


def inspect_dataset_stats(s1_df: pd.DataFrame, s2_df: pd.DataFrame,
                          s3_df: pd.DataFrame, gt_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Computes diagnostic statistics and distributions across datasets."""
    stats = {
        "source1_shape": s1_df.shape,
        "source2_shape": s2_df.shape,
        "source3_shape": s3_df.shape,
        "source1_countries": s1_df["country"].value_counts().to_dict(),
        "source2_countries": s2_df["country"].value_counts().to_dict(),
        "source3_countries": s3_df["country"].value_counts().to_dict(),
    }

    if gt_df is not None:
        total_s1 = len(gt_df)
        singletons = (gt_df["n_matches"] == 0).sum()
        stats["ground_truth_total_s1"] = total_s1
        stats["singleton_count"] = int(singletons)
        stats["singleton_ratio"] = float(singletons / total_s1) if total_s1 > 0 else 0.0
        stats["match_count_distribution"] = gt_df["n_matches"].value_counts().head(10).to_dict()

    return stats


def load_matched_sample(s1_path: Path, s2_path: Path, s3_path: Path, gt_path: Path,
                        sample_size: int = 10000, n_distractors: int = 25000
                        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Subsamples ground truth and extracts all corresponding Source-1, Source-2, and Source-3
    entities so that true matches are 100% preserved in the evaluation pool, plus realistic distractors.
    """
    gt_df = load_ground_truth(gt_path, nrows=sample_size)
    s1_needed = set(gt_df["source1_entity_id"])
    cands_needed = set(gt_df.explode("match_list")["match_list"].dropna())
    s2_needed = {c for c in cands_needed if c.startswith("S2-")}
    s3_needed = {c for c in cands_needed if c.startswith("S3-")}

    # Stream S1
    cols = ["entity_id", "name", "address", "country"]
    s1_rows = []
    with open(s1_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if parts and parts[0] in s1_needed:
                s1_rows.append(parts[:4])
                if len(s1_rows) == len(s1_needed):
                    break
    s1_df = pd.DataFrame(s1_rows, columns=cols)

    # Stream S2
    s2_rows = []
    with open(s2_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if not parts:
                continue
            is_target = parts[0] in s2_needed
            if is_target or len(s2_rows) < n_distractors:
                s2_rows.append(parts[:4])
                if is_target:
                    s2_needed.discard(parts[0])
            if not s2_needed and len(s2_rows) >= n_distractors:
                break
    s2_df = pd.DataFrame(s2_rows, columns=cols)

    # Stream S3
    s3_rows = []
    with open(s3_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if not parts:
                continue
            is_target = parts[0] in s3_needed
            if is_target or len(s3_rows) < n_distractors:
                s3_rows.append(parts[:4])
                if is_target:
                    s3_needed.discard(parts[0])
            if not s3_needed and len(s3_rows) >= n_distractors:
                break
    s3_df = pd.DataFrame(s3_rows, columns=cols)

    return s1_df, s2_df, s3_df, gt_df

