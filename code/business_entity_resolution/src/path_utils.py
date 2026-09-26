"""
Smart path auto-detection utility.
Detects test datasets, output directories, and caches across any machine
(e.g., D:\\test_data, D:\\output, or relative paths inside the project folder).
"""

from pathlib import Path
import os
import sys

_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _CURRENT_DIR.parent
_WORKSPACE_ROOT = _PROJECT_DIR.parent.parent


def get_project_root() -> Path:
    return _PROJECT_DIR


def detect_test_dir(custom_path=None) -> Path:
    """Finds test data directory containing test_source1.tsv."""
    if custom_path:
        p = Path(custom_path)
        if (p / "test_source1.tsv").exists() or (p / "train_source1.tsv").exists():
            return p

    candidates = [
        Path("D:/test_data"),
        Path("D:/"),
        _WORKSPACE_ROOT / "dataset" / "test",
        _PROJECT_DIR / "dataset" / "test",
        _WORKSPACE_ROOT / "test_data",
        _PROJECT_DIR / "test_data",
        Path.cwd() / "dataset" / "test",
        Path.cwd() / "test_data",
        Path.cwd(),
    ]

    for c in candidates:
        if (c / "test_source1.tsv").exists():
            return c.resolve()

    # Fallback to train dataset if test not present
    for c in [_WORKSPACE_ROOT / "dataset" / "train", _PROJECT_DIR / "dataset" / "train"]:
        if (c / "train_source1.tsv").exists():
            return c.resolve()

    return Path("D:/test_data")


def detect_output_dir(custom_path=None) -> Path:
    """Finds or creates output directory. Uses D:/output if available, else local."""
    if custom_path:
        p = Path(custom_path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    # If D: exists, prefer D:/output
    d_drive = Path("D:/")
    if d_drive.exists():
        try:
            d_out = Path("D:/output")
            d_out.mkdir(parents=True, exist_ok=True)
            # Test write
            test_f = d_out / ".write_test"
            test_f.write_text("ok")
            test_f.unlink()
            return d_out
        except Exception:
            pass

    local_out = _WORKSPACE_ROOT / "output"
    local_out.mkdir(parents=True, exist_ok=True)
    return local_out.resolve()


def detect_cache_dir(output_dir: Path) -> Path:
    """Finds existing cache folder containing blocking_index.pkl or other_norm.pkl."""
    candidates = [
        output_dir / "_cache",
        Path("D:/output/_cache"),
        _WORKSPACE_ROOT / "output" / "_cache",
        _PROJECT_DIR / "output" / "_cache",
        _PROJECT_DIR / "_cache",
        Path.cwd() / "_cache",
    ]

    for c in candidates:
        if (c / "blocking_index.pkl").exists() or (c / "other_norm.pkl").exists():
            return c.resolve()

    # Default to output_dir / _cache
    target = output_dir / "_cache"
    target.mkdir(parents=True, exist_ok=True)
    return target.resolve()


def get_environment_info():
    """Returns a dictionary summarizing all detected locations."""
    test_d = detect_test_dir()
    out_d = detect_output_dir()
    cache_d = detect_cache_dir(out_d)

    has_s1 = (test_d / "test_source1.tsv").exists() or (test_d / "train_source1.tsv").exists()
    has_cache_norm = (cache_d / "other_norm.pkl").exists()
    has_cache_index = (cache_d / "blocking_index.pkl").exists()

    return {
        "test_dir": str(test_d),
        "test_dir_valid": has_s1,
        "output_dir": str(out_d),
        "cache_dir": str(cache_d),
        "has_norm_cache": has_cache_norm,
        "has_index_cache": has_cache_index,
        "cache_status": "READY (Instant Startup)" if (has_cache_norm and has_cache_index) else "Will be created on 1st run",
        "cpu_count": os.cpu_count() or 4
    }


if __name__ == "__main__":
    import json
    print(json.dumps(get_environment_info(), indent=2))
