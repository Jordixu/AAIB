"""Resolve all project paths without any hardcoded roots."""
from pathlib import Path


def project_root() -> Path:
    """Walk up from this file until we find pyproject.toml."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("pyproject.toml not found - are you inside the project?")


def outputs_root() -> Path:
    return project_root() / "outputs"


def preprocessed_root() -> Path:
    return outputs_root() / "preprocessed"


def shards_root() -> Path:
    return preprocessed_root() / "shards"


def runs_root() -> Path:
    return outputs_root() / "runs"


def cv_summary_root() -> Path:
    return outputs_root() / "cv_summary"


def eda_root() -> Path:
    return outputs_root() / "eda"


def manifest_path() -> Path:
    return preprocessed_root() / "manifest.csv"


def norm_stats_path() -> Path:
    return preprocessed_root() / "normalization_stats.json"


def corner_mask_path() -> Path:
    return preprocessed_root() / "corner_keep_mask_336.npy"


def configs_root() -> Path:
    return project_root() / "configs"


def logs_root() -> Path:
    return outputs_root() / "logs"
