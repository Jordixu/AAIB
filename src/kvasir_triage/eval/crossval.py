"""Aggregate metrics across one or more validation runs."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from kvasir_triage.utils import load_json, save_json


def aggregate_folds(fold_metrics: list[dict]) -> dict:
    """Compute mean and std of each scalar metric across runs."""
    keys = [k for k in fold_metrics[0] if isinstance(fold_metrics[0][k], (int, float))]
    agg: dict = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if m[k] is not None]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg


def load_fold_metrics(run_dirs: list[Path]) -> list[dict]:
    """Load threshold_calibration.json from each run dir."""
    out = []
    for d in run_dirs:
        p = d / "threshold_calibration.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing threshold calibration: {p}")
        out.append(load_json(p))
    return out


def write_cv_summary(fold_metrics: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    agg = aggregate_folds(fold_metrics)
    save_json({"folds": fold_metrics, "aggregate": agg}, out_dir / "cv_metrics.json")
