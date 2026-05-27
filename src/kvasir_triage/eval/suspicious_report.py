"""Suspicious-frame report helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _meta_value(meta: dict[str, list[Any]], key: str, idx: int, default: Any = "") -> Any:
    values = meta.get(key)
    if values is None:
        return default
    return values[idx]


def _gradcam_lookup(run_dir: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    gradcam_root = run_dir / "gradcam"
    if not gradcam_root.exists():
        return rows

    for csv_path in gradcam_root.rglob("activation_mass.csv"):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        for row_idx, row in df.iterrows():
            rel_path = row.get("rel_path")
            if not isinstance(rel_path, str) or not rel_path:
                continue
            thumb_path = row.get("thumbnail_path", "")
            if not isinstance(thumb_path, str) or not thumb_path:
                thumb_path = f"thumbnails/tp_{row_idx:04d}.jpg"
            thumb_abs = csv_path.parent / thumb_path
            rows[rel_path] = {
                "gradcam_csv": str(csv_path),
                "gradcam_thumbnail": str(thumb_abs),
                "gradcam_done": thumb_abs.exists(),
            }
    return rows


def build_suspicious_frame_report(
    logits: np.ndarray,
    labels: np.ndarray,
    metas: list[dict[str, Any]],
    run_dir: Path,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Return one row per frame predicted suspicious at the given threshold."""
    probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))
    preds = (probs >= threshold).astype(int)
    gradcams = _gradcam_lookup(run_dir)

    rows: list[dict[str, Any]] = []
    for idx, (prob, pred, label, meta) in enumerate(zip(probs, preds, labels, metas)):
        if int(pred) != 1:
            continue
        rel_path = str(meta.get("rel_path", ""))
        gradcam = gradcams.get(rel_path, {})
        rows.append({
            "rank": len(rows) + 1,
            "val_index": idx,
            "prob_suspicious": float(prob),
            "threshold": float(threshold),
            "predicted_label": "Suspicious",
            "true_binary_label": int(label),
            "is_true_positive": bool(int(label) == 1),
            "subclass": meta.get("subclass", ""),
            "original_path": meta.get("path", ""),
            "rel_path": rel_path,
            "filename": meta.get("filename", ""),
            "stem": meta.get("stem", ""),
            "fold": meta.get("fold", ""),
            "shard_rel_path": meta.get("shard_rel_path", ""),
            "local_idx": meta.get("local_idx", ""),
            "bbox_norm_area": meta.get("bbox_norm_area", ""),
            "gradcam_done": bool(gradcam.get("gradcam_done", False)),
            "gradcam_thumbnail": gradcam.get("gradcam_thumbnail", ""),
            "gradcam_csv": gradcam.get("gradcam_csv", ""),
        })

    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values("prob_suspicious", ascending=False).reset_index(drop=True)
        report["rank"] = np.arange(1, len(report) + 1)
    return report


def save_suspicious_frame_report(
    logits: np.ndarray,
    labels: np.ndarray,
    metas: list[dict[str, Any]],
    run_dir: Path,
    threshold: float = 0.5,
    stem: str = "suspicious_frames_best",
) -> dict[str, str]:
    """Save suspicious-frame report as CSV and XLSX when the Excel engine is installed."""
    report = build_suspicious_frame_report(logits, labels, metas, run_dir, threshold)
    csv_path = run_dir / f"{stem}.csv"
    report.to_csv(csv_path, index=False)

    paths = {"csv": str(csv_path)}
    xlsx_path = run_dir / f"{stem}.xlsx"
    try:
        report.to_excel(xlsx_path, index=False)
        paths["xlsx"] = str(xlsx_path)
    except Exception as exc:
        paths["xlsx_error"] = str(exc)
    return paths
