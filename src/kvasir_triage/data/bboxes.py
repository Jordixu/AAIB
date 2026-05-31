"""Load Kvasir boundary-box annotations as normalized polygons."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _read_metadata(metadata_csv: Path) -> pd.DataFrame:
    """Read the semicolon-separated Kvasir metadata with a CSV fallback."""
    try:
        md = pd.read_csv(metadata_csv, sep=";")
        if len(md.columns) == 1:
            md = pd.read_csv(metadata_csv)
    except Exception:
        md = pd.read_csv(metadata_csv)
    return md


def load_bbox_annotations(metadata_csv: Path | str, image_size: int = 336) -> pd.Series:
    """Return JSON-encoded normalized boundary polygons keyed by image stem."""
    metadata_csv = Path(metadata_csv)
    md = _read_metadata(metadata_csv)
    x_cols = ["x1", "x2", "x3", "x4"]
    y_cols = ["y1", "y2", "y3", "y4"]
    required = ["filename", *x_cols, *y_cols]
    if not all(column in md.columns for column in required):
        return pd.Series(dtype=str)

    annotations: dict[str, list[list[list[float]]]] = {}
    for _, row in md.iterrows():
        points = []
        for x_col, y_col in zip(x_cols, y_cols):
            x = pd.to_numeric(row[x_col], errors="coerce")
            y = pd.to_numeric(row[y_col], errors="coerce")
            if pd.isna(x) or pd.isna(y):
                points = []
                break
            points.append([
                float(max(0.0, min(1.0, float(x) / image_size))),
                float(max(0.0, min(1.0, float(y) / image_size))),
            ])
        if not points:
            continue
        stem = Path(str(row["filename"]).replace("\\", "/")).stem
        annotations.setdefault(stem, []).append(points)

    return pd.Series({
        stem: json.dumps(polygons, separators=(",", ":"))
        for stem, polygons in annotations.items()
    })
