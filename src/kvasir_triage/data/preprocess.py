"""Image preprocessing: resize + mask + uint8 + write NPZ shards."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from kvasir_triage.data.masks import make_corner_polygon_mask, apply_mask


TARGET_SIZE = 336
SHARD_SIZE = 512


def preprocess_image_to_uint8(
    image_path: Path | str,
    mask: np.ndarray | None = None,
    target_size: int = TARGET_SIZE,
) -> np.ndarray:
    """Load, resize, apply corner mask, return uint8 HWC array."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((target_size, target_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.uint8)
    if mask is None:
        mask = make_corner_polygon_mask(target_size)
    return apply_mask(arr, mask)


def iter_shards(
    rows: list[dict],
    shard_size: int = SHARD_SIZE,
) -> Iterator[tuple[int, list[dict]]]:
    """Yield (shard_idx, batch_of_rows) without loading images."""
    for i in range(0, len(rows), shard_size):
        yield i // shard_size, rows[i:i + shard_size]


def write_npz_shard(
    rows: list[dict],
    out_path: Path,
    mask: np.ndarray | None = None,
    target_size: int = TARGET_SIZE,
) -> None:
    """Load images for this batch, preprocess, write uncompressed NPZ."""
    if mask is None:
        mask = make_corner_polygon_mask(target_size)

    images = []
    labels = []
    subclasses = []
    rel_paths = []
    bbox_norm_areas = []

    for row in rows:
        arr = preprocess_image_to_uint8(row["path"], mask, target_size)
        images.append(arr)
        labels.append(row["binary_label"])
        subclasses.append(row["subclass"])
        rel_paths.append(row["rel_path"])
        bbox_norm_areas.append(float(row.get("bbox_norm_area") or float("nan")))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_path),
        images=np.stack(images),        # (N, H, W, 3) uint8
        labels=np.array(labels, dtype=np.int8),
        subclasses=np.array(subclasses, dtype="<U64"),
        rel_paths=np.array(rel_paths, dtype="<U512"),
        bbox_norm_area=np.array(bbox_norm_areas, dtype=np.float32),
    )
