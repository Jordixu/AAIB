"""Grad-CAM interpretability using pytorch-grad-cam library."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
    _GRADCAM_AVAILABLE = True
except ImportError:
    _GRADCAM_AVAILABLE = False


def _get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Return the layer to hook for Grad-CAM."""
    if model_name.startswith("densenet"):
        return model.features.denseblock4
    elif model_name.startswith("efficientnet"):
        # last MBConv block in EfficientNet-B0
        return model.features[-1]
    elif model_name.startswith("mobilenet"):
        return model.features[-1]
    elif model_name == "tiny":
        return model.features[-2]  # last ReLU
    raise ValueError(f"No Grad-CAM target layer known for model '{model_name}'")


def compute_activation_mass(
    heatmap: np.ndarray,
    bbox_mask: np.ndarray | None,
) -> float | None:
    """Fraction of Grad-CAM activation that falls inside the bounding box.

    Returns None if no bbox mask is provided.
    """
    if bbox_mask is None or heatmap.sum() == 0:
        return None
    # resize bbox_mask to heatmap size if needed
    h, w = heatmap.shape
    if bbox_mask.shape != (h, w):
        from PIL import Image as PILImage
        bm = PILImage.fromarray(bbox_mask.astype(np.uint8) * 255).resize((w, h), PILImage.NEAREST)
        bbox_mask = np.array(bm).astype(bool)
    return float((heatmap * bbox_mask).sum() / heatmap.sum())


def _denormalize_image(img: torch.Tensor, mean: list[float], std: list[float]) -> np.ndarray:
    """Convert a normalized CHW tensor back to RGB float HWC in [0, 1]."""
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    img_np = img.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip((img_np * std_arr) + mean_arr, 0.0, 1.0)


def _parse_bbox_annotations(value: Any) -> list[list[list[float]]]:
    """Return normalized polygons from the JSON value carried by the dataset."""
    if not isinstance(value, str) or not value:
        return []
    try:
        annotations = json.loads(value)
    except json.JSONDecodeError:
        return []
    return annotations if isinstance(annotations, list) else []


def _polygon_pixels(
    polygon: list[list[float]],
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    return [
        (
            int(round(float(x) * (width - 1))),
            int(round(float(y) * (height - 1))),
        )
        for x, y in polygon
    ]


def _bbox_mask(
    annotations: list[list[list[float]]],
    width: int,
    height: int,
) -> np.ndarray | None:
    if not annotations:
        return None
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in annotations:
        draw.polygon(_polygon_pixels(polygon, width, height), fill=255)
    return np.array(mask, dtype=bool)


def _draw_bbox_boundaries(
    image: np.ndarray,
    annotations: list[list[list[float]]],
) -> np.ndarray:
    """Draw the original dataset boundaries over a Grad-CAM RGB image."""
    rendered = Image.fromarray(image)
    draw = ImageDraw.Draw(rendered)
    for polygon in annotations:
        points = _polygon_pixels(polygon, rendered.width, rendered.height)
        if not points:
            continue
        closed = [*points, points[0]]
        draw.line(closed, fill=(0, 0, 0), width=5, joint="curve")
        draw.line(closed, fill=(80, 255, 80), width=3, joint="curve")
    return np.array(rendered)


def run_gradcam_audit(
    model: nn.Module,
    model_name: str,
    loader,
    out_dir: Path,
    device: torch.device,
    max_samples: int = 200,
    mean: list[float] | None = None,
    std: list[float] | None = None,
    threshold: float = 0.5,
    outcome: str = "true-positive",
    max_samples_per_class: int | None = None,
) -> Path:
    """Run Grad-CAM on selected positive-label outcomes. Saves CSV + thumbnails."""
    if not _GRADCAM_AVAILABLE:
        raise RuntimeError("Install grad-cam: pip install grad-cam")
    if outcome not in {"true-positive", "false-negative"}:
        raise ValueError(f"Unsupported Grad-CAM outcome: {outcome}")

    target_layer = _get_target_layer(model, model_name)
    cam = GradCAM(model=model, target_layers=[target_layer])

    out_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir = out_dir / "thumbnails"
    thumb_dir.mkdir(exist_ok=True)
    for old_thumbnail in thumb_dir.glob("*.jpg"):
        old_thumbnail.unlink()

    rows: list[dict[str, Any]] = []
    selected_per_subclass: dict[str, int] = defaultdict(int)
    n = 0
    mean = mean or [0.0, 0.0, 0.0]
    std = std or [1.0, 1.0, 1.0]
    targets = [BinaryClassifierOutputTarget(1)]

    for imgs, labels, meta in loader:
        if n >= max_samples:
            break
        imgs_dev = imgs.to(device)
        with torch.no_grad():
            logits = model(imgs_dev).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= threshold).astype(int)

        for i in range(len(imgs)):
            lbl = int(labels[i])
            pred = preds[i]
            is_selected = (
                (outcome == "true-positive" and lbl == 1 and pred == 1)
                or (outcome == "false-negative" and lbl == 1 and pred == 0)
            )
            if not is_selected:
                continue
            subclass = meta["subclass"][i]
            if (
                max_samples_per_class is not None
                and selected_per_subclass[subclass] >= max_samples_per_class
            ):
                continue
            annotations = _parse_bbox_annotations(meta["bbox_annotations"][i])
            if not annotations:
                continue

            inp = imgs[i:i+1].to(device)
            heatmap = cam(input_tensor=inp, targets=targets)[0]  # (H, W)

            bbox_mask = _bbox_mask(annotations, heatmap.shape[1], heatmap.shape[0])
            mass = compute_activation_mass(heatmap, bbox_mask)

            row = {
                "subclass": subclass,
                "rel_path": meta["rel_path"][i],
                "prob": float(probs[i]),
                "outcome": outcome,
                "bbox_count": len(annotations),
                "activation_mass": mass,
            }

            # save thumbnail
            img_np = _denormalize_image(imgs[i], mean, std)
            vis = show_cam_on_image(img_np.astype(np.float32), heatmap, use_rgb=True)
            vis = _draw_bbox_boundaries(vis, annotations)
            prefix = "tp" if outcome == "true-positive" else "fn"
            safe_subclass = subclass.lower().replace(" ", "_").replace("/", "_")
            class_idx = selected_per_subclass[subclass]
            fname = thumb_dir / f"{prefix}_{safe_subclass}_{class_idx:02d}.jpg"
            Image.fromarray(vis).save(fname)
            row["thumbnail_path"] = str(fname.relative_to(out_dir))
            rows.append(row)
            selected_per_subclass[subclass] += 1
            n += 1

    csv_path = out_dir / "activation_mass.csv"
    if rows:
        fieldnames = [
            "subclass",
            "rel_path",
            "prob",
            "outcome",
            "bbox_count",
            "activation_mass",
            "thumbnail_path",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return csv_path
