"""Grad-CAM interpretability using pytorch-grad-cam library."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
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


def run_gradcam_audit(
    model: nn.Module,
    model_name: str,
    loader,
    out_dir: Path,
    device: torch.device,
    max_samples: int = 200,
) -> Path:
    """Run Grad-CAM on True Positives with bboxes. Saves CSV + thumbnails."""
    if not _GRADCAM_AVAILABLE:
        raise RuntimeError("Install grad-cam: pip install grad-cam")

    target_layer = _get_target_layer(model, model_name)
    cam = GradCAM(model=model, target_layers=[target_layer])

    out_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir = out_dir / "thumbnails"
    thumb_dir.mkdir(exist_ok=True)

    rows: list[dict[str, Any]] = []
    n = 0

    for imgs, labels, meta in loader:
        if n >= max_samples:
            break
        imgs_dev = imgs.to(device)
        with torch.no_grad():
            logits = model(imgs_dev).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)

        for i in range(len(imgs)):
            lbl = int(labels[i])
            pred = preds[i]
            if lbl != 1 or pred != 1:
                continue  # only True Positives

            inp = imgs[i:i+1].to(device)
            heatmap = cam(input_tensor=inp)[0]  # (H, W)

            mass = None  # bbox masks not available at this stage

            row = {
                "subclass": meta["subclass"][i],
                "rel_path": meta["rel_path"][i],
                "prob": float(probs[i]),
                "activation_mass": mass,
            }
            rows.append(row)

            # save thumbnail
            img_np = (imgs[i].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            vis = show_cam_on_image(img_np / 255.0, heatmap, use_rgb=True)
            fname = thumb_dir / f"tp_{n:04d}.jpg"
            Image.fromarray(vis).save(fname)
            n += 1

    csv_path = out_dir / "activation_mass.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return csv_path
