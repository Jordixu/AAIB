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
            if lbl != 1 or pred != 1:
                continue  # only True Positives

            inp = imgs[i:i+1].to(device)
            heatmap = cam(input_tensor=inp, targets=targets)[0]  # (H, W)

            mass = None  # bbox masks not available at this stage

            row = {
                "subclass": meta["subclass"][i],
                "rel_path": meta["rel_path"][i],
                "prob": float(probs[i]),
                "activation_mass": mass,
            }

            # save thumbnail
            img_np = _denormalize_image(imgs[i], mean, std)
            vis = show_cam_on_image(img_np.astype(np.float32), heatmap, use_rgb=True)
            fname = thumb_dir / f"tp_{n:04d}.jpg"
            Image.fromarray(vis).save(fname)
            row["thumbnail_path"] = str(fname.relative_to(out_dir))
            rows.append(row)
            n += 1

    csv_path = out_dir / "activation_mass.csv"
    fieldnames = ["subclass", "rel_path", "prob", "activation_mass", "thumbnail_path"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path
