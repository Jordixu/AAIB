"""Train and eval transform builders using albumentations + torchvision."""
from __future__ import annotations

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2


def _norm_stats(fold_stats: dict) -> tuple[list, list]:
    mean = fold_stats["mean"]
    std = fold_stats["std"]
    return mean, std


def build_train_transform(mean: list[float], std: list[float]) -> A.Compose:
    """Training augmentation with orientation, color, contrast, blur, and distortion.

    Rotation/distortion use fill=0 so the corner mask handles black borders.
    CLAHE boosts local contrast for subtle lesions; GridDistortion mimics bowel
    wall deformation during capsule endoscopy.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=360, border_mode=0, p=0.7),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.5),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, border_mode=0, p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def build_eval_transform(mean: list[float], std: list[float]) -> A.Compose:
    """No augmentation, just normalize."""
    return A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def build_tta_transforms(mean: list[float], std: list[float]) -> list[A.Compose]:
    """8 deterministic TTA variants: all combinations of h-flip, v-flip, transpose."""
    norm = [A.Normalize(mean=mean, std=std), ToTensorV2()]
    combos = []
    for hflip in (False, True):
        for vflip in (False, True):
            for transpose in (False, True):
                ops = []
                if hflip:
                    ops.append(A.HorizontalFlip(p=1.0))
                if vflip:
                    ops.append(A.VerticalFlip(p=1.0))
                if transpose:
                    ops.append(A.Transpose(p=1.0))
                combos.append(A.Compose(ops + norm))
    return combos


def make_transforms_for_fold(
    norm_stats: dict,
    train_fold: int,
) -> tuple[A.Compose, A.Compose]:
    """Return (train_transform, eval_transform) for a given training fold.

    train_fold=0 means fold 0 is train (use fold_0 stats), fold 1 is val.
    """
    key = f"fold_{train_fold}"
    mean, std = _norm_stats(norm_stats[key])
    return build_train_transform(mean, std), build_eval_transform(mean, std)


def make_tta_transforms_for_fold(
    norm_stats: dict,
    train_fold: int,
) -> list[A.Compose]:
    """Return deterministic TTA transforms using the training fold normalization."""
    key = f"fold_{train_fold}"
    mean, std = _norm_stats(norm_stats[key])
    return build_tta_transforms(mean, std)


class AlbumentationsWrapper:
    """Wrap an albumentations Compose so it accepts uint8 HWC ndarray."""

    def __init__(self, transform) -> None:
        self.transform = transform

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        result = self.transform(image=image)
        return result["image"]
