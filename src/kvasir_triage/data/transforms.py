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
    """Augmentation for training. No Hue shift, no ElasticTransform.

    RandomRotation uses fill=0 so the corner mask handles the black border
    naturally - no additional correction needed.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=360, border_mode=0, p=0.7),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def build_eval_transform(mean: list[float], std: list[float]) -> A.Compose:
    """No augmentation, just normalize."""
    return A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def make_transforms_for_fold(
    norm_stats: dict,
    train_fold: int,
) -> tuple[A.Compose, A.Compose]:
    """Return (train_transform, eval_transform) for a given training fold.

    train_fold=0 means fold 0 is train (use fold_0 stats), fold 1 is val.
    train_fold=1 means fold 1 is train (use fold_1 stats), fold 0 is val.
    """
    key = f"fold_{train_fold}"
    mean, std = _norm_stats(norm_stats[key])
    return build_train_transform(mean, std), build_eval_transform(mean, std)


class AlbumentationsWrapper:
    """Wrap an albumentations Compose so it accepts uint8 HWC ndarray."""

    def __init__(self, transform) -> None:
        self.transform = transform

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        result = self.transform(image=image)
        return result["image"]
