"""Weighted sampler for class-balanced training."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def make_weighted_sampler(labels: list[int] | np.ndarray) -> WeightedRandomSampler:
    """Return a sampler that gives each class equal expected frequency.

    With LRU shard mode this thrashes the cache because sample order is fully
    random. Use KvasirInMemoryDataset when possible, or increase lru_size.
    """
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    class_weight = 1.0 / counts
    weight_map = dict(zip(classes, class_weight))
    sample_weight = np.array([weight_map[l] for l in labels])
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weight).double(),
        num_samples=len(sample_weight),
        replacement=True,
    )