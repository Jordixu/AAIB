"""Weighted samplers for balanced or rare-class oversampling."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def make_weighted_sampler(
    groups: list[int | str] | np.ndarray,
    weight_power: float = 1.0,
    num_samples_multiplier: float = 1.0,
) -> WeightedRandomSampler:
    """Return a sampler that gives each group equal expected frequency.

    With LRU shard mode this thrashes the cache because sample order is fully
    random. Use KvasirInMemoryDataset when possible, or increase lru_size.

    weight_power controls how aggressively rare groups are oversampled:
      1.0 = inverse frequency, 0.5 = sqrt inverse frequency, 0.0 = uniform.
    """
    groups = np.asarray(groups)
    unique_groups, counts = np.unique(groups, return_counts=True)
    group_weight = (1.0 / counts.astype(np.float64)) ** weight_power
    weight_map = dict(zip(unique_groups, group_weight))
    sample_weight = np.array([weight_map[group] for group in groups])
    num_samples = max(1, int(round(len(sample_weight) * num_samples_multiplier)))
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weight).double(),
        num_samples=num_samples,
        replacement=True,
    )
