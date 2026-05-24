"""Shared classification head builders."""
import torch.nn as nn


def make_head(in_features: int, dropout: float = 0.0) -> nn.Sequential:
    """AdaptiveAvgPool2d(1) -> Flatten -> (optional Dropout) -> Linear(in_features, 1)."""
    layers: list[nn.Module] = [
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(in_features, 1))
    return nn.Sequential(*layers)
