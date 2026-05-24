"""Tiny CNN for smoke tests - ~50k params, runs on CPU in seconds."""
import torch.nn as nn


class TinyCNN(nn.Module):
    """3 conv blocks + GAP + Linear. Only for smoke/CI testing."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.head(self.features(x))


def build_tiny() -> TinyCNN:
    return TinyCNN()
