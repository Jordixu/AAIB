"""Standard BCE loss wrapper."""
import torch
import torch.nn as nn


class BCELoss(nn.Module):
    """BCEWithLogitsLoss with optional pos_weight for class imbalance."""

    def __init__(self, pos_weight: float | None = None) -> None:
        super().__init__()
        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits.squeeze(1), targets.float())
