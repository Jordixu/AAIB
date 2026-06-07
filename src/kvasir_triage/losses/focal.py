"""Binary focal loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """Focal loss for binary classification (input is raw logits).

    gamma=0, alpha=None reduces to standard BCE, which is a useful sanity check.
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = 0.25) -> None:
        super().__init__()
        # 2.0 / 0.25 are the values from the focal loss paper - tried tuning
        # them once on a held-out sweep, didn't beat the defaults, left as-is
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.squeeze(1)
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        focal_term = (1 - p_t) ** self.gamma
        loss = focal_term * bce

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean()
