"""Conditional BCE -> Focal loss switch based on per-subclass recall."""
from __future__ import annotations

import torch.nn as nn

from kvasir_triage.losses.bce import BCELoss
from kvasir_triage.losses.focal import BinaryFocalLoss


# Subclasses in each group for the conditional check
SOFT_TISSUE = {"Erythema", "Erosion"}
BLEEDING = {"Blood - fresh"}

# Recall thresholds that trigger the switch
SOFT_TISSUE_RECALL_THRESHOLD = 0.85
BLEEDING_RECALL_THRESHOLD = 0.95


class ConditionalLoss:
    """Wraps the active loss and hot-swaps to Focal when soft-tissue recall falls.

    After each validation epoch call .maybe_switch(per_subclass_recall) to
    check whether the switch condition is met.
    """

    def __init__(self, cfg: dict) -> None:
        self.active: nn.Module = BCELoss(pos_weight=cfg.get("bce_pos_weight"))
        self.switched = False
        self._focal_gamma = cfg.get("focal_gamma", 2.0)
        self._focal_alpha = cfg.get("focal_alpha", 0.25)

    def __call__(self, logits, targets):
        return self.active(logits, targets)

    def parameters(self):
        return self.active.parameters()

    def maybe_switch(self, per_subclass_recall: dict[str, float]) -> bool:
        """Switch to Focal if conditions are met. Returns True if switched this call."""
        if self.switched:
            return False
        soft_recall = max(
            (per_subclass_recall.get(c, 1.0) for c in SOFT_TISSUE), default=1.0
        )
        bleed_recall = min(
            (per_subclass_recall.get(c, 0.0) for c in BLEEDING), default=0.0
        )
        if soft_recall < SOFT_TISSUE_RECALL_THRESHOLD and bleed_recall > BLEEDING_RECALL_THRESHOLD:
            self.active = BinaryFocalLoss(gamma=self._focal_gamma, alpha=self._focal_alpha)
            self.switched = True
            return True
        return False

    @property
    def name(self) -> str:
        return "focal" if self.switched else "bce"
