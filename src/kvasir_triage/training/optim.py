"""Optimizer and scheduler factories."""
import torch
import torch.nn as nn


def make_optimizer(
    model: nn.Module,
    cfg: dict,
    param_groups: list[dict] | None = None,
) -> torch.optim.Optimizer:
    params = param_groups if param_groups is not None else model.parameters()
    return torch.optim.AdamW(
        params,
        lr=cfg.get("lr", 3e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )


def make_scheduler(optimizer: torch.optim.Optimizer, cfg: dict):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",  # maximize val PR-AUC
        factor=cfg.get("lr_factor", 0.5),
        patience=cfg.get("lr_patience", 3),
        min_lr=cfg.get("min_lr", 1e-6),
    )
