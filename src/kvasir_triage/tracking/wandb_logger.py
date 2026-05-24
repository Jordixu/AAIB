"""W&B logger - no-op when wandb is disabled or not installed."""
from __future__ import annotations

from typing import Any


class WandbLogger:
    def __init__(self, cfg: dict, run_name: str) -> None:
        self._enabled = cfg.get("wandb_enabled", False)
        self._run = None
        if self._enabled:
            try:
                import wandb
                self._run = wandb.init(
                    project=cfg.get("wandb_project", "kvasir-triage"),
                    name=run_name,
                    config=cfg,
                )
            except Exception as e:
                print(f"W&B init failed, logging disabled: {e}")
                self._enabled = False

    def log_epoch(self, entry: dict) -> None:
        if self._run:
            self._run.log(entry)

    def log_step(self, entry: dict) -> None:
        if self._run:
            self._run.log(entry)

    def log_hardware(self, hw: dict) -> None:
        if self._run:
            self._run.config.update({"hardware": hw})

    def finish(self) -> None:
        if self._run:
            self._run.finish()
