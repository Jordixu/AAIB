"""Atomic checkpoint save/load - keeps best.pt and last.pt."""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch


def save_checkpoint(state: dict, path: Path) -> None:
    """Write atomically - use a temp file then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        torch.save(state, tmp)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def load_checkpoint(path: Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def save_best_and_last(
    state: dict,
    run_dir: Path,
    is_best: bool,
) -> None:
    # last.pt strips optimizer/scaler to halve disk usage; best.pt keeps full state for resuming
    last_state = {k: v for k, v in state.items() if k not in ("optimizer", "scaler")}
    save_checkpoint(last_state, run_dir / "checkpoints" / "last.pt")
    if is_best:
        save_checkpoint(state, run_dir / "checkpoints" / "best.pt")
