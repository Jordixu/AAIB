"""Always-on local logger writing to metrics.jsonl."""
from __future__ import annotations

import json
from pathlib import Path


class JsonlLogger:
    """Appends one JSON line per epoch to metrics.jsonl."""

    def __init__(self, run_dir: Path) -> None:
        self._path = run_dir / "metrics.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log_epoch(self, entry: dict) -> None:
        with open(self._path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_step(self, entry: dict) -> None:
        # steps are too frequent for jsonl - skip
        pass

    def finish(self) -> None:
        pass
