import json
import time
from pathlib import Path
from typing import Any


def load_json(path: Path | str) -> Any:
    with open(path) as f:
        return json.load(f)


def save_json(obj: Any, path: Path | str, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)


class Timer:
    """Simple wall-clock stopwatch."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def reset(self) -> None:
        self._start = time.perf_counter()
