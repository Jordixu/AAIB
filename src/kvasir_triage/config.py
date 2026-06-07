"""Load YAML configs and expose them as a plain dict (or dot-access namespace)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from kvasir_triage.paths import configs_root


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base, recursing into nested dicts."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(config_path: Path | str, overrides: dict[str, Any] | None = None) -> dict:
    """Load a YAML config, merging base.yaml first, then applying any overrides."""
    config_path = Path(config_path)
    base_path = configs_root() / "base.yaml"

    cfg: dict = {}
    if base_path.exists():
        with open(base_path) as f:
            cfg = yaml.safe_load(f) or {}

    with open(config_path) as f:
        specific = yaml.safe_load(f) or {}

    cfg = _deep_merge(cfg, specific)

    if overrides:
        cfg = _deep_merge(cfg, overrides)

    return cfg
