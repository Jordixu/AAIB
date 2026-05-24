"""Build a model by name from config."""
import torch.nn as nn

from kvasir_triage.models.densenet import build_densenet161
from kvasir_triage.models.efficientnet import build_efficientnet_b0
from kvasir_triage.models.tiny import build_tiny


_BUILDERS = {
    "densenet161_pretrained": lambda cfg: build_densenet161(
        pretrained=True,
        drop_first_downsampling=cfg.get("drop_first_downsampling", True),
        dropout=cfg.get("dropout", 0.0),
    ),
    "densenet161_scratch": lambda cfg: build_densenet161(
        pretrained=False,
        drop_first_downsampling=cfg.get("drop_first_downsampling", True),
        dropout=cfg.get("dropout", 0.0),
    ),
    "efficientnet_b0": lambda cfg: build_efficientnet_b0(
        pretrained=cfg.get("pretrained", True),
        dropout=cfg.get("dropout", 0.2),
    ),
    "tiny": lambda cfg: build_tiny(),
}


def build_model(name: str, cfg: dict | None = None) -> nn.Module:
    """Instantiate a model by registry name. cfg is the full run config."""
    if name not in _BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_BUILDERS)}")
    return _BUILDERS[name](cfg or {})
