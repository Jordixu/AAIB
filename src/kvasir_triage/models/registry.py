"""Build a model by name from config."""
import torch.nn as nn

from kvasir_triage.models import densenet as densenet_helpers
from kvasir_triage.models import efficientnet as efficientnet_helpers
from kvasir_triage.models.densenet import build_densenet161
from kvasir_triage.models.efficientnet import (
    build_efficientnet_b0,
    build_efficientnet_v2_s,
    build_efficientnet_v2_m,
    build_efficientnet_v2_l,
)
from kvasir_triage.models.tiny import build_tiny


_BUILDERS = {
    "densenet161_pretrained": lambda cfg: build_densenet161(
        pretrained=cfg.get("pretrained", True),
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
    "efficientnet_v2_s": lambda cfg: build_efficientnet_v2_s(
        pretrained=cfg.get("pretrained", True),
        dropout=cfg.get("dropout", 0.2),
    ),
    "efficientnet_v2_m": lambda cfg: build_efficientnet_v2_m(
        pretrained=cfg.get("pretrained", True),
        dropout=cfg.get("dropout", 0.2),
    ),
    "efficientnet_v2_l": lambda cfg: build_efficientnet_v2_l(
        pretrained=cfg.get("pretrained", True),
        dropout=cfg.get("dropout", 0.2),
    ),
    "tiny": lambda cfg: build_tiny(),
}

_TRANSFER_HELPERS = {
    "densenet161_pretrained": densenet_helpers,
    "densenet161_scratch": densenet_helpers,
    "efficientnet_b0": efficientnet_helpers,
    "efficientnet_v2_s": efficientnet_helpers,
    "efficientnet_v2_m": efficientnet_helpers,
    "efficientnet_v2_l": efficientnet_helpers,
}


def build_model(name: str, cfg: dict | None = None) -> nn.Module:
    """Instantiate a model by registry name. cfg is the full run config."""
    if name not in _BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_BUILDERS)}")
    return _BUILDERS[name](cfg or {})


def get_param_groups(
    model_name: str,
    model: nn.Module,
    backbone_lr: float,
    head_lr: float,
) -> list[dict]:
    helpers = _TRANSFER_HELPERS.get(model_name)
    if helpers is None:
        return [{"params": model.parameters(), "lr": head_lr}]
    return helpers.get_param_groups(model, backbone_lr, head_lr)


def freeze_backbone(model_name: str, model: nn.Module) -> None:
    helpers = _TRANSFER_HELPERS.get(model_name)
    if helpers is not None:
        helpers.freeze_backbone(model)


def unfreeze_backbone(model_name: str, model: nn.Module) -> None:
    helpers = _TRANSFER_HELPERS.get(model_name)
    if helpers is not None:
        helpers.unfreeze_backbone(model)
