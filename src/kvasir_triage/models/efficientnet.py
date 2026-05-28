"""EfficientNet variants for VCE triage."""
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    efficientnet_v2_m, EfficientNet_V2_M_Weights,
    efficientnet_v2_l, EfficientNet_V2_L_Weights,
)


def _replace_head(model: nn.Module, in_features: int, dropout: float) -> nn.Module:
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 1),
    )
    return model


def get_param_groups(
    model: nn.Module,
    backbone_lr: float,
    head_lr: float,
) -> list[dict]:
    return [
        {"params": model.features.parameters(), "lr": backbone_lr},
        {"params": model.classifier.parameters(), "lr": head_lr},
    ]


def freeze_backbone(model: nn.Module) -> None:
    for param in model.features.parameters():
        param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    for param in model.features.parameters():
        param.requires_grad = True


def build_efficientnet_b0(pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    return _replace_head(efficientnet_b0(weights=weights), 1280, dropout)


def build_efficientnet_v2_s(pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
    return _replace_head(efficientnet_v2_s(weights=weights), 1280, dropout)


def build_efficientnet_v2_m(pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None
    return _replace_head(efficientnet_v2_m(weights=weights), 1280, dropout)


def build_efficientnet_v2_l(pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1 if pretrained else None
    return _replace_head(efficientnet_v2_l(weights=weights), 1280, dropout)
