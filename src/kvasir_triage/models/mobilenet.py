"""MobileNetV3 variants for VCE triage."""
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


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


def build_mobilenet_v3_large(pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    """MobileNetV3-Large with a binary classification head."""
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    model = mobilenet_v3_large(weights=weights)
    in_features = model.classifier[0].in_features
    hidden_features = model.classifier[0].out_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, hidden_features),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(hidden_features, 1),
    )
    return model
