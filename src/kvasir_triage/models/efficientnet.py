"""EfficientNet-B0 for VCE triage."""
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def build_efficientnet_b0(pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    """EfficientNet-B0 with a binary classification head (outputs 1 logit).

    torchvision's EfficientNet already applies AdaptiveAvgPool2d + flatten
    before the classifier, so we only need Dropout + Linear here.
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(1280, 1),
    )
    return model
