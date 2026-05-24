"""DenseNet-161 variants for VCE triage."""
import torch.nn as nn
from torchvision.models import densenet161, DenseNet161_Weights


def _remove_first_downsampling(features: nn.Module) -> None:
    """Patch features.conv0 stride (2,2)->(1,1) and remove features.pool0.

    EDA showed bbox p10 < 2% of frame area, meaning small lesions occupy
    very few pixels. Striding by 2 in the first block halves spatial resolution
    before any feature extraction happens, which hurts recall for those cases.
    """
    features.conv0.stride = (1, 1)
    features.pool0 = nn.Identity()


def build_densenet161(
    pretrained: bool = True,
    drop_first_downsampling: bool = True,
    dropout: float = 0.0,
) -> nn.Module:
    """DenseNet-161 with a binary classification head (outputs 1 logit).

    torchvision's DenseNet applies adaptive_avg_pool2d + flatten before
    the classifier, so we only need (optional Dropout +) Linear here.
    """
    weights = DenseNet161_Weights.IMAGENET1K_V1 if pretrained else None
    model = densenet161(weights=weights)

    if drop_first_downsampling:
        _remove_first_downsampling(model.features)

    layers: list[nn.Module] = []
    if dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(2208, 1))
    model.classifier = nn.Sequential(*layers) if len(layers) > 1 else layers[0]
    return model
