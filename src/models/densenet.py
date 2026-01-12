from __future__ import annotations

from typing import Literal

from torch import nn
from torchvision import models


def xavier_init_(m: nn.Module) -> None:
    """Xavier init for Conv2d/Linear weights + zero biases (matches notebook intent)."""
    for module in m.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if getattr(module, "weight", None) is not None:
                nn.init.xavier_normal_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias, 0)


FineTuneMode = Literal["none", "last_block", "all"]


def build_densenet121(
    num_classes: int,
    pretrained: bool = True,
    fine_tune: FineTuneMode = "none",
    dropout: float = 0.3,
    hidden_dim: int = 512,
) -> nn.Module:
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet121(weights=weights)

    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    )

    for p in model.features.parameters():
        p.requires_grad = False

    if fine_tune == "all":
        for p in model.features.parameters():
            p.requires_grad = True
    elif fine_tune == "last_block":
        for p in model.features.denseblock4.parameters():
            p.requires_grad = True
        for p in model.features.norm5.parameters():
            p.requires_grad = True
    elif fine_tune == "none":
        pass
    else:
        raise ValueError(f"Unknown fine_tune={fine_tune}")

    return model


def build_from_scratch(num_classes: int, dropout: float = 0.3, hidden_dim: int = 512) -> nn.Module:
    """DenseNet121 trained from scratch (no ImageNet weights). Applies Xavier init."""
    model = models.densenet121(weights=None)
    xavier_init_(model)

    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    )
    return model
