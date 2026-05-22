from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .constants import DEFAULT_MODEL


class FeatureClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, images: torch.Tensor, return_features: bool = False):
        features = self.backbone(images)
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        logits = self.classifier(features)
        return (logits, features) if return_features else logits


class TinyBackbone(nn.Module):
    num_features = 32

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, self.num_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.num_features),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.features(images), start_dim=1)


def build_model(
    model_name: str = DEFAULT_MODEL,
    num_classes: int = 65,
    pretrained: bool = True,
    weights_dir: str | Path | None = None,
) -> FeatureClassifier:
    if model_name == "tiny_cnn":
        backbone = TinyBackbone()
        return FeatureClassifier(backbone, backbone.num_features, num_classes)
    import timm

    cache_dir = None if weights_dir is None else str(Path(weights_dir).expanduser().resolve())
    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,
        cache_dir=cache_dir,
    )
    feature_dim = getattr(backbone, "num_features", None)
    if feature_dim is None:
        raise ValueError(f"Cannot infer feature dimension from timm model {model_name}")
    return FeatureClassifier(backbone, int(feature_dim), num_classes)
