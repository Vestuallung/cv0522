from __future__ import annotations

from itertools import combinations

import torch
from torch.nn import functional as F


def mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0:
        raise ValueError("Mixup alpha must be positive")
    lam = float(torch.distributions.Beta(alpha, alpha).sample(()))
    permutation = torch.randperm(images.shape[0], device=images.device)
    targets = F.one_hot(labels, num_classes=num_classes).to(dtype=images.dtype)
    mixed_images = lam * images + (1.0 - lam) * images[permutation]
    mixed_targets = lam * targets + (1.0 - lam) * targets[permutation]
    return mixed_images, mixed_targets


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def _covariance(features: torch.Tensor) -> torch.Tensor:
    centered = features - features.mean(dim=0, keepdim=True)
    return centered.transpose(0, 1).matmul(centered) / max(features.shape[0] - 1, 1)


def coral_loss(features: torch.Tensor, domains: torch.Tensor) -> torch.Tensor:
    features = features.float()
    covariances: list[torch.Tensor] = []
    for domain in torch.unique(domains):
        selected = features[domains == domain]
        if selected.shape[0] >= 2:
            covariances.append(_covariance(selected))
    if len(covariances) < 2:
        return features.new_zeros(())
    losses = [(left - right).square().mean() for left, right in combinations(covariances, 2)]
    return torch.stack(losses).mean()
