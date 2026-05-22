from __future__ import annotations

import torch
from torchvision import transforms

from officehome_dg.data import build_transforms
from officehome_dg.methods import coral_loss, mixup_batch, soft_cross_entropy
from officehome_dg.models import build_model


def test_mixup_and_soft_loss_are_finite():
    images = torch.randn(4, 3, 16, 16)
    labels = torch.tensor([0, 1, 2, 1])
    mixed_images, targets = mixup_batch(images, labels, alpha=0.2, num_classes=3)
    logits = torch.randn(4, 3)
    assert mixed_images.shape == images.shape
    assert torch.allclose(targets.sum(dim=1), torch.ones(4))
    assert torch.isfinite(soft_cross_entropy(logits, targets))


def test_coral_loss_and_model_head_are_finite():
    features = torch.randn(6, 8)
    domains = torch.tensor([0, 0, 1, 1, 2, 2])
    model = build_model("tiny_cnn", num_classes=65, pretrained=False)
    logits, embedded = model(torch.randn(2, 3, 32, 32), return_features=True)
    assert torch.isfinite(coral_loss(features, domains))
    assert logits.shape == (2, 65)
    assert embedded.shape[0] == 2


def test_randaugment_only_enters_training_transform():
    training = build_transforms(32, "randaugment", training=True)
    validation = build_transforms(32, "randaugment", training=False)
    assert any(isinstance(operation, transforms.RandAugment) for operation in training.transforms)
    assert not any(isinstance(operation, transforms.RandAugment) for operation in validation.transforms)
