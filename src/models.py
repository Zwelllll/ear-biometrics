"""
Phase 3, part 2: MODELS.

Three transfer-learning backbones, pretrained on ImageNet, with the final
classifier replaced to match our number of identities.

    resnet50          25.6M params   the paper's best model on AMI
    mobilenet_v2       3.5M params   the lightweight comparison
    efficientnet_b0    5.3M params   replaces the paper's EfficientNet-B7

Why B0 and not B7: B7 has 66M parameters and is designed for 600x600 input.
AMI has 700 images across 100 subjects. B7 memorises that in a few epochs and
tells you nothing, while costing ~10x the GPU time of B0 -- which matters a lot
on free-tier compute. The paper's own tables show every model hitting ~100%
train accuracy on AMI, i.e. all of them overfit it regardless of size, so the
extra capacity buys nothing. We also expose b7 here so you can run it ONCE for
a like-for-like number against their headline result if quota allows.

FREEZING: the paper describes different strategies per model (MobileNet: train
only the final layer; EfficientNet/VGG19: freeze front layers; ResNet50: replace
the FC layer, unclear what else). That inconsistency makes their cross-model
comparison hard to interpret. We use ONE strategy for all models -- full
fine-tuning -- and say so, which is what makes an architecture comparison
meaningful. Set freeze_backbone=True to compare against the frozen-feature case.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

SUPPORTED = ("resnet50", "mobilenet_v2", "efficientnet_b0", "efficientnet_b7")


def build_model(name: str, num_classes: int, pretrained: bool = True,
                freeze_backbone: bool = False) -> nn.Module:
    if name not in SUPPORTED:
        raise ValueError(f"unknown model {name!r}; valid: {SUPPORTED}")

    if name == "resnet50":
        w = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.resnet50(weights=w)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
        head_names = ("fc",)

    elif name == "mobilenet_v2":
        w = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.mobilenet_v2(weights=w)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        head_names = ("classifier",)

    elif name == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=w)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        head_names = ("classifier",)

    else:  # efficientnet_b7
        w = models.EfficientNet_B7_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b7(weights=w)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        head_names = ("classifier",)

    if freeze_backbone:
        for pname, param in m.named_parameters():
            param.requires_grad = any(pname.startswith(h) for h in head_names)

    return m


def get_embedding_dim(name: str) -> int:
    """Feature dimension before the classifier -- needed for Phase 7."""
    return {
        "resnet50": 2048,
        "mobilenet_v2": 1280,
        "efficientnet_b0": 1280,
        "efficientnet_b7": 2560,
    }[name]


def extract_features(model: nn.Module, name: str, x: torch.Tensor) -> torch.Tensor:
    """
    Penultimate-layer embeddings, for Phase 7 verification.

    Note for the report: these come from a network trained with plain softmax
    cross-entropy, so they are NOT optimised to be metric-friendly. Cosine
    similarity on them is a weak baseline. Phase 7 compares this against an
    ArcFace head, which is trained for exactly this purpose.
    """
    if name == "resnet50":
        modules = list(model.children())[:-1]     # drop fc
        feat = nn.Sequential(*modules)(x)
        return torch.flatten(feat, 1)
    # mobilenet / efficientnet share the same structure
    feat = model.features(x)
    feat = model.avgpool(feat)
    return torch.flatten(feat, 1)


def count_params(model: nn.Module) -> tuple[float, float]:
    """Returns (total_millions, trainable_millions) -- for the Phase 8 table."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e6, trainable / 1e6
