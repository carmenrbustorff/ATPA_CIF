"""
PyTorch Model Scaffolds for BirdCLEF+ 2026 (Track B).

Provides ready-to-train model definitions for multi-label classification
of bird species from mel-spectrogram inputs.

Includes:
  - Lightweight 2-D CNN (fast first-iteration baseline)
  - EfficientNetB0 transfer-learning model (torchvision)

All models output a sigmoid-activated vector of length ``num_classes``
suitable for multi-label (binary cross-entropy) training.
"""

from __future__ import annotations

import logging
from typing import Optional

from config import N_MELS, NUM_SPECIES, TIME_FRAMES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PyTorch CNN baseline
# ---------------------------------------------------------------------------

def build_simple_cnn_torch(
    input_channels: int = 1,
    n_mels: int = N_MELS,
    time_frames: int = TIME_FRAMES,
    num_classes: int = NUM_SPECIES,
    dropout_rate: float = 0.3,
):
    """
    Lightweight 2-D CNN baseline (PyTorch).

    Input tensor shape: ``(batch, input_channels, n_mels, time_frames)``.

    Returns
    -------
    torch.nn.Module
    """
    import torch
    import torch.nn as nn

    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            return torch.sigmoid(self.classifier(x))

    return SimpleCNN()


# ---------------------------------------------------------------------------
# PyTorch EfficientNet transfer-learning model
# ---------------------------------------------------------------------------

def build_efficientnet_torch(
    num_classes: int = NUM_SPECIES,
    dropout_rate: float = 0.3,
    freeze_base: bool = True,
    pretrained: bool = True,
):
    """
    EfficientNetB0 transfer-learning model (PyTorch / torchvision).

    The model includes a 1→3 channel projection so it can accept
    single-channel mel-spectrograms ``(batch, 1, H, W)`` directly.

    Parameters
    ----------
    freeze_base:
        If True (default), all base layers except the classifier head are
        frozen.  Call ``unfreeze_top_layers_torch(model)`` after warm-up.
    pretrained:
        If True (default), load ImageNet-pretrained weights.

    Returns
    -------
    torch.nn.Module
    """
    import torch.nn as nn

    try:
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base = efficientnet_b0(weights=weights)
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for EfficientNet. "
            "Install it with: pip install torchvision"
        ) from exc

    if freeze_base:
        for param in base.parameters():
            param.requires_grad = False

    in_features = base.classifier[1].in_features
    base.classifier = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, num_classes),
        nn.Sigmoid(),
    )
    for param in base.classifier.parameters():
        param.requires_grad = True

    # Wrap with a channel-expansion head so the model accepts 1-channel input
    class EfficientNet1ch(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.channel_proj = nn.Conv2d(1, 3, kernel_size=1, bias=False)
            self.backbone = backbone

        def forward(self, x):
            x = self.channel_proj(x)
            return self.backbone(x)

    return EfficientNet1ch(base)


# ---------------------------------------------------------------------------
# Fine-tuning helper
# ---------------------------------------------------------------------------

def unfreeze_top_layers_torch(model, num_layers: int = 20) -> None:
    """
    Unfreeze the top ``num_layers`` of the EfficientNet backbone for
    fine-tuning.

    Call this after the warm-up training phase (e.g., after epoch 3).
    Works on models returned by ``build_efficientnet_torch``.
    """
    # Handle the EfficientNet1ch wrapper
    backbone = getattr(model, "backbone", model)
    all_layers = list(backbone.features.children())
    n_total = len(all_layers)
    freeze_up_to = max(0, n_total - num_layers)

    for i, layer in enumerate(all_layers):
        trainable = i >= freeze_up_to
        for param in layer.parameters():
            param.requires_grad = trainable

    # Always keep the classifier trainable
    for param in backbone.classifier.parameters():
        param.requires_grad = True

    logger.info(
        "Unfroze top %d / %d feature layers of EfficientNet for fine-tuning.",
        num_layers, n_total,
    )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

AVAILABLE_MODELS = {
    "simple_cnn_torch": build_simple_cnn_torch,
    "efficientnet_torch": build_efficientnet_torch,
}


def get_model(name: str, **kwargs):
    """
    Retrieve a model by name.

    Parameters
    ----------
    name:
        One of ``"simple_cnn_torch"`` or ``"efficientnet_torch"``.
    **kwargs:
        Additional keyword arguments forwarded to the builder function.

    Returns
    -------
    torch.nn.Module
    """
    if name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model {name!r}. Available: {list(AVAILABLE_MODELS)}"
        )
    return AVAILABLE_MODELS[name](**kwargs)
