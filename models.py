"""
Model Scaffolds for BirdCLEF+ 2026 (Track B).

Provides ready-to-train model definitions for multi-label classification
across 234 bird species from mel-spectrogram inputs.

Includes:
  - Lightweight 2-D CNN (fast first-iteration baseline)
  - EfficientNetB0 transfer-learning model (TensorFlow/Keras)
  - PyTorch CNN baseline
  - PyTorch EfficientNet transfer-learning model
  - BirdNET-inspired fine-tunable feature extractor stub

All models output a 234-dimensional sigmoid-activated vector suitable for
multi-label (binary cross-entropy) training.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

NUM_SPECIES = 234   # BirdCLEF+ 2026 Track B


# ---------------------------------------------------------------------------
# TensorFlow / Keras models
# ---------------------------------------------------------------------------

def build_simple_cnn_tf(
    input_shape: Tuple[int, int, int] = (64, 216, 1),
    num_classes: int = NUM_SPECIES,
    dropout_rate: float = 0.3,
):
    """
    Lightweight 2-D CNN for rapid initial iterations.

    Input shape: (n_mels, time_frames, channels).
    Default is (64, 216, 1) — ~5 s clip at 22 kHz, hop=512, n_mels=64.

    Returns
    -------
    tf.keras.Model
    """
    import tensorflow as tf  # type: ignore
    from tensorflow.keras import layers  # type: ignore

    inputs = tf.keras.Input(shape=input_shape, name="spectrogram")

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="sigmoid", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name="SimpleCNN")
    return model


def build_efficientnet_tf(
    input_shape: Tuple[int, int, int] = (64, 216, 3),
    num_classes: int = NUM_SPECIES,
    dropout_rate: float = 0.3,
    freeze_base: bool = True,
):
    """
    EfficientNetB0 transfer-learning model (TensorFlow/Keras).

    The base model is initialised with ImageNet weights.
    The spectrogram should be replicated to 3 channels before calling this model
    (e.g., ``tf.repeat(spec, 3, axis=-1)``).

    Parameters
    ----------
    input_shape:
        (height, width, 3) – spectrogram dimensions with 3 channels.
    freeze_base:
        If True (default), the EfficientNet base layers are frozen for initial
        training. Call ``unfreeze_top_layers(model)`` to fine-tune later.

    Returns
    -------
    tf.keras.Model
    """
    import tensorflow as tf  # type: ignore
    from tensorflow.keras import layers  # type: ignore
    from tensorflow.keras.applications import EfficientNetB0  # type: ignore

    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg",
    )
    base.trainable = not freeze_base

    inputs = tf.keras.Input(shape=input_shape, name="spectrogram_3ch")
    x = base(inputs, training=False)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="sigmoid", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name="EfficientNetB0_BirdCLEF")
    return model


def unfreeze_top_layers_tf(model, num_layers: int = 20) -> None:
    """
    Unfreeze the top ``num_layers`` of the base model for fine-tuning.

    Call this after the first warm-up training phase.
    """
    base = model.layers[1]          # second layer is the base model
    base.trainable = True
    for layer in base.layers[:-num_layers]:
        layer.trainable = False
    logger.info("Unfroze top %d layers of %s for fine-tuning.", num_layers, base.name)


def compile_tf_model(model, learning_rate: float = 1e-3):
    """
    Compile a TF/Keras model with sensible defaults for multi-label classification.
    """
    import tensorflow as tf  # type: ignore

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(multi_label=True, name="auc"),
            "binary_accuracy",
        ],
    )
    return model


# ---------------------------------------------------------------------------
# PyTorch models
# ---------------------------------------------------------------------------

def build_simple_cnn_torch(
    input_channels: int = 1,
    n_mels: int = 64,
    time_frames: int = 216,
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
    import torch.nn as nn  # type: ignore

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


def build_efficientnet_torch(
    num_classes: int = NUM_SPECIES,
    dropout_rate: float = 0.3,
    freeze_base: bool = True,
    pretrained: bool = True,
):
    """
    EfficientNetB0 transfer-learning model (PyTorch / torchvision).

    Expects RGB input: ``(batch, 3, H, W)``.

    Parameters
    ----------
    freeze_base:
        If True (default), all layers except the classifier head are frozen.
    pretrained:
        If True (default), load ImageNet-pretrained weights.

    Returns
    -------
    torch.nn.Module
    """
    import torch.nn as nn  # type: ignore

    try:
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights  # type: ignore
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base = efficientnet_b0(weights=weights)
    except ImportError:
        raise ImportError(
            "torchvision is required for EfficientNet. "
            "Install it with: pip install torchvision"
        )

    if freeze_base:
        for param in base.parameters():
            param.requires_grad = False

    in_features = base.classifier[1].in_features
    base.classifier = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, num_classes),
        nn.Sigmoid(),
    )
    # Re-enable classifier gradients
    for param in base.classifier.parameters():
        param.requires_grad = True

    return base


# ---------------------------------------------------------------------------
# BirdNET-inspired feature extractor stub
# ---------------------------------------------------------------------------

def build_birdnet_inspired_tf(
    input_shape: Tuple[int, int, int] = (144, 144, 1),
    num_classes: int = NUM_SPECIES,
):
    """
    BirdNET-inspired architecture stub (TensorFlow/Keras).

    Mimics the depthwise-separable + residual block pattern used in BirdNET v2.4.
    Replace this with actual BirdNET weights when available under an open licence.

    Input: mel-spectrogram of shape (144, 144, 1).

    Returns
    -------
    tf.keras.Model
    """
    import tensorflow as tf  # type: ignore
    from tensorflow.keras import layers  # type: ignore

    def _sep_block(x, filters: int, stride: int = 1):
        shortcut = x
        x = layers.DepthwiseConv2D((3, 3), padding="same", strides=stride)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, (1, 1), padding="same")(x)
        x = layers.BatchNormalization()(x)
        if shortcut.shape[-1] != filters or stride > 1:
            shortcut = layers.Conv2D(
                filters, (1, 1), strides=stride, padding="same"
            )(shortcut)
        return layers.Add()([x, shortcut])

    inputs = tf.keras.Input(shape=input_shape, name="mel_spec")
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same", activation="relu")(inputs)
    x = _sep_block(x, 64)
    x = _sep_block(x, 64, stride=2)
    x = _sep_block(x, 128)
    x = _sep_block(x, 128, stride=2)
    x = _sep_block(x, 256)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="sigmoid", name="predictions")(x)
    return tf.keras.Model(inputs, outputs, name="BirdNET_Inspired")


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

AVAILABLE_MODELS = {
    "simple_cnn_tf": build_simple_cnn_tf,
    "efficientnet_tf": build_efficientnet_tf,
    "birdnet_tf": build_birdnet_inspired_tf,
    "simple_cnn_torch": build_simple_cnn_torch,
    "efficientnet_torch": build_efficientnet_torch,
}


def get_model(name: str, **kwargs):
    """
    Retrieve a model builder by name.

    Parameters
    ----------
    name:
        One of the keys in :data:`AVAILABLE_MODELS`.
    **kwargs:
        Additional keyword arguments forwarded to the builder function.

    Returns
    -------
    A compiled TF/Keras model or an instantiated PyTorch nn.Module.
    """
    if name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model {name!r}. Available: {list(AVAILABLE_MODELS)}"
        )
    model = AVAILABLE_MODELS[name](**kwargs)
    if name.endswith("_tf"):
        model = compile_tf_model(model)
    return model
