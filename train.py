"""
train.py — Standalone PyTorch training script for BirdCLEF 2026.

Trains either the lightweight SimpleCNN or EfficientNetB0 on mel-spectrogram
inputs produced by BirdCLEFDataset.  Mixed-precision training (AMP) keeps GPU
memory usage low on the shared NVIDIA L4.

Usage
-----
    # from the ATPA_CIF root directory, with the shared .venv activated:
    python train.py \\
        --epochs 20 \\
        --batch-size 32 \\
        --lr 1e-3 \\
        --model simple_cnn_torch \\
        --output-dir experiments/run1

CLI arguments
-------------
    --epochs        Number of full passes over the training set (default: 10)
    --batch-size    Samples per batch (default: 32)
    --lr            Initial learning rate for Adam (default: 1e-3)
    --model         Model architecture: simple_cnn_torch | efficientnet_torch
    --output-dir    Directory where best_model.pt and train_log.csv are saved
    --val-split     Fraction of data held out for validation (default: 0.2)
    --seed          Random seed for reproducibility (default: 42)

Output
------
    <output-dir>/best_model.pt   — state_dict of the best checkpoint (by val AUC)
    <output-dir>/train_log.csv   — per-epoch metrics
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from data_loader import BirdCLEFDataset, METADATA_CSV, AUDIO_DIR, NUM_WORKERS
from models import build_simple_cnn_torch, build_efficientnet_torch, NUM_SPECIES

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BirdCLEF 2026 — PyTorch training script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument(
        "--model",
        default="simple_cnn_torch",
        choices=["simple_cnn_torch", "efficientnet_torch"],
        help="Model architecture",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/default"),
        help="Directory for checkpoints and logs",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of data held out for validation",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(name: str, num_classes: int) -> nn.Module:
    """Instantiate the requested model with the correct output dimension."""
    if name == "simple_cnn_torch":
        # AdaptiveAvgPool2d makes this spatial-size agnostic;
        # override defaults to match BirdCLEFDataset output (1, 128, 313).
        return build_simple_cnn_torch(
            input_channels=1,
            n_mels=128,
            time_frames=313,
            num_classes=num_classes,
        )
    if name == "efficientnet_torch":
        return build_efficientnet_torch(num_classes=num_classes)
    raise ValueError(f"Unknown model: {name!r}")


# ---------------------------------------------------------------------------
# Channel adapter for EfficientNet (expects 3-channel input)
# ---------------------------------------------------------------------------

class ChannelAdapter(nn.Module):
    """
    Wraps a model that expects RGB input.

    When ``expand_channels=True`` a single-channel spectrogram tensor is
    replicated across 3 channels before being forwarded to the base model.
    """

    def __init__(self, base: nn.Module, expand_channels: bool = False) -> None:
        super().__init__()
        self.base = base
        self.expand_channels = expand_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.expand_channels:
            x = x.expand(-1, 3, -1, -1)  # (B, 1, H, W) → (B, 3, H, W)
        return self.base(x)


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler,
    device: torch.device,
    num_classes: int,
    is_train: bool,
) -> tuple[float, float]:
    """
    Run one epoch (train or validation).

    Returns
    -------
    avg_loss : float
    macro_auc : float  (returns 0.0 if only one class present in batch)
    """
    model.train(is_train)

    total_loss = 0.0
    all_targets: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for spectrograms, labels in loader:
            spectrograms = spectrograms.to(device, non_blocking=True)
            labels = labels.to(device)

            # One-hot encode integer labels → (B, num_classes) float targets
            targets = torch.zeros(
                labels.size(0), num_classes, dtype=torch.float32, device=device
            )
            targets.scatter_(1, labels.unsqueeze(1), 1.0)

            if is_train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)

            with autocast():
                # Models apply sigmoid internally; outputs are probabilities in [0, 1]
                outputs = model(spectrograms)          # (B, num_classes)
                loss = criterion(outputs, targets)

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * spectrograms.size(0)

            # Collect for AUC (cast to float32 to handle any fp16 from AMP)
            all_targets.append(targets.detach().cpu().numpy().astype(np.float32))
            all_probs.append(outputs.detach().cpu().numpy().astype(np.float32))

    avg_loss = total_loss / len(loader.dataset)

    y_true = np.concatenate(all_targets, axis=0)   # (N, num_classes)
    y_score = np.concatenate(all_probs, axis=0)     # (N, num_classes)

    try:
        # Only compute AUC for classes that are actually present in this split
        present_mask = y_true.sum(axis=0) > 0
        macro_auc = roc_auc_score(
            y_true[:, present_mask],
            y_score[:, present_mask],
            average="macro",
        )
    except ValueError as exc:
        logger.warning("AUC computation skipped: %s", exc)
        macro_auc = 0.0

    return avg_loss, macro_auc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        dev_idx = torch.cuda.current_device()
        logger.info(
            "GPU: %s  |  VRAM: %.1f GB",
            torch.cuda.get_device_name(dev_idx),
            torch.cuda.get_device_properties(dev_idx).total_memory / 1024 ** 3,
        )

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = args.output_dir / "best_model.pt"
    log_csv_path = args.output_dir / "train_log.csv"

    # ------------------------------------------------------------------
    # Dataset + train/val split
    # ------------------------------------------------------------------
    logger.info("Loading dataset metadata...")
    full_dataset = BirdCLEFDataset(
        metadata_csv=METADATA_CSV,
        audio_dir=AUDIO_DIR,
        augment=False,  # augmentation enabled below for train subset only
    )
    num_classes = full_dataset.num_classes
    logger.info("Classes: %d", num_classes)

    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.val_split,
        random_state=args.seed,
        shuffle=True,
    )

    # Enable augmentation on training data only via a second dataset instance
    train_dataset = BirdCLEFDataset(
        metadata_csv=METADATA_CSV,
        audio_dir=AUDIO_DIR,
        augment=True,
    )
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

    logger.info(
        "Train: %d samples  |  Val: %d samples", len(train_subset), len(val_subset)
    )

    loader_kwargs = dict(
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    logger.info("Building model: %s", args.model)
    base_model = build_model(args.model, num_classes)
    needs_rgb = args.model == "efficientnet_torch"
    model = ChannelAdapter(base_model, expand_channels=needs_rgb).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Parameters — total: %s  |  trainable: %s",
        f"{total_params:,}",
        f"{trainable_params:,}",
    )

    # ------------------------------------------------------------------
    # Loss, optimiser, AMP scaler
    # ------------------------------------------------------------------
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # ------------------------------------------------------------------
    # CSV log header
    # ------------------------------------------------------------------
    with open(log_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_auc", "val_loss", "val_auc", "elapsed_s"])

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_auc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_loss, train_auc = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, num_classes,
            is_train=True,
        )

        # Validate
        val_loss, val_auc = run_epoch(
            model, val_loader, criterion, None, scaler, device, num_classes,
            is_train=False,
        )

        elapsed = time.time() - t0

        logger.info(
            "Epoch %d/%d | train_loss=%.4f  train_auc=%.4f | "
            "val_loss=%.4f  val_auc=%.4f | %.1fs",
            epoch, args.epochs,
            train_loss, train_auc,
            val_loss, val_auc,
            elapsed,
        )

        # Save best checkpoint
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_ckpt_path)
            logger.info(
                "  ✓ New best val AUC: %.4f — checkpoint saved to %s",
                best_val_auc,
                best_ckpt_path,
            )

        # Append to CSV log
        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_auc, val_loss, val_auc, f"{elapsed:.1f}"])

        # Free unused GPU memory before the next epoch
        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info(
        "Training complete. Best val AUC: %.4f  |  Checkpoint: %s",
        best_val_auc,
        best_ckpt_path,
    )


if __name__ == "__main__":
    main()
