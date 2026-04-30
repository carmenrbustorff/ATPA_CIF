"""
PyTorch Training Script for BirdCLEF+ 2026 (Track B).

Usage
-----
    python train.py \\
        --model simple_cnn_torch \\
        --epochs 10 \\
        --batch-size 32 \\
        --lr 1e-3 \\
        --data-dir ~/birdclef-data \\
        --output-dir experiments/run_001

    python train.py \\
        --model efficientnet_torch \\
        --epochs 5 \\
        --batch-size 16 \\
        --lr 5e-4 \\
        --unfreeze-epoch 3 \\
        --data-dir ~/birdclef-data

Features
--------
- Mixed precision training (torch.cuda.amp)
- Per-epoch macro ROC-AUC validation (skips classes with no true positives)
- Saves best_model.pt on validation AUC improvement
- torch.cuda.empty_cache() after every epoch
- Experiment results written to <output-dir>/metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import GradScaler, autocast

from config import (
    AUDIO_DIR,
    NUM_SPECIES,
    TRAIN_CSV,
)
from data_loader import build_train_val_dataloaders
from models import get_model, unfreeze_top_layers_torch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation metric
# ---------------------------------------------------------------------------

def compute_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Macro-averaged ROC-AUC skipping classes that have no true positive labels.

    This matches the official BirdCLEF+ 2026 evaluation metric.
    """
    valid_cols = [c for c in range(y_true.shape[1]) if y_true[:, c].sum() > 0]
    if not valid_cols:
        return 0.0
    return float(
        roc_auc_score(
            y_true[:, valid_cols],
            y_pred[:, valid_cols],
            average="macro",
        )
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler,
    epoch: int,
) -> float:
    """Run one training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for specs, labels in loader:
        specs = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            preds = model(specs)
            loss = criterion(preds, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run validation. Returns (mean_loss, macro_auc)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for specs, labels in loader:
        specs = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            preds = model(specs)
            loss = criterion(preds, labels)

        total_loss += loss.item()
        n_batches += 1
        all_preds.append(preds.cpu().float().numpy())
        all_labels.append(labels.cpu().float().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    auc = compute_auc(y_true, y_pred)
    mean_loss = total_loss / max(n_batches, 1)
    return mean_loss, auc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    # Resolve paths
    data_dir = Path(args.data_dir).expanduser()
    train_csv = data_dir / "train.csv"
    audio_dir = data_dir / "train_audio"
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_csv.exists():
        logger.error("train.csv not found at %s", train_csv)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    logger.info("Building train/val DataLoaders from %s …", train_csv)
    train_loader, val_loader, species_list = build_train_val_dataloaders(
        train_csv=train_csv,
        audio_dir=audio_dir,
        batch_size=args.batch_size,
        val_batch_size=args.batch_size,
    )
    num_classes = len(species_list)
    logger.info("Classes: %d", num_classes)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    logger.info("Building model: %s", args.model)
    model = get_model(args.model, num_classes=num_classes)
    model = model.to(device)

    # ------------------------------------------------------------------
    # Optimiser, loss, AMP scaler
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    criterion = nn.BCELoss()
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_auc = 0.0
    best_epoch = 0
    history: list[dict] = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        # EfficientNet: unfreeze top layers after warm-up
        if args.model == "efficientnet_torch" and epoch == args.unfreeze_epoch:
            logger.info("Epoch %d: unfreezing top 20 EfficientNet layers.", epoch)
            unfreeze_top_layers_torch(model, num_layers=20)
            # Re-create optimizer to include newly unfrozen params
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr * 0.1,  # lower LR for fine-tuning
            )

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, epoch
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        if device.type == "cuda":
            torch.cuda.empty_cache()

        elapsed = time.time() - t_start
        logger.info(
            "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | val_auc=%.4f | %.0fs",
            epoch, args.epochs, train_loss, val_loss, val_auc, elapsed,
        )

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_auc": round(val_auc, 6),
            "elapsed_s": round(elapsed, 1),
        }
        history.append(row)

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            best_path = output_dir / "best_model.pt"
            torch.save(model.state_dict(), best_path)
            logger.info("  ↑ New best AUC=%.4f — saved to %s", best_auc, best_path)

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    total_time = time.time() - t_start
    metrics = {
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "num_classes": num_classes,
        "best_val_auc": round(best_auc, 6),
        "best_epoch": best_epoch,
        "final_val_auc": round(history[-1]["val_auc"], 6) if history else 0.0,
        "training_time_s": round(total_time, 1),
        "device": str(device),
        "history": history,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Metrics written to %s", metrics_path)
    logger.info(
        "Training complete. Best val AUC=%.4f at epoch %d (%.0fs total).",
        best_auc, best_epoch, total_time,
    )
    # Print for agent capture
    print("METRICS:", json.dumps({k: v for k, v in metrics.items() if k != "history"}))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BirdCLEF+ 2026 PyTorch Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default="simple_cnn_torch",
        choices=["simple_cnn_torch", "efficientnet_torch"],
        help="Model architecture.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size (use 16 for EfficientNet to fit GPU memory).",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--unfreeze-epoch", type=int, default=3,
        help="Epoch at which to unfreeze top EfficientNet layers (efficientnet_torch only).",
    )
    parser.add_argument(
        "--data-dir", type=str, default=str(Path("~/birdclef-data").expanduser()),
        help="Path to the BirdCLEF data root (must contain train.csv and train_audio/).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="experiments/latest",
        help="Directory for model checkpoints and metrics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args())
