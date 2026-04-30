"""
Evaluation Script for BirdCLEF+ 2026 (Track B).

Loads a trained ``.pt`` checkpoint, runs it on the validation split, and
computes per-class ROC-AUC scores using the official competition metric
(macro-average, skipping classes with no true positive labels).

Also sweeps sigmoid thresholds (0.1–0.9) to find the per-class threshold
that maximises macro-F1, saving results to ``experiments/thresholds.json``.

Usage
-----
    python evaluate.py \\
        --checkpoint experiments/latest/best_model.pt \\
        --model simple_cnn_torch \\
        --data-dir ~/birdclef-data \\
        --output-dir experiments/latest
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.cuda.amp import autocast

from config import AUDIO_DIR, TRAIN_CSV
from data_loader import build_train_val_dataloaders
from models import get_model

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
# Metric helpers
# ---------------------------------------------------------------------------

def competition_auc(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, list[float]]:
    """
    Compute macro ROC-AUC skipping classes with no true positive labels.

    Returns
    -------
    (macro_auc, per_class_auc_list)
        Classes with no positives receive AUC = NaN in the per-class list.
    """
    n_classes = y_true.shape[1]
    per_class = [float("nan")] * n_classes
    valid_cols = []

    for c in range(n_classes):
        if y_true[:, c].sum() > 0:
            per_class[c] = float(roc_auc_score(y_true[:, c], y_pred[:, c]))
            valid_cols.append(c)

    macro_auc = (
        float(np.mean([per_class[c] for c in valid_cols])) if valid_cols else 0.0
    )
    return macro_auc, per_class


def find_best_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: list[float] | None = None,
) -> list[float]:
    """
    For each class, sweep thresholds and return the one that maximises F1.
    Classes with no positives default to threshold=0.5.
    """
    if thresholds is None:
        thresholds = [round(t, 1) for t in np.arange(0.1, 1.0, 0.1)]

    n_classes = y_true.shape[1]
    best_thresholds = [0.5] * n_classes

    for c in range(n_classes):
        if y_true[:, c].sum() == 0:
            continue
        best_f1 = -1.0
        for thr in thresholds:
            preds_bin = (y_pred[:, c] >= thr).astype(int)
            f1 = float(f1_score(y_true[:, c], preds_bin, zero_division=0))
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds[c] = thr

    return best_thresholds


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, loader, device: torch.device):
    """Collect all predictions and labels from the val loader."""
    model.eval()
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for specs, labels in loader:
        specs = specs.to(device, non_blocking=True)
        with autocast():
            preds = model(specs)
        all_preds.append(preds.cpu().float().numpy())
        all_labels.append(labels.cpu().float().numpy())

    return (
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_labels, axis=0),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).expanduser()
    train_csv = data_dir / TRAIN_CSV.name      # train.csv
    audio_dir = data_dir / AUDIO_DIR.name      # train_audio
    checkpoint = Path(args.checkpoint).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_csv.exists():
        logger.error("train.csv not found: %s", train_csv)
        sys.exit(1)
    if not checkpoint.exists():
        logger.error("Checkpoint not found: %s", checkpoint)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------
    # Data (validation split only)
    # ------------------------------------------------------------------
    logger.info("Building val DataLoader…")
    _, val_loader, species_list = build_train_val_dataloaders(
        train_csv=train_csv,
        audio_dir=audio_dir,
        batch_size=args.batch_size,
    )
    num_classes = len(species_list)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    logger.info("Loading checkpoint: %s", checkpoint)
    model = get_model(args.model, num_classes=num_classes)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    logger.info("Model loaded.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    logger.info("Running inference on validation set…")
    y_pred, y_true = run_inference(model, val_loader, device)

    # ------------------------------------------------------------------
    # AUC
    # ------------------------------------------------------------------
    macro_auc, per_class_auc = competition_auc(y_true, y_pred)
    logger.info("Macro ROC-AUC (competition metric): %.4f", macro_auc)

    valid_count = sum(1 for v in per_class_auc if not np.isnan(v))
    logger.info(
        "Classes with at least one positive: %d / %d", valid_count, num_classes
    )

    # ------------------------------------------------------------------
    # Threshold sweep
    # ------------------------------------------------------------------
    logger.info("Sweeping thresholds to maximise per-class F1…")
    best_thresholds = find_best_thresholds(y_true, y_pred)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    per_class_results = {
        species_list[c]: {
            "auc": None if np.isnan(per_class_auc[c]) else round(per_class_auc[c], 4),
            "threshold": round(best_thresholds[c], 2),
        }
        for c in range(num_classes)
    }

    results = {
        "checkpoint": str(checkpoint),
        "model": args.model,
        "macro_auc": round(macro_auc, 6),
        "valid_classes": valid_count,
        "total_classes": num_classes,
        "per_class": per_class_results,
    }

    # Save full results
    results_path = output_dir / "eval_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Evaluation results saved to %s", results_path)

    # Save thresholds in a flat format for the submission script
    thresholds_path = output_dir / "thresholds.json"
    thresholds_out = {species_list[c]: best_thresholds[c] for c in range(num_classes)}
    thresholds_path.write_text(json.dumps(thresholds_out, indent=2), encoding="utf-8")
    logger.info("Thresholds saved to %s", thresholds_path)

    print(json.dumps({"macro_auc": round(macro_auc, 6), "valid_classes": valid_count}))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BirdCLEF+ 2026 Evaluation Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the .pt model checkpoint.",
    )
    parser.add_argument(
        "--model", type=str, default="simple_cnn_torch",
        choices=["simple_cnn_torch", "efficientnet_torch"],
        help="Model architecture (must match the checkpoint).",
    )
    parser.add_argument(
        "--data-dir", type=str, default=str(Path("~/birdclef-data").expanduser()),
        help="Path to the BirdCLEF data root.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="experiments/latest",
        help="Directory for saving evaluation results and thresholds.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for inference.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args())
