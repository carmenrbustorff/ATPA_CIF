"""train.py — BirdCLEF 2026 PyTorch training (stratified + early stopping).

Usage:
    python train.py --model simple_cnn_torch --epochs 5
    python train.py --model efficientnet_torch --epochs 10 --batch-size 16 --lr 5e-4 --augment
"""
import argparse, json, time, traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score

from models import build_simple_cnn_torch, build_efficientnet_torch
from config import NUM_SPECIES
from data_loader import build_train_val_dataloaders


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["simple_cnn_torch", "efficientnet_torch"], required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--early-stopping-patience", type=int, default=3)
    p.add_argument("--output-dir", type=Path, default=Path("experiments"))
    p.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    p.add_argument("--llm-name", default="manual")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-cache", action="store_true", 
                  help="Use precomputed cached spectrograms (requires cache_spectrograms.py first)")
    p.add_argument("--cache-dir", type=Path, default=Path("/tmp/birdclef-specs"),
                  help="Directory with cached spectrograms (if --use-cache is set)")
    return p.parse_args()


def to_multilabel(y, num_classes):
    if y.dim() == 1:
        out = torch.zeros(y.size(0), num_classes, device=y.device)
        out.scatter_(1, y.long().unsqueeze(1), 1.0)
        return out
    return y.float()


def build_model(name, num_classes):
    if name == "simple_cnn_torch":
        return build_simple_cnn_torch(num_classes=num_classes)
    return build_efficientnet_torch(num_classes=num_classes)


def forward_batch(model, x, expand_rgb):
    if expand_rgb:
        x = x.expand(-1, 3, -1, -1)
    return model(x)  # Returns logits


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes, expand_rgb):
    model.eval()
    losses, y_true_list, y_pred_list = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y_device = y.to(device)
        y_multilabel = to_multilabel(y_device, num_classes)
        with autocast(device_type=device.type):
            logits = forward_batch(model, x, expand_rgb)  # logits
            loss = criterion(logits, y_multilabel)  # BCEWithLogitsLoss
        losses.append(loss.item())
        y_true_list.append(y.cpu().numpy())  # store original class indices
        # Apply sigmoid to get probabilities for AUC
        probs = torch.sigmoid(logits).float().cpu().numpy()
        y_pred_list.append(probs)
    
    y_true_idx = np.concatenate(y_true_list)  # shape: (n_samples,)
    y_prob = np.concatenate(y_pred_list)  # shape: (n_samples, num_classes)
    
    # Convert class indices to one-hot encoding
    y_true_onehot = np.zeros((y_true_idx.shape[0], num_classes))
    y_true_onehot[np.arange(y_true_idx.shape[0]), y_true_idx] = 1.0
    
    # Only compute AUC for classes that appear in validation set
    present = y_true_onehot.sum(axis=0) > 0
    if present.sum() == 0:
        auc = 0.0
    else:
        try:
            auc = roc_auc_score(y_true_onehot[:, present], y_prob[:, present], average="macro")
        except (ValueError, ZeroDivisionError):
            auc = 0.0
    return float(np.mean(losses)), float(auc)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, num_classes, expand_rgb):
    model.train()
    losses = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = to_multilabel(y.to(device), num_classes)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type):
            logits = forward_batch(model, x, expand_rgb)  # logits
            loss = criterion(logits, y)  # BCEWithLogitsLoss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
    return float(np.mean(losses))


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    run_dir = args.output_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "status": "started", "llm_name": args.llm_name, "model": args.model,
        "config": {k: str(v) for k, v in vars(args).items()},
        "best_val_auc": None, "best_epoch": None,
        "epochs_completed": 0, "epoch_log": [],
        "early_stopped": False, "error_message": None,
    }

    def write_metrics():
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
    write_metrics()

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        # Load dataset (cached or on-the-fly)
        train_loader, val_loader = build_train_val_dataloaders(
            val_split=args.val_split,
            batch_size=args.batch_size,
            augment=args.augment,
            random_state=args.seed,
            use_cache=args.use_cache,
            cache_dir=args.cache_dir,
        )
        print(f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}  Classes: {NUM_SPECIES}")

        model = build_model(args.model, NUM_SPECIES).to(device)
        expand_rgb = (args.model == "efficientnet_torch")
        criterion = nn.BCEWithLogitsLoss()  # logits input + sigmoid internally
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=args.weight_decay
        )
        scaler = GradScaler(device=device.type, enabled=(device.type == "cuda"))

        best_auc = 0.0
        patience_counter = 0
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler,
                                          device, NUM_SPECIES, expand_rgb)
            val_loss, val_auc = validate(model, val_loader, criterion, device, NUM_SPECIES, expand_rgb)
            elapsed = time.time() - t0

            print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_auc={val_auc:.4f}  ({elapsed:.1f}s)")

            metrics["epoch_log"].append({
                "epoch": epoch, "train_loss": train_loss,
                "val_loss": val_loss, "val_auc": val_auc, "time_s": elapsed,
            })
            metrics["epochs_completed"] = epoch

            if val_auc > best_auc:
                best_auc = val_auc
                metrics["best_val_auc"] = val_auc
                metrics["best_epoch"] = epoch
                torch.save(model.state_dict(), run_dir / "best_model.pt")
                patience_counter = 0
            else:
                patience_counter += 1

            write_metrics()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if patience_counter >= args.early_stopping_patience:
                print(f"Early stopping (no improvement for {args.early_stopping_patience} epochs)")
                metrics["early_stopped"] = True
                break

        metrics["status"] = "success"

    except torch.cuda.OutOfMemoryError as e:
        metrics["status"] = "failed_oom"
        metrics["error_message"] = str(e)
    except Exception as e:
        metrics["status"] = "failed_other"
        metrics["error_message"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    finally:
        write_metrics()
        print(f"Done: status={metrics['status']}, best_auc={metrics['best_val_auc']}")


if __name__ == "__main__":
    main()