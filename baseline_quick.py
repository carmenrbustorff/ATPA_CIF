"""Quick baseline runner using a metadata subset for fast comparison.
Creates experiments/baseline_quick_{timestamp}/ with metrics.json and best_model.pt
"""
from pathlib import Path
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from data_loader import BirdCLEFDataset
from backups.models import build_simple_cnn_torch
from config import NUM_SPECIES

OUT_DIR = Path("/tmp")
RUN_ID = f"baseline_quick_{int(time.time())}"
RUN_DIR = OUT_DIR / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

METADATA = Path("/tmp/train_subset.csv")
AUDIO_DIR = Path("/mnt/disks/data/birdclef/train_audio")

print('Using metadata:', METADATA)

dataset = BirdCLEFDataset(metadata_csv=METADATA, audio_dir=AUDIO_DIR, augment=False)
labels = [lbl for (_, lbl) in dataset._samples]

# Keep small for speed
indices = list(range(len(labels)))
if len(indices) > 400:
    indices = indices[:400]
    labels = labels[:400]

train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)
train_set = Subset(dataset, train_idx)
val_set = Subset(dataset, val_idx)

train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_simple_cnn_torch(num_classes=NUM_SPECIES).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

best_auc = 0.0
for epoch in range(1, 3):
    model.train()
    train_losses = []
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y_multi = torch.zeros(x.size(0), NUM_SPECIES, device=device)
        y_multi.scatter_(1, y.view(-1,1).long(), 1.0)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y_multi)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        if i >= 50:
            break
    # Validation
    model.eval()
    y_trues = []
    y_preds = []
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_preds.append(probs)
            y_trues.append(y.cpu().numpy())
            if i >= 20:
                break
    if y_preds:
        y_preds = np.concatenate(y_preds, axis=0)
        y_trues = np.concatenate(y_trues, axis=0)
        y_true_onehot = np.zeros((y_trues.shape[0], NUM_SPECIES))
        y_true_onehot[np.arange(y_trues.shape[0]), y_trues] = 1.0
        present = y_true_onehot.sum(axis=0) > 0
        try:
            auc = float(roc_auc_score(y_true_onehot[:, present], y_preds[:, present], average='macro'))
        except Exception:
            auc = 0.0
    else:
        auc = 0.0
    print(f"Epoch {epoch} train_loss={np.mean(train_losses):.4f} val_auc={auc:.4f}")
    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), RUN_DIR / 'best_model.pt')

metrics = {
    'best_val_auc': best_auc,
    'epochs': 2,
    'train_samples': len(train_idx),
    'val_samples': len(val_idx),
}
(RUN_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=2))
print('Done. Metrics:', metrics)
print('Run dir:', RUN_DIR)
