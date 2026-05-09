"""baseline.py — minimal BirdCLEF baseline.

Mirrors the structure of the MNIST CNN notebook from class:
load data → build model → compile → fit → evaluate.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score
import numpy as np

from models import build_simple_cnn_torch
from config import NUM_SPECIES
from data_loader import BirdCLEFDataset, METADATA_CSV, AUDIO_DIR

# 1. Load data — same as `(x_train, y_train), (x_test, y_test) = mnist.load_data()`
dataset = BirdCLEFDataset(metadata_csv=METADATA_CSV, audio_dir=AUDIO_DIR)
n_val = int(len(dataset) * 0.2)
train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True,
                           num_workers=3, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_set, batch_size=32,
                         num_workers=3, pin_memory=True, persistent_workers=True)

# 2. Build model — same as `keras.Sequential([...])` from the CNN notebook
device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_simple_cnn_torch(num_classes=NUM_SPECIES).to(device)

# 3. Compile — same as `model.compile(optimizer="adam", loss=...)`
# NOTE: build_simple_cnn_torch already applies sigmoid internally,
# so we use BCELoss (not BCEWithLogitsLoss) to avoid double-sigmoid.
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 4. Fit — same as `model.fit(x_train, y_train, epochs=5)`
for epoch in range(5):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device).float()
        if y.dim() == 1:  # convert integer labels to multi-hot if needed
            y = torch.nn.functional.one_hot(y.long(), NUM_SPECIES).float()
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    # 5. Evaluate — same as `model.evaluate(x_test, y_test)`
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            if y.dim() == 1:
                y = torch.nn.functional.one_hot(y.long(), NUM_SPECIES).float()
            ys.append(y.numpy())
            ps.append(model(x).cpu().numpy())  # already in [0,1] from model's sigmoid
    y_true, y_prob = np.concatenate(ys), np.concatenate(ps)
    present = y_true.sum(axis=0) > 0
    auc = roc_auc_score(y_true[:, present], y_prob[:, present], average="macro")
    print(f"Epoch {epoch+1}: val AUC = {auc:.4f}")

# 6. Save the model
torch.save(model.state_dict(), "baseline_model.pt")
print("Done.")