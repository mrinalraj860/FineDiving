"""
Train DifficultyRegressor on dive-difficulty regression
(using only train_split.pkl videos).  CPU-only.
"""

import os
import ast
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchinfo import summary
from sklearn.metrics import mean_absolute_error, mean_squared_error

from DataLoaderForDifficulty import DiveDifficultyDataset, pad_collate
from ModelForDifficulty import DifficultyRegressor

# ------------------------------------------------------------------
# 1 ─── Paths and Hyper-params
# ------------------------------------------------------------------
PT_DIR          = "videosTensors"
TRAIN_SPLIT_PKL = "/Users/mrinalraj/Documents/FineDiving/train_test_split/train_split.pkl"
CSV_DIFFICULTY  = "/Users/mrinalraj/Downloads/WebDownload/Preprocess/DifficultyAndDiveScore_AllVides.csv"
NUM_EPOCHS      = 30
BATCH_SIZE      = 8
LR              = 1e-4
DEVICE          = "cpu"

os.makedirs("plots", exist_ok=True)

# ------------------------------------------------------------------
# 2 ─── Build difficulty map  {video_id_str : difficulty(float)}
# ------------------------------------------------------------------
df_diff = pd.read_csv(CSV_DIFFICULTY)
raw_diff = dict(zip(df_diff["video_id_str"], df_diff["all_difficulties"]))

all_difficulties = {}
for k, v in raw_diff.items():
    if isinstance(v, str) and v.startswith("["):
        try:
            all_difficulties[k] = float(ast.literal_eval(v)[0])
        except Exception:
            continue
    else:
        all_difficulties[k] = float(v)

# ------------------------------------------------------------------
# 3 ─── Allowed video IDs from train_split.pkl
# ------------------------------------------------------------------
with open(TRAIN_SPLIT_PKL, "rb") as f:
    train_file_list = pickle.load(f)           # list of tuples (x, y)

allowed_ids = {f"{x}_{y}" for x, y in train_file_list}

# ------------------------------------------------------------------
# 4 ─── Dataset restricted to allowed_ids
# ------------------------------------------------------------------
base_ds = DiveDifficultyDataset(pt_dir=PT_DIR, all_difficulties=all_difficulties)
keep_idx = [i for i, vid in enumerate(base_ds.video_ids) if vid in allowed_ids]
train_ds = Subset(base_ds, keep_idx)

print(f"Dataset size (train only) : {len(train_ds)} videos")

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=pad_collate, num_workers=0
)

# peek at one batch to display shapes
tmp_tracks, tmp_vis, tmp_diff, tmp_len, tmp_vid = next(iter(train_loader))
print("First batch tracks shape :", tmp_tracks.shape)   # [B, T, N, 2]
print("First batch vis shape    :", tmp_vis.shape)      # [B, T, N]
print("First batch diff shape   :", tmp_diff.shape)     # [B]
print("All video IDs in batch  :", tmp_vid)            # [B]
print("First Difference in batch  :", tmp_diff)         # str


# ------------------------------------------------------------------
# 5 ─── Model & Optimiser
# ------------------------------------------------------------------
model = DifficultyRegressor(num_points=1000).to(DEVICE)
summary(model, input_size=tmp_tracks.shape)             # model summary

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# ------------------------------------------------------------------
# 6 ─── Training loop
# ------------------------------------------------------------------
hist = {"epoch": [], "mse": [], "mae": [], "rmse": []}

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_losses, epoch_targets, epoch_preds = [], [], []

    for tracks, vis, diff, lengths, vids in train_loader:
        pred = model(tracks)                  # [B]
        loss = criterion(pred, diff)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        epoch_targets.extend(diff.numpy())
        epoch_preds.extend(pred.detach().numpy())

    mse  = np.mean(epoch_losses)
    mae  = mean_absolute_error(epoch_targets, epoch_preds)
    from sklearn.metrics import root_mean_squared_error

    rmse = root_mean_squared_error(epoch_targets, epoch_preds)

    hist["epoch"].append(epoch)
    hist["mse"].append(mse)
    hist["mae"].append(mae)
    hist["rmse"].append(rmse)

    print(f"Epoch {epoch:02d} | MSE {mse:.4f} | MAE {mae:.4f} | RMSE {rmse:.4f}")

# ------------------------------------------------------------------
# 7 ─── Save metrics, plots, model
# ------------------------------------------------------------------
pd.DataFrame(hist).to_csv("plots/diff_train_metrics.csv", index=False)

plt.figure(figsize=(8,5))
plt.plot(hist["epoch"], hist["mse"], marker='o', color='red')
plt.title("Training MSE over Epochs")
plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.grid(True)
plt.savefig("plots/diff_train_loss.png"); plt.close()

torch.save(model.state_dict(), "difficulty_regressor.pt")
print("✅ Training complete — metrics & model saved.")