# """
# Train DifficultyRegressor on dive-difficulty regression
# (using only train_split.pkl videos).  CPU-only.
# """

# import os
# import ast
# import pickle
# import torch
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset
# from torchinfo import summary
# from sklearn.metrics import mean_absolute_error, mean_squared_error

# from DataLoaderForDifficulty import DiveDifficultyDataset, pad_collate
# from ModelForDifficulty import DifficultyRegressor

# # ------------------------------------------------------------------
# # 1 ─── Paths and Hyper-params
# # ------------------------------------------------------------------
# PT_DIR          = "videosTensors"
# TRAIN_SPLIT_PKL = "/Users/mrinalraj/Documents/FineDiving/train_test_split/train_split.pkl"
# CSV_DIFFICULTY  = "/Users/mrinalraj/Downloads/WebDownload/Preprocess/DifficultyAndDiveScore_AllVides.csv"
# NUM_EPOCHS      = 30
# BATCH_SIZE      = 8
# LR              = 1e-4
# DEVICE          = "cpu"

# os.makedirs("plots", exist_ok=True)

# # ------------------------------------------------------------------
# # 2 ─── Build difficulty map  {video_id_str : difficulty(float)}
# # ------------------------------------------------------------------
# df_diff = pd.read_csv(CSV_DIFFICULTY)
# raw_diff = dict(zip(df_diff["video_id_str"], df_diff["all_difficulties"]))

# all_difficulties = {}
# for k, v in raw_diff.items():
#     if isinstance(v, str) and v.startswith("["):
#         try:
#             all_difficulties[k] = float(ast.literal_eval(v)[0])
#         except Exception:
#             continue
#     else:
#         all_difficulties[k] = float(v)



# # ------------------------------------------------------------------
# # 📊 Step 2.5 ─── Check distribution of difficulty scores
# # ------------------------------------------------------------------
# from collections import Counter
# import seaborn as sns

# # Convert all difficulties to 1 decimal bin (e.g. 3.1 → 3.1, 3.14 → 3.1)
# difficulty_rounded = [round(v, 1) for v in all_difficulties.values()]
# difficulty_freq = Counter(difficulty_rounded)

# # Print top 10 most common difficulties
# print("\n🔍 Top difficulty counts:")
# for diff, count in difficulty_freq.most_common(10):
#     print(f"  Difficulty {diff:.1f} → {count} samples")

# # Optional: plot histogram
# plt.figure(figsize=(8, 4))
# sns.histplot(difficulty_rounded, bins=20, kde=False, color="skyblue")
# plt.title("Distribution of Difficulty Scores")
# plt.xlabel("Difficulty")
# plt.ylabel("Frequency")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("plots/diff_histogram.png")
# plt.close()

# # ------------------------------------------------------------------
# # 3 ─── Allowed video IDs from train_split.pkl
# # ------------------------------------------------------------------
# with open(TRAIN_SPLIT_PKL, "rb") as f:
#     train_file_list = pickle.load(f)           # list of tuples (x, y)

# allowed_ids = {f"{x}_{y}" for x, y in train_file_list}

# # ------------------------------------------------------------------
# # 4 ─── Dataset restricted to allowed_ids
# # ------------------------------------------------------------------
# from collections import Counter
# from torch.utils.data import WeightedRandomSampler

# # Group allowed train indices
# base_ds = DiveDifficultyDataset(pt_dir=PT_DIR, all_difficulties=all_difficulties)
# keep_idx = [i for i, vid in enumerate(base_ds.video_ids) if vid in allowed_ids]
# train_ds = Subset(base_ds, keep_idx)

# # Extract difficulty scores for those
# train_diffs = [base_ds.diff[base_ds.video_ids[i]] for i in keep_idx]
# diff_bins = [round(x, 1) for x in train_diffs]
# freq = Counter(diff_bins)

# # Compute inverse-frequency sample weights
# total = sum(freq.values())
# bin_weight = {k: total / (len(freq) * v) for k, v in freq.items()}  # inverse proportional
# weights = [bin_weight[round(x, 1)] for x in train_diffs]

# sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# print(f"✅ Train dataset: {len(train_ds)} samples")

# train_loader = DataLoader(
#     train_ds, batch_size=BATCH_SIZE, sampler=sampler,
#     collate_fn=pad_collate, num_workers=0
# )

# # 🔍 Print difficulty imbalance summary
# print("\n📊 Weighted Sampling Summary:")
# for k, v in sorted(freq.items()):
#     print(f"  Difficulty {k:>4} → {v} samples | weight={bin_weight[k]:.4f}")

# # peek at one batch to display shapes
# tmp_tracks, tmp_vis, tmp_diff, tmp_len, tmp_vid = next(iter(train_loader))
# print("First batch tracks shape :", tmp_tracks.shape)   # [B, T, N, 2]
# print("First batch vis shape    :", tmp_vis.shape)      # [B, T, N]
# print("First batch diff shape   :", tmp_diff.shape)     # [B]
# print("All video IDs in batch  :", tmp_vid)            # [B]
# print("First Difference in batch  :", tmp_diff)         # str


# # ------------------------------------------------------------------
# # 5 ─── Model & Optimiser
# # ------------------------------------------------------------------
# model = DifficultyRegressor(num_points=1000).to(DEVICE)
# summary(model, input_size=tmp_tracks.shape)             # model summary

# optimizer = optim.Adam(model.parameters(), lr=LR)
# criterion = nn.MSELoss()

# # ------------------------------------------------------------------
# # 6 ─── Training loop
# # ------------------------------------------------------------------
# hist = {"epoch": [], "mse": [], "mae": [], "rmse": []}

# for epoch in range(1, NUM_EPOCHS + 1):
#     model.train()
#     epoch_losses, epoch_targets, epoch_preds = [], [], []

#     for tracks, vis, diff, lengths, vids in train_loader:
#         pred = model(tracks)                  # [B]
#         loss = criterion(pred, diff)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         epoch_losses.append(loss.item())
#         epoch_targets.extend(diff.numpy())
#         epoch_preds.extend(pred.detach().numpy())

#     mse  = np.mean(epoch_losses)
#     mae  = mean_absolute_error(epoch_targets, epoch_preds)
#     from sklearn.metrics import root_mean_squared_error

#     rmse = root_mean_squared_error(epoch_targets, epoch_preds)

#     hist["epoch"].append(epoch)
#     hist["mse"].append(mse)
#     hist["mae"].append(mae)
#     hist["rmse"].append(rmse)

#     print(f"Epoch {epoch:02d} | MSE {mse:.4f} | MAE {mae:.4f} | RMSE {rmse:.4f}")

# # ------------------------------------------------------------------
# # 7 ─── Save metrics, plots, model
# # ------------------------------------------------------------------
# pd.DataFrame(hist).to_csv("plots/diff_train_metrics.csv", index=False)

# plt.figure(figsize=(8,5))
# plt.plot(hist["epoch"], hist["mse"], marker='o', color='red')
# plt.title("Training MSE over Epochs")
# plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.grid(True)
# plt.savefig("plots/diff_train_loss.png"); plt.close()

# torch.save(model.state_dict(), "difficulty_regressor.pt")
# print("✅ Training complete — metrics & model saved.")


"""
Train DifficultyRegressor on dive-difficulty regression
(uses only train_split.pkl videos) with *both*
  • WeightedRandomSampler  AND
  • per-sample weighted MSE
to mitigate class imbalance.
"""

import os, ast, pickle, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchinfo import summary
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from collections import Counter
import seaborn as sns

from DataLoaderForDifficulty import DiveDifficultyDataset, pad_collate
from ModelForDifficulty import DifficultyRegressor

# ───────────────────────────── CONFIG ─────────────────────────────
PT_DIR          = "videosTensors"
TRAIN_SPLIT_PKL = "/Users/mrinalraj/Documents/FineDiving/train_test_split/train_split.pkl"
CSV_DIFF        = "/Users/mrinalraj/Downloads/WebDownload/Preprocess/DifficultyAndDiveScore_AllVides.csv"
NUM_EPOCHS      = 50
BATCH_SIZE      = 16
LR              = 1e-4
DEVICE          = "cpu"
os.makedirs("plots", exist_ok=True)

# ───────────────── 1. Build {video_id : difficulty} ───────────────
df_diff = pd.read_csv(CSV_DIFF)
raw = dict(zip(df_diff["video_id_str"], df_diff["all_difficulties"]))

all_diffs = {}
for k, v in raw.items():
    try:
        all_diffs[k] = float(ast.literal_eval(v)[0]) if isinstance(v, str) and v.startswith("[") else float(v)
    except Exception:
        continue

# ───────────────── 2. Inspect overall distribution ────────────────
rounded = [round(v, 1) for v in all_diffs.values()]
freq_all = Counter(rounded)
plt.figure(figsize=(8,4))
sns.histplot(rounded, bins=20, color="skyblue")
plt.title("Overall Difficulty Distribution"); plt.grid(True)
plt.savefig("plots/diff_histogram.png"); plt.close()

# ───────────────── 3. Restrict to train_split IDs ─────────────────
with open(TRAIN_SPLIT_PKL, "rb") as f:
    train_pairs = pickle.load(f)
train_ids = {f"{x}_{y}" for x, y in train_pairs}

base_ds   = DiveDifficultyDataset(PT_DIR, all_diffs)
train_idx = [i for i, vid in enumerate(base_ds.video_ids) if vid in train_ids]
train_ds  = Subset(base_ds, train_idx)

print(f"✅ Train dataset: {len(train_ds)} samples")

# ───────────────── 4. Build weights (inverse-frequency, 0.1 bins) ─
train_diffs = [base_ds.diff[base_ds.video_ids[i]] for i in train_idx]
bins = [round(x, 1) for x in train_diffs]
freq = Counter(bins)
total = sum(freq.values())
bin_w  = {b: total / (len(freq) * c) for b, c in freq.items()}  # inverse proportional
sample_w = [bin_w[round(x, 1)] for x in train_diffs]

sampler = WeightedRandomSampler(weights=sample_w,
                                num_samples=len(sample_w),
                                replacement=True)

print("\n📊 Weighted Sampling Summary:")
for b in sorted(freq):
    print(f"  Diff {b:>4} → {freq[b]} samples | weight={bin_w[b]:.4f}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          collate_fn=pad_collate, num_workers=0)

# ───────────────── 5. Model & optimiser ───────────────────────────
tracks_ex, _, _, _, _ = next(iter(train_loader))
model = DifficultyRegressor(num_points=1000).to(DEVICE)
summary(model, input_size=tracks_ex.shape)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss(reduction="none")   # we'll weight manually

# Pre-compute quick lookup  vid → weight
vid2weight = {
    base_ds.video_ids[i]: bin_w[round(base_ds.diff[base_ds.video_ids[i]], 1)]
    for i in train_idx
}

# ───────────────── 6. Training loop ───────────────────────────────
hist = {"epoch": [], "mse": [], "mae": [], "rmse": []}

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    losses, tgts, preds = [], [], []

    for tracks, vis, diff, lengths, vids in train_loader:
        pred = model(tracks)

        # fetch per-sample weights for this batch
        w = torch.tensor([vid2weight[v] for v in vids],
                         dtype=torch.float32, device=DEVICE)
        loss_vec = criterion(pred, diff)          # [B]
        loss = (loss_vec * w).mean()              # weighted mean

        optimizer.zero_grad(); loss.backward(); optimizer.step()

        losses.append(loss.item())
        tgts.extend(diff.numpy()); preds.extend(pred.detach().numpy())

    mse  = np.mean(losses)
    mae  = mean_absolute_error(tgts, preds)
    rmse = root_mean_squared_error(tgts, preds)

    hist["epoch"].append(epoch)
    hist["mse"].append(mse)
    hist["mae"].append(mae)
    hist["rmse"].append(rmse)

    print(f"Epoch {epoch:02d} | MSE {mse:.4f} | MAE {mae:.4f} | RMSE {rmse:.4f}")

# ───────────────── 7. Save stuff ──────────────────────────────────
pd.DataFrame(hist).to_csv("plots/diff_train_metrics.csv", index=False)
plt.figure(); plt.plot(hist["epoch"], hist["mse"], marker='o', color='red')
plt.title("Training MSE (weighted)"); plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.grid(True)
plt.savefig("plots/diff_train_loss.png"); plt.close()

torch.save(model.state_dict(), "difficulty_regressor.pt")
print("\n✅ Training complete — weighted model saved to difficulty_regressor.pt")