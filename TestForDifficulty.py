import os, pickle, ast
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    r2_score, root_mean_squared_error
)
from DataLoaderForDifficulty import DiveDifficultyDataset, pad_collate
from ModelForDifficulty import DifficultyRegressor

# ───────────────────────────── Config ─────────────────────────────
PT_DIR          = "videosTensors"
TEST_SPLIT_PKL  = "/Users/mrinalraj/Documents/FineDiving/train_test_split/test_split.pkl"
CSV_DIFF        = "/Users/mrinalraj/Downloads/WebDownload/Preprocess/DifficultyAndDiveScore_AllVides.csv"
MODEL_PATH      = "difficulty_regressor.pt"
DEVICE          = "cpu"
os.makedirs("plots", exist_ok=True)

# ───────────── Load difficulty dictionary from CSV ────────────────
df_diff = pd.read_csv(CSV_DIFF)
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

# ───────────── Load test_split ─────────────
with open(TEST_SPLIT_PKL, "rb") as f:
    test_pairs = pickle.load(f)

test_ids = {f"{x}_{y}" for x, y in test_pairs}
base_ds = DiveDifficultyDataset(pt_dir=PT_DIR, all_difficulties=all_difficulties)
test_idx = [i for i, vid in enumerate(base_ds.video_ids) if vid in test_ids]
test_ds = Subset(base_ds, test_idx)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=pad_collate, num_workers=0)

print(f"🧪 Test set size: {len(test_ds)}")

# ───────────── Load model and evaluate ─────────────
model = DifficultyRegressor(num_points=1000).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

all_preds, all_targets, all_vids = [], [], []
with torch.no_grad():
    for tracks, vis, diff, lengths, vids in test_loader:
        preds = model(tracks)
        all_preds.extend(preds.numpy())
        all_targets.extend(diff.numpy())
        all_vids.extend(vids)

all_preds   = np.array(all_preds)
all_targets = np.array(all_targets)

# ───────────── Compute Metrics ─────────────
mse  = mean_squared_error(all_targets, all_preds)
mae  = mean_absolute_error(all_targets, all_preds)
rmse = root_mean_squared_error(all_targets, all_preds)
r2   = r2_score(all_targets, all_preds)
exact = np.sum(np.round(all_preds, 3) == np.round(all_targets, 3))

# ───────────── Report ─────────────
print("\n📊 Difficulty Regression Test Metrics:")
print(f"MSE   : {mse:.4f}")
print(f"MAE   : {mae:.4f}")
print(f"RMSE  : {rmse:.4f}")
print(f"R²    : {r2:.4f}")
print(f"Exact predictions (rounded 3dp): {exact} / {len(all_targets)}")

# ───────────── Save predictions ─────────────
df_out = pd.DataFrame({
    "video_id": all_vids,
    "target"  : all_targets,
    "prediction": all_preds
})
df_out.to_csv("plots/diff_test_preds.csv", index=False)
print("✅ Saved predictions → plots/diff_test_preds.csv")