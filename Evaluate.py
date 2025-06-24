import torch
from torch.utils.data import DataLoader
from DataLoader import MotionDataset, motion_collate_fn
from Model import MotionCNN
# from LabelMappings import index_to_class  # Optional, for readable names
import numpy as np
from collections import defaultdict


original_to_index = {
    36: 0,
    15: 1,
    2: 2,
    1: 3,
    25: 4,
    4: 5,
    3: 6,
    17: 7,
    31: 8,
    6: 9,
    37: 10,
    27: 11,
    14: 12,
    13: 13,
    33: 14,
    32: 15,
    30: 16,
    23: 17,
    24: 18,
    35: 19,
    34: 20,
    29: 21,
    16: 22,
    5: 23,
    12: 24,
    7: 25,
    21: 26,
    22: 27,
    19: 28
}

index_to_class = {
    0: "Entry",
    1: "2.5 Soms.Pike",
    2: "Back",
    3: "Forward",
    4: "3.5 Soms.Tuck",
    5: "Inward",
    6: "Reverse",
    7: "3.5 Soms.Pike",
    8: "1.5 Twists",
    9: "Arm.Back",
    10: "0.5 Som.Pike",
    11: "4.5 Soms.Tuck",
    12: "2 Soms.Pike",
    13: "1.5 Soms.Pike",
    14: "2.5 Twists",
    15: "2 Twists",
    16: "1 Twist",
    17: "2.5 Soms.Tuck",
    18: "3 Soms.Tuck",
    19: "3.5 Twists",
    20: "3 Twists",
    21: "0.5 Twist",
    22: "3 Soms.Pike",
    23: "Arm.Forward",
    24: "1 Som.Pike",
    25: "Arm.Reverse",
    26: "1.5 Soms.Tuck",
    27: "2 Soms.Tuck",
    28: "4.5 Soms.Pike"
}

# === CONFIG ===
MODEL_PATH = "motion_cnn_trained_50.pt"
PT_FOLDER = "Tracks"
NUM_CLASSES = 29
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === LOAD ===
model = MotionCNN(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

dataset = MotionDataset(PT_FOLDER)
dataloader = DataLoader(dataset, batch_size=8, collate_fn=motion_collate_fn)

# === EVALUATE ===
correct = defaultdict(int)
total = defaultdict(int)

with torch.no_grad():
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(DEVICE).float(), labels.to(DEVICE)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)

        for t, p in zip(labels, preds):
            total[t.item()] += 1
            if t.item() == p.item():
                correct[t.item()] += 1

# === REPORT ===
print("\n📊 Per-Class Accuracy:")
for cls in range(NUM_CLASSES):
    acc = 100 * correct[cls] / total[cls] if total[cls] > 0 else 0
    class_name = index_to_class.get(cls, f"Class-{cls}")
    print(f"{class_name:20s} ({cls:02d}): {acc:.2f}%")