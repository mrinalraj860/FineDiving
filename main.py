import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from Model import MotionCNN
from DataLoader import MotionDataset, motion_collate_fn
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchinfo import summary
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from MotionTransformer import MotionTransformer  # Assuming you have a MotionTransformer class defined

# === CONFIG ===
PT_FOLDER = "Tracks"
NUM_CLASSES = 29
EPOCHS = 50
BATCH_SIZE = 8
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === INIT ===
os.makedirs("plots", exist_ok=True)
model = MotionTransformer(num_classes=NUM_CLASSES).to(DEVICE)
dataset = MotionDataset(PT_FOLDER)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=motion_collate_fn)


# criterion = nn.CrossEntropyLoss()
all_labels = [int(label) for _,label,x in dataset]
present_classes = np.unique(all_labels)
all_class_indices = np.arange(NUM_CLASSES)

# Compute weights only for present classes
weights_present = compute_class_weight(class_weight='balanced', classes=present_classes, y=all_labels)

# Initialize all weights with default value (e.g., 0 or 1)
full_weights = np.zeros(NUM_CLASSES, dtype=np.float32)

# Assign computed weights to correct positions
for cls, w in zip(present_classes, weights_present):
    full_weights[cls] = w

print("Class weights:", full_weights)
# Convert to torch tensor
class_weights_tensor = torch.tensor(full_weights, dtype=torch.float32).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=LR)

# print(dataset[0])
print("Classes in dataset:", np.unique(all_labels))
missing = set(range(NUM_CLASSES)) - set(np.unique(all_labels))
print("Missing classes in training set:", missing)

# === PRINT MODEL SUMMARY ===
summary(model, input_size=(1, 64, 1000, 3))

from sklearn.metrics import precision_recall_fscore_support

train_losses = []
train_accuracies = []
per_class_metrics = []
all_preds = []
all_targets = []
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    epoch_preds = []
    epoch_targets = []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(DEVICE).float(), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        epoch_preds.extend(predicted.cpu().numpy())
        epoch_targets.extend(labels.cpu().numpy())

    acc = correct / total * 100
    train_losses.append(total_loss)
    train_accuracies.append(acc)
    all_preds.extend(epoch_preds)
    all_targets.extend(epoch_targets)

    precision, recall, f1, _ = precision_recall_fscore_support(
        epoch_targets, epoch_preds, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    per_class_metrics.append({
        'Epoch': epoch + 1,
        **{f'Precision_Class_{i}': p for i, p in enumerate(precision)},
        **{f'Recall_Class_{i}': r for i, r in enumerate(recall)},
        **{f'F1_Class_{i}': f for i, f in enumerate(f1)}
    })

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss:.4f}  Accuracy: {acc:.2f}%")

# === SAVE PER-CLASS PRECISION / RECALL / F1 ===
df_metrics = pd.DataFrame(per_class_metrics)
df_metrics.to_csv("plots/per_class_metrics.csv", index=False)

# === SAVE MODEL ===
torch.save(model.state_dict(), "motion_cnn_trained_Transformer.pt")

# === SAVE METRICS TO CSV ===
df = pd.DataFrame({
    'Epoch': list(range(1, EPOCHS + 1)),
    'Loss': train_losses,
    'Accuracy (%)': train_accuracies
})
df.to_csv("plots/training_metrics.csv", index=False)

# === PLOT LOSS ===
plt.figure(figsize=(8, 5))
plt.plot(df['Epoch'], df['Loss'], marker='o', color='red')
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("plots/training_loss.png")
plt.close()

# === PLOT ACCURACY ===
plt.figure(figsize=(8, 5))
plt.plot(df['Epoch'], df['Accuracy (%)'], marker='s', color='blue')
plt.title("Training Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.savefig("plots/training_accuracy.png")
plt.close()

# === PLOT BOTH ===
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Loss'], label="Loss", color='red', marker='o')
plt.plot(df['Epoch'], df['Accuracy (%)'], label="Accuracy", color='blue', marker='s')
plt.title("Loss and Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.savefig("plots/training_combined.png")
plt.close()

# === CONFUSION MATRIX (final epoch) ===
cm = confusion_matrix(all_targets, all_preds, labels=list(range(NUM_CLASSES)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(xticks_rotation=45, cmap='Blues')
plt.title("Confusion Matrix (Final Epoch)")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png")
plt.close()

print("✅ Training complete. Metrics, plots, and confusion matrix saved in 'plots/' folder.")