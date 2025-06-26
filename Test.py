# === TEST SPLIT EVALUATION =====================================================
from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from Model import MotionCNN
from DataLoader import MotionDataset, motion_collate_fn
import matplotlib.pyplot as plt
import os
from Model import MotionCNN
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchinfo import summary
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from Model import MotionCNN  
import pickle

BATCH_SIZE = 8
PT_FOLDER = "videosTensors"
DEVICE = "cpu"  
NUM_CLASSES = 29

with open("/Users/mrinalraj/Documents/FineDiving/train_test_split/test_split.pkl", "rb") as f:
    test_file_list = pickle.load(f)

test_dataset = MotionDataset(PT_FOLDER, file_list=test_file_list)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=motion_collate_fn)

model = MotionCNN(num_classes=NUM_CLASSES).to(DEVICE)

model.load_state_dict(torch.load("motion_cnn_trained_CNN.pt", map_location=DEVICE))
model.eval()

test_preds = []
test_targets = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE).float(), labels.to(DEVICE)
        outputs = model(inputs)
        predicted = torch.argmax(outputs, dim=1)

        test_preds.extend(predicted.cpu().numpy())
        test_targets.extend(labels.cpu().numpy())


test_acc = np.mean(np.array(test_preds) == np.array(test_targets)) * 100
print(f"\nTest Accuracy: {test_acc:.2f}%")


print("\n Classification Report (Test Set):")
print(classification_report(
    test_targets, test_preds,
    labels=list(range(NUM_CLASSES)),
    target_names=[f"Class_{i}" for i in range(NUM_CLASSES)],
    zero_division=0
))
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, _ = precision_recall_fscore_support(
    test_targets, test_preds, labels=list(range(NUM_CLASSES)), zero_division=0
)
df_test = pd.DataFrame({
    'Class': [f'Class_{i}' for i in range(NUM_CLASSES)],
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1
})
df_test.to_csv("plots/test_metrics.csv", index=False)


cm_test = confusion_matrix(test_targets, test_preds, labels=list(range(NUM_CLASSES)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test)
disp.plot(xticks_rotation=45, cmap='Oranges')
plt.title("Test Confusion Matrix")
plt.tight_layout()
plt.savefig("plots/test_confusion_matrix.png")
plt.close()

cm_df = pd.DataFrame(
    cm_test,
    index=[f"Actual_{i}" for i in range(NUM_CLASSES)],
    columns=[f"Pred_{i}" for i in range(NUM_CLASSES)]
)
cm_df.to_csv("plots/test_confusion_matrix.csv")

print("Test results saved to: plots/test_metrics.csv, plots/test_confusion_matrix.csv and plots/test_confusion_matrix.png")