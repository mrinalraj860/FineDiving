import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from Model import MotionCNN
from DataLoader import MotionDataset, motion_collate_fn


# === CONFIG ===
PT_FOLDER = "Tracks"
NUM_CLASSES = 29
EPOCHS = 20
BATCH_SIZE = 8
LR = 1e-3
DEVICE = "mps" if torch.cuda.is_available() else "cpu"

# === INIT ===
model = MotionCNN(num_classes=NUM_CLASSES).to(DEVICE)
dataset = MotionDataset(PT_FOLDER)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=motion_collate_fn)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# === TRAIN LOOP ===
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(DEVICE).float(), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)  # [B, NUM_CLASSES]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = correct / total * 100
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss:.4f}  Accuracy: {acc:.2f}%")

# Save the model
torch.save(model.state_dict(), "motion_cnn_trained.pt")