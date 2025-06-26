import torch.nn as nn
import torch.nn.functional as F

class MotionCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(MotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # in: [B, 3, T, N]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((16, 16))
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):  # x: [B, T, N, 3]
        x = x.permute(0, 3, 1, 2)  # -> [B, 3, T, N]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)  # reshape is safer after permute/pooling
        x = F.relu(self.fc1(x))
        return self.fc2(x)