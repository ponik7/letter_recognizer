from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # (64, 1, 28, 28)
            nn.MaxPool2d(2),  # (64, 16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),  # (64, 32, 12, 12)
            nn.Dropout(0.5),
            nn.MaxPool2d(2),  # (64, 32, 6, 6)
            nn.ReLU(),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(1152, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Dropout(0.5),
            nn.Linear(64, 27)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return F.log_softmax(x, -1)

