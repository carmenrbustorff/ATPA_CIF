import torch
import torch.nn as nn

class BirdCLEFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 234)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.pool(x).view(x.size(0), -1)
        return torch.sigmoid(self.classifier(x))

# Load checkpoint
model = BirdCLEFModel()
state = torch.load("experiments/iter_0049_20260513_145457/model.pt", map_location="cpu")
model.load_state_dict(state)
print("Checkpoint loaded successfully.")