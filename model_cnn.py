# model_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    

    def __init__(self, num_classes=2, input_size=(3, 224, 224), dropout=0.5):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 224 -> 224
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 112 -> 112
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# 56 -> 56
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 28
        )

        # determine flatten dimension dynamically with a dummy tensor
        dummy = torch.zeros(1, *input_size)
        feat = self.features(dummy)
        flatten_dim = feat.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Provide a convenience constructor for importers
def build_model(num_classes=2):
    return SimpleCNN(num_classes=num_classes)
