# compare_models.py

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from custom_dataset import CustomImageDataset
from model_cnn import SimpleCNN  # your custom CNN model class


# -----------------------------
# CONFIG
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
BATCH_SIZE = 32

print(f"\n🔥 Using Device: {DEVICE}\n")


# -----------------------------
# TEST TRANSFORMS
# -----------------------------
test_tfms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# -----------------------------
# LOAD TEST DATASET
# -----------------------------
test_dataset = CustomImageDataset("data/test", transform=test_tfms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"📌 Test Samples: {len(test_dataset)}\n")


# -----------------------------
# MODEL LOADER FUNCTIONS
# -----------------------------

def load_custom_cnn():
    model = SimpleCNN(num_classes=2)
    model.load_state_dict(torch.load(r"cnn_model.pth", map_location=DEVICE))
    return model.to(DEVICE)


def load_resnet18():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(r"saved_models/resnet_best.pth", map_location=DEVICE))
    return model.to(DEVICE)


def load_mobilenet():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(r"saved_models/best_mobilenet.pth", map_location=DEVICE))
    return model.to(DEVICE)


def load_efficientnet():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(r"efficientnet_best.pth", map_location=DEVICE))
    return model.to(DEVICE)


# -----------------------------
# EVALUATION FUNCTION
# -----------------------------
def evaluate(model):
    model.eval()
    preds, labels_list = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)

            outputs = model(imgs)
            predictions = outputs.argmax(1).cpu().numpy()

            preds.extend(predictions)
            labels_list.extend(labels.numpy())

    return accuracy_score(labels_list, preds)


# -----------------------------
# MODEL COMPARISON
# -----------------------------
results = {}

print("🔍 Evaluating Models...\n")

models_dict = {
    "Custom CNN": load_custom_cnn,
    "ResNet18": load_resnet18,
    "MobileNetV2": load_mobilenet,
    "EfficientNetB0": load_efficientnet
}

for name, loader in models_dict.items():
    print(f"➡ Loading {name}...")
    model = loader()
    acc = evaluate(model)
    results[name] = acc
    print(f"   🎯 Test Accuracy: {acc:.4f}\n")


# -----------------------------
# SHOW RESULTS
# -----------------------------
print("\n📊 FINAL MODEL COMPARISON\n")
for name, acc in results.items():
    print(f"{name:20} : {acc:.4f}")

# -----------------------------
# BAR CHART
# -----------------------------
plt.figure(figsize=(7, 5))
plt.bar(results.keys(), results.values())
plt.title("Model Comparison on Test Set")
plt.ylabel("Accuracy")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
