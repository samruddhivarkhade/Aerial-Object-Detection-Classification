import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from custom_dataset import CustomImageDataset
import os

# -----------------------------------------------------
# Device
# -----------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🔥 Using Device: {device}\n")

# -----------------------------------------------------
# Data Transforms
# -----------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------------------------------
# Load Dataset
# -----------------------------------------------------
train_dataset = CustomImageDataset("data/train", transform=train_transform)
val_dataset   = CustomImageDataset("data/validation", transform=test_transform)
test_dataset  = CustomImageDataset("data/test", transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# -----------------------------------------------------
# Load Pretrained MobileNetV2
# -----------------------------------------------------
mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# Freeze feature layers
for param in mobilenet.features.parameters():
    param.requires_grad = False

# Replace classification head (1000 → 2)
mobilenet.classifier[1] = nn.Linear(mobilenet.classifier[1].in_features, 2)

mobilenet = mobilenet.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mobilenet.classifier.parameters(), lr=0.0008)

# -----------------------------------------------------
# Training Function
# -----------------------------------------------------
def train(model, loader):
    model.train()
    correct, total = 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total

# -----------------------------------------------------
# Validation Function
# -----------------------------------------------------
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

# -----------------------------------------------------
# Training Loop
# -----------------------------------------------------
best_val_acc = 0
os.makedirs("saved_models", exist_ok=True)

print("\n🚀 Starting MobileNetV2 Training...\n")

for epoch in range(5):
    train_acc = train(mobilenet, train_loader)
    val_acc = evaluate(mobilenet, val_loader)

    print(f"Epoch {epoch+1}/5 | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(mobilenet.state_dict(), "saved_models/best_mobilenet.pth")
        print("  ✅ Saved Best Model")

# -----------------------------------------------------
# Test Evaluation
# -----------------------------------------------------
print("\n📌 Evaluating on Test Set...\n")

mobilenet.load_state_dict(torch.load("saved_models/best_mobilenet.pth"))
mobilenet.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = mobilenet(images)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

test_acc = np.mean(np.array(y_true) == np.array(y_pred))
print(f"🎯 Test Accuracy: {test_acc:.4f}\n")

# -----------------------------------------------------
# Classification Report
# -----------------------------------------------------
print("📄 Classification Report:")
print(classification_report(y_true, y_pred, target_names=["bird", "drone"]))

# -----------------------------------------------------
# Confusion Matrix
# -----------------------------------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["bird", "drone"],
            yticklabels=["bird", "drone"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - MobileNetV2")
plt.show()
