# efficientnet.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import classification_report
import numpy as np

from custom_dataset import CustomImageDataset


# -----------------------------
# CONFIG
# -----------------------------
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print(f"\n🔥 Using Device: {DEVICE}\n")


# -----------------------------
# TRANSFORMS
# -----------------------------
train_tfms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_tfms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# -----------------------------
# LOAD DATASETS
# -----------------------------
train_dataset = CustomImageDataset("data/train", transform=train_tfms)
val_dataset = CustomImageDataset("data/validation", transform=test_tfms)
test_dataset = CustomImageDataset("data/test", transform=test_tfms)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# -----------------------------
# LOAD EfficientNetB0
# -----------------------------
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# Freeze all feature layers
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

model = model.to(DEVICE)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)


# -----------------------------
# TRAINING FUNCTION
# -----------------------------
def train_one_epoch():
    model.train()
    correct, total = 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


# -----------------------------
# VALIDATION FUNCTION
# -----------------------------
def evaluate(loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            outputs = model(imgs)
            preds = outputs.argmax(1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


# -----------------------------
# TRAIN LOOP
# -----------------------------
best_val_acc = 0

print("\n🚀 Starting EfficientNetB0 Training...\n")

for epoch in range(1, EPOCHS + 1):
    train_acc = train_one_epoch()
    val_acc = evaluate(val_loader)

    print(f"Epoch {epoch}/{EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "efficientnet_best.pth")
        print("  ✅ Saved Best Model")


# -----------------------------
# TESTING
# -----------------------------
print("\n📌 Evaluating on Test Set...\n")

model.load_state_dict(torch.load("efficientnet_best.pth"))
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        preds = outputs.argmax(1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"🎯 Test Accuracy: {test_acc:.4f}\n")

print("📄 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["bird", "drone"]))
