import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
#  DEVICE SETUP
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🔥 Using Device: {device}\n")


# ----------------------------
#  TRANSFORMS
# ----------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ----------------------------
#  LOAD DATASETS
# ----------------------------
def load_datasets():
    train_ds = ImageFolder("data/train", transform=train_transform)
    val_ds = ImageFolder("data/validation", transform=test_transform)
    test_ds = ImageFolder("data/test", transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader, train_ds.classes


# ----------------------------
#  RESNET MODEL
# ----------------------------
def build_resnet(num_classes):
    model = models.resnet18(weights="IMAGENET1K_V1")  # pretrained=True alternative

    # Freeze base layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace FC layer for 2 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model.to(device)


# ----------------------------
#  TRAINING LOOP
# ----------------------------
def train_model(model, train_loader, val_loader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

    best_val_acc = 0
    history = {"train_acc": [], "val_acc": []}

    print("\n🚀 Starting ResNet18 Training...\n")

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, pred = outputs.max(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                _, pred = out.max(1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("saved_models", exist_ok=True)
            torch.save(model.state_dict(), "saved_models/resnet_best.pth")
            print("  ✅ Saved Best Model")

    # Save history
    with open("resnet_history.json", "w") as f:
        json.dump(history, f)

    return history


# ----------------------------
#  TEST EVALUATION
# ----------------------------
def evaluate(model, test_loader, class_names):
    print("\n📌 Evaluating on Test Set...\n")

    model.eval()
    preds, true = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, pred = outputs.max(1)
            preds.extend(pred.cpu().numpy())
            true.extend(labels.numpy())

    preds = np.array(preds)
    true = np.array(true)

    accuracy = (preds == true).mean()
    print(f"🎯 Test Accuracy: {accuracy:.4f}\n")

    # Classification report
    print("📄 Classification Report:")
    print(classification_report(true, preds, target_names=class_names))

    # Confusion matrix plot
    cm = confusion_matrix(true, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("ResNet18 Confusion Matrix")
    plt.show()


# ----------------------------
#  MAIN
# ----------------------------
def main():
    train_loader, val_loader, test_loader, class_names = load_datasets()

    model = build_resnet(num_classes=len(class_names))

    history = train_model(model, train_loader, val_loader, epochs=5)

    # Load best model
    model.load_state_dict(torch.load("saved_models/resnet_best.pth"))

    evaluate(model, test_loader, class_names)


if __name__ == "__main__":
    main()
