import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from custom_dataset import CustomImageDataset
from model_cnn import SimpleCNN

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------
# Evaluation Function
# -------------------------------
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)


# -------------------------------
# Plot Training Curves
# -------------------------------
def plot_graphs(history):
    # Accuracy Plot
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.savefig("accuracy_plot.png")
    plt.close()

    # Loss Plot
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig("loss_plot.png")
    plt.close()


# -------------------------------
# Main Training Function
# -------------------------------
def main():

    print("📌 Loading datasets...")

    train_dataset = CustomImageDataset("data/train", transform=True, augment=True)
    val_dataset   = CustomImageDataset("data/validation", transform=True, augment=False)
    test_dataset  = CustomImageDataset("data/test", transform=True, augment=False)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model
    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # History tracking
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    best_val_acc = 0
    EPOCHS = 5

    print("\n🚀 Starting Training...\n")

    for epoch in range(1, EPOCHS + 1):

        model.train()
        total_loss = 0
        correct = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = correct / len(train_loader.dataset)

        val_loss, val_acc = evaluate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch}/{EPOCHS} | Train Acc: {train_acc:.4f}  Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("saved_models", exist_ok=True)
            torch.save(model.state_dict(), "saved_models/best_model.pth")
            print("  ✅ Saved Best Model")

    # Save training history
    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    print("\n📊 Plotting graphs...")
    plot_graphs(history)

    print("\n📌 Evaluating on Test Set...")
    model.load_state_dict(torch.load("saved_models/best_model.pth"))
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)

            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())

    # Metrics
    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\n🎯 Final Test Accuracy: {test_acc:.4f}")

    print("\n📄 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Bird", "Drone"]))

    # Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Bird", "Drone"],
                yticklabels=["Bird", "Drone"])

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/confusion_matrix.png")
    plt.close()

    print("\n🎉 Training Finished Successfully!")


# -------------------------------
# Run Script
# -------------------------------
if __name__ == "__main__":
    main()
