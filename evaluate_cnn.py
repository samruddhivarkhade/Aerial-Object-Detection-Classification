import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from custom_dataset import CustomImageDataset
from model_cnn import SimpleCNN

# ----------------------- LOAD TEST DATA -----------------------
test_dataset = CustomImageDataset(
    root_dir="dataset/test",
    transform=True,   # ensures resizing + normalization
    augment=False     # no augmentation for test
)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ----------------------- LOAD MODEL -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=2)
model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
model.to(device)
model.eval()

criterion = nn.CrossEntropyLoss()

# ----------------------- EVALUATION -----------------------
test_loss = 0
correct = 0
total = 0
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

test_accuracy = 100 * correct / total
test_loss = test_loss / len(test_loader)

print("\n📌 FINAL TEST RESULTS")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")

# ----------------------- CONFUSION MATRIX -----------------------
cm = confusion_matrix(all_labels, all_preds)
class_names = ["Bird", "Drone"]

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - CNN Model")
plt.savefig("confusion_matrix.png")
plt.show()

# ----------------------- CLASSIFICATION REPORT -----------------------
print("\n📌 Classification Report")
print(classification_report(all_labels, all_preds, target_names=class_names))
