import os
import matplotlib.pyplot as plt
import random
import cv2

# ---- Set your dataset path ----
DATASET_PATH = r"C:\Users\hp\Desktop\Aerial Object Classification\data"

# ---- Folder paths ----
train_path = os.path.join(DATASET_PATH, "train")
test_path = os.path.join(DATASET_PATH, "test")
val_path = os.path.join(DATASET_PATH, "validation")

classes = ["bird", "drone"]   # folders must be exactly these names

def count_images(data_path):
    print(f"\nDataset: {data_path}")
    for cls in classes:
        cls_path = os.path.join(data_path, cls)
        count = len(os.listdir(cls_path))
        print(f"  {cls}: {count} images")

def show_sample_images(data_path, num_samples=4):
    print("\nShowing sample images...")

    plt.figure(figsize=(8, 8))

    for i, cls in enumerate(classes):
        cls_path = os.path.join(data_path, cls)
        img_name = random.choice(os.listdir(cls_path))
        img_path = os.path.join(cls_path, img_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(2, 2, i + 1)
        plt.imshow(img)
        plt.title(cls)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# ------------ RUN CHECKS ------------
print("📌 Checking TRAIN dataset:")
count_images(train_path)

print("\n📌 Checking TEST dataset:")
count_images(test_path)

print("\n📌 Checking VALIDATION dataset:")
count_images(val_path)

show_sample_images(train_path)
