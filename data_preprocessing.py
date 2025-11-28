import os
import cv2
import numpy as np

DATASET_PATH = r"C:\Users\hp\Desktop\Aerial Object Classification\data"
IMAGE_SIZE = (224, 224)

def load_images_from_folder(folder):
    images = []
    labels = []
    
    for label_index, label in enumerate(["bird", "drone"]):
        class_folder = os.path.join(folder, label)
        for filename in os.listdir(class_folder):
            img_path = os.path.join(class_folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, IMAGE_SIZE)
            img = img / 255.0  # normalize
            images.append(img)
            labels.append(label_index)
    
    return np.array(images), np.array(labels)

print("📌 Loading TRAIN data...")
X_train, y_train = load_images_from_folder(os.path.join(DATASET_PATH, "train"))

print("📌 Loading VALIDATION data...")
X_val, y_val = load_images_from_folder(os.path.join(DATASET_PATH, "validation"))

print("📌 Loading TEST data...")
X_test, y_test = load_images_from_folder(os.path.join(DATASET_PATH, "test"))

print("\n✔ Shape Summary:")
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

print("\n🎉 Data preprocessing completed successfully!")
