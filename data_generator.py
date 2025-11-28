import os
import cv2
import numpy as np

IMAGE_SIZE = (224, 224)

class DataGenerator:
    def __init__(self, folder):
        self.folder = folder
        self.classes = ["bird", "drone"]
        self.filepaths = []

        for label_index, label in enumerate(self.classes):
            class_folder = os.path.join(folder, label)
            for filename in os.listdir(class_folder):
                self.filepaths.append((os.path.join(class_folder, filename), label_index))
    
    def __len__(self):
        return len(self.filepaths)

    def load_batch(self, batch_size=32):
        """Yield batches instead of loading entire dataset"""
        for i in range(0, len(self.filepaths), batch_size):
            batch_files = self.filepaths[i : i + batch_size]
            images = []
            labels = []

            for path, label in batch_files:
                img = cv2.imread(path)
                if img is None:
                    continue
                img = cv2.resize(img, IMAGE_SIZE)
                img = img.astype("float32") / 255.0  # Normalize
                images.append(img)
                labels.append(label)

            yield np.array(images), np.array(labels)
