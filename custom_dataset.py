import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import random


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=True, augment=False):
        self.root_dir = root_dir
        self.transform_flag = transform
        self.augment_flag = augment

        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_folder = os.path.join(root_dir, cls)
            for img_file in os.listdir(cls_folder):
                if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cls_folder, img_file), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def _augment(self, img):
        """Simple augmentations only when augment=True"""
        if random.random() < 0.5:
            img = cv2.flip(img, 1)  # horizontal flip

        if random.random() < 0.3:
            angle = random.randint(-15, 15)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h))

        return img

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (224, 224))

        # Apply augmentation
        if self.augment_flag:
            img = self._augment(img)

        # Convert to numpy (copy fixes negative stride)
        img_np = np.array(img).copy()

        # Normalize
        img_np = img_np / 255.0

        # Convert to torch tensor (C, H, W)
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float()

        return img_t, label
