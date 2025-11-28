import cv2
from augmentation import augment

img_path = "data/train/bird/"  # any bird image path
import os

file = os.listdir(img_path)[0]
img = cv2.imread(img_path + file)

aug_img = augment(img)

cv2.imwrite("augmented_sample.jpg", aug_img)

print("Augmentation successful. Check 'augmented_sample.jpg'.")
