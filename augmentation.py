import cv2
import numpy as np
import random

def random_rotate(image):
    angle = random.randint(-20, 20)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(image, M, (w, h))

def random_flip(image):
    if random.random() > 0.5:
        return cv2.flip(image, 1)  # horizontal flip
    return image

def random_zoom(image):
    h, w = image.shape[:2]
    zoom_factor = random.uniform(1.0, 1.3)
    nh, nw = int(h * zoom_factor), int(w * zoom_factor)
    zoomed = cv2.resize(image, (nw, nh))

    # crop center
    x = (nw - w) // 2
    y = (nh - h) // 2
    return zoomed[y:y+h, x:x+w]

def random_brightness(image):
    factor = random.uniform(0.7, 1.3)
    image = image * factor
    return np.clip(image, 0, 255).astype("uint8")

def add_noise(image):
    # Create very small, lightweight noise
    noise = np.random.normal(0, 10, image.shape).astype("float32")

    noisy = image.astype("float32") + noise
    return np.clip(noisy, 0, 255).astype("uint8")


def augment(image):
    if random.random() > 0.5:
        image = random_rotate(image)
    if random.random() > 0.5:
        image = random_flip(image)
    if random.random() > 0.5:
        image = random_zoom(image)
    if random.random() > 0.5:
        image = random_brightness(image)
    if random.random() > 0.5:
        image = add_noise(image)

    return image
