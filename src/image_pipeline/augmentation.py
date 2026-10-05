"""
Image augmentation – flip, rotation, brightness/contrast jitter.
"""
import cv2
import numpy as np
import random

def augment_single(img: np.ndarray, target_size=224) -> np.ndarray:
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((target_size / 2, target_size / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (target_size, target_size), borderMode=cv2.BORDER_REFLECT)
    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-30, 30)
    img = np.clip(img * alpha + beta / 255.0, 0, 1).astype(np.float32)
    return img

def augment_batch(images: np.ndarray) -> np.ndarray:
    return np.array([augment_single(img) for img in images])
