"""
Pixel normalisation and CLAHE contrast enhancement.
"""
import cv2
import numpy as np

def apply_clahe(image: np.ndarray, clip_limit=2.0, grid=(8, 8)) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def normalize_batch(images: np.ndarray, use_clahe=True) -> np.ndarray:
    normed = []
    for img in images:
        if use_clahe:
            img = apply_clahe(img)
        normed.append(img.astype(np.float32) / 255.0)
    return np.array(normed)
