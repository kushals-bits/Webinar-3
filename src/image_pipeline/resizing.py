"""Resizing utilities – aspect-ratio preserving letterbox and direct resize."""
import numpy as np
from PIL import Image

def resize_batch(images: np.ndarray, target_size=(224, 224)) -> np.ndarray:
    resized = []
    for img_arr in images:
        img = Image.fromarray(img_arr).resize(target_size, Image.LANCZOS)
        resized.append(np.array(img))
    return np.array(resized)
