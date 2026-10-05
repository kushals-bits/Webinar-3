"""Image loader – scans class subdirectories and returns image arrays + labels."""
import os
import numpy as np
from PIL import Image
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def load_images_from_dirs(root: str, target_size=(128, 128)):
    images, labels, paths = [], [], []
    root = Path(root)
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name
        for img_file in sorted(cls_dir.iterdir()):
            if img_file.suffix.lower() not in EXTS:
                continue
            img = Image.open(img_file).convert("RGB").resize(target_size)
            images.append(np.array(img))
            labels.append(cls_name)
            paths.append(str(img_file))
    return np.array(images), np.array(labels), paths
