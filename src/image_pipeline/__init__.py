"""
Image Preprocessing Pipeline Package (Webinar 3 – L4).
Covers loading, resizing, CLAHE normalisation, augmentation, and PCA reduction.
"""
from .loading import load_images_from_dirs
from .resizing import resize_batch
from .normalization import normalize_batch
from .augmentation import augment_batch
from .pipeline import run_image_pipeline
