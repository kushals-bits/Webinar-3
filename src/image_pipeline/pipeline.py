# -*- coding: utf-8 -*-
"""
Master Image & Biosignal Pipeline Orchestrator (Module 10).
Handles:
1. Skin lesion images: CLAHE contrast enhancement, PCA dimensionality reduction, class weighting
2. Biosignals: Stethoscope PCG audio filtering and STFT spectrograms
"""

import os
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

from .loading import load_images_from_dirs
from .normalization import normalize_batch
from .augmentation import augment_batch
from .signal_processor import run_signal_pipeline

def run_image_pipeline(image_dir: str) -> dict:
    print("\n" + "=" * 70)
    print("  [MODULE 10] Running Dermatology Image Preprocessing Pipeline")
    print("=" * 70)
    
    images, labels, paths = load_images_from_dirs(image_dir)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n = len(images)
    print(f"  Loaded {n} lesion images across {len(le.classes_)} classes: {list(le.classes_)}")

    # 1. Baseline: Raw unnormalized flattened pixels + Logistic Regression
    X_raw = images.reshape(n, -1).astype(np.float32) / 255.0
    Xtr, Xte, ytr, yte = train_test_split(X_raw, y, test_size=0.2, random_state=42, stratify=y)

    import warnings
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        lr = LogisticRegression(max_iter=100, random_state=42)
        lr.fit(Xtr, ytr)
    baseline_pred = lr.predict(Xte)
    baseline_acc = accuracy_score(yte, baseline_pred)
    baseline_f1 = f1_score(yte, baseline_pred, average="macro")

    # 2. Preprocessed: CLAHE Contrast Enhancement + Fast Randomized PCA + Class-Weighted SVM
    print("  Applying CLAHE local contrast enhancement and channel normalization...")
    normed = normalize_batch(images, use_clahe=True)
    X_normed = normed.reshape(n, -1)

    print("  Fitting Randomized PCA (50 principal components)...")
    pca = PCA(n_components=50, svd_solver="randomized", random_state=42)
    X_pca = pca.fit_transform(X_normed)
    n_components = X_pca.shape[1]
    exp_var = float(np.sum(pca.explained_variance_ratio_))
    print(f"  Dimensionality reduced: {X_raw.shape[1]:,} pixels -> {n_components} PCA components ({exp_var*100:.1f}% variance retained)")

    Xtr2, Xte2, ytr2, yte2 = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

    # Balanced class weights to handle vision imbalance
    svm = SVC(C=5.0, kernel="rbf", class_weight="balanced", random_state=42)
    svm.fit(Xtr2, ytr2)
    svm_pred = svm.predict(Xte2)
    svm_acc = accuracy_score(yte2, svm_pred)
    svm_f1 = f1_score(yte2, svm_pred, average="macro")

    print(f"  [RESULT] Image Baseline (Raw Pixels) -> Accuracy: {baseline_acc:.4f} | F1-Macro: {baseline_f1:.4f}")
    print(f"  [RESULT] Image Processed (CLAHE+PCA) -> Accuracy: {svm_acc:.4f} | F1-Macro: {svm_f1:.4f} (Delta: +{svm_f1-baseline_f1:.4f})")

    return {
        "modality": "Image (Module 10)",
        "model_name": "RBF SVM (CLAHE + Randomized PCA-50)",
        "baseline_acc": float(baseline_acc),
        "baseline_f1": float(baseline_f1),
        "after_acc": float(svm_acc),
        "after_f1": float(svm_f1),
        "delta_acc": float(svm_acc - baseline_acc),
        "delta_f1": float(svm_f1 - baseline_f1),
        "n_samples": n,
        "n_pca_components": n_components,
        "pca": pca,
    }
