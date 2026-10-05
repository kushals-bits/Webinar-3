# -*- coding: utf-8 -*-
"""
Master Text Pipeline Orchestrator (Module 9).
Compares:
1. Baseline: Raw text + Naive Bag-of-Words + SGDClassifier
2. Cleaned + Negation-Preserving Clinical Stopwords + Sublinear TF-IDF (1,2-grams) + Calibrated LinearSVC
3. Feature analysis: Top discriminatory clinical terms
"""

import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from .cleaner import TextCleaner
from .nltk_pipeline import NLTKPipeline

# Critical clinical stopwords: keep negations so "not febrile" != "febrile"
NEGATIONS = {"not", "no", "never", "none", "cannot", "nor", "neither", "without"}
CLINICAL_STOP_WORDS = list(set(ENGLISH_STOP_WORDS) - NEGATIONS)

def run_text_pipeline(csv_path: str) -> dict:
    print("\n" + "=" * 70)
    print("  [MODULE 9] Running Clinical Text Preprocessing Pipeline")
    print("=" * 70)
    
    df = pd.read_csv(csv_path)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_samples = len(texts)

    # 1. Baseline: Raw noisy text, simple Bag-of-Words with standard English stop words
    print(f"  Step 1: Baseline evaluation on {n_samples} raw clinical texts...")
    bow = CountVectorizer(max_features=5000, stop_words="english")
    X_bow = bow.fit_transform(texts)
    Xtr, Xte, ytr, yte = train_test_split(X_bow, y, test_size=0.2, random_state=42, stratify=y)
    
    sgd = SGDClassifier(random_state=42, max_iter=1000)
    sgd.fit(Xtr, ytr)
    baseline_pred = sgd.predict(Xte)
    baseline_acc = accuracy_score(yte, baseline_pred)
    baseline_f1 = f1_score(yte, baseline_pred, average="macro")

    # 2. Preprocessing: Cleaning + CamelCase + OCR Repair + Negation-preserving TF-IDF
    print("  Step 2: Cleaning (CamelCase splitting, OCR repair, HTML unescape, PHI de-identification)...")
    cleaner = TextCleaner(remove_phi=True, split_camel=True, fix_ocr=True)
    cleaned = cleaner.clean_batch(texts)

    print("  Step 3: Vectorization with Negation-Preserving Sublinear TF-IDF (1,2-grams)...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
        stop_words=CLINICAL_STOP_WORDS
    )
    X_tfidf = tfidf.fit_transform(cleaned)

    Xtr2, Xte2, ytr2, yte2 = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)
    svc = CalibratedClassifierCV(LinearSVC(max_iter=5000, random_state=42))
    svc.fit(Xtr2, ytr2)
    after_pred = svc.predict(Xte2)
    after_acc = accuracy_score(yte2, after_pred)
    after_f1 = f1_score(yte2, after_pred, average="macro")

    # Extract top keywords
    feat_names = np.array(tfidf.get_feature_names_out())
    mean_tfidf = np.asarray(X_tfidf.mean(axis=0)).flatten()
    top_idx = mean_tfidf.argsort()[-10:][::-1]
    top_kw = pd.DataFrame({"keyword": feat_names[top_idx], "mean_tfidf": mean_tfidf[top_idx]})

    print(f"  [RESULT] Text Baseline  -> Accuracy: {baseline_acc:.4f} | F1-Macro: {baseline_f1:.4f}")
    print(f"  [RESULT] Text Processed -> Accuracy: {after_acc:.4f} | F1-Macro: {after_f1:.4f} (Delta: +{after_f1-baseline_f1:.4f})")

    return {
        "modality": "Text (Module 9)",
        "model_name": "Calibrated LinearSVC + Negation TF-IDF",
        "baseline_acc": float(baseline_acc),
        "baseline_f1": float(baseline_f1),
        "after_acc": float(after_acc),
        "after_f1": float(after_f1),
        "delta_acc": float(after_acc - baseline_acc),
        "delta_f1": float(after_f1 - baseline_f1),
        "n_samples": n_samples,
        "vocab_size": len(feat_names),
        "top_keywords": top_kw,
    }
