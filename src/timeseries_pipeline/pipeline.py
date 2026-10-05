# -*- coding: utf-8 -*-
"""
Master Time-Series Pipeline Orchestrator (Module 11).
Compares:
1. Baseline: Raw irregular vitals with random split (demonstrating Data Leakage pitfall)
2. Preprocessed: 5-min Equidistant Resampling + Spline Imputation + Circadian Detrending
   + Autoregressive Lags + Strict Chronological Temporal Split
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from .resampler import resample_vitals_stream
from .decomposition import decompose_vitals_seasonality
from .features import build_temporal_features

def run_timeseries_pipeline(csv_path: str) -> dict:
    print("\n" + "=" * 70)
    print("  [MODULE 11] Running Remote Patient Monitoring Time-Series Pipeline")
    print("=" * 70)
    
    df_raw = pd.read_csv(csv_path)
    n_raw = len(df_raw)
    print(f"  Loaded raw vitals stream: {n_raw:,} records with irregular timestamps.")

    # Clinical Target: Predict high-risk tachycardia episodes in next observation (t+1)
    # 1. Baseline: Raw irregular vitals without resampling, imputation, or lag features
    df_base = df_raw.copy().sort_values("timestamp")
    df_base["target"] = (df_base["heart_rate"].shift(-1) > 85).astype(float)
    df_base = df_base.dropna().copy()
    df_base["target"] = df_base["target"].astype(int)

    feature_base = ["spo2", "systolic_bp", "temperature"]
    X_base = df_base[feature_base].values
    y_base = df_base["target"].values

    # Evaluate Baseline on strict Chronological split (last 25% future holdout)
    split_base = int(0.75 * len(X_base))
    Xtr_b, Xte_b = X_base[:split_base], X_base[split_base:]
    ytr_b, yte_b = y_base[:split_base], y_base[split_base:]

    lr_base = LogisticRegression(random_state=42, max_iter=200)
    lr_base.fit(Xtr_b, ytr_b)
    base_pred = lr_base.predict(Xte_b)
    baseline_acc = accuracy_score(yte_b, base_pred)
    baseline_f1 = f1_score(yte_b, base_pred, average="macro")

    # Teaching Demonstration: The "Data Leakage Trap"
    # Show how naive random shuffle split creates an artificially inflated score due to autocorrelation leakage
    Xtr_leak, Xte_leak, ytr_leak, yte_leak = train_test_split(X_base, y_base, test_size=0.25, random_state=42)
    lr_leak = LogisticRegression(random_state=42, max_iter=200).fit(Xtr_leak, ytr_leak)
    leakage_f1 = f1_score(yte_leak, lr_leak.predict(Xte_leak), average="macro")
    print(f"  [Pitfall Demo] Naive Random Split Leakage F1: {leakage_f1:.4f} vs Real Future Split F1: {baseline_f1:.4f}")

    # 2. Production Preprocessing Pipeline:
    # A. Resample to 5-min equidistant grid with Spline/Linear imputation
    print("  Step 1: Resampling irregular timestamps to aligned 5-minute grid...")
    df_resampled = resample_vitals_stream(df_raw, freq="5min", method="linear")

    # B. Extract Circadian Seasonality
    print("  Step 2: Circadian decomposition (24-hour biological cycle extraction)...")
    decomp = decompose_vitals_seasonality(df_resampled, col="heart_rate", period=288)

    # C. Engineer Autoregressive Lags and Rolling Windows
    print("  Step 3: Creating lag features (t-1, t-2, t-3, t-6) and rolling statistics...")
    df_feat = build_temporal_features(df_resampled, lags=[1, 2, 3, 6], rolling_windows=[3, 6, 12])

    # Target: Predict next-step tachycardia (t+1)
    df_feat["target"] = (df_feat["heart_rate"].shift(-1) > 85).astype(float)
    df_feat = df_feat.dropna().copy()
    df_feat["target"] = df_feat["target"].astype(int)
    
    feature_cols = [c for c in df_feat.columns if c not in ["timestamp", "patient_id", "target", "heart_rate"]]
    X = df_feat[feature_cols].values
    y = df_feat["target"].values

    # D. Strict Chronological Train/Test Split (No Data Leakage!)
    split_idx = int(0.75 * len(X))
    Xtr, Xte = X[:split_idx], X[split_idx:]
    ytr, yte = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    Xtr_scaled = scaler.fit_transform(Xtr)
    Xte_scaled = scaler.transform(Xte)  # Fitted ONLY on past training data!

    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    model.fit(Xtr_scaled, ytr)
    after_pred = model.predict(Xte_scaled)
    after_acc = accuracy_score(yte, after_pred)
    after_f1 = f1_score(yte, after_pred, average="macro")

    print(f"  [RESULT] Time-Series Baseline  (Raw irregular point) -> Accuracy: {baseline_acc:.4f} | F1-Macro: {baseline_f1:.4f}")
    print(f"  [RESULT] Time-Series Processed (Resampled + Lags)    -> Accuracy: {after_acc:.4f} | F1-Macro: {after_f1:.4f} (Delta: +{after_f1-baseline_f1:.4f})")

    return {
        "modality": "Time-Series (Module 11)",
        "model_name": "Gradient Boosting (5m Resample + Lags + No Leakage)",
        "baseline_acc": float(baseline_acc),
        "baseline_f1": float(baseline_f1),
        "after_acc": float(after_acc),
        "after_f1": float(after_f1),
        "delta_acc": float(after_acc - baseline_acc),
        "delta_f1": float(after_f1 - baseline_f1),
        "n_samples": len(df_feat),
        "n_features": len(feature_cols),
        "resampled_records": len(df_resampled),
    }
