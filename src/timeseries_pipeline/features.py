# -*- coding: utf-8 -*-
"""
Module 11: Rolling Windows & Lag Feature Engineering.
Avoids temporal data leakage by strictly shifting features prior to evaluation.
"""

import pandas as pd
import numpy as np

def build_temporal_features(
    df: pd.DataFrame,
    lags: list = [1, 2, 3, 6],
    rolling_windows: list = [3, 6, 12]  # 15m, 30m, 60m windows for 5min data
) -> pd.DataFrame:
    """
    Constructs autoregressive lag features and rolling statistics.
    """
    feat_df = df.copy()
    vitals = ["heart_rate", "spo2", "systolic_bp", "temperature"]
    
    # 1. Autoregressive Lag Features
    for v in vitals:
        if v in feat_df.columns:
            for lag in lags:
                feat_df[f"{v}_lag{lag}"] = feat_df[v].shift(lag)
                
    # 2. Rolling Window Statistics (mean, std, min, max)
    for v in vitals:
        if v in feat_df.columns:
            for w in rolling_windows:
                # shift(1) guarantees no future data leakage into the current prediction step
                shifted = feat_df[v].shift(1)
                feat_df[f"{v}_roll_mean_{w}"] = shifted.rolling(window=w).mean()
                feat_df[f"{v}_roll_std_{w}"] = shifted.rolling(window=w).std().fillna(0)
                feat_df[f"{v}_roll_min_{w}"] = shifted.rolling(window=w).min()
                feat_df[f"{v}_roll_max_{w}"] = shifted.rolling(window=w).max()
                
    # Drop warm-up rows containing NaNs due to shifting
    feat_df = feat_df.dropna().reset_index(drop=True)
    return feat_df
