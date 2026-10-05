# -*- coding: utf-8 -*-
"""
Module 11: Irregular Timestamp Resampling & Multi-Method Interpolation.
Transforms asynchronous IoT sensor packets into an aligned, equidistant time grid.
"""

import pandas as pd
import numpy as np

def resample_vitals_stream(
    df: pd.DataFrame,
    freq: str = "5min",
    method: str = "linear",
    max_gap_limit: int = 12
) -> pd.DataFrame:
    """
    Resamples irregular vitals to a fixed frequency (e.g. 5min) and performs interpolation.
    
    Parameters:
    - df: DataFrame with 'timestamp' column
    - freq: Resampling frequency string (e.g. '5min', '15min')
    - method: 'linear', 'time', 'cubic', or 'ffill'
    - max_gap_limit: Max consecutive missing steps allowed to interpolate
    """
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp")
    
    # Set timestamp as datetime index
    data = data.set_index("timestamp")
    
    # Numeric columns to resample
    num_cols = ["heart_rate", "spo2", "systolic_bp", "temperature"]
    
    # 1. Resample to regular intervals using mean
    resampled = data[num_cols].resample(freq).mean()
    
    # 2. Multi-method interpolation
    if method in ["linear", "time"]:
        interpolated = resampled.interpolate(method=method, limit=max_gap_limit)
    elif method == "cubic":
        interpolated = resampled.interpolate(method="spline", order=2, limit=max_gap_limit)
    elif method == "ffill":
        interpolated = resampled.ffill(limit=max_gap_limit)
    else:
        interpolated = resampled.interpolate(method="linear")
        
    # Backward fill any remaining leading NaNs
    interpolated = interpolated.bfill()
    
    # Reset index so timestamp is once again a column
    res = interpolated.reset_index()
    res["patient_id"] = data["patient_id"].iloc[0] if "patient_id" in data.columns else "PAT1001"
    return res
