# -*- coding: utf-8 -*-
"""
Module 11: Time-Series Detrending & Circadian Seasonality Decomposition.
Extracts 24-hour diurnal biological cycles from vital signs streams.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

def decompose_vitals_seasonality(
    df: pd.DataFrame,
    col: str = "heart_rate",
    period: int = 288,  # 24 hours at 5-min intervals (24 * 12 = 288)
    model: str = "additive"
):
    """
    Decomposes a continuous vital sign series into Trend, Seasonal, and Residual components.
    """
    series = df[col].copy()
    if series.isnull().any():
        series = series.interpolate(method="linear").bfill()
        
    # Apply classical additive seasonal decomposition
    decomposition = seasonal_decompose(series, model=model, period=period, extrapolate_trend="period")
    
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    detrended = series - trend
    deseasonalized = series - seasonal
    
    return {
        "decomposition": decomposition,
        "trend": trend,
        "seasonal": seasonal,
        "residual": residual,
        "detrended": detrended,
        "deseasonalized": deseasonalized,
    }
