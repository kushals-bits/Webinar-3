# -*- coding: utf-8 -*-
"""
Module 10 (Signal Track): Biosignal & Audio Preprocessing.
Handles:
- Phonocardiogram (PCG) digital stethoscope audio / biosignal processing
- Sampling rate & Nyquist frequency compliance (f_Nyquist = f_s / 2)
- Zero-phase Butterworth bandpass filtering (25 Hz - 450 Hz for heart sounds)
- Baseline wander and ambient friction noise suppression
- Time-Frequency transformation: Short-Time Fourier Transform (STFT) & Spectrograms
- Feature extraction for classifier (spectral centroid, band energies, rolloff)
"""

import os
import json
import numpy as np
from pathlib import Path
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

class BiosignalPreprocessor:
    """Zero-phase Butterworth filter and STFT spectrogram extractor."""
    def __init__(self, fs: int = 2000, lowcut: float = 25.0, highcut: float = 450.0, filter_order: int = 4):
        self.fs = fs
        self.nyquist = 0.5 * fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = filter_order
        
        # Design Butterworth bandpass filter
        low = lowcut / self.nyquist
        high = highcut / self.nyquist
        self.b, self.a = signal.butter(self.order, [low, high], btype="band")

    def bandpass_filter(self, raw_signal: np.ndarray) -> np.ndarray:
        """Applies zero-phase forward-backward Butterworth bandpass filter."""
        return signal.filtfilt(self.b, self.a, raw_signal)

    def compute_spectrogram(self, sig: np.ndarray, nperseg: int = 128, noverlap: int = 64):
        """Computes Short-Time Fourier Transform (STFT) power spectrogram."""
        f, t, Sxx = signal.spectrogram(sig, fs=self.fs, nperseg=nperseg, noverlap=noverlap)
        log_spectrogram = np.log10(Sxx + 1e-10)
        return f, t, log_spectrogram

    def extract_features(self, sig: np.ndarray) -> np.ndarray:
        """Extracts key clinical spectral and acoustic features."""
        filtered = self.bandpass_filter(sig)
        f, t, Sxx = signal.spectrogram(filtered, fs=self.fs, nperseg=128, noverlap=64)
        
        # 1. Band energies
        s1_s2_band = np.mean(Sxx[(f >= 25) & (f <= 150), :])
        murmur_band = np.mean(Sxx[(f > 150) & (f <= 450), :])
        ratio = murmur_band / (s1_s2_band + 1e-8)
        
        # 2. Spectral centroid
        centroid = np.sum(f[:, None] * Sxx, axis=0) / (np.sum(Sxx, axis=0) + 1e-8)
        mean_centroid = np.mean(centroid)
        
        # 3. Zero crossing rate
        zcr = np.mean(np.abs(np.diff(np.sign(filtered)))) / 2.0
        
        # 4. Signal RMS energy
        rms = np.sqrt(np.mean(filtered**2))
        
        return np.array([s1_s2_band, murmur_band, ratio, mean_centroid, zcr, rms], dtype=np.float32)

def run_signal_pipeline(signal_dir: str) -> dict:
    """Executes the Biosignal/Audio preprocessing pipeline and benchmarks results."""
    sig_path = Path(signal_dir)
    signals = np.load(sig_path / "pcg_signals.npy")
    labels_str = np.load(sig_path / "pcg_labels.npy")
    with open(sig_path / "signal_meta.json", "r") as f:
        meta = json.load(f)
    fs = meta.get("sampling_rate", 2000)

    y = (labels_str == "murmur").astype(int)
    n_samples = len(signals)

    # 1. Baseline: Raw downsampled signal points directly into classifier
    raw_downsampled = signals[:, ::10]  # Downsample raw waveform
    Xtr, Xte, ytr, yte = train_test_split(raw_downsampled, y, test_size=0.25, random_state=42, stratify=y)
    
    rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_base.fit(Xtr, ytr)
    base_pred = rf_base.predict(Xte)
    baseline_acc = accuracy_score(yte, base_pred)
    baseline_f1 = f1_score(yte, base_pred, average="macro")

    # 2. Preprocessed: Bandpass filtering + STFT Spectral Feature Extraction
    proc = BiosignalPreprocessor(fs=fs)
    features = np.array([proc.extract_features(s) for s in signals])

    Xtr2, Xte2, ytr2, yte2 = train_test_split(features, y, test_size=0.25, random_state=42, stratify=y)
    rf_proc = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_proc.fit(Xtr2, ytr2)
    proc_pred = rf_proc.predict(Xte2)
    after_acc = accuracy_score(yte2, proc_pred)
    after_f1 = f1_score(yte2, proc_pred, average="macro")

    print(f"  [RESULT] Signal Baseline   -> Accuracy: {baseline_acc:.4f} | F1-Macro: {baseline_f1:.4f}")
    print(f"  [RESULT] Signal Processed  -> Accuracy: {after_acc:.4f} | F1-Macro: {after_f1:.4f} (Delta: +{after_f1-baseline_f1:.4f})")

    return {
        "modality": "Biosignal (Module 10)",
        "model_name": "Random Forest + STFT Spectrogram Features",
        "baseline_acc": float(baseline_acc),
        "baseline_f1": float(baseline_f1),
        "after_acc": float(after_acc),
        "after_f1": float(after_f1),
        "delta_acc": float(after_acc - baseline_acc),
        "delta_f1": float(after_f1 - baseline_f1),
        "n_samples": n_samples,
    }
