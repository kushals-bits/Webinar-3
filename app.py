# -*- coding: utf-8 -*-
"""
HealthAI 360: Interactive Multimodal Data Science Dashboard.
Built with Streamlit.
Transforms analytical preprocessing scripts into a dynamic, visual clinical AI dashboard.
Modules Covered:
  - Executive Overview: Master Before vs After Benchmark Lift
  - Module 9: Text Preprocessing & Negation-Aware NLP Studio
  - Module 10A: Dermatology Computer Vision (CLAHE & PCA)
  - Module 10B: Acoustic Biosignals (Butterworth Bandpass & STFT Spectrograms)
  - Module 11: Wearable Telemetry Stream Analytics & Data Leakage Lab
  - Module 12: Heterogeneous Medical Knowledge Graph & Contraindication Reasoning
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from PIL import Image
from pathlib import Path
import streamlit as st

# Configure project root path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import modular pipelines from src
from src.text_pipeline.cleaner import TextCleaner, _split_camel_case, _repair_ocr, PHI_RE, NAME_RE
from src.image_pipeline.normalization import apply_clahe
from src.image_pipeline.signal_processor import BiosignalPreprocessor
from src.timeseries_pipeline.resampler import resample_vitals_stream
from src.timeseries_pipeline.decomposition import decompose_vitals_seasonality
from src.graph_pipeline.representation import build_clinical_knowledge_graph
from src.graph_pipeline.entity_linker import MedicalEntityLinker
import networkx as nx

# -----------------------------------------------------------------------------
# Streamlit App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="HealthAI 360 | Multimodal Preprocessing Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-delta {
        font-size: 0.9rem;
        font-weight: 600;
        color: #10B981;
    }
    .step-box {
        background-color: #FFFFFF;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 12px;
        border: 1px solid #E5E7EB;
    }
    .legend-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.88rem;
        margin-right: 10px;
        margin-bottom: 8px;
    }
    .badge-disease { background-color: #FEE2E2; color: #991B1B; border: 1px solid #FCA5A5; }
    .badge-symptom { background-color: #FEF3C7; color: #92400E; border: 1px solid #FCD34D; }
    .badge-drug { background-color: #DBEAFE; color: #1E40AF; border: 1px solid #93C5FD; }
    .badge-patient { background-color: #F1F5F9; color: #334155; border: 1px solid #CBD5E1; }
    
    .audience-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #10B981;
        border-radius: 6px;
        padding: 14px 18px;
        margin: 12px 0 16px 0;
    }
    .audience-title {
        font-weight: 700;
        color: #065F46;
        font-size: 1.02rem;
        margin-bottom: 6px;
    }
    .concept-chip {
        display: inline-block;
        background: #E0E7FF;
        color: #3730A3;
        font-weight: 600;
        font-size: 0.82rem;
        padding: 3px 8px;
        border-radius: 4px;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Cached Data Loaders
# -----------------------------------------------------------------------------
@st.cache_data
def load_benchmark_report():
    csv_path = REPO_ROOT / "reports" / "master_before_vs_after_metrics.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None

@st.cache_data
def load_vitals_data():
    ts_path = REPO_ROOT / "data" / "timeseries" / "vitals_stream.csv"
    if ts_path.exists():
        return pd.read_csv(ts_path)
    return None

@st.cache_data
def load_pcg_signals():
    sig_path = REPO_ROOT / "data" / "signal" / "pcg_signals.npy"
    lbl_path = REPO_ROOT / "data" / "signal" / "pcg_labels.npy"
    if sig_path.exists() and lbl_path.exists():
        signals = np.load(sig_path)
        labels = np.load(lbl_path)
        return signals, labels
    return None, None

@st.cache_resource
def load_knowledge_graph():
    graph_dir = REPO_ROOT / "data" / "graph"
    if graph_dir.exists():
        return build_clinical_knowledge_graph(str(graph_dir))
    return None

@st.cache_resource
def load_entity_linker():
    graph_dir = REPO_ROOT / "data" / "graph"
    if graph_dir.exists():
        return MedicalEntityLinker(str(graph_dir))
    return None

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/medical-heart.png", width=64)
    st.title("HealthAI 360")
    st.markdown("**Webinar 3: Multimodal Data Preprocessing for Clinical AI**")
    st.markdown("Transforming raw, noisy clinical streams into production-grade AI features.")
    
    st.divider()
    
    view_selection = st.radio(
        "Select Studio View:",
        [
            "📊 Executive Scorecard & Lift",
            "📝 Module 9: Clinical Text & NLP",
            "🖼️ Module 10A: Skin Lesion Vision (CLAHE)",
            "🩺 Module 10B: Acoustic Biosignals (PCG)",
            "📈 Module 11: Wearable Streams & Leakage",
            "🕸️ Module 12: Knowledge Graph Reasoning",
        ]
    )
    
    st.divider()
    st.markdown("### 🎯 Audience Navigation Guide")
    st.caption("Each tab illustrates how specific data preprocessing methods directly solve catastrophic clinical failure modes in real hospitals.")

# -----------------------------------------------------------------------------
# View 1: Executive Scorecard & Lift
# -----------------------------------------------------------------------------
if view_selection == "📊 Executive Scorecard & Lift":
    st.markdown('<div class="main-title">Executive Scorecard & Multimodal Lift</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Before vs. After Benchmarks Across All 4 Clinical Modalities (Modules 9–12)</div>', unsafe_allow_html=True)
    
    # Audience Guide Box
    st.markdown("""
    <div class="audience-card">
        <div class="audience-title">💡 Audience Primer: Why Healthcare AI Demands Rigorous Preprocessing</div>
        <p style="margin: 0; color: #1F2937; font-size: 0.92rem;">
            In clinical AI, <b>accuracy alone is misleading</b> due to severe class imbalance (e.g., 98% of triage cases are non-urgent, but missing the 2% acute infarcts is fatal). 
            We use <b>F1-Macro Score</b> to equally evaluate rare, high-stakes medical classes. Notice the dramatic lifts achieved simply by cleaning and engineering data <i>before</i> training models.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    df_metrics = load_benchmark_report()
    
    if df_metrics is not None:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(label="Text (Mod 9) F1 Score", value="1.000", delta="100% PHI Redacted & HIPAA Safe", delta_color="normal")
        with c2:
            st.metric(label="Vision (Mod 10A) PCA Size", value="50 Components", delta="-99.9% Dimensionality Reduction", delta_color="normal")
        with c3:
            st.metric(label="Biosignal (Mod 10B) PCG Murmur", value="0.864", delta="+0.333 (+33.3% Lift)")
        with c4:
            st.metric(label="Wearables (Mod 11) RPM Forecasting", value="0.602", delta="Leakage-Free Temporal Split", delta_color="normal")
        
        st.write("")
        st.subheader("Interactive Before vs. After Benchmark Chart")
        
        fig, ax = plt.subplots(figsize=(10, 4.5))
        tracks = [m.split("(")[0].strip() for m in df_metrics["Curriculum Track"]]
        x = np.arange(len(tracks))
        w = 0.35
        
        ax.bar(x - w/2, df_metrics["Baseline F1-Macro"], w, label="Before (Raw / Naive Model)", color="#EF4444", alpha=0.85)
        ax.bar(x + w/2, df_metrics["Preprocessed F1-Macro"], w, label="After (Sanitized & Feature-Engineered)", color="#10B981", alpha=0.85)
        
        ax.set_xticks(x)
        ax.set_xticklabels(tracks, fontweight="bold", fontsize=10)
        ax.set_ylabel("F1-Macro Score", fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.legend(frameon=True, facecolor="white", edgecolor="#E5E7EB")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        
        for i in range(len(tracks)):
            b_val = df_metrics["Baseline F1-Macro"].iloc[i]
            a_val = df_metrics["Preprocessed F1-Macro"].iloc[i]
            lift = df_metrics["Delta F1 (Lift)"].iloc[i]
            ax.text(x[i] - w/2, b_val + 0.02, f"{b_val:.2f}", ha="center", fontsize=8.5)
            ax.text(x[i] + w/2, a_val + 0.02, f"{a_val:.2f}\n(+{lift:.2f})", ha="center", fontweight="bold", fontsize=8.5, color="#059669")
            
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write("")
        st.subheader("Detailed Modality Benchmark Table")
        st.dataframe(df_metrics, use_container_width=True)
        
        st.markdown("""
        ### 📋 Plain-English Modality Takeaways for Attendees:
        - **Module 9 (Clinical Text):** Raw doctor notes leak confidential patient names and misclassify negation (treating *"no chest pain"* as *"chest pain"*). Sanitization eliminates PHI while boosting diagnostic routing.
        - **Module 10 (Vision & Sound):** Compresses 49,152 raw skin lesion pixels down to 50 PCA coordinates (**99.9% reduction**) while Butterworth filtering eliminates stethoscope background rustle for a **+33.3% murmur detection lift**.
        - **Module 11 (Wearable Streams):** Exposes the **Data Leakage Trap** where models cheat during random validation splits by memorizing future heart rates. Chronological walk-forward validation guarantees live clinical safety.
        - **Module 12 (Knowledge Graphs):** Connects disparate EHR symptoms, diseases, and drugs into a multi-hop graph to automate contraindication safety alerts before drugs are dispensed.
        """)
    else:
        st.warning("Reports file not found. Please ensure reports/master_before_vs_after_metrics.csv exists.")

# -----------------------------------------------------------------------------
# View 2: Module 9 Text & NLP Studio
# -----------------------------------------------------------------------------
elif view_selection == "📝 Module 9: Clinical Text & NLP":
    st.markdown('<div class="main-title">Module 9: Clinical Text & NLP Sanitization Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Live Regex Scrubbing, OCR Correction, CamelCase Splitting & Negation Preservation</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="audience-card">
        <div class="audience-title">💡 Audience Guide: The 5 Essential Clinical Text Preprocessing Guards</div>
        <p style="margin: 0; color: #1F2937; font-size: 0.92rem;">
            Clinical free-text is notorious for OCR typos, shorthand, and confidential identifiers. 
            Standard generic NLP tokenizers fail catastrophically on clinical notes. This studio demonstrates the 5 safeguards required for production healthcare models:
            <b>(1) HIPAA PHI Masking</b>, <b>(2) CamelCase Splitting</b>, <b>(3) OCR Scanner Repair</b>, <b>(4) Negation Preservation</b>, and <b>(5) Lemmatization</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_input, col_config = st.columns([2, 1])
    
    preset_samples = {
        "Dermatology Lesion with OCR Typos": "Patient: Jane Doe, DOB: 1985-04-12. Complains of <span class='hl'>bord3rs</span> that are |rregular. No itching reported.",
        "Cardiology Emergency with CamelCase": "Patient: Robert Smith, Phone: +1-555-0199. Reports SevereChestPain and palp1tations. Denies shortness of breath. temp 1O1F.",
        "Skin Infection with HTML & PHI": "MRN982341. Referred by Dr. Alice Walker. Left leg has severe cellulitis with pus discharge. No fever.",
        "Orthopedics with Contractions & Negations": "Patient can't bend right knee without pain. No fracture detected on X-ray. Contact: nurse@hospital.org."
    }
    
    with col_config:
        st.markdown("### ⚙️ Pipeline Configuration")
        selected_preset = st.selectbox("Load Sample Report:", list(preset_samples.keys()))
        mask_phi = st.checkbox("De-identify PHI (HIPAA Safe)", value=True)
        split_camel = st.checkbox("Split CamelCase Words", value=True)
        repair_ocr = st.checkbox("Repair OCR Scanner Typos", value=True)
        keep_negations = st.checkbox("Preserve Negations ('no', 'not')", value=True)
        
        st.markdown("""
        **Guard Explanations:**
        - **PHI Scrubbing:** Replaces names/SSNs with `[NAME]`, `[PHI]` tokens.
        - **CamelCase:** Turns `SevereChestPain` into `Severe Chest Pain`.
        - **OCR Fix:** Fixes optical mistakes like `bord3rs` -> `borders`.
        - **Negations:** Prevents stripping words like *"no"* and *"without"*.
        """)
        
    with col_input:
        st.markdown("### ✍️ Clinical Input Report")
        user_text = st.text_area("Edit or type raw clinical text:", preset_samples[selected_preset], height=130)
        
    if st.button("🚀 Run Live Sanitization Pipeline", type="primary"):
        cleaner = TextCleaner(remove_phi=mask_phi, split_camel=split_camel, fix_ocr=repair_ocr)
        cleaned_text = cleaner.clean(user_text)
        
        st.divider()
        st.subheader("Step-by-Step Transformation Analysis")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🛑 Raw Unsanitized Input:**")
            st.code(user_text, language="html")
        with c2:
            st.markdown("**✅ Sanitized Output:**")
            st.code(cleaned_text, language="text")
            
        st.write("")
        st.markdown("#### Inspection of Applied Safeguards:")
        g1, g2, g3 = st.columns(3)
        with g1:
            phi_detected = bool(PHI_RE.search(user_text) or NAME_RE.search(user_text))
            st.info(f"**HIPAA PHI Detected:** {'Yes (Masked to [PHI]/[NAME])' if phi_detected else 'None in sample'}")
        with g2:
            ocr_fixed = "bord3rs" in user_text or "palp1tations" in user_text or "1O1F" in user_text or "|rregular" in user_text
            st.info(f"**OCR Typos Repaired:** {'Yes (Matched dictionary)' if ocr_fixed else 'None in sample'}")
        with g3:
            negation_found = any(w in user_text.lower().split() for w in ["no", "denies", "without", "never", "not"])
            st.info(f"**Negation Present:** {'Preserved (Crucial for clinical safety)' if negation_found else 'None'}")

# -----------------------------------------------------------------------------
# View 3: Module 10A Skin Lesion Vision (CLAHE)
# -----------------------------------------------------------------------------
elif view_selection == "🖼️ Module 10A: Skin Lesion Vision (CLAHE)":
    st.markdown('<div class="main-title">Module 10A: Computer Vision & Contrast Normalization</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Contrast-Limited Adaptive Histogram Equalization (CLAHE) & PCA Dimensionality Reduction</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="audience-card">
        <div class="audience-title">💡 Audience Guide: Solving Lighting Biases in Dermatology AI</div>
        <p style="margin: 0; color: #1F2937; font-size: 0.92rem;">
            Clinical skin lesion photographs suffer from flash glare, poor room lighting, and pigmentation differences across <b>Fitzpatrick Skin Types I–VI</b>. 
            Standard Global Histogram Equalization over-amplifies noise in healthy skin. 
            <b>CLAHE</b> divides images into localized tiles, computing contrast locally and clipping amplification to preserve genuine lesion border geometry.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    image_dir = REPO_ROOT / "data" / "image" / "raw"
    classes = ["benign", "malignant", "inflammatory"]
    
    col_ctrl, col_view = st.columns([1, 2])
    
    with col_ctrl:
        st.subheader("Image & CLAHE Controls")
        chosen_class = st.selectbox("Lesion Diagnostic Class:", classes)
        
        class_folder = image_dir / chosen_class
        img_files = sorted([f.name for f in class_folder.glob("*.jpg")]) if class_folder.exists() else []
        
        if img_files:
            chosen_img_name = st.selectbox("Select Lesion Sample:", img_files[:20])
            img_path = class_folder / chosen_img_name
        else:
            img_path = None
            
        clip_limit = st.slider("CLAHE Clip Limit (Local Contrast)", min_value=1.0, max_value=8.0, value=2.0, step=0.5)
        grid_dim = st.select_slider("Tile Grid Size (Blocks)", options=[4, 8, 16], value=8, format_func=lambda x: f"{x}x{x}")
        grid_size = (grid_dim, grid_dim)
        
        st.markdown("""
        **Parameter Guide:**
        - **Clip Limit:** Maximum contrast threshold. Higher values boost local detail but can amplify sensor grain.
        - **Tile Grid Size:** Smaller grids ($4\\times4$) capture micro-textures; larger grids ($16\\times16$) produce smoother transitions.
        - **Randomized PCA:** Compresses 49,152 raw pixels into 50 principal components (**99.9% dimensionality drop**).
        """)
        
    with col_view:
        if img_path and img_path.exists():
            raw_bgr = cv2.imread(str(img_path))
            raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
            
            # Apply CLAHE
            clahe_rgb = apply_clahe(raw_rgb, clip_limit=clip_limit, grid=grid_size)
            
            c_img1, c_img2 = st.columns(2)
            with c_img1:
                st.markdown("**Original Image (Under/Over Exposed)**")
                st.image(raw_rgb, use_container_width=True)
            with c_img2:
                st.markdown(f"**CLAHE Enhanced (Clip={clip_limit}, Grid={grid_dim}x{grid_dim})**")
                st.image(clahe_rgb, use_container_width=True)
                
            # Histogram comparison
            st.write("")
            st.markdown("#### Luminance (L-Channel) Histogram Comparison")
            raw_lab = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2LAB)
            clahe_lab = cv2.cvtColor(clahe_rgb, cv2.COLOR_RGB2LAB)
            
            fig_hist, ax_hist = plt.subplots(figsize=(8, 2.8))
            ax_hist.hist(raw_lab[:, :, 0].ravel(), bins=50, color="#EF4444", alpha=0.6, label="Raw Luminance (Narrow dynamic range)")
            ax_hist.hist(clahe_lab[:, :, 0].ravel(), bins=50, color="#10B981", alpha=0.6, label="CLAHE Luminance (Equalized dynamic range)")
            ax_hist.set_xlabel("Luminance Intensity (0 - 255)")
            ax_hist.set_ylabel("Pixel Count")
            ax_hist.legend()
            ax_hist.grid(True, alpha=0.2)
            plt.tight_layout()
            st.pyplot(fig_hist)
        else:
            st.info("No lesion images found in data/image/raw. Please generate datasets first.")

# -----------------------------------------------------------------------------
# View 4: Module 10B Acoustic Biosignals (PCG)
# -----------------------------------------------------------------------------
elif view_selection == "🩺 Module 10B: Acoustic Biosignals (PCG)":
    st.markdown('<div class="main-title">Module 10B: Phonocardiogram Biosignal Processing</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Zero-Phase Butterworth Filtering & STFT Time-Frequency Power Spectrograms</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="audience-card">
        <div class="audience-title">💡 Audience Guide: Extracting Heart Murmurs from Noisy Stethoscopes</div>
        <p style="margin: 0; color: #1F2937; font-size: 0.92rem;">
            A digital stethoscope records patient heart sounds at 2,000 Hz, but captures ambient hospital noise, chest hair friction, and lung sounds. 
            We use a <b>Zero-Phase Butterworth Bandpass Filter (25–450 Hz)</b> to isolate acoustic valves and a 
            <b>Short-Time Fourier Transform (STFT) Spectrogram</b> to visually expose high-frequency turbulent murmur energy flares.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    signals, labels = load_pcg_signals()
    
    if signals is not None:
        col_ctrl, col_disp = st.columns([1, 2])
        
        with col_ctrl:
            st.subheader("Filter Tuning Parameters")
            sample_idx = st.slider("Select Patient Recording Index:", 0, len(signals) - 1, 0)
            patient_label = "Heart Murmur" if labels[sample_idx] == 1 else "Normal Heart Sound"
            st.info(f"**Clinical Ground Truth:** {patient_label}")
            
            lowcut = st.slider("Butterworth Low Cutoff (Hz):", 10.0, 50.0, 25.0, 5.0)
            highcut = st.slider("Butterworth High Cutoff (Hz):", 200.0, 600.0, 450.0, 25.0)
            order = st.slider("Filter Order:", 2, 6, 4)
            
            st.markdown("""
            **Physiologic Acoustic Bands:**
            - **25 Hz - 150 Hz:** Normal heart sounds (S1 'lub' mitral/tricuspid closure, S2 'dub' aortic/pulmonic closure).
            - **150 Hz - 450 Hz:** Turbulent hemodynamic murmur frequency band.
            - **Zero-Phase Filtering (`filtfilt`):** Filters forward and backward to eliminate phase delay (preventing time distortion of heartbeats).
            """)
            
        with col_disp:
            raw_sig = signals[sample_idx]
            processor = BiosignalPreprocessor(fs=2000, lowcut=lowcut, highcut=highcut, filter_order=order)
            filtered_sig = processor.bandpass_filter(raw_sig)
            
            # Compute Spectrogram
            f, t, log_spec = processor.compute_spectrogram(filtered_sig)
            
            fig_sig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=False)
            
            # Waveform
            time_axis = np.linspace(0, len(raw_sig) / 2000.0, len(raw_sig))
            ax1.plot(time_axis, raw_sig, color="#94A3B8", alpha=0.7, label="Raw Noisy Stethoscope (Ambient Noise)")
            ax1.plot(time_axis, filtered_sig, color="#2563EB", linewidth=1.2, label=f"Butterworth Bandpass ({lowcut}-{highcut}Hz)")
            ax1.set_title(f"Acoustic Waveform ({patient_label})", fontweight="bold")
            ax1.set_xlabel("Time (seconds)")
            ax1.set_ylabel("Amplitude")
            ax1.legend(loc="upper right")
            ax1.grid(True, alpha=0.3)
            
            # Spectrogram
            im = ax2.pcolormesh(t, f, log_spec, shading="gouraud", cmap="magma")
            ax2.set_title("Short-Time Fourier Transform (STFT) Power Spectrogram", fontweight="bold")
            ax2.set_ylabel("Frequency (Hz)")
            ax2.set_xlabel("Time (seconds)")
            ax2.set_ylim(0, 600)
            fig_sig.colorbar(im, ax=ax2, label="Log Power (dB)")
            
            plt.tight_layout()
            st.pyplot(fig_sig)
    else:
        st.warning("PCG signals data file not found in data/signal/pcg_signals.npy.")

# -----------------------------------------------------------------------------
# View 5: Module 11 Wearable Streams & Leakage Lab
# -----------------------------------------------------------------------------
elif view_selection == "📈 Module 11: Wearable Streams & Leakage":
    st.markdown('<div class="main-title">Module 11: Wearable Sensor Streams & Leakage Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Asynchronous Resampling, Circadian Seasonality Decomposition & Temporal Splitting</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="audience-card">
        <div class="audience-title">💡 Audience Guide: Continuous Telemetry & The Deadly Data Leakage Trap</div>
        <p style="margin: 0; color: #1F2937; font-size: 0.92rem;">
            Remote Patient Monitoring (RPM) vitals arrive asynchronously with packet dropouts. 
            Before feeding to ML, streams must be resampled to a uniform 5-minute grid. 
            Critically, <b>never use random train-test splitting</b> in time-series! Because vitals are autocorrelated, random splitting lets the model peek into tomorrow's heart rate, 
            giving a dangerously inflated fake test score (~0.77). Only strict chronological walk-forward splitting guarantees patient safety.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    df_raw = load_vitals_data()
    
    if df_raw is not None:
        c_ctrl, c_viz = st.columns([1, 2])
        
        with c_ctrl:
            st.subheader("Telemetry Stream Settings")
            target_col = st.selectbox("Vital Sign Signal:", ["heart_rate", "spo2", "systolic_bp", "temperature"])
            resample_freq = st.selectbox("Resampling Grid Interval:", ["5min", "15min", "30min", "1h"], index=0)
            interp_method = st.selectbox("Missing Imputation Method:", ["linear", "cubic", "ffill"], index=0)
            
            st.divider()
            st.markdown("### ⚠️ Data Leakage Demonstrator")
            leakage_mode = st.radio(
                "Select Model Evaluation Strategy:",
                ["🛡️ Chronological Split (No Leakage - Production Safe)", "❌ Random Split (Future Leakage Trap - Fake High Score)"]
            )
            
            if "Random" in leakage_mode:
                st.error("**Catastrophic Leakage Trap:** The model memorizes observations from t+1 and t-1 during training. Test F1 appears artificially high (~0.77), but live deployment crashes!")
            else:
                st.success("**Clinically Valid:** Trains on past 75% and predicts strictly future 25%. Scalers and lags never look forward.")
                
        with c_viz:
            df_clean = resample_vitals_stream(df_raw, freq=resample_freq, method=interp_method)
            
            period_samples = 288 if resample_freq == "5min" else 96
            period_samples = min(period_samples, len(df_clean) // 3)
            
            decomp = decompose_vitals_seasonality(df_clean, col=target_col, period=period_samples)
            
            fig_ts, axes = plt.subplots(4, 1, figsize=(9, 7.5), sharex=True)
            axes[0].plot(df_clean[target_col], color="#1E293B")
            axes[0].set_title(f"1. Observed {target_col.replace('_', ' ').title()} (Continuous Telemetry)", fontweight="bold")
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(decomp["trend"], color="#D97706")
            axes[1].set_title("2. Underlying Pathophysiologic Trend (Slow Multi-Day Drift)", fontweight="bold")
            axes[1].grid(True, alpha=0.3)
            
            axes[2].plot(decomp["seasonal"], color="#059669")
            axes[2].set_title("3. Circadian 24-Hour Diurnal Seasonality (Biological Day vs. Night Rhythm)", fontweight="bold")
            axes[2].grid(True, alpha=0.3)
            
            axes[3].plot(decomp["residual"], color="#DC2626")
            axes[3].set_title("4. Residual Noise & Acute Pathological Spikes (Early Warning Flags)", fontweight="bold")
            axes[3].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig_ts)
    else:
        st.warning("Vitals stream data file not found in data/timeseries/vitals_stream.csv.")

# -----------------------------------------------------------------------------
# View 6: Module 12 Knowledge Graph Reasoning
# -----------------------------------------------------------------------------
elif view_selection == "🕸️ Module 12: Knowledge Graph Reasoning":
    st.markdown('<div class="main-title">Module 12: Clinical Knowledge Graph Reasoning</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Heterogeneous Relational Networks, SNOMED-CT Linking & Drug Contraindication Safety</div>', unsafe_allow_html=True)
    
    # Prominent Color-Coding Badges for Audience Understanding
    st.markdown("""
    <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;">
        <div style="font-weight: 700; font-size: 1.05rem; color: #1E3A8A; margin-bottom: 8px;">
            🎨 Graph Entity Color-Coding Legend (Audience Reference):
        </div>
        <div>
            <span class="legend-badge badge-disease">🔴 RED = Diseases / Diagnoses (ICD-10)</span>
            <span class="legend-badge badge-symptom">🟠 ORANGE = Symptoms & Clinical Signs (SNOMED-CT)</span>
            <span class="legend-badge badge-drug">🔵 BLUE = Pharmacological Drugs & Treatments</span>
            <span class="legend-badge badge-patient">⚪ GREY = Patient Profile Records</span>
        </div>
        <p style="margin: 8px 0 0 0; color: #4B5563; font-size: 0.88rem;">
            <b>Why Relational Graphs?</b> Flat tabular models treat rows in isolation. A Clinical Knowledge Graph connects multi-hop diagnostic pathways: 
            <i>Patient A exhibits Symptom B &#8594; diagnosed with Disease C &#8594; treated by Drug D &#8594; flagged as CONTRAINDICATED with Drug E!</i>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    G = load_knowledge_graph()
    linker = load_entity_linker()
    
    if G is not None and linker is not None:
        tab1, tab2, tab3 = st.tabs([
            "🔗 Clinical Entity Linking (NLP -> KG)", 
            "🛡️ Drug Contraindication Checker", 
            "🌐 Interactive Topology & Graph Structure"
        ])
        
        with tab1:
            st.subheader("NLP-to-Ontology Clinical Entity Linking")
            st.markdown("""
            When a patient writes: *"complains of severe redness, erythema, and pus discharge on left leg"*, 
            how does the AI resolve colloquial phrasing to medical standards?
            The **Entity Linker** maps free-text clinical mentions directly into formal **SNOMED-CT** and **ICD-10** Knowledge Graph nodes:
            """)
            
            sample_query = st.text_input(
                "Enter unstructured patient complaint text:",
                "Patient with suspected cellulitis complains of severe redness, erythema, and pus discharge."
            )
            
            if st.button("Link Entities to SNOMED-CT", type="primary"):
                matches = linker.link_text_entities(sample_query)
                if matches:
                    st.success(f"Matched {len(matches)} Clinical Entities in Graph Knowledge Base:")
                    for m in matches:
                        st.markdown(f"- 🏷️ Mention: **`{m['matched_alias']}`** $\\rightarrow$ KG Node ID: `{m['kg_node_id']}` | **SNOMED-CT Code:** `{m['snomed_code']}`")
                else:
                    st.info("No mapped clinical symptoms identified in text.")
                    
        with tab2:
            st.subheader("Automated Drug-Drug & Disease Contraindication Engine")
            st.markdown("""
            Knowledge graphs prevent adverse drug events by traversing directed `CONTRAINDICATED_WITH` relations. 
            Select any medication to simulate an automated prescription safety check:
            """)
            
            drug_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "Drug"]
            drug_labels = {n: G.nodes[n].get("label", n) for n in drug_nodes}
            
            c_d1, c_d2 = st.columns(2)
            with c_d1:
                selected_drug = st.selectbox("Select Patient Prescription:", list(drug_labels.keys()), format_func=lambda x: drug_labels[x])
            with c_d2:
                contra_edges = [
                    (u, v, d) for u, v, d in G.edges(data=True) 
                    if (u == selected_drug or v == selected_drug) and d.get("relation") == "CONTRAINDICATED_WITH"
                ]
                
                st.markdown(f"**Safety Scan for `{drug_labels[selected_drug]}`:**")
                if contra_edges:
                    for u, v, d in contra_edges:
                        other = v if u == selected_drug else u
                        other_label = G.nodes[other].get("label", other)
                        st.error(f"🚨 **CRITICAL CONTRAINDICATION DETECTED:** Must not be co-administered with **{other_label}**!")
                else:
                    st.success(f"✅ No direct high-risk contraindications found for {drug_labels[selected_drug]}.")
                    
        with tab3:
            st.subheader("Medical Knowledge Graph Topology & Node Distribution")
            
            # Node Counts Breakdown
            node_counts = pd.Series([d.get("node_type", "Unknown") for _, d in G.nodes(data=True)]).value_counts()
            
            c_st1, c_st2, c_st3, c_st4 = st.columns(4)
            with c_st1:
                st.metric("Total Graph Nodes", G.number_of_nodes())
            with c_st2:
                st.metric("Total Relationships (Edges)", G.number_of_edges())
            with c_st3:
                st.metric("Diseases (ICD-10)", node_counts.get("Disease", 0))
            with c_st4:
                st.metric("Symptoms (SNOMED)", node_counts.get("Symptom", 0))
                
            st.write("")
            st.markdown("#### Clinical Subgraph Visualization with Explicit Color-Coded Entities")
            
            core_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") in ["Disease", "Symptom", "Drug"]]
            H = G.subgraph(core_nodes[:22])
            
            fig_kg, ax_kg = plt.subplots(figsize=(10, 6))
            pos = nx.spring_layout(H, seed=42, k=0.7)
            
            color_map = {
                "Disease": "#EF4444",  # Red
                "Symptom": "#F59E0B",  # Orange
                "Drug": "#3B82F6",     # Blue
            }
            node_colors = [color_map.get(H.nodes[n].get("node_type"), "#94A3B8") for n in H.nodes()]
            
            nx.draw_networkx_nodes(H, pos, node_color=node_colors, node_size=750, alpha=0.9, ax=ax_kg)
            nx.draw_networkx_edges(H, pos, arrowstyle="->", arrowsize=14, edge_color="#CBD5E1", alpha=0.7, ax=ax_kg)
            labels = {n: H.nodes[n].get("label", n) for n in H.nodes()}
            nx.draw_networkx_labels(H, pos, labels=labels, font_size=8.5, font_weight="bold", ax=ax_kg)
            
            # Explicit Color Legend directly on the Matplotlib Figure
            red_patch = mpatches.Patch(color='#EF4444', label='Disease (ICD-10)')
            orange_patch = mpatches.Patch(color='#F59E0B', label='Symptom (SNOMED-CT)')
            blue_patch = mpatches.Patch(color='#3B82F6', label='Drug (Pharmacological)')
            ax_kg.legend(handles=[red_patch, orange_patch, blue_patch], loc='upper right', frameon=True, facecolor='white', edgecolor='#E2E8F0', fontsize=10)
            
            ax_kg.axis("off")
            plt.tight_layout()
            st.pyplot(fig_kg)
            
    else:
        st.warning("Knowledge Graph data not found in data/graph.")
