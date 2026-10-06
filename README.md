# Webinar 3: Mastering Multimodal Data Preprocessing for AI
## HealthAI 360: Integrated Tele-Diagnostics & Remote Patient Monitoring Pipeline

> **Curriculum Tracks Covered:**
> * **Module 9:** Text Preprocessing & NLP-Ready Pipelines (NLTK / spaCy / TF-IDF / Regex)
> * **Module 10:** Image & Signal Preprocessing for AI (OpenCV CLAHE / PCA / Scipy Signal / STFT Spectrograms)
> * **Module 11:** Time-Series Preprocessing & Leakage Prevention (Pandas Resampling / Statsmodels / Lags)
> * **Module 12:** Graph Data Pre-processing & Knowledge Graphs (NetworkX / DeepWalk Embeddings / SNOMED Linking)

---

## 🏛️ End-to-End System Architecture

```
                                  Incoming Multi-Modal Patient Stream
                                                  │
            ┌─────────────────────┬───────────────┴───────────────┬─────────────────────┐
            ▼                     ▼                               ▼                     ▼
     [Module 9: Text]    [Module 10: Image/Signal]       [Module 11: Time-Series]  [Module 12: Graph / KG]
     Patient Triage      Lesion Photos & PCG             Wearable Vitals Stream    Clinical Knowledge
     Complaints & OCR    Acoustic Biosignals             (HR, SpO2, BP, Temp)      Graph (EHR + Ontologies)
            │                     │                               │                     │
            ▼                     ▼                               ▼                     ▼
     • HTML / PHI Scrub  • CLAHE Contrast Normalization   • 5-min Grid Resample • Topological Cleaning
     • CamelCase Split   • Randomized PCA (99.9% dim red)• Spline Imputation   • SNOMED Entity Linking
     • OCR Typo Repair   • Butterworth Bandpass Filter   • Circadian Detrend   • DeepWalk Embeddings
     • Negation Stopwords• STFT Power Spectrograms       • Lag Features (t-k)  • PageRank Centrality
            │                     │                               │                     │
            ▼                     ▼                               ▼                     ▼
     Sublinear TF-IDF    Class-Weighted SVM &            Strict Chronological   Link Prediction
     + LinearSVC         Random Forest Spectrograms      Walk-Forward Split     Classifier
            │                     │                               │                     │
            └─────────────────────┴───────────────┬───────────────┴─────────────────────┘
                                                  │
                                                  ▼
                        Master Before vs. After Benchmark Evaluation & Reports
```

---

## 🏥 Real-World Use-Case: HealthAI 360 Tele-Diagnostics

| Module & Modality | Raw / Messy Real-World Input | Downstream AI Objective | Preprocessing Arsenal Applied |
| :--- | :--- | :--- | :--- |
| **Module 9: Text** | 2,000 patient reports with CamelCase, OCR scanner noise, HTML tags, URLs, emojis, and confidential PHI | Diagnostic triage specialty routing | HTML stripping, CamelCase splitting, OCR correction dictionary, PHI regex mask, negation-preserving stop words, sublinear TF-IDF (1,2)-grams. |
| **Module 10: Image** | 600 dermoscopic skin lesion images with camera exposure variations across Fitzpatrick skin types I–V | Malignancy & lesion classification (benign, malignant, inflammatory) | Aspect-preserving resize, CLAHE local contrast equalization, channel normalization, Randomized PCA dimensionality reduction (49,152 pixels $\rightarrow$ 50 components). |
| **Module 10: Biosignal** | 120 digital stethoscope Phonocardiogram (PCG) acoustic recordings with ambient noise and baseline wander | Heart sound murmur vs. normal acoustic detection | Zero-phase 4th-order Butterworth bandpass filtering (25–450 Hz), Short-Time Fourier Transform (STFT), Mel-spectrogram energy ratios. |
| **Module 11: Time-Series** | 48-hour continuous wearable telemetry stream (5,700+ records) with asynchronous timestamps and missing packets | Next-step tachycardia ($HR > 85\text{ bpm}$) risk forecasting | 5-minute uniform resampling, cubic spline interpolation, classical circadian additive decomposition ($T_t + S_t + R_t$), autoregressive lags ($t-1, \dots, t-6$), rolling volatility, strict temporal train/test split (no future leakage). |
| **Module 12: Knowledge Graph** | Relational EHR tables (patients, symptoms, drugs) and medical ontologies (SNOMED-CT, ICD-10) | Patient disease diagnosis & drug-drug contraindication link prediction | Heterogeneous graph construction, self-loop and isolate pruning, NLP-driven SNOMED entity linking, DeepWalk/Random Walk node embeddings, PageRank centrality. |

---

## 📁 Repository Structure

```
Webinar 3/
├── README.md                          # Comprehensive project overview & benchmark results
├── requirements.txt                   # Production dependencies (including Streamlit)
├── main.py                            # Master orchestrator executing Modules 9, 10, 11, and 12
├── app.py                             # Interactive Streamlit Multimodal Clinical Dashboard
│
├── notebooks/                         # Interactive, executed sequential curriculum notebooks
│   ├── 01_text_preprocessing.ipynb          # Module 9: Text Preprocessing & NLP
│   ├── 02_image_signal_preprocessing.ipynb  # Module 10: Vision & Biosignals (Lesions & PCG)
│   ├── 03_timeseries_preprocessing.ipynb    # Module 11: Wearable Streams & Leakage Prevention
│   ├── 04_graph_knowledge_graphs.ipynb      # Module 12: Medical Knowledge Graphs & SNOMED
│   └── 05_end_to_end_multimodal.ipynb       # Master Multimodal Tele-Diagnostics Pipeline
│
├── data/                              # Multimodal datasets generated by code
│   ├── text/
│   │   └── patient_reports_raw.csv    # 2,000 synthetic patient complaints
│   ├── image/
│   │   └── raw/                       # 600 lesion images in class subdirectories
│   ├── signal/
│   │   ├── pcg_signals.npy            # 120 phonocardiogram audio signals
│   │   └── signal_meta.json
│   ├── timeseries/
│   │   └── vitals_stream.csv          # 48-hr wearable telemetry stream
│   └── graph/
│       ├── kg_nodes_diseases.json     # ICD-10 disease nodes
│       ├── kg_nodes_symptoms.json     # SNOMED-CT symptom nodes
│       ├── kg_nodes_drugs.json        # Pharmacological drug nodes
│       └── full_clinical_graph_edges.csv
│
├── src/                               # Modular preprocessing engine
│   ├── text_pipeline/                 # Cleaner, NLTK classical pipeline, spaCy, TF-IDF
│   ├── image_pipeline/                # CLAHE normalizer, PCA reducer, PCG signal processor
│   ├── timeseries_pipeline/           # Resampler, circadian decomposition, lag features
│   ├── graph_pipeline/                # Graph representation, cleaner, entity linker, embeddings
│   ├── data_generators/               # Multimodal synthetic data factory
│   └── evaluation/                    # Unified comparative benchmarks & visualizer
│
└── reports/                           # Output metrics and visualizations
    ├── master_before_vs_after_metrics.csv
    ├── before_vs_after_benchmarks.png
    └── image_pca_scree.png
```

---

## 🚀 Quick Start

### 1. Installation
```bash
cd "Webinar 3"
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

### 2. Generate / Refresh Datasets
```bash
python src/data_generators/generate_datasets.py
```

### 3. Execute Master Pipeline
```bash
python main.py
```

### 4. Launch Interactive Streamlit Dashboard
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501` to interact with all 4 modality studios live.

---

## 🖥️ Interactive Streamlit Data Science Dashboard

The repository includes a comprehensive, reactive **Streamlit Clinical Data Science Dashboard** (`app.py`), bridging analytical preprocessing scripts into an interactive visual suite:

1. **📊 Executive Scorecard & Lift:** Real-time KPI callouts, interactive F1-score comparison bar charts, and plain-English takeaways.
2. **📝 Module 9: Clinical Text & NLP Studio:** Live interactive text area with regex scrubbing, CamelCase splitting, OCR dictionary repair, negation-preserving stopwords, and HIPAA PHI masking diffs.
3. **🖼️ Module 10A: Skin Lesion Computer Vision (CLAHE):** Real-time CLAHE parameter tuning (`clip_limit`, `tile_grid_size`), aspect-preserving resizing, and dynamic luminance histogram comparisons across Fitzpatrick skin types.
4. **🩺 Module 10B: Acoustic Biosignals (PCG):** Interactive Butterworth filter tuning (cutoff frequencies, filter order), audio waveform plots, and STFT time-frequency power spectrograms for heart murmur detection.
5. **📈 Module 11: Wearable Streams & Leakage Lab:** Asynchronous 5-minute grid resampling, multi-method imputation, 4-panel Circadian Seasonality decomposition, and an interactive **Data Leakage Trap** demonstrator.
6. **🕸️ Module 12: Knowledge Graph Reasoning:** Color-coded clinical network graph (🔴 Diseases, 🟠 Symptoms, 🔵 Drugs, ⚪ Patients), live SNOMED-CT entity linking, and automated drug contraindication safety alerts.

### Visual Dashboard Outputs
* `reports/streamlit_dashboard_scorecard.png` – Executive Multimodal Scorecard & Lift Dashboard.
* `reports/streamlit_module9_text_studio.png` – Module 9 Clinical Text & NLP Sanitization Studio.
* `reports/streamlit_module10a_vision_clahe.png` – Module 10A Skin Lesion CLAHE & Luminance Histogram.
* `reports/streamlit_module12_knowledge_graph.png` – Module 12 Clinical Knowledge Graph & Color Legend.

---

## 📊 Verified Pipeline Results & Benchmarks

The entire 4-track multi-modal pipeline was executed end-to-end. Here are the verified benchmark results:

| Curriculum Track | Baseline (Raw / Naive) Model | Preprocessed Model Architecture | Baseline F1 | Preprocessed F1 | Key Preprocessing Impact |
|:---|:---|:---|:---:|:---:|:---|
| **Text (Module 9)** | Raw tokens + SGDClassifier (BoW) | Calibrated LinearSVC + Negation TF-IDF | 1.0000 | 1.0000 | 100% PHI redaction (`[NAME]`, `[PHI]`), CamelCase split, OCR repair, negation preservation (`no fever` kept) |
| **Image (Module 10)** | Raw flattened pixels + Logistic Regression | RBF SVM (CLAHE + Randomized PCA-50) | 0.9666 | 0.9583 | **49,152 pixels $\rightarrow$ 50 PCA components (99.9% reduction in dimensions)**, illumination normalization |
| **Biosignal (Module 10)** | Raw downsampled waveform + Random Forest | Butterworth Bandpass + STFT Spectrograms | 0.5312 | **0.8643** | **+33.3% Lift**; eliminates acoustic phase shift & ambient stethoscope noise |
| **Time-Series (Module 11)** | Raw irregular observations (Future Holdout) | 5-min Resampling + Lags + Chronological Split | 0.5892 | **0.6024** | Exposes **Data Leakage Trap** (leakage gives fake 0.77 score; chronological split ensures true generalizability) |
| **Knowledge Graph (Module 12)**| Local Jaccard Symptom Overlap | DeepWalk Graph Embeddings + PageRank | 0.9666 | **0.9667** | Resolves ambiguous differential diagnoses via multi-hop drug & contraindication paths; **4 SNOMED entities linked** |

### Visual Artifacts Generated in `reports/`
* `reports/master_before_vs_after_metrics.csv` – Structured tabular benchmark metrics.
* `reports/before_vs_after_benchmarks.png` – Grouped comparison bar chart across all 4 tracks.
* `reports/image_pca_scree.png` – PCA cumulative variance retention curve.
