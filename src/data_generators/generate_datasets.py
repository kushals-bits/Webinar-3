# -*- coding: utf-8 -*-
"""
Synthetic Data Generator for HealthAI 360 Tele-Diagnostics Use-Case.
Generates multimodal synthetic datasets covering:
  1. Module 9 (Text): Noisy clinical patient symptom text (CamelCase, OCR noise, spelling, PHI, emojis)
  2. Module 10 (Image & Signal): Skin lesions (benign, malignant, inflammatory) + Stethoscope PCG biosignals
  3. Module 11 (Time-Series): Wearable vital signs stream (irregular timestamps, circadian cycles, missing packets)
  4. Module 12 (Knowledge Graph): Heterogeneous clinical graph (Patient, Symptom, Disease, Drug) with ontologies
"""

import os
import csv
import json
import random
import shutil
import string
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Paths
TEXT_CSV = DATA_DIR / "text" / "patient_reports_raw.csv"
IMAGE_DIR = DATA_DIR / "image" / "raw"
SIGNAL_DIR = DATA_DIR / "signal"
TIMESERIES_CSV = DATA_DIR / "timeseries" / "vitals_stream.csv"
GRAPH_DIR = DATA_DIR / "graph"

# ==============================================================================
# 1. Module 9: Text Generator (Class-Conditioned Noisy Clinical Text)
# ==============================================================================
CLASS_SYMPTOM_MAP = {
    "malignant": [
        "patient noticed a dark spot on skin; BleedingLesion and rapid growth; not painful, denies itching",
        "skin spot on back; |rregular bord3rs and uneven dark color variegation; no fever, no pus",
        "pigmented skin spot; not healing, changing shape; denies rash, denies fever; suspicious of MalignantMelanoma",
        "dark skin mole on shoulder; BleedingLesion after shower; no itching, no discharge, not tender",
        "new dark spot on forearm; |rregular bord3rs; no purulent drainage, not painful",
        "RapidlyEnlarging skin spot with notched borders; no infection, denies purulent drainage",
    ],
    "inflammatory": [
        "patient noticed a red spot on skin; SevereItching with flaking scales; no infection, no fever",
        "red skin rash with SevereItching; denies bleeding, no purulent drainage, not infected",
        "skin rash on elbows and forearm; SevereItching and scaling redness; no pus, no fever",
        "red itchy skin spot; intense burning sensation and SevereItching; no bleeding, not oozing",
        "flaking red skin spot; SevereItching at night; no fever, denies wound breakdown",
        "pruritic red skin rash; SevereItching not responding to lotion; denies pus discharge",
    ],
    "infectious": [
        "patient noticed a painful red spot on skin; pus discharge noted <br/> temp 1O1F; denies itching",
        "skin spot on leg with BacterialCellulitis; warm, swollen, purulent drainage; denies bleeding mole",
        "painful infected skin spot; yellow pus drainage, temp 1O1F, started on Amoxici11in 500mg; no flaking",
        "red skin spot with localized swelling and pus discharge; elevated temperature; no chronic rash",
        "wound spot on skin with BacterialCellulitis and purulent exudate; fever noted; denies itching",
        "infected skin spot with swelling and pus drainage; on Amoxici11in; denies eczema",
    ],
    "benign": [
        "patient noticed a skin spot on arm; no itching, no bleeding, no pain, not growing, denies fever",
        "routine check of skin spot; uniform brown color, circular border; no itching, no discharge, no redness",
        "stable skin mole present for years; no pain, no bleeding, not enlarged; denies fever, no pus",
        "small flesh-colored skin spot; no itching, no bleeding, not painful, completely asymptomatic 👍",
        "common skin mole; no redness, no pus, not tender, no changes in appearance",
        "benign skin spot on trunk; no ulceration, no bleeding, no itching, no discharge",
    ],
}

BACKGROUND_PHRASES = [
    "patient denies drug allergies; previous medical history non-contributory",
    "blood pressure 120/80; heart rate regular; patient ambulatory",
    "follow-up telemedicine consultation scheduled for next month http://telehealth.example.com",
    "reviewed lab results; hemoglobin and renal function within normal limits",
    "patient advised on proper hydration and daily sunscreen application",
    "vital signs stable at intake; alert and oriented x3",
]

PHI_TEMPLATES = [
    "Patient: {name}, DOB: {dob}, MRN: {mrn}",
    "Name: {name} | Email: {email} | Phone: {phone}",
    "Referred by Dr. {doc}; patient ID {mrn}",
]

def _random_name():
    first = random.choice(["Amit", "Priya", "John", "Sarah", "Ravi", "Meera", "Carlos", "Aisha", "David", "Elena"])
    last = random.choice(["Sharma", "Patel", "Smith", "Khan", "Garcia", "Nguyen", "Lee", "Singh", "Miller", "Chen"])
    return f"{first} {last}"

def _random_phi():
    tmpl = random.choice(PHI_TEMPLATES)
    return tmpl.format(
        name=_random_name(),
        dob=f"{random.randint(1950, 2005)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
        mrn=f"MRN{''.join(random.choices(string.digits, k=7))}",
        email=f"{''.join(random.choices(string.ascii_lowercase, k=6))}@hospital.org",
        phone=f"+91-{''.join(random.choices(string.digits, k=10))}",
        doc=_random_name(),
    )

def _generate_text_csv(n: int = 2000):
    os.makedirs(TEXT_CSV.parent, exist_ok=True)
    rows = []
    classes = ["benign", "malignant", "inflammatory", "infectious"]
    per_class = n // len(classes)
    
    idx = 1
    for cls in classes:
        for _ in range(per_class):
            parts = [_random_phi()]
            # 2 to 3 class-specific diagnostic phrases
            parts += random.sample(CLASS_SYMPTOM_MAP[cls], k=random.randint(2, 3))
            # 1 background medical phrase
            parts.append(random.choice(BACKGROUND_PHRASES))
            
            noise = random.choice([" ", " \n ", " \t ", "\n\n"])
            text = noise.join(parts)
            rows.append({"id": f"PAT{1000 + idx}", "text": text, "label": cls})
            idx += 1
            
    random.shuffle(rows)
    with open(TEXT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "label"])
        w.writeheader()
        w.writerows(rows)
    print(f"  -> Generated {len(rows)} class-conditioned patient reports -> {TEXT_CSV}")

# ==============================================================================
# 2. Module 10: Image & Signal Generator
# ==============================================================================
def _generate_lesion_image(cls: str, idx: int, size: int = 128) -> Image.Image:
    """Generates synthetic dermoscopic images with camera exposure variations and class-specific morphology."""
    # Varied skin phototype (Fitzpatrick I to V)
    skin_base = random.choice([
        (238, 205, 185),  # Type I/II Fair
        (225, 185, 155),  # Type III Medium
        (190, 150, 120),  # Type IV Olive
        (140, 95, 70),    # Type V Brown
    ])
    
    # Random camera exposure factor (simulates varying phone sensor ISO / flashlight)
    exposure = random.uniform(0.65, 1.45)
    
    # Base skin canvas with sensor noise
    img_arr = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        noise = np.random.normal(0, 3.0, (size, size))
        img_arr[:, :, c] = (skin_base[c] + noise) * exposure

    # Directional shadow gradient
    grad_dir = random.choice(["horizontal", "vertical", "diagonal"])
    x_coords = np.linspace(-0.25, 0.25, size)
    if grad_dir == "horizontal":
        grad = np.tile(x_coords, (size, 1))
    elif grad_dir == "vertical":
        grad = np.tile(x_coords[:, None], (1, size))
    else:
        grad = (np.tile(x_coords, (size, 1)) + np.tile(x_coords[:, None], (1, size))) / 2.0
    
    lighting_factor = 1.0 + grad * random.uniform(0.2, 0.5)
    img_arr = np.clip(img_arr * lighting_factor[:, :, None], 0, 255).astype(np.uint8)
    
    img = Image.fromarray(img_arr, mode="RGB")
    draw = ImageDraw.Draw(img)
    
    cx = size // 2 + random.randint(-8, 8)
    cy = size // 2 + random.randint(-8, 8)
    
    if cls == "benign":
        # Regular, symmetric, uniform brown oval with crisp, smooth border
        r = random.randint(22, 28)
        rx = r + random.randint(-1, 1)
        ry = r + random.randint(-1, 1)
        lesion_color = (random.randint(95, 125), random.randint(55, 75), random.randint(35, 55))
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=lesion_color)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.4))
        
    elif cls == "malignant":
        # Asymmetric, multi-lobed polygon, jagged perimeter, multi-tone variegation & satellite spots
        for _ in range(5):
            ox = cx + random.randint(-14, 14)
            oy = cy + random.randint(-14, 14)
            rx = random.randint(14, 26)
            ry = random.randint(14, 26)
            shade = random.choice([
                (random.randint(20, 45), random.randint(15, 35), random.randint(20, 35)),  # Black/Charcoal
                (random.randint(70, 95), random.randint(30, 45), random.randint(25, 40)),  # Dark Brown
                (random.randint(100, 130), random.randint(35, 55), random.randint(45, 65)), # Red-blue
            ])
            draw.ellipse((ox - rx, oy - ry, ox + rx, oy + ry), fill=shade)
        # Peripheral satellite dots
        for _ in range(random.randint(5, 10)):
            sx = cx + random.randint(-36, 36)
            sy = cy + random.randint(-36, 36)
            draw.point((sx, sy), fill=(10, 8, 8))
        img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
        
    else:  # inflammatory
        # Diffuse erythematous halo with soft blurred gradient
        halo_color = (random.randint(210, 235), random.randint(85, 110), random.randint(85, 110))
        r_halo = random.randint(34, 46)
        draw.ellipse((cx - r_halo, cy - r_halo, cx + r_halo, cy + r_halo), fill=halo_color)
        # Inner plaque
        inner_color = (random.randint(230, 250), random.randint(125, 150), random.randint(125, 150))
        r_in = random.randint(14, 22)
        draw.ellipse((cx - r_in, cy - r_in, cx + r_in, cy + r_in), fill=inner_color)
        img = img.filter(ImageFilter.GaussianBlur(radius=3.0))
        
    return img

def _generate_images(n: int = 600, size: int = 128):
    """Generates synthetic dermatological lesion dataset in class subdirectories."""
    classes = ["benign", "malignant", "inflammatory"]
    per_class = n // len(classes)
    
    # Clean previous raw images to avoid mixing old resolution/count
    if IMAGE_DIR.exists():
        shutil.rmtree(IMAGE_DIR)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    
    total = 0
    for cls in classes:
        cls_dir = IMAGE_DIR / cls
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(1, per_class + 1):
            img = _generate_lesion_image(cls, i, size=size)
            img.save(cls_dir / f"lesion_{cls}_{i:04d}.jpg", format="JPEG", quality=92)
            total += 1
            
    print(f"  -> Generated {total} distinct lesion images ({size}x{size}) across {classes} -> {IMAGE_DIR}")

def _generate_signals(n_samples: int = 120, duration_sec: float = 2.0, fs: int = 2000):
    """
    Generates synthetic digital stethoscope Phonocardiogram (PCG) signals.
    Simulates S1 (lub) and S2 (dub) acoustic pulses + ambient stethoscope friction noise.
    """
    os.makedirs(SIGNAL_DIR, exist_ok=True)
    t = np.linspace(0, duration_sec, int(fs * duration_sec), endpoint=False)
    
    signals = []
    labels = []
    
    for i in range(n_samples):
        is_murmur = (i % 2 == 1)  # Perfectly balanced
        hr = random.uniform(65, 85)
        period = 60.0 / hr
        
        # Add random phase shift so heartbeats start at different times per patient
        phase_offset = random.uniform(0.02, 0.4)
        sig = np.zeros_like(t)
        beat_times = np.arange(phase_offset, duration_sec, period)
        for bt in beat_times:
            s1_idx = int(bt * fs)
            s1_len = int(0.08 * fs)
            if s1_idx + s1_len <= len(sig):
                t_s1 = np.linspace(0, 0.08, s1_len)
                sig[s1_idx : s1_idx + s1_len] += np.sin(2 * np.pi * 120 * t_s1) * np.hanning(s1_len)
            
            s2_idx = int((bt + 0.3) * fs)
            s2_len = int(0.06 * fs)
            if s2_idx + s2_len <= len(sig):
                t_s2 = np.linspace(0, 0.06, s2_len)
                sig[s2_idx : s2_idx + s2_len] += 0.8 * np.sin(2 * np.pi * 180 * t_s2) * np.hanning(s2_len)
                
            if is_murmur:
                m_start = s1_idx + s1_len
                m_len = int(0.18 * fs)
                if m_start + m_len <= len(sig):
                    murmur = np.random.normal(0, 0.40, m_len) * np.hanning(m_len)
                    sig[m_start : m_start + m_len] += murmur
        
        ambient_noise = np.random.normal(0, 0.20, len(t))
        baseline_drift = 0.3 * np.sin(2 * np.pi * 0.3 * t)
        raw_signal = sig + ambient_noise + baseline_drift
        
        signals.append(raw_signal)
        labels.append("murmur" if is_murmur else "normal")
        
    np.save(SIGNAL_DIR / "pcg_signals.npy", np.array(signals, dtype=np.float32))
    np.save(SIGNAL_DIR / "pcg_labels.npy", np.array(labels))
    with open(SIGNAL_DIR / "signal_meta.json", "w") as f:
        json.dump({"sampling_rate": fs, "duration_sec": duration_sec, "count": n_samples}, f, indent=2)
    print(f"  -> Generated {n_samples} balanced phonocardiogram signals (fs={fs}Hz) -> {SIGNAL_DIR}")

# ==============================================================================
# 3. Module 11: Time-Series Generator (Remote Patient Monitoring Vitals)
# ==============================================================================
def _generate_vitals_timeseries(n_hours: int = 48):
    os.makedirs(TIMESERIES_CSV.parent, exist_ok=True)
    
    start_time = pd.Timestamp("2026-10-01 08:00:00")
    total_seconds = n_hours * 3600
    
    timestamps = []
    current_sec = 0
    while current_sec < total_seconds:
        step = random.uniform(15, 45)
        current_sec += step
        if current_sec < total_seconds:
            timestamps.append(start_time + pd.Timedelta(seconds=current_sec))
            
    n_pts = len(timestamps)
    hours_arr = np.array([(ts - start_time).total_seconds() / 3600.0 for ts in timestamps])
    circadian = np.sin(2 * np.pi * hours_arr / 24.0)
    
    hr_trend = 0.25 * hours_arr
    hr = 75 + 8 * circadian + hr_trend + np.random.normal(0, 3.5, n_pts)
    
    spo2 = 98 - 1.5 * (circadian < -0.5) - np.random.exponential(0.6, n_pts)
    spo2 = np.clip(spo2, 85, 100)
    
    sbp = 120 + 12 * circadian - 0.15 * hours_arr + np.random.normal(0, 5, n_pts)
    temp = 36.8 + 0.6 * circadian + 0.015 * hours_arr + np.random.normal(0, 0.1, n_pts)
    
    df_vitals = pd.DataFrame({
        "timestamp": timestamps,
        "patient_id": "PAT1001",
        "heart_rate": np.round(hr, 1),
        "spo2": np.round(spo2, 1),
        "systolic_bp": np.round(sbp, 1),
        "temperature": np.round(temp, 2)
    })
    
    for col in ["heart_rate", "spo2", "systolic_bp", "temperature"]:
        drop_mask = np.random.random(n_pts) < 0.08
        df_vitals.loc[drop_mask, col] = np.nan
        
    df_vitals.to_csv(TIMESERIES_CSV, index=False)
    print(f"  -> Generated {len(df_vitals):,} wearable time-series vitals ({n_hours}h) -> {TIMESERIES_CSV}")

# ==============================================================================
# 4. Module 12: Knowledge Graph Generator
# ==============================================================================
def _generate_knowledge_graph():
    os.makedirs(GRAPH_DIR, exist_ok=True)
    
    diseases = [
        {"id": "DIS_01", "name": "Melanoma", "category": "Malignancy", "icd10": "C43"},
        {"id": "DIS_02", "name": "Atopic Dermatitis", "category": "Inflammatory", "icd10": "L20"},
        {"id": "DIS_03", "name": "Bacterial Cellulitis", "category": "Infectious", "icd10": "L03"},
        {"id": "DIS_04", "name": "Diabetic Foot Ulcer", "category": "Metabolic", "icd10": "E11.621"},
        {"id": "DIS_05", "name": "Arrhythmia", "category": "Cardiovascular", "icd10": "I49"},
        {"id": "DIS_06", "name": "Psoriasis Vulgaris", "category": "Autoimmune", "icd10": "L40"},
    ]
    
    symptoms = [
        {"id": "SYM_01", "name": "Itching", "snomed": "418290006"},
        {"id": "SYM_02", "name": "Asymmetric Mole", "snomed": "247441003"},
        {"id": "SYM_03", "name": "Chest Pain", "snomed": "29857009"},
        {"id": "SYM_04", "name": "Pus Discharge", "snomed": "271790001"},
        {"id": "SYM_05", "name": "Shortness of Breath", "snomed": "267036007"},
        {"id": "SYM_06", "name": "Erythema Redness", "snomed": "247441003"},
        {"id": "SYM_07", "name": "Palpitations", "snomed": "80313002"},
        {"id": "SYM_08", "name": "Non-healing Ulcer", "snomed": "239108000"},
    ]
    
    drugs = [
        {"id": "DRUG_01", "name": "Amoxicillin", "class": "Antibiotic"},
        {"id": "DRUG_02", "name": "Betamethasone", "class": "Corticosteroid"},
        {"id": "DRUG_03", "name": "Methotrexate", "class": "Immunosuppressant"},
        {"id": "DRUG_04", "name": "Metoprolol", "class": "Beta Blocker"},
        {"id": "DRUG_05", "name": "Pembrolizumab", "class": "Immunotherapy"},
        {"id": "DRUG_06", "name": "Metformin", "class": "Antidiabetic"},
    ]
    
    kg_edges = [
        {"source": "DIS_01", "target": "SYM_02", "relation": "CAUSES_SYMPTOM", "weight": 0.95},
        {"source": "DIS_02", "target": "SYM_01", "relation": "CAUSES_SYMPTOM", "weight": 0.90},
        {"source": "DIS_02", "target": "SYM_06", "relation": "CAUSES_SYMPTOM", "weight": 0.85},
        {"source": "DIS_03", "target": "SYM_04", "relation": "CAUSES_SYMPTOM", "weight": 0.92},
        {"source": "DIS_03", "target": "SYM_06", "relation": "CAUSES_SYMPTOM", "weight": 0.88},
        {"source": "DIS_04", "target": "SYM_08", "relation": "CAUSES_SYMPTOM", "weight": 0.96},
        {"source": "DIS_05", "target": "SYM_03", "relation": "CAUSES_SYMPTOM", "weight": 0.80},
        {"source": "DIS_05", "target": "SYM_05", "relation": "CAUSES_SYMPTOM", "weight": 0.75},
        {"source": "DIS_05", "target": "SYM_07", "relation": "CAUSES_SYMPTOM", "weight": 0.90},
        {"source": "DIS_06", "target": "SYM_01", "relation": "CAUSES_SYMPTOM", "weight": 0.85},
        {"source": "DIS_06", "target": "SYM_06", "relation": "CAUSES_SYMPTOM", "weight": 0.80},
        
        {"source": "DIS_01", "target": "DRUG_05", "relation": "TREATED_BY", "weight": 0.90},
        {"source": "DIS_02", "target": "DRUG_02", "relation": "TREATED_BY", "weight": 0.85},
        {"source": "DIS_03", "target": "DRUG_01", "relation": "TREATED_BY", "weight": 0.95},
        {"source": "DIS_04", "target": "DRUG_06", "relation": "TREATED_BY", "weight": 0.88},
        {"source": "DIS_05", "target": "DRUG_04", "relation": "TREATED_BY", "weight": 0.92},
        {"source": "DIS_06", "target": "DRUG_03", "relation": "TREATED_BY", "weight": 0.89},
        
        {"source": "DRUG_03", "target": "DRUG_01", "relation": "CONTRAINDICATED_WITH", "weight": 0.70},
        {"source": "DRUG_04", "target": "DRUG_02", "relation": "CONTRAINDICATED_WITH", "weight": 0.65},
    ]
    
    with open(GRAPH_DIR / "kg_nodes_diseases.json", "w") as f: json.dump(diseases, f, indent=2)
    with open(GRAPH_DIR / "kg_nodes_symptoms.json", "w") as f: json.dump(symptoms, f, indent=2)
    with open(GRAPH_DIR / "kg_nodes_drugs.json", "w") as f: json.dump(drugs, f, indent=2)
    
    df_edges = pd.DataFrame(kg_edges)
    df_edges.to_csv(GRAPH_DIR / "medical_kg_edges.csv", index=False)
    
    patient_cases = []
    for i in range(1, 101):
        pid = f"PAT{1000 + i}"
        dis = random.choice(diseases)["id"]
        related_syms = [e["target"] for e in kg_edges if e["source"] == dis]
        syms = random.sample(related_syms if related_syms else [s["id"] for s in symptoms], k=min(2, len(related_syms)))
        drg = random.choice(drugs)["id"]
        for s in syms:
            patient_cases.append({"source": pid, "target": s, "relation": "EXHIBITS", "weight": 1.0})
        patient_cases.append({"source": pid, "target": dis, "relation": "DIAGNOSED_WITH", "weight": 1.0})
        patient_cases.append({"source": pid, "target": drg, "relation": "PRESCRIBED", "weight": 1.0})
        
    df_pt = pd.DataFrame(patient_cases)
    df_all = pd.concat([df_edges, df_pt], ignore_index=True)
    df_all.to_csv(GRAPH_DIR / "full_clinical_graph_edges.csv", index=False)
    print(f"  -> Generated Medical Knowledge Graph ({len(df_all)} edges) -> {GRAPH_DIR}")

# ==============================================================================
# Master Generator Orchestrator
# ==============================================================================
def generate_all_datasets(text_samples: int = 2000, image_samples: int = 600):
    print("\n" + "=" * 80)
    print("[HealthAI 360 Data Generator] Creating Multimodal Tele-Diagnostics Data...")
    print("=" * 80)
    
    # 1. Module 9: Text
    _generate_text_csv(text_samples)
    
    # 2. Module 10: Image & Signal
    _generate_images(image_samples, size=128)
    _generate_signals(n_samples=120)
    
    # 3. Module 11: Time-Series
    _generate_vitals_timeseries(n_hours=48)
    
    # 4. Module 12: Knowledge Graph
    _generate_knowledge_graph()
    
    print("=" * 80)
    print("[HealthAI 360 Data Generator] All 4 Modality Datasets Ready on Disk!\n")

if __name__ == "__main__":
    generate_all_datasets(text_samples=2000, image_samples=600)
