# -*- coding: utf-8 -*-
"""
HealthAI 360: Integrated Tele-Diagnostics & Remote Patient Monitoring Pipeline.
Master End-to-End Orchestrator executing Modules 9, 10, 11, and 12.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Ensure src is on sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.text_pipeline.pipeline import run_text_pipeline
from src.image_pipeline.pipeline import run_image_pipeline
from src.image_pipeline.signal_processor import run_signal_pipeline
from src.timeseries_pipeline.pipeline import run_timeseries_pipeline
from src.graph_pipeline.pipeline import run_graph_pipeline
from src.evaluation.comparison import build_master_comparison_report, print_master_comparison
from src.evaluation.visualizer import generate_all_report_visuals

def main():
    print("=" * 80)
    print("   HEALTHAI 360: INTEGRATED TELE-DIAGNOSTICS & REMOTE PATIENT MONITORING")
    print("   Comprehensive Data Preprocessing Pipeline (Modules 9, 10, 11, and 12)")
    print("=" * 80)

    data_dir = BASE_DIR / "data"
    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Module 9: Text Preprocessing Pipeline
    text_csv = data_dir / "text" / "patient_reports_raw.csv"
    text_results = run_text_pipeline(str(text_csv))

    # 2. Module 10: Image Preprocessing Pipeline
    image_dir = data_dir / "image" / "raw"
    image_results = run_image_pipeline(str(image_dir))

    # 2b. Module 10: Biosignal / Audio Preprocessing Pipeline
    signal_dir = data_dir / "signal"
    signal_results = run_signal_pipeline(str(signal_dir))

    # 3. Module 11: Time-Series Preprocessing Pipeline
    timeseries_csv = data_dir / "timeseries" / "vitals_stream.csv"
    timeseries_results = run_timeseries_pipeline(str(timeseries_csv))

    # 4. Module 12: Knowledge Graph Preprocessing Pipeline
    graph_dir = data_dir / "graph"
    graph_results = run_graph_pipeline(str(graph_dir))

    # 5. Master Comparative Evaluation
    all_results = [
        text_results,
        image_results,
        signal_results,
        timeseries_results,
        graph_results,
    ]
    master_df = build_master_comparison_report(all_results)
    print_master_comparison(master_df)

    # Export master benchmark CSV
    csv_out = reports_dir / "master_before_vs_after_metrics.csv"
    master_df.to_csv(csv_out, index=False)
    print(f"\n[Master Pipeline] Exported master benchmark metrics to: {csv_out}")

    # Generate charts
    generate_all_report_visuals(all_results, master_df, image_results=image_results)
    print("[Master Pipeline] All 4-track multimodal pipelines completed successfully!\n")

if __name__ == "__main__":
    main()
