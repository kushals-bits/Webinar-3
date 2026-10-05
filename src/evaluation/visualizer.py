# -*- coding: utf-8 -*-
"""Generate comparison report visualisations and save to reports/."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports")

def generate_all_report_visuals(results_list: list, master_df: pd.DataFrame, image_results: dict = None):
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Grouped bar chart - before vs after across all modules
    fig, ax = plt.subplots(figsize=(12, 6))
    modalities = [m.split("(")[0].strip() for m in master_df["Curriculum Track"].tolist()]
    x = np.arange(len(modalities))
    w = 0.35
    ax.bar(x - w/2, master_df["Baseline F1-Macro"], w, label="Before (Raw / Naive)", color="#e74c3c", alpha=0.9)
    ax.bar(x + w/2, master_df["Preprocessed F1-Macro"], w, label="After (Sanitized & Engineered)", color="#2ecc71", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(modalities, fontweight="bold", fontsize=11)
    ax.set_ylabel("F1-Macro Score", fontsize=12)
    ax.set_title("HealthAI 360 - Before vs After Data Preprocessing Impact (Modules 9 - 12)", fontsize=14, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#ccc", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    
    for i in range(len(modalities)):
        b_val = master_df["Baseline F1-Macro"].iloc[i]
        a_val = master_df["Preprocessed F1-Macro"].iloc[i]
        ax.text(x[i] - w/2, b_val + 0.02, f"{b_val:.2f}", ha="center", fontsize=9)
        ax.text(x[i] + w/2, a_val + 0.02, f"{a_val:.2f}", ha="center", fontweight="bold", fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(REPORTS_DIR, "before_vs_after_benchmarks.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"  [Visualizer] Saved benchmark chart: {chart_path}")

    # 2. PCA scree plot (image)
    if image_results and "pca" in image_results and image_results["pca"] is not None:
        pca = image_results["pca"]
        fig, ax = plt.subplots(figsize=(8, 5))
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        ax.plot(range(1, len(cumvar)+1), cumvar, marker="o", markersize=3, color="#2980b9")
        ax.axhline(y=0.95, color="#e74c3c", linestyle="--", label="95% Variance Retained")
        ax.set_xlabel("Number of Principal Components", fontsize=11)
        ax.set_ylabel("Cumulative Explained Variance", fontsize=11)
        ax.set_title("PCA Scree Plot - Skin Lesion High-Dimensionality Reduction", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        pca_path = os.path.join(REPORTS_DIR, "image_pca_scree.png")
        plt.savefig(pca_path, dpi=150)
        plt.close()
        print(f"  [Visualizer] Saved PCA scree plot: {pca_path}")
