# -*- coding: utf-8 -*-
"""Master Before-vs-After Comparison Benchmarking across Modules 9, 10, 11, and 12."""
import pandas as pd
from typing import List

def build_master_comparison_report(results_list: List[dict]) -> pd.DataFrame:
    """Builds a unified comparison DataFrame from multimodal pipeline results."""
    rows = []
    for r in results_list:
        rows.append({
            "Curriculum Track": r.get("modality", "Unknown"),
            "Model Architecture": r.get("model_name", "Model"),
            "Baseline Acc": round(r.get("baseline_acc", 0.0), 4),
            "Baseline F1-Macro": round(r.get("baseline_f1", 0.0), 4),
            "Preprocessed Acc": round(r.get("after_acc", 0.0), 4),
            "Preprocessed F1-Macro": round(r.get("after_f1", 0.0), 4),
            "Delta F1 (Lift)": round(r.get("delta_f1", r.get("after_f1", 0.0) - r.get("baseline_f1", 0.0)), 4),
        })
    return pd.DataFrame(rows)

def print_master_comparison(df: pd.DataFrame):
    """Prints a styled terminal table of before-vs-after metrics."""
    print("\n" + "=" * 115)
    print("      HEALTHAI 360 TELE-DIAGNOSTICS: MASTER BEFORE vs AFTER PREPROCESSING BENCHMARK")
    print("=" * 115)
    print(df.to_string(index=False))
    print("=" * 115)
