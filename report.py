import json
import pandas as pd
import numpy as np
from pathlib import Path


# -----------------------------
# Load results
# -----------------------------
def load_results(base_dir: str):
    base = Path(base_dir)

    metrics = json.loads((base / "metrics.json").read_text())
    voting = pd.read_csv(base / "ensemble_voting_details.csv")
    preds = pd.read_csv(base / "predictions.csv")
    workers = pd.read_csv(base / "worker_reliability.csv")

    return metrics, voting, preds, workers


# -----------------------------
# Helpers
# -----------------------------
def classify_column_health(m):
    if m["f1"] == 0:
        return "failed"
    if m["f1"] < 0.7:
        return "risky"
    if m["f1"] < 0.85:
        return "acceptable"
    return "good"


def is_missing(val: str) -> bool:
    if val is None:
        return True
    v = str(val).strip().lower()
    return v in {"", "nan", "null", "none", "empty", "na"}


# -----------------------------
# Main report builder
# -----------------------------
def build_report(base_dir: str) -> dict:
    metrics, voting, preds, workers = load_results(base_dir)

    overall = metrics["overall"]
    per_column = metrics["per_column"]

    total_cells = len(voting)

    # =========================================================
    # 1️⃣ COMPLETENESS — Missing value rate (t-assess)
    # =========================================================
    voting["is_missing"] = voting["dirty_value"].apply(is_missing)
    missing_rate = voting["is_missing"].mean()

    missing_by_column = (
        voting[voting["is_missing"]]
        .groupby("column")
        .size()
        .sort_values(ascending=False)
        .to_dict()
    )

    # =========================================================
    # 2️⃣ CONSISTENCY — Outlier / anomaly ratio (t-assess)
    #   = error cells / non-missing cells
    # =========================================================
    non_missing_cells = voting[~voting["is_missing"]]
    anomaly_ratio = (
        non_missing_cells["ensemble_prediction"].sum()
        / max(1, len(non_missing_cells))
    )

    anomaly_by_column = (
        non_missing_cells[non_missing_cells["ensemble_prediction"] == 1]
        .groupby("column")
        .size()
        .sort_values(ascending=False)
        .to_dict()
    )

    # =========================================================
    # 3️⃣ FAIRNESS — Attribute density skew (t-assess)
    #   = top-k error concentration
    # =========================================================
    error_cells = voting[voting["ensemble_prediction"] == 1]
    error_counts = error_cells["column"].value_counts()

    total_errors = error_counts.sum()
    top3_error_ratio = (
        error_counts.head(3).sum() / total_errors if total_errors > 0 else 0.0
    )

    # =========================================================
    # Column performance summary
    # =========================================================
    column_report = []
    failed_cols, risky_cols = [], []

    for col, m in per_column.items():
        status = classify_column_health(m)

        if status == "failed":
            failed_cols.append(col)
        if status == "risky":
            risky_cols.append(col)

        column_report.append({
            "column": col,
            "status": status,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"]
        })

    # =========================================================
    # Ensemble / worker analysis
    # =========================================================
    worker_stats = {
        "num_workers": len(workers),
        "avg_reliability": float(workers["reliability"].mean()),
        "min_reliability": float(workers["reliability"].min()),
        "max_reliability": float(workers["reliability"].max())
    }

    # =========================================================
    # Recommendations
    # =========================================================
    recs = []

    if failed_cols:
        recs.append(
            f"Columns {failed_cols} show zero recall and require new rules or constraints."
        )

    if risky_cols:
        recs.append(
            f"Columns {risky_cols} have low recall or precision and should be prioritized."
        )

    if anomaly_ratio < 0.05 and overall["recall"] < 0.75:
        recs.append(
            "The detector is conservative: anomaly ratio is low but recall is limited."
        )

    if top3_error_ratio > 0.6:
        recs.append(
            "Errors are highly concentrated in a few attributes; targeted fixes may yield large gains."
        )

    # =========================================================
    # Final report object
    # =========================================================
    return {
        "summary": {
            "accuracy": overall["accuracy"],
            "precision": overall["precision"],
            "recall": overall["recall"],
            "f1": overall["f1"],
            "error_rate": 1 - overall["accuracy"]
        },

        # 🧪 t-assess aligned blocks
        "completeness": {
            "missing_rate": missing_rate,
            "missing_by_column": missing_by_column
        },

        "consistency": {
            "anomaly_ratio": anomaly_ratio,
            "anomaly_by_column": anomaly_by_column
        },

        "fairness": {
            "error_density_skew": top3_error_ratio,
            "interpretation": (
                f"{top3_error_ratio:.1%} of detected errors concentrate in top-3 attributes"
                if total_errors > 0 else "No detected errors"
            )
        },

        "column_analysis": column_report,
        "ensemble_analysis": worker_stats,
        "recommendations": recs
    }


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True,
                    help="Directory containing metrics.json and ensemble_voting_details.csv")
    ap.add_argument("--out", default="report.json")
    args = ap.parse_args()

    report = build_report(args.input_dir)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[OK] Report written to {args.out}")