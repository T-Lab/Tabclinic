import json
import pandas as pd
import argparse
from pathlib import Path


# ===============================
# Metrics computation
# ===============================

def compute_metrics(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec) if (prec + rec) > 0 else 0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }


def is_missing(val):
    if pd.isna(val):
        return True
    v = str(val).strip().lower()
    return v in {"", "nan", "null", "none", "empty", "na"}


# ===============================
# Main merge logic
# ===============================

def merge_runs(run_dirs, out_path):

    all_voting = []

    for run in run_dirs:
        path = Path(run) / "ensemble_voting_details.csv"
        print(f"Loading {path}")
        df = pd.read_csv(path)
        all_voting.append(df)

    combined = pd.concat(all_voting, ignore_index=True)

    print(f"Total cells after merge: {len(combined)}")

    # ===============================
    # Summary
    # ===============================

    overall = compute_metrics(
        combined["ground_truth"],
        combined["ensemble_prediction"]
    )

    summary = {
        "accuracy": overall["accuracy"],
        "precision": overall["precision"],
        "recall": overall["recall"],
        "f1": overall["f1"],
        "error_rate": 1 - overall["accuracy"]
    }

    # ===============================
    # Completeness
    # ===============================

    combined["is_missing"] = combined["dirty_value"].apply(is_missing)

    missing_rate = combined["is_missing"].mean()

    missing_by_column = (
        combined[combined["is_missing"]]
        .groupby("column")
        .size()
        .sort_values(ascending=False)
        .to_dict()
    )

    # ===============================
    # Consistency
    # ===============================

    non_missing = combined[~combined["is_missing"]]

    anomaly_ratio = (
        non_missing["ensemble_prediction"].sum()
        / max(1, len(non_missing))
    )

    anomaly_by_column = (
        non_missing[non_missing["ensemble_prediction"] == 1]
        .groupby("column")
        .size()
        .sort_values(ascending=False)
        .to_dict()
    )

    # ===============================
    # Fairness
    # ===============================

    error_cells = combined[combined["ensemble_prediction"] == 1]
    error_counts = error_cells["column"].value_counts()
    total_errors = error_counts.sum()

    top3_error_ratio = (
        error_counts.head(3).sum() / total_errors
        if total_errors > 0 else 0.0
    )

    # ===============================
    # Column-level analysis
    # ===============================

    column_report = []

    for col in combined["column"].unique():
        sub = combined[combined["column"] == col]

        m = compute_metrics(
            sub["ground_truth"],
            sub["ensemble_prediction"]
        )

        status = (
            "failed" if m["f1"] == 0
            else "risky" if m["f1"] < 0.7
            else "acceptable" if m["f1"] < 0.85
            else "good"
        )

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

    # ===============================
    # Recommendations
    # ===============================

    failed_cols = [c["column"] for c in column_report if c["status"] == "failed"]
    risky_cols = [c["column"] for c in column_report if c["status"] == "risky"]

    recs = []

    if failed_cols:
        recs.append(
            f"Columns {failed_cols} show zero recall and require new rules or constraints."
        )

    if risky_cols:
        recs.append(
            f"Columns {risky_cols} have low recall or precision and should be prioritized."
        )

    if anomaly_ratio < 0.05 and summary["recall"] < 0.75:
        recs.append(
            "The detector is conservative: anomaly ratio is low but recall is limited."
        )

    # ===============================
    # Final report
    # ===============================

    merged_report = {
        "summary": summary,
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
            "interpretation":
                f"{top3_error_ratio:.1%} of detected errors concentrate in top-3 attributes"
        },
        "column_analysis": column_report,
        "recommendations": recs
    }

    with open(out_path, "w") as f:
        json.dump(merged_report, f, indent=2)

    print(f"\nMerged report saved to {out_path}")


# ===============================
# CLI
# ===============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="Run directories")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    merge_runs(args.runs, args.out)