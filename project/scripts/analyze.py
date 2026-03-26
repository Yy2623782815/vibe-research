from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns


def make_accuracy_bar(summary_csv: Path, output_path: Path) -> None:
    df = pd.read_csv(summary_csv)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="model", y="top1_accuracy")
    plt.title("Top-1 Accuracy by Model")
    plt.ylabel("Top-1 Accuracy")
    plt.xlabel("Model")
    plt.ylim(0, 1)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def make_acc_latency_scatter(summary_csv: Path, output_path: Path) -> None:
    df = pd.read_csv(summary_csv)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="avg_inference_time_sec", y="top1_accuracy", hue="model", s=120)
    for _, r in df.iterrows():
        plt.text(r["avg_inference_time_sec"], r["top1_accuracy"], r["model"], fontsize=9)
    plt.title("Accuracy vs Latency")
    plt.xlabel("Average Inference Time (sec / image)")
    plt.ylabel("Top-1 Accuracy")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def make_per_class_summary(per_class_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(per_class_csv)
    pivot = df.pivot_table(index="label", columns="model", values="class_accuracy")

    model_cols = pivot.columns.tolist()
    if not model_cols:
        raise ValueError("No model columns found in per_class_csv.")

    best_model = pivot[model_cols].idxmax(axis=1)
    best_accuracy = pivot[model_cols].max(axis=1)

    summary = pivot.copy()
    summary["best_model"] = best_model
    summary["best_accuracy"] = best_accuracy

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.reset_index().to_csv(output_csv, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--summary-csv", type=str, default="project/results/summary.csv")
    parser.add_argument("--per-class-csv", type=str, default="project/results/per_class_accuracy.csv")
    parser.add_argument("--analysis-dir", type=str, default="project/analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    summary_csv = Path(args.summary_csv)
    per_class_csv = Path(args.per_class_csv)

    bar_path = analysis_dir / "accuracy_bar.png"
    scatter_path = analysis_dir / "accuracy_latency_scatter.png"
    per_class_summary = analysis_dir / "per_class_summary.csv"

    make_accuracy_bar(summary_csv, bar_path)
    make_acc_latency_scatter(summary_csv, scatter_path)
    make_per_class_summary(per_class_csv, per_class_summary)

    active = mlflow.active_run()
    if active is not None:
        mlflow.log_artifact(str(bar_path), artifact_path="charts")
        mlflow.log_artifact(str(scatter_path), artifact_path="charts")
        mlflow.log_artifact(str(per_class_summary), artifact_path="analysis")

    print(f"Saved analysis artifacts to {analysis_dir}")


if __name__ == "__main__":
    main()
