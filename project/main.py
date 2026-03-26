from __future__ import annotations

import argparse
from pathlib import Path

import mlflow

from scripts.analyze import make_acc_latency_scatter, make_accuracy_bar, make_per_class_summary
from scripts.data_prep import build_smoke_or_full, build_subset_csv
from scripts.evaluate import EvalConfig, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ImageNet validation subset benchmark")
    parser.add_argument("--imagenet-val-dir", type=str, required=True, help="ImageFolder format val directory")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--smoke-size", type=int, default=100)
    parser.add_argument("--full-size", type=int, default=1200)
    parser.add_argument("--samples-per-class", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--work-dir", type=str, default="project")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)
    data_dir = work_dir / "data"
    results_dir = work_dir / "results"
    analysis_dir = work_dir / "analysis"

    subset_csv = data_dir / "subset.csv"
    run_csv = data_dir / "run_subset.csv"

    build_subset_csv(
        imagenet_val_dir=args.imagenet_val_dir,
        output_csv=subset_csv,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )

    run_size = args.smoke_size if args.mode == "smoke" else args.full_size
    build_smoke_or_full(base_subset_csv=subset_csv, output_csv=run_csv, n_samples=run_size, seed=args.seed)

    mlflow.set_experiment("imagenet-model-benchmark")
    with mlflow.start_run(run_name=f"{args.mode}_run"):
        mlflow.log_params(
            {
                "mode": args.mode,
                "run_size": run_size,
                "samples_per_class": args.samples_per_class,
                "seed": args.seed,
            }
        )

        eval_cfg = EvalConfig(
            input_csv=run_csv,
            output_dir=results_dir,
            models_to_run=["resnet50", "mobilenet_v3_large", "convnext_tiny", "swin_t"],
            batch_size=args.batch_size,
            device=args.device,
        )
        evaluate(eval_cfg)

        make_accuracy_bar(results_dir / "summary.csv", analysis_dir / "accuracy_bar.png")
        make_acc_latency_scatter(results_dir / "summary.csv", analysis_dir / "accuracy_latency_scatter.png")
        make_per_class_summary(results_dir / "per_class_accuracy.csv", analysis_dir / "per_class_summary.csv")

        mlflow.log_artifact(str(analysis_dir / "accuracy_bar.png"), artifact_path="charts")
        mlflow.log_artifact(str(analysis_dir / "accuracy_latency_scatter.png"), artifact_path="charts")
        mlflow.log_artifact(str(analysis_dir / "per_class_summary.csv"), artifact_path="analysis")

    print(f"Finished {args.mode} run. See {results_dir} and {analysis_dir}")


if __name__ == "__main__":
    main()
