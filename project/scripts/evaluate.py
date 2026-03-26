from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import mlflow
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models


MODEL_BUILDERS: Dict[str, Callable[[], Tuple[torch.nn.Module, object]]] = {
    "resnet50": lambda: (
        models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
        models.ResNet50_Weights.DEFAULT,
    ),
    "mobilenet_v3_large": lambda: (
        models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT),
        models.MobileNet_V3_Large_Weights.DEFAULT,
    ),
    "convnext_tiny": lambda: (
        models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT),
        models.ConvNeXt_Tiny_Weights.DEFAULT,
    ),
    "swin_t": lambda: (
        models.swin_t(weights=models.Swin_T_Weights.DEFAULT),
        models.Swin_T_Weights.DEFAULT,
    ),
}


@dataclass
class EvalConfig:
    input_csv: Path
    output_dir: Path
    models_to_run: List[str]
    batch_size: int
    device: str


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.inference_mode()
def run_model_eval(
    model_name: str,
    input_df: pd.DataFrame,
    batch_size: int,
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[int, float]]:
    model, weights = MODEL_BUILDERS[model_name]()
    model.eval().to(device)
    preprocess = weights.transforms()

    paths = input_df["image_path"].tolist()
    labels = input_df["label"].astype(int).tolist()

    all_top5: List[List[int]] = []
    all_top1: List[int] = []
    all_latency: List[float] = []

    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]

        images = [preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
        x = torch.stack(images).to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        probs = torch.softmax(logits, dim=1)
        top5 = torch.topk(probs, k=5, dim=1).indices.cpu().numpy()
        top1 = top5[:, 0]

        per_image_latency = elapsed / len(batch_paths)
        all_latency.extend([per_image_latency] * len(batch_paths))

        for pred5, pred1 in zip(top5, top1):
            all_top5.append(pred5.tolist())
            all_top1.append(int(pred1))

    pred_df = input_df.copy()
    pred_df["model"] = model_name
    pred_df["pred_top1"] = all_top1
    pred_df["pred_top5"] = [" ".join(map(str, t5)) for t5 in all_top5]
    pred_df["top1_correct"] = (pred_df["pred_top1"].astype(int) == pred_df["label"].astype(int)).astype(int)
    pred_df["top5_correct"] = pred_df.apply(
        lambda r: int(str(r["label"]) in str(r["pred_top5"]).split()), axis=1
    )
    pred_df["inference_time_sec"] = all_latency

    metrics = {
        "top1_accuracy": float(pred_df["top1_correct"].mean()),
        "top5_accuracy": float(pred_df["top5_correct"].mean()),
        "avg_inference_time_sec": float(np.mean(all_latency)),
        "num_parameters": int(count_parameters(model)),
    }
    per_class_acc = pred_df.groupby("label")["top1_correct"].mean().to_dict()
    return pred_df, metrics, per_class_acc


def evaluate(config: EvalConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    input_df = pd.read_csv(config.input_csv)

    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")

    summary_rows = []
    per_class_rows = []

    mlflow.log_params(
        {
            "input_csv": str(config.input_csv),
            "batch_size": config.batch_size,
            "device": str(device),
            "models": ",".join(config.models_to_run),
        }
    )

    for model_name in config.models_to_run:
        pred_df, metrics, per_class_acc = run_model_eval(model_name, input_df, config.batch_size, device)

        pred_csv = config.output_dir / f"predictions_{model_name}.csv"
        pred_df.to_csv(pred_csv, index=False)
        mlflow.log_artifact(str(pred_csv), artifact_path="predictions")

        summary_rows.append({"model": model_name, **metrics})
        for lbl, acc in per_class_acc.items():
            per_class_rows.append({"model": model_name, "label": int(lbl), "class_accuracy": float(acc)})

        mlflow.log_metrics({f"{model_name}_{k}": v for k, v in metrics.items() if k != "num_parameters"})
        mlflow.log_metric(f"{model_name}_num_parameters", float(metrics["num_parameters"]))

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = config.output_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    mlflow.log_artifact(str(summary_csv), artifact_path="summary")

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_csv = config.output_dir / "per_class_accuracy.csv"
    per_class_df.to_csv(per_class_csv, index=False)
    mlflow.log_artifact(str(per_class_csv), artifact_path="analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ImageNet models")
    parser.add_argument("--input-csv", type=str, default="project/data/run_subset.csv")
    parser.add_argument("--output-dir", type=str, default="project/results")
    parser.add_argument("--models", nargs="+", default=["resnet50", "mobilenet_v3_large", "convnext_tiny", "swin_t"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EvalConfig(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
        models_to_run=args.models,
        batch_size=args.batch_size,
        device=args.device,
    )
    mlflow.set_experiment("imagenet-model-benchmark")
    with mlflow.start_run(run_name="evaluate_script_run"):
        evaluate(cfg)
    print(f"Done. Results are saved to {cfg.output_dir}")


if __name__ == "__main__":
    main()
