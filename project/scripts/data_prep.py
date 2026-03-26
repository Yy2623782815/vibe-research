from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
from torchvision.datasets import ImageFolder


def build_subset_csv(
    imagenet_val_dir: str | Path,
    output_csv: str | Path,
    samples_per_class: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a stratified ImageNet subset CSV: image_path,label."""
    dataset = ImageFolder(root=str(imagenet_val_dir))

    rows = []
    for img_path, label in dataset.samples:
        rows.append({"image_path": str(Path(img_path).resolve()), "label": int(label)})

    df = pd.DataFrame(rows)

    class_counts = df["label"].value_counts().sort_index()
    too_small = class_counts[class_counts < samples_per_class]
    if not too_small.empty:
        raise ValueError(
            "Some classes have fewer images than requested samples_per_class: "
            f"{too_small.to_dict()}"
        )

    sampled = (
        df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(n=samples_per_class, random_state=seed))
        .reset_index(drop=True)
    )

    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(output_csv, index=False)
    return sampled


def build_smoke_or_full(
    base_subset_csv: str | Path,
    output_csv: str | Path,
    n_samples: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a smoke/full run CSV from the base stratified subset CSV."""
    df = pd.read_csv(base_subset_csv)
    if n_samples >= len(df):
        sampled = df.copy()
    else:
        sampled = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(output_csv, index=False)
    return sampled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stratified ImageNet subset CSV")
    parser.add_argument("--imagenet-val-dir", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default="project/data/subset.csv")
    parser.add_argument("--samples-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_subset_csv(
        imagenet_val_dir=args.imagenet_val_dir,
        output_csv=args.output_csv,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    print(f"Saved {len(df)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
