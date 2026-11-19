"""Train the paper models on the full course panel across several seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from price_trends.data import (
    ImageDataset,
    IterableImageDataset,
    image_pixel_stats,
    load_real_data,
    temporal_split,
)
from price_trends.models import LogisticBaseline, MLPBaseline, PriceTrendCNN, TemporalConv
from price_trends.references import PAPER_20D, REPRODUCIBILITY_SEED
from price_trends.services.evaluation import classification_metrics, long_short_metrics
from price_trends.services.training import fit, predict_proba, set_seed

CONFIGS = {
    "logistic": LogisticBaseline,
    "mlp": MLPBaseline,
    "cnn": PriceTrendCNN,
    "minimal": lambda: TemporalConv(filters=16),
    "minimal_1f": lambda: TemporalConv(filters=1),
}

ABLATION_CONFIGS = {
    "cnn_2blocks": lambda: PriceTrendCNN(filters=(64, 128)),
    "cnn_3blocks": lambda: PriceTrendCNN(filters=(64, 128, 256)),
    "cnn_4blocks": lambda: PriceTrendCNN(filters=(64, 128, 256, 512)),
    "cnn_narrow": lambda: PriceTrendCNN(filters=(32, 64, 128)),
    "cnn_wide": lambda: PriceTrendCNN(filters=(128, 256, 512)),
}


def evaluate(model, bundle, loader, device: str) -> tuple[dict[str, float], np.ndarray]:
    probabilities = predict_proba(model, loader, device)
    metrics = classification_metrics(bundle.labels, probabilities)
    metrics.update(
        {f"portfolio_{k}": v for k, v in long_short_metrics(probabilities, bundle.returns, bundle.dates, quantile=0.1).items()}
    )
    metrics.update(
        {f"portfolio_20pct_{k}": v for k, v in long_short_metrics(probabilities, bundle.returns, bundle.dates, quantile=0.2).items()}
    )
    return metrics, probabilities


def write_results(output: Path, seeds, mean, std, args, results: list[dict[str, object]]) -> None:
    (output / "results.json").write_text(
        json.dumps(
            {
                "seeds_requested": list(seeds),
                "seeds_completed": sorted({int(item["seed"]) for item in results}),
                "target": args.target,
                "return_column": args.return_column,
                "eval_batch_size": args.eval_batch_size,
                "normalization_mean": mean,
                "normalization_std": std,
                "normalization_source": "train split only",
                "max_epochs": args.epochs,
                "batch_size": args.batch_size,
                "runs": results,
            },
            indent=2,
        )
    )


def normalization_stats(splits, batch_size: int, cache: Path | None) -> tuple[float, float]:
    if cache and cache.exists():
        stored = json.loads(cache.read_text())
        return stored["mean"], stored["std"]
    mean, std = image_pixel_stats(ImageDataset(splits["train"]), batch_size=batch_size)
    if cache:
        cache.write_text(json.dumps({"mean": mean, "std": std, "source": "train split only"}))
    return mean, std


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/extracted")
    parser.add_argument("--target", default="Retx_20d_label")
    parser.add_argument("--return-column", default="Retx_20d")
    parser.add_argument("--output", default="artifacts/full_seeds")
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=PAPER_20D.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--stats-file", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--configs", default="main")
    parser.add_argument("--patience", type=int, default=PAPER_20D.early_stopping_patience)
    args = parser.parse_args()
    tag = args.target.removeprefix("Retx_").removesuffix("_label")
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    available = {**CONFIGS, **ABLATION_CONFIGS}
    torch.backends.cudnn.benchmark = args.device.startswith("cuda")
    if args.configs == "main":
        selected = dict(CONFIGS)
    elif args.configs == "ablations":
        selected = dict(ABLATION_CONFIGS)
    else:
        selected = {name: available[name] for name in args.configs.split(",")}
    seeds = tuple(int(value) for value in args.seeds.split(","))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    bundle = load_real_data(args.data_dir, target=args.target, horizon="20d", return_column=args.return_column, lazy=True)
    splits = temporal_split(bundle)
    mean, std = normalization_stats(splits, args.batch_size, Path(args.stats_file) if args.stats_file else None)
    print(json.dumps({"normalization_mean": mean, "normalization_std": std, "source": "train split only"}), flush=True)
    datasets = {
        name: IterableImageDataset(
            part,
            batch_size=args.batch_size if name == "train" else args.eval_batch_size,
            mean=mean,
            std=std,
            shuffle=name == "train",
            seed=REPRODUCIBILITY_SEED,
        )
        for name, part in splits.items()
    }
    loaders = {
        name: DataLoader(dataset, batch_size=None)
        for name, dataset in datasets.items()
    }

    results: list[dict[str, object]] = []
    for seed in seeds:
        for name, factory in selected.items():
            set_seed(seed)
            model = factory()
            print(json.dumps({"start": name, "seed": seed}), flush=True)
            history = fit(
                model,
                loaders["train"],
                loaders["val"],
                epochs=args.epochs,
                patience=args.patience,
                device=args.device,
                checkpoint=output / f"seed_{seed}_{name}_{tag}.pt",
                amp=args.device.startswith("cuda"),
            )
            metrics, probabilities = evaluate(model, splits["test"], loaders["test"], args.device)
            frame = pd.DataFrame(
                {
                    "Date": np.asarray(splits["test"].dates),
                    "StockID": np.asarray(splits["test"].stock_ids),
                    "label": splits["test"].labels,
                    "return": splits["test"].returns,
                    "probability": probabilities,
                }
            )
            frame.to_parquet(output / f"seed_{seed}_{name}_{tag}_predictions.parquet")
            results.append(
                {
                    "seed": seed,
                    "name": name,
                    "epochs_run": len(history),
                    "best_epoch": int(min(history, key=lambda row: row["val"]["loss"])["epoch"]),
                    "best_val_loss": float(min(row["val"]["loss"] for row in history)),
                    "history": history,
                    "metrics": metrics,
                }
            )
            print(json.dumps({"seed": seed, "name": name, "epochs": len(history), "metrics": metrics}), flush=True)
            write_results(output, seeds, mean, std, args, results)
    write_results(output, seeds, mean, std, args, results)


if __name__ == "__main__":
    main()
