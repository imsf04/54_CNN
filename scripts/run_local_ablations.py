"""Run fixed-budget local ablations, decile portfolios, and multi-sample Grad-CAM."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from price_trends.data import (
    DataBundle,
    ImageDataset,
    image_pixel_stats,
    load_real_data,
    temporal_split,
)
from price_trends.models import LogisticBaseline, PriceTrendCNN
from price_trends.references import PAPER_20D, REPRODUCIBILITY_SEED
from price_trends.services.evaluation import evaluate_model
from price_trends.services.interpretation import grad_cam
from price_trends.services.training import fit, predict_proba, set_seed


def _limit_for_check(bundle: DataBundle, max_samples: int) -> DataBundle:
    rng = np.random.default_rng(REPRODUCIBILITY_SEED)
    years = np.asarray([int(str(date)[:4]) for date in bundle.dates])
    per_year = max(1, max_samples // len(np.unique(years)))
    indices = np.concatenate(
        [rng.choice(np.flatnonzero(years == year), size=min(per_year, int((years == year).sum())), replace=False) for year in np.unique(years)]
    )[:max_samples]
    rng.shuffle(indices)
    returns = None if bundle.returns is None else bundle.returns[indices]
    stock_ids = None if bundle.stock_ids is None else bundle.stock_ids[indices]
    return DataBundle(images=bundle.images[indices], labels=bundle.labels[indices], dates=bundle.dates[indices], stock_ids=stock_ids, returns=returns, horizon=bundle.horizon)


def _configs() -> list[tuple[str, type[torch.nn.Module], dict[str, object]]]:
    return [
        ("logistic_baseline", LogisticBaseline, {}),
        ("cnn_2blocks", PriceTrendCNN, {"filters": (64, 128)}),
        ("cnn_3blocks", PriceTrendCNN, {"filters": (64, 128, 256)}),
        ("cnn_4blocks", PriceTrendCNN, {"filters": (64, 128, 256, 512)}),
        ("cnn_narrow", PriceTrendCNN, {"filters": (32, 64, 128)}),
        ("cnn_wide", PriceTrendCNN, {"filters": (128, 256, 512)}),
    ]


def _pick_examples(labels: np.ndarray, probabilities: np.ndarray, dates: np.ndarray, stock_ids: np.ndarray) -> list[dict[str, object]]:
    predicted = (probabilities >= 0.5).astype(int)
    categories = {
        "tp": np.flatnonzero((labels == 1) & (predicted == 1)),
        "tn": np.flatnonzero((labels == 0) & (predicted == 0)),
        "fp": np.flatnonzero((labels == 0) & (predicted == 1)),
        "fn": np.flatnonzero((labels == 1) & (predicted == 0)),
    }
    selected: list[dict[str, object]] = []
    for category, indices in categories.items():
        if len(indices) < 4:
            raise ValueError(f"Grad-CAM protocol needs four {category} examples, found {len(indices)}")
        chosen = indices[np.linspace(0, len(indices) - 1, 4, dtype=int)]
        selected.extend(
            {"category": category, "index": int(index), "date": str(dates[index]), "stock_id": str(stock_ids[index]), "label": int(labels[index]), "probability": float(probabilities[index]), "prediction": int(predicted[index])}
            for index in chosen
        )
    return selected


def _gradcam_grid(model: torch.nn.Module, dataset: ImageDataset, examples: list[dict[str, object]], mean: float, std: float, output: Path, device: str) -> None:
    model.eval()
    fig, axes = plt.subplots(4, 4, figsize=(14, 11), constrained_layout=True)
    for axis, example in zip(axes.flat, examples):
        image, _ = dataset[int(example["index"])]
        batch = image.unsqueeze(0).to(device)
        cam = grad_cam(model, batch).detach().cpu().numpy()
        raw = image[0].numpy() * std + mean
        axis.imshow(raw, cmap="gray", aspect="auto")
        axis.imshow(cam, cmap="magma", alpha=0.55, aspect="auto")
        axis.set_title(f"{example['category'].upper()} {example['date'][:10]} p={example['probability']:.2f}")
        axis.set_axis_off()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="artifacts/local_ablations_normalized")
    parser.add_argument("--max-samples", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seeds", default="7,17,27")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.epochs < 1 or args.max_samples < 1:
        raise ValueError("epochs and max-samples must be positive")
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if not seeds:
        raise ValueError("at least one seed is required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    set_seed(REPRODUCIBILITY_SEED)
    bundle = load_real_data(args.data_dir, target="Retx_20d_label", horizon="20d", return_column="Retx_20d")
    bundle = _limit_for_check(bundle, args.max_samples)
    splits = temporal_split(bundle)
    mean, std = image_pixel_stats(ImageDataset(splits["train"]), batch_size=PAPER_20D.batch_size)
    datasets = {name: ImageDataset(split, mean=mean, std=std) for name, split in splits.items()}
    loaders = {name: DataLoader(dataset, batch_size=PAPER_20D.batch_size, shuffle=name == "train") for name, dataset in datasets.items()}

    raw_results: list[dict[str, object]] = []
    for seed in seeds:
        for index, (name, factory, kwargs) in enumerate(_configs()):
            set_seed(seed)
            model = factory(**kwargs)
            history = fit(model, loaders["train"], loaders["val"], epochs=args.epochs, patience=args.epochs + 1, device=args.device, checkpoint=output / f"seed_{seed}_{name}.pt")
            metrics = evaluate_model(model, splits["test"], loaders["test"], args.device)
            raw_results.append({"seed": seed, "name": name, "config": kwargs, "epochs_budget": args.epochs, "epochs_run": len(history), "metrics": metrics})
    best_name, best_seed = "cnn_3blocks", seeds[0]
    best_model = PriceTrendCNN(filters=(64, 128, 256))
    checkpoint = output / f"seed_{best_seed}_{best_name}.pt"
    state = torch.load(checkpoint, map_location=args.device, weights_only=False)
    best_model.load_state_dict(state["model"])
    probabilities = predict_proba(best_model, loaders["test"], args.device)
    examples = _pick_examples(splits["test"].labels, probabilities, splits["test"].dates, splits["test"].stock_ids)
    (output / "gradcam_examples.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
    _gradcam_grid(best_model, datasets["test"], examples, mean, std, output / "gradcam_16_samples.png", args.device)

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in raw_results:
        grouped[str(row["name"])].append(row["metrics"])
    summary = []
    for name, metrics_rows in grouped.items():
        summary.append({"name": name, "seeds": len(metrics_rows), "metrics_mean": {key: float(np.mean([row[key] for row in metrics_rows])) for key in metrics_rows[0]}, "metrics_std": {key: float(np.std([row[key] for row in metrics_rows], ddof=1)) if len(metrics_rows) > 1 else 0.0 for key in metrics_rows[0]}})
    payload = {"seeds": seeds, "normalization": {"mean": mean, "std": std, "source": "train split only"}, "data": {"horizon": "20d", "max_samples": args.max_samples, "n_train": len(splits["train"].labels), "n_val": len(splits["val"].labels), "n_test": len(splits["test"].labels)}, "training": {"epochs": args.epochs, "patience": args.epochs + 1, "device": args.device, "batch_size": PAPER_20D.batch_size}, "best_cnn_for_gradcam": {"name": best_name, "seed": best_seed}, "raw_results": raw_results, "summary": summary}
    (output / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "name", "epochs_budget", "epochs_run", "accuracy", "balanced_accuracy", "roc_auc", "average_precision", "portfolio_sharpe", "portfolio_20pct_sharpe"])
        writer.writeheader()
        for row in raw_results:
            writer.writerow({"seed": row["seed"], "name": row["name"], "epochs_budget": row["epochs_budget"], "epochs_run": row["epochs_run"], **{key: row["metrics"][key] for key in writer.fieldnames[4:]}})
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
