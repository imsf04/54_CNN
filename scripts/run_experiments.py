"""Run cited CNN experiments on real course data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from torch.utils.data import DataLoader

from price_trends.adapters.runtime import select_device
from price_trends.data import (
    DataBundle,
    ImageDataset,
    image_pixel_stats,
    load_real_data,
    temporal_split,
)
from price_trends.models import LogisticBaseline, MLPBaseline, PriceTrendCNN
from price_trends.references import PAPER_20D, REPRODUCIBILITY_SEED
from price_trends.services.evaluation import evaluate_model
from price_trends.services.training import fit, set_seed


def _target_for_horizon(horizon: str) -> str:
    return f"Retx_{horizon}_label"


def _limit_for_check(bundle: DataBundle, max_samples: int) -> DataBundle:
    """Keep a small, year-balanced real-data slice for pipeline checks."""

    rng = np.random.default_rng(REPRODUCIBILITY_SEED)
    years = np.asarray([int(str(date)[:4]) for date in bundle.dates])
    unique_years = np.unique(years)
    per_year = max(1, max_samples // len(unique_years))
    indices = np.concatenate(
        [
            rng.choice(np.flatnonzero(years == year), size=min(per_year, int((years == year).sum())), replace=False)
            for year in unique_years
        ]
    )
    rng.shuffle(indices)
    indices = indices[:max_samples]
    returns = None if bundle.returns is None else bundle.returns[indices]
    stock_ids = None if bundle.stock_ids is None else bundle.stock_ids[indices]
    return DataBundle(images=bundle.images[indices], labels=bundle.labels[indices], dates=bundle.dates[indices], stock_ids=stock_ids, returns=returns, horizon=bundle.horizon)


def run_one(
    data_dir: str,
    horizon: str,
    output: Path,
    max_epochs: int,
    device: str,
    max_samples: int | None,
) -> dict[str, object]:
    # The course archive contains 20-day images only. Horizon experiments use
    # the corresponding future-return label while retaining that image input.
    target = _target_for_horizon(horizon)
    bundle = load_real_data(data_dir, target=target, horizon=horizon, return_column=f"Retx_{horizon}")
    if max_samples is not None:
        bundle = _limit_for_check(bundle, max_samples)
    splits = temporal_split(bundle)
    mean, std = image_pixel_stats(ImageDataset(splits["train"]), batch_size=PAPER_20D.batch_size)
    datasets = {name: ImageDataset(split, mean=mean, std=std) for name, split in splits.items()}
    loaders = {
        name: DataLoader(dataset, batch_size=PAPER_20D.batch_size, shuffle=name == "train")
        for name, dataset in datasets.items()
    }
    factories = {"logistic": LogisticBaseline, "mlp": MLPBaseline, "cnn": PriceTrendCNN}
    runs = {}
    for name, factory in factories.items():
        model = factory()
        history = fit(model, loaders["train"], loaders["val"], epochs=max_epochs, device=device, checkpoint=output / f"{horizon}_{name}.pt")
        runs[name] = {"history": history, "metrics": evaluate_model(model, splits["test"], loaders["test"], device)}
    return {
        "horizon": horizon,
        "target": target,
        "models": runs,
        "n_train": len(splits["train"].labels),
        "n_val": len(splits["val"].labels),
        "n_test": len(splits["test"].labels),
        "normalization": {"mean": mean, "std": std, "source": "train split only"},
        "device": str(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="artifacts/experiments")
    parser.add_argument("--horizons", nargs="+", default=["20d"], choices=["5d", "20d", "60d"])
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=None, help="仅用于真实数据链路检查的年度均衡样本上限")
    args = parser.parse_args()

    set_seed(REPRODUCIBILITY_SEED)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    results: dict[str, object] = {"references": {"paper": "docs/references/jiang_kelly_xiu_2023.pdf", "course": "Final Project.pdf"}, "runs": {}}
    for horizon in args.horizons:
        results["runs"][horizon] = run_one(args.data_dir, horizon, output, args.max_epochs, device, args.max_samples)
    (output / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
