"""Measure how much of the CNN signal survives masking older days of the window."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from price_trends.data import (  # noqa: E402
    ImageDataset,
    IterableImageDataset,
    image_pixel_stats,
    load_real_data,
    temporal_split,
)
from price_trends.models import PriceTrendCNN  # noqa: E402


def load_model(checkpoint: Path, device: str) -> PriceTrendCNN:
    model = PriceTrendCNN()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model" in state:
        state = state["model"]
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
    return model.to(device).eval()


def predict_mask(
    model: PriceTrendCNN,
    loader: DataLoader,
    keep: int,
    device: str,
    limit: int,
) -> np.ndarray:
    probabilities: list[np.ndarray] = []
    seen = 0
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            if keep < images.shape[-1]:
                images[:, :, :, : images.shape[-1] - keep] = 0.0
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(images)
            probabilities.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
            seen += len(logits)
            if limit and seen >= limit:
                break
    return np.concatenate(probabilities)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/raw/extracted")
    parser.add_argument("--target", default="Retx_20d_label")
    parser.add_argument("--return-column", default="Retx_20d")
    parser.add_argument("--sizes", default="3,5,10,20,60")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--stats-file", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="artifacts/analysis/window_mask.json")
    args = parser.parse_args()
    torch.set_num_threads(1)

    bundle = load_real_data(
        args.data_dir,
        target=args.target,
        horizon="20d",
        return_column=args.return_column,
        lazy=True,
    )
    splits = temporal_split(bundle)
    test = splits["test"]
    labels = np.asarray(test.labels).astype(int)

    if args.stats_file:
        stored = json.loads(Path(args.stats_file).read_text())
        mean, std = stored["mean"], stored["std"]
    else:
        mean, std = image_pixel_stats(ImageDataset(splits["train"]), batch_size=args.batch_size)

    dataset = IterableImageDataset(test, batch_size=args.batch_size, mean=mean, std=std)
    loader = DataLoader(dataset, batch_size=None)
    model = load_model(Path(args.checkpoint), args.device)

    rows: list[dict[str, float]] = []
    baseline: float | None = None
    for keep in (int(value) for value in args.sizes.split(",")):
        probabilities = predict_mask(model, loader, keep, args.device, args.limit)
        y = labels[: len(probabilities)]
        auc = float(roc_auc_score(y, probabilities))
        ap = float(average_precision_score(y, probabilities))
        if keep == 60:
            baseline = auc
        rows.append({"kept_days": keep, "roc_auc": auc, "average_precision": ap})

    for row in rows:
        row["auc_change"] = row["roc_auc"] - baseline if baseline is not None else float("nan")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "target": args.target,
                "samples": len(labels) if not args.limit else min(args.limit, len(labels)),
                "normalization_mean": mean,
                "normalization_std": std,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
