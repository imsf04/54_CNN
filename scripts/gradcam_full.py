"""Render the fixed sixteen Grad-CAM examples with the full-data CNN checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from price_trends.data import ImageDataset, load_real_data, temporal_split
from price_trends.models import PriceTrendCNN
from price_trends.services.interpretation import grad_cam

NORMALIZATION = {"mean": 0.10689710380321306, "std": 0.3089823829956947}


def pick_examples(labels: np.ndarray, probabilities: np.ndarray, dates: np.ndarray, stock_ids: np.ndarray) -> list[dict[str, object]]:
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
        order = indices[np.argsort(-np.abs(probabilities[indices] - 0.5))[:200]]
        order = order[np.argsort(dates[order])]
        chosen = order[np.linspace(0, len(order) - 1, 4, dtype=int)]
        selected.extend(
            {
                "category": category,
                "index": int(index),
                "date": str(dates[index])[:10],
                "stock_id": str(stock_ids[index]),
                "label": int(labels[index]),
                "probability": float(probabilities[index]),
                "prediction": int(predicted[index]),
            }
            for index in chosen
        )
    return selected


def load_model(checkpoint: Path, device: str) -> torch.nn.Module:
    model = PriceTrendCNN()
    state = torch.load(checkpoint, map_location=device, weights_only=False)["model"]
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
    model.to(device).eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/extracted")
    parser.add_argument("--checkpoint", default="artifacts/kaggle_output_1118/cnn_20d.pt")
    parser.add_argument("--predictions", default="artifacts/kaggle_output_1118/test_predictions.parquet")
    parser.add_argument("--output", default="artifacts/analysis/gradcam_full_16.png")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    bundle = load_real_data(args.data_dir, target="Retx_20d_label", horizon="20d", return_column="Retx_20d", lazy=True)
    splits = temporal_split(bundle)
    test = splits["test"]
    frame = pd.read_parquet(args.predictions)
    test_keys = pd.MultiIndex.from_arrays(
        [pd.to_datetime(test.dates).strftime("%Y-%m-%d"), np.asarray(test.stock_ids).astype(str)]
    )
    if not test_keys.is_unique:
        raise ValueError("test split contains duplicate date-stock pairs")
    keys = pd.MultiIndex.from_arrays(
        [pd.to_datetime(frame["Date"]).dt.strftime("%Y-%m-%d"), frame["StockID"].astype(str)]
    )
    positions = test_keys.get_indexer(keys)
    if (positions < 0).any():
        raise ValueError("prediction rows are not aligned with the temporal test split")
    probabilities = np.empty(len(test.labels), dtype=float)
    probabilities[positions] = frame["cnn_probability"].to_numpy(dtype=float)
    examples = pick_examples(test.labels, probabilities, test.dates, test.stock_ids)

    model = load_model(Path(args.checkpoint), args.device)
    dataset = ImageDataset(test, mean=NORMALIZATION["mean"], std=NORMALIZATION["std"])
    fig, axes = plt.subplots(4, 4, figsize=(13, 10), constrained_layout=True)
    for axis, example in zip(axes.flat, examples):
        image, _ = dataset[int(example["index"])]
        cam = grad_cam(model, image.unsqueeze(0).to(args.device)).detach().cpu().numpy()
        raw = image[0].numpy() * NORMALIZATION["std"] + NORMALIZATION["mean"]
        axis.imshow(raw, cmap="gray", aspect="auto", vmin=0, vmax=1)
        axis.imshow(cam, cmap="magma", alpha=0.55, aspect="auto")
        axis.set_title(f"{str(example['category']).upper()} {example['date']} p={example['probability']:.2f}", fontsize=9)
        axis.set_axis_off()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    metadata = {"checkpoint": str(args.checkpoint), "normalization": NORMALIZATION, "examples": examples}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata["examples"], indent=1))


if __name__ == "__main__":
    main()
