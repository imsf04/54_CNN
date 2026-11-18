"""Measure whether the columns Grad-CAM highlights actually drive CNN predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from price_trends.data import ImageDataset, load_real_data, temporal_split
from price_trends.models import PriceTrendCNN
from price_trends.services.interpretation import grad_cam

MEAN = 0.10689710380321306
STD = 0.3089823829956947


def load_model(checkpoint: Path) -> PriceTrendCNN:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = state["model"] if "model" in state else state
    weights = {key.removeprefix("module."): value for key, value in weights.items()}
    model = PriceTrendCNN()
    model.load_state_dict(weights)
    model.eval()
    return model


def select(frame: pd.DataFrame, count: int, rng: np.random.Generator) -> np.ndarray:
    confident = frame.index[frame["cnn_probability"] > 0.6].to_numpy()
    years = frame.loc[confident, "Date"].str[:4].astype(int).to_numpy()
    chosen = []
    for year in np.unique(years):
        pool = confident[years == year]
        chosen.append(rng.choice(pool, size=min(10, len(pool)), replace=False))
    return np.sort(np.concatenate(chosen))[:count]


def probability(model: PriceTrendCNN, image: torch.Tensor) -> float:
    with torch.no_grad():
        return float(torch.softmax(model(image), dim=1)[0, 1])


def masked(image: torch.Tensor, columns: np.ndarray) -> torch.Tensor:
    altered = image.clone()
    altered[:, :, :, columns] = 0.0
    return altered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/kaggle_output_1118/cnn_20d.pt")
    parser.add_argument("--predictions", default="artifacts/kaggle_output_1118/test_predictions.parquet")
    parser.add_argument("--data-dir", default="data/raw/extracted")
    parser.add_argument("--output", default="artifacts/analysis/occlusion.json")
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()

    frame = pd.read_parquet(args.predictions)
    frame["Date"] = frame["Date"].astype(str)
    bundle = load_real_data(args.data_dir, target="Retx_20d_label", horizon="20d", return_column="Retx_20d", lazy=True)
    test = temporal_split(bundle)["test"]
    dataset = ImageDataset(test, mean=MEAN, std=STD)
    model = load_model(Path(args.checkpoint))

    rng = np.random.default_rng(7)
    indices = select(frame.reset_index(drop=True), args.samples, rng)
    drops = {"top_cam": [], "random": [], "recent": [], "oldest": []}
    top_wins = 0
    for index in indices:
        image, _ = dataset[int(index)]
        image = image.unsqueeze(0)
        cam = grad_cam(model, image).mean(dim=0).detach().numpy()
        base = probability(model, image)
        columns = np.argsort(-cam)[:3]
        random_columns = rng.choice(60, size=3, replace=False)
        candidates = {
            "top_cam": columns,
            "random": random_columns,
            "recent": np.arange(57, 60),
            "oldest": np.arange(0, 3),
        }
        values = {name: base - probability(model, masked(image, np.asarray(cols))) for name, cols in candidates.items()}
        for name, value in values.items():
            drops[name].append(value)
        top_wins += int(values["top_cam"] > max(values["random"], values["recent"]))
    summary = {}
    for name, values in drops.items():
        array = np.asarray(values)
        summary[name] = {
            "mean_drop": float(array.mean()),
            "std": float(array.std(ddof=1)),
            "t_stat": float(array.mean() / array.std(ddof=1) * np.sqrt(len(array))),
            "share_positive": float((array > 0).mean()),
        }
    report = {
        "samples": int(len(indices)),
        "checkpoint": args.checkpoint,
        "top_cam_wins": int(top_wins),
        "top_cam_win_rate": float(top_wins / len(indices)),
        **summary,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
