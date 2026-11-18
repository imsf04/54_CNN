"""Check that evaluation batch size does not move validation probabilities."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from price_trends.data import IterableImageDataset, load_real_data, temporal_split
from price_trends.models import PriceTrendCNN
from price_trends.services.training import predict_proba

MEAN = 0.10689710380321306
STD = 0.3089823829956947


def main() -> None:
    bundle = load_real_data(
        "/root/autodl-tmp/data/extracted", target="Retx_20d_label", horizon="20d", return_column="Retx_20d", lazy=True
    )
    splits = temporal_split(bundle)
    checkpoint = torch.load("/root/autodl-tmp/full_seeds/seed_7_cnn_20d.pt", map_location="cuda")
    model = PriceTrendCNN().to("cuda")
    model.load_state_dict(checkpoint["model"])
    outputs = {}
    for batch_size in (128, 1024):
        dataset = IterableImageDataset(
            splits["val"], batch_size=batch_size, mean=MEAN, std=STD, shuffle=False, seed=7
        )
        loader = DataLoader(dataset, batch_size=None, pin_memory=True)
        outputs[batch_size] = predict_proba(model, loader, "cuda")
    difference = abs(outputs[128].astype("float64") - outputs[1024].astype("float64"))
    print({"samples": len(outputs[128]), "max_abs_diff": float(difference.max()), "mean_abs_diff": float(difference.mean())})


if __name__ == "__main__":
    main()
