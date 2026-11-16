"""Fast synthetic smoke test; it is not a project result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from torch.utils.data import DataLoader

from price_trends.data import ImageDataset, make_synthetic_data, temporal_split
from price_trends.models import PriceTrendCNN
from price_trends.services.training import fit, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/smoke")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    set_seed(7)
    splits = temporal_split(make_synthetic_data(48))
    model = PriceTrendCNN()
    history = fit(
        model,
        DataLoader(ImageDataset(splits["train"]), batch_size=8, shuffle=True),
        DataLoader(ImageDataset(splits["val"]), batch_size=8),
        epochs=args.epochs,
        checkpoint=Path(args.output) / "checkpoint.pt",
    )
    metrics = {
        "history": history,
        "test_split_size": len(splits["test"].labels),
        "status": "passed",
        "note": "synthetic smoke test only",
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
