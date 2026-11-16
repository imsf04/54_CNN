from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from torch.utils.data import DataLoader
from price_trends.data import ImageDataset, make_synthetic_data, temporal_split
from price_trends.model import PriceTrendCNN
from price_trends.train import fit, run_epoch

def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", default="artifacts/smoke"); args = p.parse_args()
    splits = temporal_split(make_synthetic_data(48)); train, val = splits["train"], splits["val"]
    model = PriceTrendCNN(); history = fit(model, DataLoader(ImageDataset(train), batch_size=8, shuffle=True), DataLoader(ImageDataset(val), batch_size=8), checkpoint=Path(args.output) / "checkpoint.pt")
    metrics = {"history": history, "test_split_size": len(splits["test"].labels), "status": "passed"}
    Path(args.output).mkdir(parents=True, exist_ok=True); (Path(args.output) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
if __name__ == "__main__": main()
