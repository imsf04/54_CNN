from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

METRICS = ("roc_auc", "average_precision", "portfolio_sharpe")


def load(path: Path, horizon: str) -> pd.DataFrame:
    payload = json.loads(path.read_text())
    frame = pd.DataFrame(payload["runs"])
    frame["horizon"] = horizon
    return frame


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["horizon", "name"], sort=False)
    rows = []
    for (horizon, name), group in grouped:
        row = {"horizon": horizon, "model": name, "seeds": len(group)}
        for metric in METRICS:
            values = group["metrics"].map(lambda item: item[metric])
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_sd"] = values.std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", default=[], help="horizon=path")
    parser.add_argument("--output", default="artifacts/horizons_summary.csv")
    args = parser.parse_args()

    frames = []
    for item in args.run:
        horizon, raw = item.split("=", 1)
        frames.append(load(Path(raw), horizon))
    summary = summarise(pd.concat(frames, ignore_index=True))
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(target, index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
