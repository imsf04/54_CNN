"""Summarise multi-seed full-data runs into per-configuration mean and standard deviation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

METRICS = (
    "accuracy",
    "roc_auc",
    "average_precision",
    "portfolio_cumulative_return",
    "portfolio_annualized_volatility",
    "portfolio_sharpe",
    "portfolio_20pct_sharpe",
)


def load_runs(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text())
    rows = []
    for run in payload["runs"]:
        row = {"seed": run["seed"], "name": run["name"], "epochs_run": run["epochs_run"], "best_epoch": run["best_epoch"]}
        row.update({metric: run["metrics"][metric] for metric in METRICS})
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("name")
    summary = grouped[list(METRICS)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["seeds"] = grouped.size()
    return summary.round(6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", default="artifacts/full_seeds/results.json")
    parser.add_argument("--ablations", default="artifacts/full_ablations/results.json")
    parser.add_argument("--output", default="artifacts/full_summary")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frames = {}
    for label, path in (("main", args.main), ("ablations", args.ablations)):
        if Path(path).exists():
            frames[label] = load_runs(Path(path))

    combined = pd.concat(frames.values(), ignore_index=True) if frames else pd.DataFrame()
    combined.to_csv(output / "runs.csv", index=False)
    summary = summarise(combined)
    summary.to_csv(output / "summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
