"""Aggregate the three-seed full-data runs and build the seed-averaged evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analyze_predictions import COSTS_BPS, MODELS, build_report, write_series

SEEDS = (7, 17, 27)


def flatten(prefix: str, payload: object) -> dict[str, float]:
    if isinstance(payload, dict):
        out = {}
        for key, value in payload.items():
            out.update(flatten(f"{prefix}.{key}" if prefix else key, value))
        return out
    if isinstance(payload, list):
        return {}
    if isinstance(payload, bool) or not isinstance(payload, (int, float)):
        return {}
    return {prefix: float(payload)}


def summarise(reports: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    rows = [flatten("", report) for report in reports]
    frame = pd.DataFrame(rows)
    return {
        key: {"mean": float(frame[key].mean()), "std": float(frame[key].std(ddof=1))} for key in frame.columns
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="artifacts/full_seeds/predictions")
    parser.add_argument("--output", default="artifacts/analysis")
    args = parser.parse_args()
    directory = Path(args.predictions)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {"seeds": list(SEEDS), "by_seed": {}, "summary": {}}
    ensemble_all: dict[str, object] = {}
    for model in MODELS:
        frames = []
        for seed in SEEDS:
            frame = pd.read_parquet(directory / f"seed_{seed}_{model}_predictions.parquet")
            frame = frame.rename(columns={"probability": f"{model}_probability"})
            frame["Date"] = frame["Date"].astype(str)
            frames.append(frame)
        base = frames[0][["Date", "StockID", "label", "return"]].copy()
        for frame in frames[1:]:
            other = frame[["Date", "StockID", "label", "return"]]
            if not base.equals(other):
                raise ValueError(f"prediction rows differ across seeds for {model}")

        probability = f"{model}_probability"
        per_seed = []
        for seed, frame in zip(SEEDS, frames, strict=True):
            columns = ["Date", "StockID", "label", "return", probability]
            report, _ = build_report(frame[columns], models=(model,))
            per_seed.append(report[model])

        ensemble = base.copy()
        ensemble[probability] = sum(frame[probability].to_numpy() for frame in frames) / len(frames)
        ensemble_report, ensemble_series = build_report(ensemble, models=(model,))
        payload["by_seed"][model] = per_seed
        payload["summary"][model] = summarise(per_seed)
        ensemble_all.update(ensemble_report)
        write_series(output, ensemble_series)

    (output / "portfolio_analysis.json").write_text(json.dumps(ensemble_all, indent=2))
    summary = payload["summary"]
    headlines = {
        model: {
            "auc": summary[model]["classification.roc_auc"],
            "sharpe": summary[model]["decile_10.sharpe"],
            "sharpe20": summary[model]["decile_20.sharpe"],
            "cumulative": summary[model]["decile_10.cumulative_return"],
            "turnover": summary[model]["decile_10.turnover_mean"],
            "net10": summary[model][f"decile_10.net_of_costs_bps.{COSTS_BPS[1]}.sharpe"],
        }
        for model in MODELS
    }
    payload["headlines"] = headlines
    (output / "seeds_summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(headlines, indent=2))


if __name__ == "__main__":
    main()
