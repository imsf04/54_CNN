"""Rebuild date-level portfolio evidence from saved out-of-sample predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

MODELS = ("logistic", "mlp", "cnn")
PERIODS_PER_YEAR = 12
QUANTILES = (0.1, 0.2)
COSTS_BPS = (5, 10, 20)
SUBPERIODS = {
    "2000-2004": (2000, 2004),
    "2005-2012": (2005, 2012),
    "2013-2019": (2013, 2019),
}


def annualized(values: np.ndarray) -> dict[str, float]:
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    return {
        "periods": int(len(values)),
        "mean": mean,
        "std": std,
        "annualized_volatility": std * np.sqrt(PERIODS_PER_YEAR),
        "sharpe": mean / std * np.sqrt(PERIODS_PER_YEAR),
        "cumulative_return": float(np.prod(1 + values) - 1),
        "t_stat": mean / std * np.sqrt(len(values)),
        "max_drawdown": float(np.min(np.cumprod(1 + values) / np.maximum.accumulate(np.cumprod(1 + values)) - 1)),
    }


def newey_west_t(values: np.ndarray, lags: int = 3) -> float:
    x = values - values.mean()
    n = len(x)
    gamma0 = float(np.dot(x, x) / n)
    variance = gamma0
    for lag in range(1, lags + 1):
        gamma = float(np.dot(x[lag:], x[:-lag]) / n)
        variance += 2 * (1 - lag / (lags + 1)) * gamma
    return float(values.mean() / np.sqrt(variance / n))


def block_bootstrap_sharpe(values: np.ndarray, block: int = 12, draws: int = 2000, seed: int = 7) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    count = n // block + 1
    starts = rng.integers(0, n - block + 1, size=(draws, count))
    blocks = np.stack([np.concatenate([values[start : start + block] for start in row]) for row in starts])
    sharpe = blocks.mean(axis=1) / blocks.std(axis=1, ddof=1) * np.sqrt(PERIODS_PER_YEAR)
    return {"ci_low": float(np.quantile(sharpe, 0.025)), "ci_high": float(np.quantile(sharpe, 0.975))}


def portfolio_weights(frame: pd.DataFrame, probability: str, quantile: float) -> pd.DataFrame:
    weights = []
    for date, group in frame.groupby("Date", sort=True):
        n = max(1, int(len(group) * quantile))
        order = group.sort_values(probability)
        selected = pd.concat([order.head(n), order.tail(n)])
        weight = np.where(np.arange(len(selected)) < n, -1 / n, 1 / n)
        weights.append(pd.Series(weight, index=selected["StockID"].to_numpy(), name=date))
    wide = pd.concat(weights, axis=1).T.fillna(0.0)
    return wide


def gross_series(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    return (weights * returns.reindex_like(weights)).sum(axis=1)


def turnover_series(weights: pd.DataFrame) -> pd.Series:
    changes = weights.diff().abs().sum(axis=1)
    return changes.iloc[1:] / 2


def classification_stats(labels: pd.Series, probabilities: pd.Series) -> dict[str, float]:
    predicted = (probabilities >= 0.5).astype(int)
    return {
        "rows": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
    }


def decile_means(frame: pd.DataFrame, probability: str) -> list[float]:
    ranks = frame.groupby("Date")[probability].rank(method="first")
    sizes = frame.groupby("Date")[probability].transform("size")
    decile = np.minimum(((ranks - 1) * 10 / sizes).astype(int), 9)
    return [float(value) for value in frame.groupby(decile)["return"].mean().tolist()]


def build_report(frame: pd.DataFrame, models: tuple[str, ...] = MODELS) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    frame = frame.copy()
    frame["Date"] = frame["Date"].astype(str)
    returns = frame.pivot(index="Date", columns="StockID", values="return").sort_index()
    years = returns.index.str[:4].astype(int)
    row_years = frame["Date"].str[:4].astype(int)

    report: dict[str, object] = {"periods": int(len(returns)), "start": returns.index[0], "end": returns.index[-1]}
    series: dict[str, pd.DataFrame] = {}
    for model in models:
        probability = f"{model}_probability"
        model_report: dict[str, object] = {}
        for quantile in QUANTILES:
            weights = portfolio_weights(frame, probability, quantile)
            gross = gross_series(weights, returns)
            turnover = turnover_series(weights)
            stats = annualized(gross.to_numpy())
            stats["newey_west_t"] = newey_west_t(gross.to_numpy())
            stats["bootstrap_sharpe"] = block_bootstrap_sharpe(gross.to_numpy())
            stats["turnover_mean"] = float(turnover.mean())
            stats["net_of_costs_bps"] = {
                str(bps): annualized(gross.to_numpy()[1:] - turnover.to_numpy() * bps * 2 / 1e4) for bps in COSTS_BPS
            }
            model_report[f"decile_{int(quantile * 100)}"] = stats
            if quantile == 0.1:
                values = pd.DataFrame({"Date": gross.index, "gross": gross.to_numpy()})
                for bps in COSTS_BPS:
                    values[f"net_{bps}"] = np.concatenate(
                        [[np.nan], gross.to_numpy()[1:] - turnover.to_numpy() * bps * 2 / 1e4]
                    )
                series[model] = values
                model_report["subperiods"] = {
                    label: annualized(gross.to_numpy()[(years >= start) & (years <= end)])
                    for label, (start, end) in SUBPERIODS.items()
                }
                model_report["classification_by_period"] = {
                    label: classification_stats(
                        frame.loc[(row_years >= start) & (row_years <= end), "label"],
                        frame.loc[(row_years >= start) & (row_years <= end), probability],
                    )
                    for label, (start, end) in SUBPERIODS.items()
                }
                model_report["yearly_mean"] = {str(year): float(value) for year, value in gross.groupby(years).mean().items()}
        model_report["decile_means"] = decile_means(frame, probability)
        model_report["classification"] = classification_stats(frame["label"], frame[probability])
        report[model] = model_report
    return report, series


def write_series(output: Path, series: dict[str, pd.DataFrame]) -> None:
    for model, values in series.items():
        values.to_csv(output / f"portfolio_{model}_decile10.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="artifacts/kaggle_output_1118/test_predictions.parquet")
    parser.add_argument("--output", default="artifacts/analysis")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    report, series = build_report(pd.read_parquet(args.predictions))
    write_series(output, series)
    (output / "portfolio_analysis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({name: report[name]["decile_10"] for name in MODELS}, indent=2))


if __name__ == "__main__":
    main()
