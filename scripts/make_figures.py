import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "artifacts" / "analysis"
FIGURES = ROOT / "reports" / "figures"
MODELS = ("logistic", "mlp", "cnn")
LABELS = {"logistic": "Logistic", "mlp": "MLP", "cnn": "CNN"}
COLORS = {"logistic": "#4477aa", "mlp": "#ee6677", "cnn": "#228833"}


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 200,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def cumulative(series: pd.Series) -> pd.Series:
    return (1.0 + series.fillna(0.0)).cumprod()


def plot_cumulative() -> None:
    figure, (left, right) = plt.subplots(1, 2, figsize=(6.4, 2.4))
    for model in MODELS:
        frame = pd.read_csv(ANALYSIS / f"portfolio_{model}_decile10.csv", parse_dates=["Date"])
        dates = frame["Date"]
        left.plot(dates, cumulative(frame["gross"]), color=COLORS[model], label=LABELS[model], linewidth=1.0)
        right.plot(dates, cumulative(frame["net_10"]), color=COLORS[model], label=LABELS[model], linewidth=1.0)
    left.set_yscale("log")
    left.set_title("Gross top-decile long-short")
    right.set_title("Net of 10 bps one-way costs")
    for axis in (left, right):
        axis.set_ylabel("Cumulative value")
        axis.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
    left.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(FIGURES / "cumulative.pdf")
    plt.close(figure)


def plot_deciles() -> None:
    report = json.loads((ANALYSIS / "portfolio_analysis.json").read_text())
    positions = np.arange(10)
    width = 0.27
    figure, axis = plt.subplots(figsize=(6.4, 2.4))
    for offset, model in zip((-width, 0.0, width), MODELS):
        means = np.asarray(report[model]["decile_means"]) * 100
        axis.bar(positions + offset, means, width=width, color=COLORS[model], label=LABELS[model], linewidth=0)
    axis.set_xticks(positions, [str(index + 1) for index in positions])
    axis.set_xlabel("Predicted probability decile")
    axis.set_ylabel("Mean monthly return (%)")
    axis.axhline(0.0, color="black", linewidth=0.5)
    axis.legend(frameon=False, ncol=3)
    figure.tight_layout()
    figure.savefig(FIGURES / "deciles.pdf")
    plt.close(figure)


def plot_cost_sensitivity() -> None:
    report = json.loads((ANALYSIS / "portfolio_analysis.json").read_text())
    costs = ("0", "5", "10", "20")
    figure, axis = plt.subplots(figsize=(3.2, 2.2))
    for model in MODELS:
        values = [report[model]["decile_10"]["sharpe"]]
        values += [report[model]["decile_10"]["net_of_costs_bps"][str(bps)]["sharpe"] for bps in (5, 10, 20)]
        axis.plot(range(len(costs)), values, marker="o", markersize=3, color=COLORS[model], label=LABELS[model])
    axis.set_xticks(range(len(costs)), costs)
    axis.set_xlabel("One-way cost (bps)")
    axis.set_ylabel("Annualised Sharpe ratio")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(FIGURES / "costs.pdf")
    plt.close(figure)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    style()
    plot_cumulative()
    plot_deciles()
    plot_cost_sensitivity()
    print("figures written to", FIGURES)


if __name__ == "__main__":
    main()
