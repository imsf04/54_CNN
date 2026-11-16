from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from price_trends.domain import DataBundle

from .training import predict_proba


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(probabilities, dtype=float)
    pred = (p >= threshold).astype(int)
    result: dict[str, float] = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
    }
    result["roc_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    result["average_precision"] = (
        float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    )
    return result


def long_short_metrics(
    probabilities: np.ndarray,
    returns: np.ndarray,
    dates: np.ndarray,
    quantile: float = 0.2,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Evaluate the paper-style date-level top-minus-bottom portfolio."""

    p, r, d = np.asarray(probabilities), np.asarray(returns, dtype=float), np.asarray(dates)
    valid = np.isfinite(p) & np.isfinite(r)
    p, r, d = p[valid], r[valid], d[valid]
    daily: list[float] = []
    for date in np.unique(d):
        indices = np.flatnonzero(d == date)
        if len(indices) < 4:
            continue
        order = indices[np.argsort(p[indices])]
        n = max(1, int(len(order) * quantile))
        daily.append(float(r[order[-n:]].mean() - r[order[:n]].mean()))
    values = np.asarray(daily, dtype=float)
    if len(values) < 2:
        return {"days": float(len(values)), "cumulative_return": 0.0, "annualized_volatility": 0.0, "sharpe": 0.0}
    std = values.std(ddof=1)
    return {
        "days": float(len(values)),
        "cumulative_return": float(np.prod(1 + values) - 1),
        "annualized_volatility": float(std * np.sqrt(periods_per_year)),
        "sharpe": float(values.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0,
    }


def evaluate_model(model, bundle: DataBundle, loader: DataLoader, device: str = "cpu") -> dict[str, float]:
    probabilities = predict_proba(model, loader, device)
    result = classification_metrics(bundle.labels, probabilities)
    if bundle.returns is not None:
        result.update({f"portfolio_{k}": v for k, v in long_short_metrics(probabilities, bundle.returns, bundle.dates).items()})
    return result
