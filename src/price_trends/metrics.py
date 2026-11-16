from __future__ import annotations
import numpy as np

def classification_metrics(y_true, probabilities, threshold=0.5):
    y = np.asarray(y_true).astype(int); p = np.asarray(probabilities); pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    accuracy = float((tp + tn) / max(1, len(y)))
    precision = float(tp / max(1, tp + fp)); recall = float(tp / max(1, tp + fn))
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "tp": tp, "tn": tn, "fp": fp, "fn": fn}

def long_short_metrics(probabilities, returns, dates, quantile=0.2, periods_per_year=252):
    p = np.asarray(probabilities); r = np.asarray(returns, dtype=float); d = np.asarray(dates)
    rows = []
    for date in np.unique(d):
        idx = np.flatnonzero(d == date)
        if len(idx) < 4: continue
        order = idx[np.argsort(p[idx])]; n = max(1, int(len(order) * quantile))
        rows.append(float(r[order[-n:]].mean() - r[order[:n]].mean()))
    daily = np.asarray(rows, dtype=float)
    if len(daily) == 0: return {"days": 0, "cumulative_return": 0.0, "annualized_volatility": 0.0, "sharpe": 0.0}
    vol = float(daily.std(ddof=1) * np.sqrt(periods_per_year)) if len(daily) > 1 else 0.0
    sharpe = float(daily.mean() / daily.std(ddof=1) * np.sqrt(periods_per_year)) if len(daily) > 1 and daily.std(ddof=1) > 0 else 0.0
    return {"days": int(len(daily)), "cumulative_return": float(np.prod(1 + daily) - 1), "annualized_volatility": vol, "sharpe": sharpe}
