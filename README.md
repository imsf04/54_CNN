# (Re-)Imag(in)ing Price Trends

Reproducible CNN project scaffold for the MSDM5054 final project. The repository runs without the course data by using a deterministic synthetic dataset, then switches to the supplied 64x60 `.dat` images and Feather labels when they are available.

## Quick start

```bash
uv sync
uv run pytest -q
uv run python scripts/smoke_train.py --output artifacts/smoke
uv run ruff check .
```

The smoke run writes a checkpoint and JSON metrics. It is intentionally small and CPU-friendly. For Kaggle or Colab, upload the repository, install `requirements.txt`, and run the same commands. Set `--data-dir` to the directory containing the image and label files when the real data is available.

## Real-data contract

The loader expects image arrays shaped `N x 64 x 60` and labels with a date column plus a binary target. Missing labels encoded as `2` are excluded. The default target is `Ret_20d`; pass `--target` to select another supported return column. The temporal split reserves 1993–1999 for development and 2000 onward for out-of-sample evaluation.

## Project layout

- `src/price_trends/`: dataset, model, training, metrics, and interpretability code
- `scripts/smoke_train.py`: one-command synthetic-data validation
- `scripts/run_experiments.py`: 5d/20d/60d runs for Logistic, MLP, and CNN baselines
- `notebooks/price_trends_start_here.ipynb`: platform entry point
- `tests/`: behavior-level smoke tests
- `docs/adr/` and `GLOSSARY.md`: decisions and canonical project terms

The course PDF is the source of the project constraints; it is not copied into the repository.

The first financial evaluation uses an uncosted date-level long-short portfolio and reports cumulative return, annualized volatility, and Sharpe ratio. The PDF verifies an official 20-day archive URL; 5-day and 60-day URLs remain configurable because they were not verified in the supplied materials. The loader accepts a manually prepared data directory when network access is unavailable.
