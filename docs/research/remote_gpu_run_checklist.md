# Remote GPU formal-run gate

Run this checklist immediately before a formal remote run. It is a gate, not a substitute for the local artifacts.

## Local gates

- [ ] `uv run pytest -q` passes.
- [ ] `uv run ruff check src scripts tests` passes.
- [ ] `git diff --check` passes.
- [ ] `artifacts/local_ablations_normalized_v2/results.json` records train-only normalization, seeds 7/17/27, 4 complete epochs, 10% decile metrics, and 20% sensitivity metrics.
- [ ] `gradcam_examples.json` has TP/TN/FP/FN four each and includes date and stock ID.
- [ ] The report and ADR match the artifact directory and selected Grad-CAM model (`cnn_3blocks`, seed 7).

## Remote read-only preflight

Before starting training, record the output of:

```bash
nvidia-smi --query-gpu=index,name,driver_version,memory.total,utilization.gpu,memory.used --format=csv
python - <<'PY'
import torch
print({'torch': torch.__version__, 'cuda': torch.version.cuda, 'devices': torch.cuda.device_count()})
PY
who
```

Do not start if either GPU is occupied by another workload. Save the preflight output beside the formal artifacts with the code revision, data archive hash, command, and environment versions.

## Formal-run invariants

- The input is the 20-day image archive only.
- The data loader must find matching 20-day image and label files; missing horizons must fail loudly.
- Train-only mean/std must be printed and saved.
- The course split is 1993–1999 development and 2000+ out-of-sample.
- The model is selected using validation loss only; test metrics are reported once after selection.
- Primary portfolio output is top/bottom 10% decile; top/bottom 20% is sensitivity only.
- Raw checkpoints, histories, predictions, metrics, and preflight output must be copied back locally.
