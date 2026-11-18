from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from price_trends.references import PAPER_20D


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
    amp: bool = False,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    total_loss = 0.0
    correct = 0
    count = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(x)
            loss = loss_fn(logits, y)
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * len(y)
        correct += int((logits.argmax(1) == y).sum())
        count += len(y)
    return {"loss": total_loss / max(1, count), "accuracy": correct / max(1, count)}


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    learning_rate: float = PAPER_20D.learning_rate,
    patience: int = PAPER_20D.early_stopping_patience,
    device: torch.device | str = "cpu",
    checkpoint: str | Path | None = None,
    amp: bool = False,
) -> list[dict[str, object]]:
    """Train with the paper's Adam/lr and two-epoch early stopping rule."""

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, amp=amp)
        val_metrics = run_epoch(model, val_loader, None, device, amp=amp)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        if val_metrics["loss"] < best_loss:
            best_loss = float(val_metrics["loss"])
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
            if checkpoint:
                path = Path(checkpoint)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": best_state, "history": history}, path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device | str = "cpu") -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []
    for x, _ in loader:
        probabilities.append(torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probabilities) if probabilities else np.empty(0, dtype="float32")
