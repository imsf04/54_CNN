from __future__ import annotations

import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader


def run_epoch(model, loader, optimizer=None, device="cpu"):
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.CrossEntropyLoss()
    total_loss = correct = count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if training: optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        if training:
            loss.backward(); optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item(); count += len(y)
    return {"loss": total_loss / max(1, count), "accuracy": correct / max(1, count)}


def fit(model, train_loader, val_loader, epochs=2, lr=1e-3, device="cpu", checkpoint=None):
    model.to(device); optimizer = torch.optim.Adam(model.parameters(), lr=lr); history = []
    best = float("inf")
    for epoch in range(1, epochs + 1):
        train = run_epoch(model, train_loader, optimizer, device)
        val = run_epoch(model, val_loader, None, device)
        row = {"epoch": epoch, "train": train, "val": val}; history.append(row)
        if checkpoint and val["loss"] < best:
            best = val["loss"]; Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "history": history}, checkpoint)
    return history
