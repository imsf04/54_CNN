from __future__ import annotations

import torch


def select_device(requested: str = "auto") -> torch.device:
    """Select CPU, CUDA, or Apple MPS without assuming one platform."""

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
