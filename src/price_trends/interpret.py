"""Backward-compatible interpretation imports and paper-backed ablations."""

from .services.interpretation import grad_cam


def ablation_configs() -> list[dict[str, object]]:
    """Architecture variants suggested by the paper's sensitivity analysis."""

    return [
        {"name": "baseline", "filters": (64, 128, 256), "blocks": 3},
        {"name": "fewer_blocks", "filters": (64, 128), "blocks": 2},
        {"name": "four_blocks", "filters": (64, 128, 256, 512), "blocks": 4},
    ]


__all__ = ["ablation_configs", "grad_cam"]
