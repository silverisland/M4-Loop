"""Device selection, reproducible seeds and tensor transfer shared by workflows."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def resolve_device(requested: str | torch.device = "auto") -> torch.device:
    """Auto prefers CUDA, then MPS, then CPU; explicit unavailable devices fail."""
    name = str(requested).lower()
    mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if mps_available else "cpu"
    try:
        device = torch.device(name)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid device {requested!r}; use auto, cpu, cuda[:N], or mps.") from exc
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {device}. Use cpu, cuda[:N], or mps.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was explicitly requested but is unavailable.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {device.index} does not exist.")
    if device.type == "mps" and not mps_available:
        raise ValueError("MPS was explicitly requested but is unavailable.")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, Any], device: torch.device) -> tuple[dict, torch.Tensor | None]:
    features = {key: value.to(device) for key, value in batch["features"].items()}
    labels = batch["labels"]
    return features, labels.to(device) if labels is not None else None
