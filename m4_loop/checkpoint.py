"""Versioned, strict checkpoints that can be loaded without the original repo."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from .config import load_config
from .model import M4LoopModel

CHECKPOINT_FORMAT = "m4_loop"
CHECKPOINT_VERSION = 1
ARCHITECTURE_VERSION = 1


def save_checkpoint(
    path: str | Path,
    model: M4LoopModel,
    *,
    epoch: int,
    global_step: int,
    metric: float,
) -> Path:
    """Atomically save the complete configuration and CPU model parameters."""
    if not math.isfinite(metric):
        raise ValueError("Checkpoint metric must be finite.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "config": load_config(model.config),
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "epoch": int(epoch),
        "global_step": int(global_step),
        "metric": float(metric),
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[M4LoopModel, dict[str, Any]]:
    """Load only this project's format, with safe loading and strict key checks."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"Cannot safely load checkpoint {path}; expected an M4-Loop checkpoint."
        ) from exc
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            "Checkpoint format is not m4_loop. Original M4 and raw state_dict "
            "checkpoints cannot be loaded by this project."
        )
    if (
        payload.get("format_version") != CHECKPOINT_VERSION
        or payload.get("architecture_version") != ARCHITECTURE_VERSION
    ):
        raise ValueError(
            "Unsupported M4-Loop checkpoint version: "
            f"format={payload.get('format_version')}, "
            f"architecture={payload.get('architecture_version')}."
        )
    required = {"config", "state_dict", "epoch", "global_step", "metric"}
    if not required.issubset(payload):
        raise ValueError(f"Incomplete checkpoint; missing keys: {sorted(required - payload.keys())}")
    if not isinstance(payload["config"], dict) or not isinstance(payload["state_dict"], dict):
        raise ValueError("Checkpoint config and state_dict must be dictionaries.")
    if (
        not isinstance(payload["epoch"], int)
        or not isinstance(payload["global_step"], int)
        or payload["epoch"] < 0
        or payload["global_step"] < 0
    ):
        raise ValueError("Checkpoint epoch and global_step must be nonnegative integers.")
    if (
        not isinstance(payload["metric"], (int, float))
        or not math.isfinite(payload["metric"])
    ):
        raise ValueError("Checkpoint metric must be a finite number.")
    config = load_config(payload["config"])
    model = M4LoopModel(config)
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(
            "Checkpoint parameters do not strictly match its M4-Loop configuration."
        ) from exc
    model.to(device)
    model.eval()
    metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    return model, metadata
