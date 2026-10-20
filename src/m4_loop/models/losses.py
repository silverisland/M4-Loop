"""Fixed-capacity, finite-label point losses; no dataset or training dependencies."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


def normalize_capacity(value: Any, batch_size: int, reference: Tensor) -> Tensor:
    if value is None:
        raise ValueError("A fixed positive capacity is required for supervised loss")
    try:
        capacity = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("capacity must be numeric") from exc
    if capacity.ndim == 0:
        capacity = capacity.expand(batch_size).reshape(batch_size, 1)
    elif capacity.ndim == 1 and capacity.shape[0] == batch_size:
        capacity = capacity[:, None]
    elif capacity.ndim == 2 and capacity.shape == (batch_size, 1):
        pass
    else:
        raise ValueError(
            "capacity must be a scalar, [B], or [B,1]; "
            f"got {tuple(capacity.shape)} for B={batch_size}. "
            "Time-varying capacities are unsupported."
        )
    if not torch.isfinite(capacity).all() or not (capacity > 0).all():
        raise ValueError("capacity must contain only finite positive values")
    return capacity


def normalized_point_loss(
    pred: Tensor, target: Any, capacity: Any, *, horizon: int, loss_type: str
) -> Tensor:
    """Mean normalized point loss over finite target values.

    Error is (pred-target)/fixed_capacity. MAE uses |error|, MSE error²,
    and Huber uses SmoothL1(error,0,beta=1). No auxiliary term is added.
    Invalid labels are excluded before arithmetic; an empty mask is an error.
    """
    if not isinstance(pred, Tensor) or pred.ndim != 2:
        raise ValueError("pred must be a Tensor with shape [B,H]")
    if pred.shape[0] < 1 or pred.shape[1] != horizon:
        raise ValueError(f"pred must have shape [B,{horizon}]")
    try:
        target = torch.as_tensor(target, dtype=pred.dtype, device=pred.device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("target must contain numeric tensor values") from exc
    if target.shape != pred.shape:
        raise ValueError(
            f"target shape {tuple(target.shape)} must equal pred shape {tuple(pred.shape)}"
        )
    if not torch.isfinite(pred).all():
        raise ValueError("pred contains non-finite values")
    capacities = normalize_capacity(capacity, pred.shape[0], pred)
    valid = torch.isfinite(target)
    if not valid.any():
        raise ValueError("The batch has no finite target labels")
    error = (pred[valid] - target[valid]) / capacities.expand_as(pred)[valid]
    if loss_type == "mae":
        return error.abs().mean()
    if loss_type == "mse":
        return error.square().mean()
    return F.smooth_l1_loss(error, torch.zeros_like(error), beta=1.0)
