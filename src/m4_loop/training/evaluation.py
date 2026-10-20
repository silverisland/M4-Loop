"""Validation metrics aggregated over finite target points."""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader

from ..models import M4LoopModel
from ..runtime import move_batch


@torch.no_grad()
def evaluate(
    model: M4LoopModel,
    loader: DataLoader,
    device: str | torch.device = "cpu",
) -> dict[str, float | int]:
    """Aggregate over finite target elements, including a final partial batch."""
    model.eval()
    device = torch.device(device)
    capacity_col = model.config["dataset"]["capacity_col"]
    point_count = 0
    loss_sum = absolute_sum = squared_sum = norm_absolute_sum = norm_squared_sum = 0.0
    for batch in loader:
        features, labels = move_batch(batch, device)
        if labels is None:
            raise ValueError("Validation requires explicit labels.")
        output = model(features, labels=labels)
        predictions = output.predictions
        if output.loss is None or not bool(torch.isfinite(output.loss)):
            raise ValueError("Validation loss is not finite.")
        if not bool(torch.isfinite(predictions).all()):
            raise ValueError("Validation predictions are not finite.")
        # CPU float64 accumulation also works with MPS, which has no float64.
        target_cpu = labels.detach().cpu().double()
        prediction_cpu = predictions.detach().cpu().double()
        capacity_cpu = features[capacity_col].detach().cpu().double().reshape(-1, 1)
        valid = torch.isfinite(target_cpu)
        count = int(valid.sum().item())
        errors = prediction_cpu - target_cpu
        normalized = errors / capacity_cpu
        point_count += count
        loss_sum += float(output.loss.item()) * count
        absolute_sum += float(errors[valid].abs().sum().item())
        squared_sum += float(errors[valid].square().sum().item())
        norm_absolute_sum += float(normalized[valid].abs().sum().item())
        norm_squared_sum += float(normalized[valid].square().sum().item())
    if point_count == 0:
        raise ValueError("Validation has no finite target values.")
    return {
        "loss": loss_sum / point_count,
        "mae": absolute_sum / point_count,
        "rmse": math.sqrt(squared_sum / point_count),
        "normalized_mae": norm_absolute_sum / point_count,
        "normalized_rmse": math.sqrt(norm_squared_sum / point_count),
        "valid_points": point_count,
    }
