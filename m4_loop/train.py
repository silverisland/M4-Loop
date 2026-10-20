"""Single-device training for explicitly supplied train/validation Parquet files."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .config import load_config
from .data import WindowDataset, collate_windows
from .model import M4LoopModel


def resolve_device(requested: str | torch.device = "auto") -> torch.device:
    """Auto prefers CUDA, then MPS, then CPU; explicit unavailable devices fail."""
    name = str(requested).lower()
    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
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


def _move_batch(batch: dict[str, Any], device: torch.device) -> tuple[dict, torch.Tensor | None]:
    features = {key: value.to(device) for key, value in batch["features"].items()}
    labels = batch["labels"]
    return features, labels.to(device) if labels is not None else None


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
        features, labels = _move_batch(batch, device)
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


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def train(
    config: dict[str, Any] | str | Path | None,
    train_source: str | Path | pd.DataFrame,
    valid_source: str | Path | pd.DataFrame,
    out_dir: str | Path,
    max_steps: int = -1,
) -> dict[str, Any]:
    """Fit a new model and save the best validation checkpoint.

    ``max_steps`` caps total optimizer steps, then still evaluates and saves.
    No pretrained M4 weights, implicit splits, or automatic window changes exist.
    """
    config = load_config(config)
    optimization = config["optimization"]
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or (
        max_steps != -1 and max_steps <= 0
    ):
        raise ValueError("max_steps must be -1 (unlimited) or a positive integer.")
    output_dir = Path(out_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(
            f"{output_dir} is not an empty directory; choose a new output directory."
        )
    device = resolve_device(optimization["device"])
    seed_everything(optimization["random_seed"])
    train_dataset = WindowDataset(train_source, config, supervised=True)
    valid_dataset = WindowDataset(valid_source, config, supervised=True)
    generator = torch.Generator().manual_seed(optimization["random_seed"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=optimization["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=optimization["num_workers"],
        collate_fn=collate_windows,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=optimization["eval_batch_size"],
        shuffle=False,
        num_workers=optimization["num_workers"],
        collate_fn=collate_windows,
    )
    model = M4LoopModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimization["learning_rate"],
        weight_decay=optimization["weight_decay"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_config.yaml").write_text(
        yaml.safe_dump(model.config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    data_stats = {"train": train_dataset.stats, "valid": valid_dataset.stats}
    _write_json(output_dir / "data_stats.json", data_stats)
    history: list[dict[str, Any]] = []
    best_metric = math.inf
    best_epoch = global_step = stale_epochs = 0
    best_validation: dict[str, Any] = {}
    stop_reason = "epochs_completed"
    checkpoint_path = output_dir / "best.pt"
    for epoch in range(1, optimization["train_epochs"] + 1):
        model.train()
        train_loss_sum = 0.0
        train_points = 0
        for batch in train_loader:
            features, labels = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            result = model(features, labels=labels)
            if result.loss is None or not bool(torch.isfinite(result.loss)):
                raise ValueError(f"Training loss became non-finite at step {global_step + 1}.")
            result.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                optimization["grad_clip"],
                error_if_nonfinite=True,
            )
            optimizer.step()
            count = int(torch.isfinite(labels).sum().item())
            train_points += count
            train_loss_sum += float(result.loss.detach().item()) * count
            global_step += 1
            if max_steps > 0 and global_step >= max_steps:
                stop_reason = "max_steps"
                break
        validation = evaluate(model, valid_loader, device)
        improved = validation["loss"] < best_metric
        if improved:
            best_metric = float(validation["loss"])
            best_epoch = epoch
            best_validation = dict(validation)
            stale_epochs = 0
            save_checkpoint(
                checkpoint_path,
                model,
                epoch=epoch,
                global_step=global_step,
                metric=best_metric,
            )
        else:
            stale_epochs += 1
        entry = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss_sum / train_points,
            "train_valid_points": train_points,
            "validation": validation,
            "improved": improved,
        }
        history.append(entry)
        _write_json(output_dir / "history.json", history)
        print(
            f"epoch={epoch} step={global_step} "
            f"train_loss={entry['train_loss']:.6f} "
            f"valid_loss={validation['loss']:.6f} best={best_metric:.6f}",
            flush=True,
        )
        if stop_reason == "max_steps":
            break
        if stale_epochs >= optimization["patience"]:
            stop_reason = "early_stopping"
            break
    metrics = {
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "best_validation": best_validation,
        "global_step": global_step,
        "epochs_completed": len(history),
        "stop_reason": stop_reason,
        "device": str(device),
        "data_stats": data_stats,
    }
    _write_json(output_dir / "metrics.json", metrics)
    return {
        **metrics,
        "best_checkpoint": str(checkpoint_path.resolve()),
        "history": history,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--out", "--out-dir", dest="out", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args(argv)
    result = train(args.config, args.train, args.valid, args.out, args.max_steps)
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
