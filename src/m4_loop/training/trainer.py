"""Training orchestration and best-checkpoint selection, independent of the CLI."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from ..checkpoint import save_checkpoint
from ..config import load_config
from ..data import WindowDataset, collate_windows
from ..models import M4LoopModel
from ..runtime import move_batch, resolve_device, seed_everything
from .evaluation import evaluate


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
    if (
        isinstance(max_steps, bool)
        or not isinstance(max_steps, int)
        or (max_steps != -1 and max_steps <= 0)
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
            features, labels = move_batch(batch, device)
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
