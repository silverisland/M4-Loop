"""Predict complete windows using a standalone M4-Loop checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint
from .data import WindowDataset, collate_windows
from .train import resolve_device


@torch.no_grad()
def predict(
    checkpoint: str | Path,
    data_source: str | Path | pd.DataFrame,
    out_path: str | Path | None = None,
    batch_size: int | None = None,
    device: str | torch.device | None = None,
) -> pd.DataFrame:
    model, _ = load_checkpoint(checkpoint, device="cpu")
    config = model.config
    # Checkpoints are portable: the training machine's explicit device may not
    # exist on the inference host. An explicit prediction override still fails
    # clearly when unavailable.
    actual_device = resolve_device(device if device is not None else "auto")
    model.to(actual_device)
    actual_batch_size = batch_size if batch_size is not None else config["optimization"]["eval_batch_size"]
    if (
        isinstance(actual_batch_size, bool)
        or not isinstance(actual_batch_size, int)
        or actual_batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer.")
    dataset = WindowDataset(data_source, config, supervised=False)
    loader = DataLoader(
        dataset,
        batch_size=actual_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_windows,
    )
    predictions: list[list[float]] = []
    for batch in loader:
        features = {
            key: value.to(actual_device) for key, value in batch["features"].items()
        }
        result = model(features)
        if not bool(torch.isfinite(result.predictions).all()):
            raise ValueError("Model produced non-finite predictions.")
        predictions.extend(result.predictions.detach().cpu().tolist())
    dataset_cfg = config["dataset"]
    metadata_columns = list(
        dict.fromkeys(
            dataset_cfg[name]
            for name in ("entity_col", "timestamp_col", "capacity_col", "target_col")
            if dataset_cfg[name] in dataset.frame.columns
        )
    )
    output = dataset.frame[metadata_columns].copy()
    output["observe_power_predict"] = predictions
    if out_path is not None:
        destination = Path(out_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        output.to_parquet(destination, engine="pyarrow", index=False)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", "--output", dest="out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    frame = predict(args.checkpoint, args.data, args.out, args.batch_size, args.device)
    print(f"Saved {len(frame)} prediction rows to {args.out.resolve()}")


if __name__ == "__main__":
    main()
