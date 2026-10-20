"""Read already-split forecasting windows without resizing or splitting them."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import feature_columns, load_config


def read_frame(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Accept a DataFrame or one Parquet file; never infer a train/valid split."""
    if isinstance(source, pd.DataFrame):
        frame = source.copy(deep=False)
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Parquet file does not exist: {path}")
        frame = pd.read_parquet(path, engine="pyarrow")
    if frame.columns.has_duplicates:
        raise ValueError("Dataset contains duplicate column names.")
    if frame.empty:
        raise ValueError("Dataset is empty.")
    return frame


def _sequence(value: Any, column: str, row: int, length: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Row {row}, {column!r}: expected a numeric sequence.") from exc
    if array.ndim != 1 or array.shape[0] != length:
        raise ValueError(
            f"Row {row}, {column!r}: expected shape ({length},), got {array.shape}; "
            "windows are never cropped or padded automatically."
        )
    return array


def _is_missing_scalar(value: Any) -> bool:
    return value is None or (np.ndim(value) == 0 and bool(pd.isna(value)))


class WindowDataset(Dataset):
    """One row is one complete window; metadata and labels are not features.

    Entirely missing supervision is removed in supervised mode and counted in
    ``stats``. Partially missing labels are retained for the model's finite-label
    mask. Inference does not inspect labels, and capacity is optional there.
    """

    def __init__(
        self,
        source: str | Path | pd.DataFrame,
        config: Mapping[str, Any] | str | Path | None = None,
        supervised: bool = True,
    ) -> None:
        self.config = load_config(config)
        self.supervised = supervised
        frame = read_frame(source)
        dataset_cfg = self.config["dataset"]
        inference_cfg = self.config["inference"]
        self.target_col = dataset_cfg["target_col"]
        self.capacity_col = dataset_cfg["capacity_col"]
        inputs = feature_columns(self.config)
        self.input_columns = [
            column
            for column in inputs
            if column != self.capacity_col or supervised or column in frame.columns
        ]
        required = set(self.input_columns)
        if supervised:
            required.update((self.target_col, self.capacity_col))
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")

        history_columns = {dataset_cfg["pv_col"], *dataset_cfg["conv_cols"]}
        future_columns = {dataset_cfg["tsfm_col"], *dataset_cfg["conv_predict_cols"]}
        lengths = {
            **{column: inference_cfg["input_len"] for column in history_columns},
            **{column: inference_cfg["pred_len"] for column in future_columns},
        }
        keep: list[int] = []
        missing_labels = 0
        valid_points = 0
        # Check the complete dataset before starting optimization, so a malformed
        # late batch cannot silently leave a misleading completed training run.
        for position in range(len(frame)):
            row = frame.iloc[position]
            if supervised:
                target = row[self.target_col]
                if _is_missing_scalar(target):
                    missing_labels += 1
                    continue
                target_array = _sequence(
                    target, self.target_col, position, inference_cfg["pred_len"]
                )
                finite_points = int(np.isfinite(target_array).sum())
                if finite_points == 0:
                    missing_labels += 1
                    continue
                valid_points += finite_points
            for column in self.input_columns:
                if column == self.capacity_col:
                    capacity = np.asarray(row[column])
                    if capacity.ndim != 0:
                        raise ValueError(
                            f"Row {position}, {column!r}: capacity must be a scalar."
                        )
                    try:
                        capacity_number = np.float32(capacity)
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise ValueError(
                            f"Row {position}, {column!r}: capacity must be numeric."
                        ) from exc
                    if not np.isfinite(capacity_number) or capacity_number <= 0:
                        raise ValueError(
                            f"Row {position}, {column!r}: capacity must be finite and positive."
                        )
                else:
                    array = _sequence(row[column], column, position, lengths[column])
                    if not np.isfinite(array).all():
                        raise ValueError(
                            f"Row {position}, {column!r}: input values must all be finite."
                        )
            keep.append(position)
        if not keep:
            raise ValueError("Dataset has no rows with at least one finite target value.")
        self.frame = frame.iloc[keep].reset_index(drop=True)
        self.stats = {
            "rows_read": len(frame),
            "rows_used": len(keep),
            "rows_without_labels": missing_labels,
            "valid_target_points": valid_points if supervised else None,
        }

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        features = {
            column: torch.tensor(np.asarray(row[column], dtype=np.float32))
            for column in self.input_columns
        }
        labels = (
            torch.tensor(np.asarray(row[self.target_col], dtype=np.float32))
            if self.supervised
            else None
        )
        return {"features": features, "labels": labels, "row_index": index}


def collate_windows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot collate an empty batch.")
    columns = rows[0]["features"]
    features = {
        column: torch.stack([row["features"][column] for row in rows])
        for column in columns
    }
    label_presence = [row["labels"] is not None for row in rows]
    if any(label_presence) and not all(label_presence):
        raise ValueError("A batch cannot mix supervised and unsupervised rows.")
    labels = torch.stack([row["labels"] for row in rows]) if all(label_presence) else None
    return {
        "features": features,
        "labels": labels,
        "row_indices": torch.tensor([row["row_index"] for row in rows], dtype=torch.long),
    }
