"""Validated forecasting windows and PyTorch batch collation."""

from .windows import WindowDataset, collate_windows

__all__ = ["WindowDataset", "collate_windows"]
