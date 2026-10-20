"""Public API for standalone M4-Loop photovoltaic forecasting."""

from typing import TYPE_CHECKING

from .config import feature_columns, load_config

if TYPE_CHECKING:
    from .models import M4LoopModel, M4LoopOutput

__version__ = "0.2.0"
__all__ = ["M4LoopModel", "M4LoopOutput", "feature_columns", "load_config"]


def __getattr__(name: str):
    # Keep CLI help/config inspection lightweight while preserving the public
    # `from m4_loop import M4LoopModel` API.
    if name in {"M4LoopModel", "M4LoopOutput"}:
        from . import models

        return getattr(models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
