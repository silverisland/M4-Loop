"""Tensor output contract shared by direct inference and supervised calls."""

from dataclasses import dataclass

from torch import Tensor


@dataclass
class M4LoopOutput:
    loss: Tensor | None
    logits: Tensor
    predictions: Tensor
    embeddings: Tensor

    def __getitem__(self, name: str) -> Tensor | None:
        if name not in ("loss", "logits", "predictions", "embeddings"):
            raise KeyError(name)
        return getattr(self, name)
