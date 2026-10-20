"""Local adapter name for scripts using a Transformer-style model entrypoint."""

from .model import M4LoopModel, M4LoopOutput

TransformersAdaptedM4LoopModel = M4LoopModel

__all__ = ["M4LoopModel", "M4LoopOutput", "TransformersAdaptedM4LoopModel"]
