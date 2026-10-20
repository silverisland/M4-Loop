"""Compatibility entrypoint; new code should use m4_loop.training."""

from .cli.train import main
from .training import evaluate, train

__all__ = ["evaluate", "main", "train"]

if __name__ == "__main__":
    main()
