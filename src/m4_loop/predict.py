"""Compatibility entrypoint; new code should use m4_loop.inference."""

from .cli.predict import main
from .inference import predict

__all__ = ["main", "predict"]

if __name__ == "__main__":
    main()
