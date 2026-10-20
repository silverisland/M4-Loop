"""Prediction CLI: translate arguments to the inference application API."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="Input windows in Parquet")
    parser.add_argument("--out", "--output", dest="out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda[:N], or mps")


def run(args: argparse.Namespace) -> None:
    from ..inference import predict

    frame = predict(args.checkpoint, args.data, args.out, args.batch_size, args.device)
    print(f"Saved {len(frame)} prediction rows to {args.out.resolve()}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predict with an M4-Loop checkpoint")
    add_arguments(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
