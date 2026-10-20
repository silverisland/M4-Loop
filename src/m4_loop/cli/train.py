"""Training CLI: translate arguments to the training application API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None, help="YAML overrides for defaults")
    parser.add_argument("--train", type=Path, required=True, help="Pre-split training Parquet")
    parser.add_argument("--valid", type=Path, required=True, help="Pre-split validation Parquet")
    parser.add_argument("--out", "--out-dir", dest="out", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=-1, help="Stop after N optimizer steps")


def run(args: argparse.Namespace) -> None:
    from ..training import train

    result = train(args.config, args.train, args.valid, args.out, args.max_steps)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "history"},
            ensure_ascii=False,
        )
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train M4-Loop on pre-split windows")
    add_arguments(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
