"""Unified command dispatcher, shared by the console script and python -m."""

from __future__ import annotations

import argparse

from . import predict, train


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="m4-loop", description="M4-Loop power forecasting")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, module, description in (
        ("train", train, "Train and select the best validation checkpoint"),
        ("predict", predict, "Export checkpoint predictions to Parquet"),
    ):
        subparser = subparsers.add_parser(name, help=description, description=description)
        module.add_arguments(subparser)
        subparser.set_defaults(handler=module.run)
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
