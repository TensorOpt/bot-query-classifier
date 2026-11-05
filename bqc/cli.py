"""Command line entry point. Commands are added by later tasks."""

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bqc")
    parser.add_subparsers(dest="command", required=True, metavar="command")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments; argparse exits with status 2 on a missing or unknown command."""
    build_parser().parse_args(argv)
    return 0
