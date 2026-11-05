"""Command line entry point. Commands are added by later tasks."""

import argparse
import sys
from collections.abc import Sequence

from bqc.wands import CHECKSUMS_PATH, RAW_PATH, WandsError, fetch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bqc")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    commands.add_parser("fetch", help="download WANDS query.csv into data/raw/")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments; argparse exits with status 2 on a missing or unknown command."""
    build_parser().parse_args(argv)  # only `fetch` exists so far
    try:
        fetch(RAW_PATH, CHECKSUMS_PATH)
    except WandsError as e:
        print(f"bqc fetch: {e}", file=sys.stderr)
        return 1
    return 0
