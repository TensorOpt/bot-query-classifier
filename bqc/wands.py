"""WANDS query loader with a pinned hash check (DESIGN.md sections 3, 10, 11.14, 11.15)."""

import csv
import hashlib
import io
from pathlib import Path
from typing import NamedTuple

HEADER = ["query_id", "query", "query_class"]


class WandsQuery(NamedTuple):
    query_id: int
    query: str


class WandsError(Exception):
    """The WANDS file or its checksum is missing, wrong or malformed."""


def load_queries(path: Path, checksums: Path) -> list[WandsQuery]:
    """Return the queries of `path` in file order, unmodified.

    `checksums` holds `sha256sum`-style lines (`<hex>  <file name>`); the line for
    `path.name` must match the SHA-256 of the file's bytes. The check runs before any
    parsing. Raises WandsError, naming the path, on a missing file, a missing or
    mismatching checksum, or a header other than `query_id, query, query_class`.
    `query_class` is ignored and may be empty.
    """
    try:
        data = path.read_bytes()
    except OSError as e:
        raise WandsError(f"cannot read WANDS file {path}: {e}") from e
    expected = _expected_hash(path, checksums)
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise WandsError(
            f"hash mismatch for {path}: expected {expected}, got {actual}"
        )
    reader = csv.reader(io.StringIO(data.decode("utf-8"), newline=""), delimiter="\t")
    if next(reader, None) != HEADER:
        raise WandsError(f"unexpected header in {path}: want {HEADER}")
    try:
        return [WandsQuery(int(row[0]), row[1]) for row in reader]
    except (ValueError, IndexError) as e:
        raise WandsError(f"malformed row in {path}: {e}") from e


def _expected_hash(path: Path, checksums: Path) -> str:
    try:
        lines = checksums.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        raise WandsError(f"cannot read checksums file {checksums}: {e}") from e
    for line in lines:
        parts = line.split()
        if len(parts) == 2 and parts[1] == path.name:
            return parts[0]
    raise WandsError(f"no checksum for {path} in {checksums}")
