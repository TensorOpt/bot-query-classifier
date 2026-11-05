"""Sanity set loader (DESIGN.md sections 8.6.1, 11.20, 11.21)."""

import csv
import io
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

from bqc.preprocess import normalise
from bqc.wands import WandsQuery

SANITY_PATH = Path(__file__).resolve().parent.parent / "data" / "sanity.csv"
HEADER = ["id", "query", "label"]


class SanityError(Exception):
    """The sanity file is malformed or overlaps WANDS."""


class SanityQuery(NamedTuple):
    id: int
    query: str


class SanitySet(NamedTuple):
    """Queries are always present, raw, in file order.

    `labels` is the list of int labels (0 or 1) aligned with `queries`, or None when
    the set is not scorable. `invalid` counts rows whose stripped label is not exactly
    `0` or `1`. `message` is the skip message when not scorable, else None.
    """

    queries: list[SanityQuery]
    labels: list[int] | None
    invalid: int
    message: str | None


def load_sanity(path: Path, wands: Iterable[WandsQuery]) -> SanitySet:
    """Read `path` (read-only) and return a SanitySet.

    Raises SanityError naming `path` on an unreadable file, a header other than
    `id,query,label` or a malformed row; naming the `id` when a query equals a WANDS
    query after normalise(), whatever the labels' state.
    """
    try:
        text = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise SanityError(f"cannot read sanity file {path}: {e}") from e
    reader = csv.reader(io.StringIO(text, newline=""))
    if next(reader, None) != HEADER:
        raise SanityError(f"unexpected header in {path}: want {HEADER}")
    queries: list[SanityQuery] = []
    raw_labels: list[str] = []
    try:
        for row in reader:
            if len(row) != 3:
                raise ValueError(f"expected 3 columns, got {len(row)}: {row}")
            queries.append(SanityQuery(int(row[0]), row[1]))
            raw_labels.append(row[2].strip())
    except ValueError as e:
        raise SanityError(f"malformed row in {path}: {e}") from e

    wands_set = {normalise(w.query) for w in wands}
    for q in queries:
        if normalise(q.query) in wands_set:
            raise SanityError(
                f"sanity query id {q.id} in {path} equals a WANDS query: {q.query!r}"
            )

    invalid = sum(label not in ("0", "1") for label in raw_labels)
    if invalid:
        message = f"sanity set: {invalid} of {len(queries)} rows unlabelled, skipping"
        return SanitySet(queries, None, invalid, message)
    return SanitySet(queries, [int(label) for label in raw_labels], 0, None)
