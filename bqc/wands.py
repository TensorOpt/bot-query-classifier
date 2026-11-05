"""WANDS query loader with a pinned hash check (DESIGN.md sections 3, 10, 11.14, 11.15)."""

import csv
import hashlib
import http.client
import io
import os
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

URL = "https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/query.csv"
PRODUCT_URL = "https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/product.csv"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_PATH = DATA_DIR / "raw" / "query.csv"
PRODUCT_PATH = DATA_DIR / "raw" / "product.csv"
CHECKSUMS_PATH = DATA_DIR / "CHECKSUMS"
TIMEOUT_SECONDS = 60
HEADER = ["query_id", "query", "query_class"]
PRODUCT_HEADER = [
    "product_id",
    "product_name",
    "product_class",
    "category hierarchy",
    "product_description",
    "product_features",
    "rating_count",
    "average_rating",
    "review_count",
]


class WandsQuery(NamedTuple):
    query_id: int
    query: str


class WandsProduct(NamedTuple):
    product_id: int
    name: str
    classes: tuple[str, ...]


class WandsError(Exception):
    """The WANDS file or its checksum is missing, wrong or malformed."""


def fetch(
    target: Path,
    checksums: Path,
    url: str = URL,
    opener: Callable = urllib.request.urlopen,
) -> None:
    """Download `url` to `target`, atomically, after checking the pinned hash.

    The bytes go to `<target>.tmp` and are renamed onto `target` only if their SHA-256
    equals the `checksums` line for `target.name`. On any failure raises WandsError and
    leaves no temp file; an existing `target` is left untouched.
    """
    expected = _expected_hash(target, checksums)
    tmp = target.with_name(target.name + ".tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with opener(url, timeout=TIMEOUT_SECONDS) as response:
            data = response.read()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise WandsError(
                f"hash mismatch for {url}: expected {expected}, got {actual}"
            )
        tmp.write_bytes(data)
        os.replace(tmp, target)
    except (OSError, http.client.HTTPException) as e:  # URLError is an OSError
        raise WandsError(f"cannot fetch {url} to {target}: {e}") from e
    finally:
        tmp.unlink(missing_ok=True)


def load_queries(path: Path, checksums: Path) -> list[WandsQuery]:
    """Return the queries of `path` in file order, unmodified.

    `checksums` holds `sha256sum`-style lines (`<hex>  <file name>`); the line for
    `path.name` must match the SHA-256 of the file's bytes. The check runs before any
    parsing. Raises WandsError, naming the path, on a missing file, a missing or
    mismatching checksum, or a header other than `query_id, query, query_class`.
    `query_class` is ignored and may be empty.
    """
    data = _read_verified(path, checksums)
    reader = _tsv_reader(data)
    if next(reader, None) != HEADER:
        raise WandsError(f"unexpected header in {path}: want {HEADER}")
    try:
        return [WandsQuery(int(row[0]), row[1]) for row in reader]
    except (ValueError, IndexError) as e:
        raise WandsError(f"malformed row in {path}: {e}") from e


def load_products(path: Path, checksums: Path) -> list[WandsProduct]:
    """Return (product_id, name, classes) of every row of `path`, raw, in file order.

    `classes` is `product_class` split on `|`, empty for an empty field. Other columns
    are matched by name and dropped. The hash check (as in `load_queries`) runs before
    parsing, and `csv.field_size_limit` is raised first (descriptions are long). Raises
    WandsError, naming the path, on a missing file, a hash problem or a header lacking
    any of the nine real column names; a row with the wrong column count names its
    row number.
    """
    data = _read_verified(path, checksums)
    csv.field_size_limit(sys.maxsize)  # a description can exceed the 131072 default
    reader = _tsv_reader(data)
    header = next(reader, None) or []
    if not all(name in header for name in PRODUCT_HEADER):
        raise WandsError(f"unexpected header in {path}: want {PRODUCT_HEADER}")
    id_i, name_i, class_i = (header.index(n) for n in PRODUCT_HEADER[:3])
    products = []
    for number, row in enumerate(reader, start=1):
        if len(row) != len(header):
            raise WandsError(
                f"row {number} of {path} has {len(row)} columns, want {len(header)}"
            )
        try:
            product_id = int(row[id_i])
        except ValueError as e:
            raise WandsError(f"malformed row {number} in {path}: {e}") from e
        classes = tuple(row[class_i].split("|")) if row[class_i] else ()
        products.append(WandsProduct(product_id, row[name_i], classes))
    return products


def _read_verified(path: Path, checksums: Path) -> bytes:
    """Read `path` and check its SHA-256 against its `checksums` line."""
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
    return data


def _tsv_reader(data: bytes):
    return csv.reader(io.StringIO(data.decode("utf-8"), newline=""), delimiter="\t")


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
