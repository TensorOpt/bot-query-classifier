"""Generation audit (DESIGN.md 4 rules 2 and 4, 8.6.1).

Three independent pieces: an exact-match check that runs on every generation, a
character-3-gram Jaccard report, and per-family length statistics.
"""
import statistics
from collections.abc import Iterable, Sequence
from typing import NamedTuple

from bqc.preprocess import normalise
from bqc.synth.registry import Sample

WANDS_KEY = "WANDS"
NEGATIVES_THRESHOLD = 0.6
SANITY_THRESHOLD = 0.5


class AuditError(ValueError):
    """A positive equals a negative or a sanity query after normalise()."""


class NearDuplicate(NamedTuple):
    """A pair above the threshold: indices into the two sequences given, and the score."""
    positive_index: int
    other_index: int
    jaccard: float


class LengthStats(NamedTuple):
    """Length in characters of the raw text; all zero when there are no strings."""
    count: int
    min: int
    max: int
    mean: float
    median: float


def check_exact(
    positives: Iterable[str],
    negatives: Iterable[str],
    sanity: Iterable[str],
) -> None:
    """Raise AuditError if a positive equals a negative or a sanity query after normalise().

    The message names the positive, the string it hit and the list hit ("negatives" or
    "sanity"). Negatives are checked before sanity queries. Both lists are required so a
    caller can't switch the gate off by omission; pass `[]` for nothing to compare.
    """
    positives_list = list(positives)
    for name, stored in (("negatives", negatives), ("sanity", sanity)):
        seen = {normalise(s): s for s in stored}
        for positive in positives_list:
            hit = seen.get(normalise(positive))
            if hit is not None:
                raise AuditError(
                    f"positive {positive!r} equals {hit!r} in {name} after normalise()")


def near_duplicates(
    positives: Sequence[str], others: Sequence[str], threshold: float
) -> list[NearDuplicate]:
    """Every (positive, other) pair whose character-3-gram Jaccard is strictly above `threshold`.

    Strings are normalised first. A string shorter than 3 characters has no 3-grams and
    scores 0 against everything, so it is never reported and never raises.
    """
    pos_grams = [_grams(p) for p in positives]
    other_grams = [_grams(o) for o in others]
    found = []
    for i, a in enumerate(pos_grams):
        if not a:
            continue
        for j, b in enumerate(other_grams):
            if not b:
                continue
            score = len(a & b) / len(a | b)
            if score > threshold:
                found.append(NearDuplicate(i, j, score))
    return found


def length_stats(
    samples: Sequence[Sample], wands: Iterable[str]
) -> dict[str, LengthStats]:
    """Length statistics per family present in `samples`, plus one entry under WANDS_KEY."""
    by_family: dict[str, list[int]] = {}
    for s in samples:
        by_family.setdefault(s.family, []).append(len(s.text))
    result = {f: _stats(lengths) for f, lengths in by_family.items()}
    result[WANDS_KEY] = _stats([len(q) for q in wands])
    return result


def _grams(s: str) -> set[str]:
    s = normalise(s)
    return {s[i:i + 3] for i in range(len(s) - 2)}


def _stats(lengths: list[int]) -> LengthStats:
    if not lengths:
        return LengthStats(0, 0, 0, 0.0, 0.0)
    return LengthStats(len(lengths), min(lengths), max(lengths),
                       statistics.fmean(lengths), statistics.median(lengths))
