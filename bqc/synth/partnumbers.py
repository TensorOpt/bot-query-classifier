"""Part-number diagnostic strings, family "P", label 0 (DESIGN.md 2, 2.2, 8.6 item 4).

Not a registered positive family: these are legitimate queries, scored by the final
models as a diagnostic and never used in any fit, threshold or positive audit. They
stay outside REGISTRY, the 480, the 600 and the Jaccard report.
"""
import random
import string
from collections.abc import Callable, Iterable, Sequence

from bqc.synth.audit import check_exact
from bqc.synth.registry import Sample

FAMILY = "P"
LABEL = 0
PER_SHAPE = 25
MAX_WORDS = 3
_SUFFIX_CHARS = string.ascii_lowercase + string.digits


def _digits(rng: random.Random) -> str:
    """3-9 digits, often zero-padded as shop part numbers are."""
    n = rng.randint(3, 9)
    return "".join(rng.choice(string.digits) for _ in range(n))


def _prefix(rng: random.Random) -> str:
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(1, 4)))


def _prefix_digits(rng: random.Random, vocabulary: Sequence[str]) -> str:
    """`ab004512`."""
    return _prefix(rng) + _digits(rng)


def _bare_digits(rng: random.Random, vocabulary: Sequence[str]) -> str:
    """`4471902`."""
    return _digits(rng)


def _part_number(rng: random.Random) -> str:
    """`bf-7731x`: prefix, optional separator, digits, a 1-2 character suffix."""
    suffix = "".join(rng.choice(_SUFFIX_CHARS) for _ in range(rng.randint(1, 2)))
    return _prefix(rng) + rng.choice(("", "-", "/")) + _digits(rng) + suffix


def _prefix_digits_suffix(rng: random.Random, vocabulary: Sequence[str]) -> str:
    return _part_number(rng)


def _words_part_number(rng: random.Random, vocabulary: Sequence[str]) -> str:
    """`leather sofa w003412`: 1-3 vocabulary words, then a part number of any shape."""
    words = [rng.choice(vocabulary) for _ in range(rng.randint(1, MAX_WORDS))]
    tail = rng.choice((_prefix_digits, _bare_digits, _prefix_digits_suffix))
    return " ".join(words + [tail(rng, vocabulary)])


_SHAPES: tuple[tuple[str, Callable[[random.Random, Sequence[str]], str]], ...] = (
    ("prefix_digits", _prefix_digits),
    ("digits", _bare_digits),
    ("prefix_digits_suffix", _prefix_digits_suffix),
    ("words_part_number", _words_part_number),
)
SHAPE_IDS = tuple(shape_id for shape_id, _ in _SHAPES)


def generate_part_numbers(
    vocabulary: Sequence[str],
    seed: int,
    negatives: Iterable[str],
    sanity: Iterable[str],
) -> list[Sample]:
    """100 distinct part-number strings, 25 per shape in SHAPE_IDS order.

    `vocabulary` is the occurrence list of all negatives' tokens (8.3); product words
    come only from it, and an empty one raises ValueError. The result goes through
    `check_exact` against `negatives` and `sanity` (both required, `[]` for nothing);
    a collision raises AuditError.
    """
    if not vocabulary:
        raise ValueError("vocabulary is empty")
    samples: list[Sample] = []
    seen: set[str] = set()
    for shape_id, render in _SHAPES:
        # A str seed is hashed with sha512, so it ignores PYTHONHASHSEED.
        rng = random.Random(f"{seed}/{FAMILY}/{shape_id}")
        count = 0
        while count < PER_SHAPE:
            text = render(rng, vocabulary)
            if text in seen:
                continue
            seen.add(text)
            samples.append(Sample(text, FAMILY, shape_id, seed))
            count += 1
    check_exact([s.text for s in samples], negatives, sanity)
    return samples
