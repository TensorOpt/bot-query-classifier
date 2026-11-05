"""Training negatives: product-name spans (DESIGN.md 3.2, 8.3, 11.25 to 11.27).

The builder receives every list as an argument. It never reads the sanity file and
opens no file.
"""
import random
import unicodedata
from collections.abc import Iterable, Sequence
from typing import NamedTuple

from bqc.preprocess import normalise
from bqc.synth.audit import drop_exact
from bqc.wands import WandsProduct

# Word-count distribution of the 480 real queries (DESIGN.md 3.2, 13.1): k -> count.
LENGTH_COUNTS = {1: 23, 2: 109, 3: 169, 4: 99, 5: 41, 6: 16, 7: 11, 8: 8, 9: 3, 10: 1}
DEFAULT_N = 5000
# Cap on the product-side vocabulary list, as in the probe (DESIGN.md 8.3, 13.19).
# Callers record it in the run config.
VOCABULARY_CAP = 10_000
BATCH_SIZE = 100
MAX_STALLED_BATCHES = 20

REMOVAL_KINDS = ("empty", "drawn", "real_queries", "part_numbers", "sanity")


class Span(NamedTuple):
    text: str
    product_id: int
    k: int
    seed: int


class PoolExhausted(Exception):
    """`n` spans can't be reached; `reached` is the number built (never padded)."""

    def __init__(self, message: str, reached: int):
        super().__init__(message)
        self.reached = reached


def fold_ascii(name: str) -> str:
    """normalise(), then NFKD, drop non-ASCII, re-collapse whitespace (3.2 step 1)."""
    decomposed = unicodedata.normalize("NFKD", normalise(name))
    return " ".join(decomposed.encode("ascii", "ignore").decode("ascii").split())


def build_spans(
    products: Sequence[WandsProduct],
    n: int,
    seed: int,
    real_queries: Iterable[str],
    part_numbers: Iterable[str],
    sanity: Iterable[str],
) -> tuple[list[Span], dict[str, int]]:
    """Return `n` product-name spans and a count per kind of removal (3.2).

    Counts have the keys in REMOVAL_KINDS and carry no removed text. Raises
    PoolExhausted, with the count reached, if `n` can't be reached or a drawn `k` has
    no product with that many words.

    Candidates are drawn in batches. A batch is filtered by `drop_exact` against the
    spans kept so far, the real queries, the part-number strings and the sanity
    queries; duplicates inside a batch are counted as "drawn" too. A pool that stops
    yielding new spans (MAX_STALLED_BATCHES batches in a row) counts as exhausted.
    """
    rng = random.Random(seed)
    names = [(p.product_id, fold_ascii(p.name).split()) for p in products]
    names = [(pid, words) for pid, words in names if words]
    ks = sorted(LENGTH_COUNTS)
    weights = [LENGTH_COUNTS[k] for k in ks]
    eligible: dict[int, list[tuple[int, list[str]]]] = {}
    other_lists = {
        "real_queries": list(real_queries),
        "part_numbers": list(part_numbers),
        "sanity": list(sanity),
    }
    kept: list[Span] = []
    counts = {kind: 0 for kind in REMOVAL_KINDS}
    stalled = 0
    while len(kept) < n and stalled < MAX_STALLED_BATCHES:
        candidates: dict[str, Span] = {}  # first occurrence of each text in the batch
        texts = []
        for _ in range(max(BATCH_SIZE, n - len(kept))):
            k = rng.choices(ks, weights)[0]
            if k not in eligible:
                eligible[k] = [(pid, w) for pid, w in names if len(w) >= k]
            if not eligible[k]:
                raise PoolExhausted(
                    f"no product has {k} words; reached {len(kept)} of {n} spans",
                    len(kept),
                )
            pid, words = rng.choice(eligible[k])
            start = rng.randrange(len(words) - k + 1)
            text = normalise(" ".join(words[start:start + k]))
            if not text:
                counts["empty"] += 1
                continue
            texts.append(text)
            candidates.setdefault(text, Span(text, pid, k, seed))
        lists = {"drawn": [s.text for s in kept], **other_lists}
        survivors, removed = drop_exact(texts, lists)
        for kind, count in removed.items():
            counts[kind] += count
        before = len(kept)
        batch_seen: set[str] = set()
        for text in survivors:
            if text in batch_seen:
                counts["drawn"] += 1
            else:
                batch_seen.add(text)
                if len(kept) < n:
                    kept.append(candidates[text])
        stalled = stalled + 1 if len(kept) == before else 0
    if len(kept) < n:
        raise PoolExhausted(
            f"reached {len(kept)} of {n} spans; the pool has no more distinct spans",
            len(kept),
        )
    return kept, counts


def token_occurrences(strings: Iterable[str], cap: int | None = None) -> list[str]:
    """Every whitespace token of the normalised strings, repeats kept, in order.

    With `cap`, only the first `cap` tokens are returned.
    """
    tokens: list[str] = []
    for string in strings:
        tokens.extend(normalise(string).split())
        if cap is not None and len(tokens) >= cap:
            return tokens[:cap]
    return tokens
