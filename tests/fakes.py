"""Deterministic fake Scorer for tests that need a model without sklearn."""
from collections.abc import Sequence

import numpy as np


class FakeScorer:
    """Scores a text as the fraction of its characters that are neither letters nor spaces.

    The rule is fixed and ignores fitting. fit() records its inputs in
    fit_texts and fit_y (None until fitted) so tests can assert leakage properties.
    The empty string scores 0.0.
    """

    def __init__(self) -> None:
        self.fit_texts: list[str] | None = None
        self.fit_y: list[int] | None = None

    def fit(self, texts: Sequence[str], y: Sequence[int]) -> "FakeScorer":
        if len(texts) != len(y):
            raise ValueError(f"texts and y differ in length: {len(texts)} != {len(y)}")
        self.fit_texts = list(texts)
        self.fit_y = list(y)
        return self

    def score(self, texts: Sequence[str]) -> np.ndarray:
        return np.array(
            [sum(not (c.isalpha() or c.isspace()) for c in t) / len(t) if t else 0.0 for t in texts],
            dtype=float,
        )
