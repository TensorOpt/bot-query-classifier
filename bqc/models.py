"""Model port: the protocol the evaluation depends on (DESIGN.md section 6.4)."""
from collections.abc import Sequence
from typing import Protocol

import numpy as np


class Scorer(Protocol):
    """Wraps vectoriser -> [SMOTE] -> estimator; evaluation code sees only this."""

    def fit(self, texts: Sequence[str], y: Sequence[int]) -> "Scorer":
        """Fit on training texts and labels (1 = illegitimate); return self."""
        ...

    def score(self, texts: Sequence[str]) -> np.ndarray:
        """Return a 1-D float array of P(illegitimate), one entry per text."""
        ...
