"""Model port: the protocol the evaluation depends on (DESIGN.md section 6.4)."""
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from bqc.features import make_vectoriser
from bqc.preprocess import normalise


class Scorer(Protocol):
    """Wraps vectoriser -> estimator; evaluation code sees only this."""

    def fit(self, texts: Sequence[str], y: Sequence[int]) -> "Scorer":
        """Fit on training texts and labels (1 = illegitimate); return self."""
        ...

    def score(self, texts: Sequence[str]) -> np.ndarray:
        """Return a 1-D float array of P(illegitimate), one entry per text."""
        ...


class LogisticRegressionScorer:
    """TF-IDF + LogisticRegression baseline (DESIGN.md 6.2).

    Calls normalise() once per text in fit and score (section 5); callers pass raw text.
    Module-level so fitted instances pickle (M3.T10).
    """

    def __init__(self) -> None:
        self._vectoriser = make_vectoriser()
        # lbfgs is deterministic, so no random_state (probe 2026-11-06, scikit-learn 1.9.1).
        self._classifier = LogisticRegression(C=1.0, max_iter=1000)

    def fit(self, texts: Sequence[str], y: Sequence[int]) -> "LogisticRegressionScorer":
        X = self._vectoriser.fit_transform([normalise(t) for t in texts])
        self._classifier.fit(X, y)
        return self

    def score(self, texts: Sequence[str]) -> np.ndarray:
        """P(label 1); raises sklearn NotFittedError before fit.

        Probe 2026-11-06 (scikit-learn 1.9.1): classes_ is [0, 1] so column 1 is P(1);
        an all-zero row scores the intercept-only probability; no convergence warnings.
        """
        if len(texts) == 0:  # sklearn rejects 0 samples (probe 2026-11-06)
            check_is_fitted(self._classifier)
            return np.empty(0)
        X = self._vectoriser.transform([normalise(t) for t in texts])
        return self._classifier.predict_proba(X)[:, 1]
