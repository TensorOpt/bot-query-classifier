"""Model port: the protocol the evaluation depends on (DESIGN.md section 6.4)."""
from collections.abc import Sequence
from typing import Protocol, Self

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils.validation import check_is_fitted

from bqc import SEED
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


class _VectorisedScorer:
    """normalise -> vectoriser -> classifier, the one home of that step (section 5).

    Calls normalise() once per text in fit and score; callers pass raw text.
    Subclasses set `_classifier`. Module-level classes so fitted instances pickle (M3.T10).
    """

    _classifier: ClassifierMixin

    def __init__(self) -> None:
        self._vectoriser = make_vectoriser()

    def fit(self, texts: Sequence[str], y: Sequence[int]) -> Self:
        X = self._vectoriser.fit_transform([normalise(t) for t in texts])
        self._classifier.fit(X, y)
        return self

    def score(self, texts: Sequence[str]) -> np.ndarray:
        """P(label 1); raises sklearn NotFittedError before fit.

        Probe 2026-11-06 (scikit-learn 1.9.1): classes_ is [0, 1] so column 1 is P(1);
        an all-zero row scores the intercept-only probability.
        """
        if len(texts) == 0:  # sklearn rejects 0 samples (probe 2026-11-06)
            check_is_fitted(self._classifier)
            return np.empty(0)
        X = self._vectoriser.transform([normalise(t) for t in texts])
        return self._classifier.predict_proba(X)[:, 1]


class LogisticRegressionScorer(_VectorisedScorer):
    """TF-IDF + LogisticRegression baseline (DESIGN.md 6.2)."""

    def __init__(self) -> None:
        super().__init__()
        # lbfgs is deterministic, so no random_state (probe 2026-11-06, scikit-learn 1.9.1).
        self._classifier = LogisticRegression(C=1.0, max_iter=1000)


class MLPScorer(_VectorisedScorer):
    """TF-IDF + MLPClassifier, the user's model (DESIGN.md 6.1).

    ConvergenceWarning is deliberately not suppressed: the runner captures it (11.12).
    Probe 2026-11-06 (scikit-learn 1.9.1): sparse input works; early stopping holds out a
    stratified 20% of the rows given to fit, and needs at least 3 + 3 rows (4 or 5 raise
    ValueError); max_iter=1 emits ConvergenceWarning and fit still returns.
    """

    def __init__(self, seed: int = SEED) -> None:
        super().__init__()
        self._classifier = MLPClassifier(
            hidden_layer_sizes=(64,), activation="relu", solver="adam", alpha=1e-3,
            early_stopping=True, validation_fraction=0.2, max_iter=500, random_state=seed)
