"""TF-IDF vectoriser factory: the one home of the configuration (DESIGN.md section 5)."""
from sklearn.feature_extraction.text import TfidfVectorizer


def make_vectoriser() -> TfidfVectorizer:
    """Return a new, unfitted TfidfVectorizer configured per DESIGN.md section 5.

    analyzer="char", ngram_range=(1, 4), lowercase=True, sublinear_tf=True,
    min_df=2. Callers fit it inside each training fold only. Empty and
    all-unknown strings transform to an all-zero row without raising.
    """
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 4),
        lowercase=True,
        sublinear_tf=True,
        min_df=2,
    )
