"""HDBSCAN label-discovery experiment (DESIGN.md section 9).

Shares only normalise() and the vectoriser factory with the training pipeline.
Labels never reach any step here: every function takes texts only.

Probes (2026-11-06, scikit-learn 1.9.1, numpy 2.5.4):
- HDBSCAN `copy` has no observable effect on dense finite input (the input array is
  unchanged for both values); it is set to True, the default from scikit-learn 1.10,
  to silence the FutureWarning (13.5) and never touch the SVD output.
- TruncatedSVD raises ValueError when n_components exceeds the vectoriser's feature
  count; it does not raise when n_components exceeds the row count (the output simply
  has fewer columns). Neither is handled: the real data (960 rows, ~25k features) is
  far from both, and the ValueError message is clear.
- HDBSCAN raises ValueError when there are fewer rows than min_samples (which is
  min_cluster_size when None) or just one row. With rows == min_cluster_size on
  structureless data it returns all -1, which is the all-noise case of 11.17.
"""
from collections.abc import Sequence

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from bqc import SEED
from bqc.features import make_vectoriser
from bqc.preprocess import normalise

GRID_N_COMPONENTS = (20, 50, 100)
GRID_MIN_CLUSTER_SIZE = (10, 15, 30)


def make_pipeline(n_components: int = 50, min_cluster_size: int = 15) -> Pipeline:
    """Unfitted vectoriser -> TruncatedSVD -> L2 Normalizer -> HDBSCAN (section 9.1)."""
    return Pipeline([
        ("vectoriser", make_vectoriser()),
        ("svd", TruncatedSVD(n_components=n_components, random_state=SEED)),
        ("normalise", Normalizer(norm="l2")),
        ("hdbscan", HDBSCAN(min_cluster_size=min_cluster_size, min_samples=None, copy=True)),
    ])


def cluster_labels(
    texts: Sequence[str], n_components: int = 50, min_cluster_size: int = 15
) -> np.ndarray:
    """HDBSCAN labels (-1 = noise) for raw texts; normalise() once per text, no labels."""
    pipeline = make_pipeline(n_components, min_cluster_size)
    pipeline.fit([normalise(t) for t in texts])
    return pipeline.named_steps["hdbscan"].labels_


def summarise(labels: np.ndarray) -> dict:
    """{"n_clusters": clusters excluding noise, "noise_fraction": share of -1 labels}."""
    labels = np.asarray(labels)
    return {
        "n_clusters": len(set(labels.tolist()) - {-1}),
        "noise_fraction": float((labels == -1).mean()),
    }


def run_grid(texts: Sequence[str]) -> list[dict]:
    """One dict per GRID cell: n_components, min_cluster_size, labels, plus summarise()."""
    cells = []
    for n_components in GRID_N_COMPONENTS:
        for min_cluster_size in GRID_MIN_CLUSTER_SIZE:
            labels = cluster_labels(texts, n_components, min_cluster_size)
            cells.append({"n_components": n_components, "min_cluster_size": min_cluster_size,
                          "labels": labels, **summarise(labels)})
    return cells
