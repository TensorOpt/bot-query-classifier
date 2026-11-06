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

Probes for the labelling metrics (2026-11-06, scikit-learn 1.9.1):
- adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score and completeness_score
  raise nothing and warn nothing on one cluster, one class, a single row, or empty input.
  One cluster with two classes gives ARI 0, AMI 0, homogeneity 0, completeness 1; two clusters
  over one class gives 0, 0, 1, 0; one cluster over one class gives 1 for all four.
- On empty input all four return 1.0, which would be wrong for the all-noise case, so
  oracle_vote() checks for zero non-noise rows itself and reports None (11.17).
"""
import random
import statistics
from collections.abc import Sequence

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from bqc import SEED
from bqc.features import make_vectoriser
from bqc.preprocess import normalise

GRID_N_COMPONENTS = (20, 50, 100)
GRID_MIN_CLUSTER_SIZE = (10, 15, 30)
SEED_KS = (10, 20, 40)
N_DRAWS = 20


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


def _check(clusters: Sequence[int], truth: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Both inputs as int arrays; ValueError when empty or of different length."""
    clusters, truth = np.asarray(clusters, dtype=int), np.asarray(truth, dtype=int)
    if len(clusters) == 0 or len(clusters) != len(truth):
        raise ValueError("clusters and truth must be non-empty and of equal length")
    return clusters, truth


def oracle_vote(clusters: Sequence[int], truth: Sequence[int]) -> dict:
    """Oracle majority vote (9.2): separability only. `truth` is 1 = illegitimate, 0 = legitimate.

    Each cluster takes its members' majority true label; an exact half-and-half tie is
    legitimate (0). Returns plain Python types:
    n_clusters, noise_fraction, non_noise (row count), ari, ami, homogeneity, completeness
    (computed on the non-noise rows only; all None when non_noise is 0, 11.17) and
    clusters: one {"cluster", "size", "purity", "label"} per cluster in ascending id order,
    purity being the majority count over the cluster size (0.5 for a tie). Noise (-1) is not
    a cluster. Raises ValueError on empty input or arrays of different length.
    """
    clusters, truth = _check(clusters, truth)
    keep = clusters != -1
    result = {"n_clusters": len(set(clusters[keep].tolist())), "non_noise": int(keep.sum()),
              "noise_fraction": float((~keep).mean())}
    metrics = {"ari": adjusted_rand_score, "ami": adjusted_mutual_info_score,
               "homogeneity": homogeneity_score, "completeness": completeness_score}
    for name, metric in metrics.items():
        result[name] = float(metric(truth[keep], clusters[keep])) if keep.any() else None
    result["clusters"] = []
    for c in sorted(set(clusters[keep].tolist())):
        members = truth[clusters == c]
        ones = int(members.sum())
        result["clusters"].append({
            "cluster": c, "size": len(members),
            "purity": max(ones, len(members) - ones) / len(members),
            "label": int(ones * 2 > len(members)),
        })
    return result


def draw_seeds(n_rows: int, k: int, draw_index: int) -> list[int]:
    """Row indices of one seed draw: `random.Random(SEED + draw_index).sample(range(n_rows), k)`.

    Over all rows, noise points included (9.2). random.sample raises ValueError when k > n_rows.
    """
    return random.Random(SEED + draw_index).sample(range(n_rows), k)


def seed_draw(clusters: Sequence[int], truth: Sequence[int], k: int, draw_index: int) -> dict:
    """Seed labelling (9.2) for one draw: {"accuracy", "fpr", "recall"} over all rows.

    A cluster is illegitimate (1) only if strictly more of its seeds are 1 than 0; ties,
    clusters without a seed and every noise point (a seed on one labels nothing) are
    legitimate. fpr is None with no true negatives, recall None with no true positives.
    """
    clusters, truth = _check(clusters, truth)
    votes = {}  # cluster id -> [seeds labelled 0, seeds labelled 1]
    for i in draw_seeds(len(clusters), k, draw_index):
        if clusters[i] != -1:
            votes.setdefault(int(clusters[i]), [0, 0])[int(truth[i])] += 1
    illegitimate = {c for c, (zeros, ones) in votes.items() if ones > zeros}
    pred = np.array([int(c) in illegitimate for c in clusters], dtype=int)
    tp = int(((pred == 1) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    positives = int((truth == 1).sum())
    negatives = len(truth) - positives
    return {
        "accuracy": float((pred == truth).mean()),
        "fpr": fp / negatives if negatives else None,
        "recall": tp / positives if positives else None,
    }


def seed_rule(clusters: Sequence[int], truth: Sequence[int]) -> list[dict]:
    """seed_draw() for k in SEED_KS over draw_index 0..N_DRAWS-1: one dict per k.

    {"k", "n_draws", "accuracy", "fpr", "recall" (means over the draws; None stays None),
    "draws": the per-draw dicts}.
    """
    rows = []
    for k in SEED_KS:
        draws = [seed_draw(clusters, truth, k, i) for i in range(N_DRAWS)]
        row = {"k": k, "n_draws": N_DRAWS}
        for metric in ("accuracy", "fpr", "recall"):
            values = [d[metric] for d in draws]
            # None depends only on the class counts, so it is the same in every draw.
            row[metric] = None if values[0] is None else statistics.fmean(values)
        row["draws"] = draws
        rows.append(row)
    return rows
