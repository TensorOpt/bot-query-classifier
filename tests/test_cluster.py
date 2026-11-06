import inspect
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

from bqc import SEED, cluster
from bqc.cluster import cluster_labels, make_pipeline, run_grid, summarise
from bqc.features import make_vectoriser
from bqc.preprocess import normalise


def blobs(n_per=30):
    """Two well-separated groups of strings over disjoint alphabets, plus raw-case noise."""
    rng = np.random.default_rng(0)
    a = ["".join(rng.choice(list("abcdef"), 10)) for _ in range(n_per)]
    b = ["".join(rng.choice(list("uvwxyz"), 10)) for _ in range(n_per)]
    return a + b


TEXTS = blobs()


class PipelineTest(unittest.TestCase):
    def test_steps_and_full_params_pinned(self):
        p = make_pipeline()
        self.assertEqual([n for n, _ in p.steps], ["vectoriser", "svd", "normalise", "hdbscan"])
        ref = {
            "vectoriser": make_vectoriser(),
            "svd": TruncatedSVD(n_components=50, random_state=SEED),
            "normalise": Normalizer(norm="l2"),
            "hdbscan": HDBSCAN(min_cluster_size=15, min_samples=None, copy=True),
        }
        for name, est in ref.items():
            self.assertEqual(p.named_steps[name].get_params(), est.get_params(), name)
            self.assertIs(type(p.named_steps[name]), type(est))

    def test_parameters_flow_through(self):
        p = make_pipeline(20, 30)
        self.assertEqual(p.named_steps["svd"].n_components, 20)
        self.assertEqual(p.named_steps["hdbscan"].min_cluster_size, 30)

    def test_defaults(self):
        self.assertEqual(make_pipeline().get_params()["svd__n_components"], 50)
        self.assertEqual(make_pipeline().get_params()["hdbscan__min_cluster_size"], 15)

    def test_rows_are_unit_norm(self):
        p = make_pipeline(5, 5)
        Z = p[:-1].fit_transform([normalise(t) for t in TEXTS])
        self.assertEqual(Z.shape, (len(TEXTS), 5))
        np.testing.assert_allclose(np.linalg.norm(Z, axis=1), 1.0)


class ClusterLabelsTest(unittest.TestCase):
    def test_same_seed_identical_labels(self):
        a = cluster_labels(TEXTS, 5, 5)
        b = cluster_labels(TEXTS, 5, 5)
        self.assertEqual(a.tolist(), b.tolist())
        self.assertEqual(a.shape, (len(TEXTS),))

    def test_finds_the_two_groups(self):
        labels = cluster_labels(TEXTS, 5, 5)
        self.assertGreaterEqual(len(set(labels[:30]) - {-1}), 1)
        self.assertTrue(set(labels[:30]).isdisjoint(set(labels[30:]) - {-1}))

    def test_normalise_once_per_text(self):
        with mock.patch("bqc.cluster.normalise", wraps=normalise) as m:
            cluster_labels(TEXTS, 5, 5)
        self.assertEqual(m.call_count, len(TEXTS))

    def test_raw_text_is_normalised(self):
        raw = [f"  {t.upper()} " for t in TEXTS]
        self.assertEqual(cluster_labels(raw, 5, 5).tolist(), cluster_labels(TEXTS, 5, 5).tolist())

    def test_all_noise_without_raising(self):
        # rows == min_cluster_size on structureless data: HDBSCAN marks every row noise
        rng = np.random.default_rng(1)
        texts = ["".join(rng.choice(list("abcdefgh"), 12)) for _ in range(15)]
        labels = cluster_labels(texts, 2, 15)
        self.assertEqual(set(labels.tolist()), {-1})
        self.assertEqual(summarise(labels), {"n_clusters": 0, "noise_fraction": 1.0})

    def test_fewer_rows_than_min_cluster_size_raises_valueerror(self):
        # documented in the module docstring (probe 2026-11-06), not handled
        with self.assertRaises(ValueError):
            cluster_labels(TEXTS[:10], 2, 15)

    def test_n_components_above_feature_count_raises_valueerror(self):
        with self.assertRaises(ValueError):
            cluster_labels(["ab", "ba", "ab", "ba"] * 3, 50, 3)

    def test_defaults_match_pipeline(self):
        params = inspect.signature(cluster_labels).parameters
        self.assertEqual(params["n_components"].default, 50)
        self.assertEqual(params["min_cluster_size"].default, 15)

    def test_signatures_take_texts_only(self):
        # exact names: a `labels=None` parameter anywhere would let labels reach a step
        for fn, names in [(cluster_labels, ["texts", "n_components", "min_cluster_size"]),
                          (make_pipeline, ["n_components", "min_cluster_size"]),
                          (run_grid, ["texts"])]:
            self.assertEqual(list(inspect.signature(fn).parameters), names, fn.__name__)

    def test_extra_positional_labels_rejected(self):
        with self.assertRaises(TypeError):
            cluster_labels(TEXTS, 5, 5, [0] * len(TEXTS))

    def test_every_step_fitted_on_all_rows(self):
        # the halves use different alphabets, so a half-fit vectoriser or SVD differs
        captured = []

        def capture(*args, **kwargs):
            captured.append(make_pipeline(*args, **kwargs))
            return captured[-1]

        with mock.patch("bqc.cluster.make_pipeline", side_effect=capture):
            labels = cluster_labels(TEXTS, 5, 5)
        ref = make_pipeline(5, 5).fit([normalise(t) for t in TEXTS])
        got = captured[0]
        self.assertEqual(got.named_steps["vectoriser"].vocabulary_,
                         ref.named_steps["vectoriser"].vocabulary_)
        np.testing.assert_allclose(got.named_steps["svd"].components_,
                                   ref.named_steps["svd"].components_)
        self.assertEqual(len(got.named_steps["hdbscan"].labels_), len(TEXTS))
        self.assertEqual(labels.tolist(), got.named_steps["hdbscan"].labels_.tolist())


class SummariseTest(unittest.TestCase):
    def test_counts(self):
        s = summarise(np.array([-1, 0, 0, 1, -1, 1, 1, 0]))
        self.assertEqual(s, {"n_clusters": 2, "noise_fraction": 0.25})

    def test_no_noise(self):
        self.assertEqual(summarise(np.array([0, 1, 1])), {"n_clusters": 2, "noise_fraction": 0.0})


class GridTest(unittest.TestCase):
    def test_grid_constants(self):
        self.assertEqual(cluster.GRID_N_COMPONENTS, (20, 50, 100))
        self.assertEqual(cluster.GRID_MIN_CLUSTER_SIZE, (10, 15, 30))

    def test_every_cell_reported_once(self):
        calls = []

        def fake(texts, n_components, min_cluster_size):
            calls.append((n_components, min_cluster_size))
            return np.array([-1, 0, 0, 0])

        with mock.patch("bqc.cluster.cluster_labels", side_effect=fake):
            cells = run_grid(["a", "b", "c", "d"])
        expected = [(n, m) for n in (20, 50, 100) for m in (10, 15, 30)]
        self.assertEqual(calls, expected)
        self.assertEqual([(c["n_components"], c["min_cluster_size"]) for c in cells], expected)
        for c in cells:
            self.assertEqual(c["labels"].tolist(), [-1, 0, 0, 0])
            self.assertEqual(c["n_clusters"], 1)
            self.assertEqual(c["noise_fraction"], 0.25)

    def test_real_grid_on_small_data(self):
        texts = blobs(150)
        cells = run_grid(texts)
        self.assertEqual(len(cells), 9)
        for c in cells:
            self.assertEqual(len(c["labels"]), len(texts))


class IsolationTest(unittest.TestCase):
    def test_does_not_import_evaluate_or_models(self):
        code = ("import sys, bqc.cluster; "
                "print('bqc.evaluate' in sys.modules, 'bqc.models' in sys.modules)")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             check=True)
        self.assertEqual(out.stdout.split(), ["False", "False"])


if __name__ == "__main__":
    unittest.main()
