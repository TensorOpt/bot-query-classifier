import pickle
import unittest
import warnings
from unittest import mock

from sklearn.exceptions import ConvergenceWarning, NotFittedError

from bqc import SEED
from bqc.features import make_vectoriser
from bqc.models import MLPScorer
from bqc.preprocess import normalise

# Early stopping holds out 20% of the rows, stratified: probed 2026-11-06 (scikit-learn
# 1.9.1), the smallest balanced input that fits is 3 + 3 rows (4 and 5 rows raise
# ValueError "test_size = 1 should be greater or equal to the number of classes").
# The fixture is 24 rows so min_df=2 keeps features.
TEXTS = ["sofa", "sofa bed", "oak table", "arm chair", "drop table users", "1=1 or 2=2",
         "select star from x", "zzqx qzzx"] * 3
Y = [0, 0, 0, 0, 1, 1, 1, 1] * 3
Q = ["sofa", "drop table", "", "zzqx"]


def fitted(seed=SEED):
    return MLPScorer(seed).fit(TEXTS, Y)


class MLPScorerTest(unittest.TestCase):
    def test_fit_returns_self(self):
        s = MLPScorer()
        self.assertIs(s.fit(TEXTS, Y), s)

    def test_same_seed_identical_scores(self):
        self.assertEqual(fitted().score(Q).tolist(), fitted().score(Q).tolist())

    def test_seed_is_passed_through(self):
        self.assertEqual(MLPScorer()._classifier.random_state, SEED)
        self.assertEqual(MLPScorer(7)._classifier.random_state, 7)

    def test_range_and_dtype(self):
        s = fitted().score(["sofa", "", "drop table", "☃"])
        self.assertEqual(s.shape, (4,))
        self.assertEqual(s.dtype.kind, "f")
        self.assertTrue(((s >= 0) & (s <= 1)).all())

    def test_scores_aligned_with_class_one(self):
        flipped = MLPScorer().fit(TEXTS, [1 - a for a in Y]).score(["sofa bed"])
        self.assertGreater(flipped[0], 0.5)
        self.assertLess(fitted().score(["sofa bed"])[0], 0.5)

    def test_empty_string_does_not_raise(self):
        s = fitted().score(["", "   "])
        self.assertEqual(s[0], s[1])

    def test_empty_input(self):
        self.assertEqual(fitted().score([]).shape, (0,))

    def test_score_normalises(self):
        m = fitted()
        self.assertEqual(m.score(["  SOFA "]).tolist(), m.score(["sofa"]).tolist())

    def test_fit_normalises(self):
        raw = [f"  {t.upper()} " for t in TEXTS]
        a = MLPScorer().fit(raw, Y).score(["sofa bed", "drop table"])
        b = fitted().score(["sofa bed", "drop table"])
        self.assertEqual(a.tolist(), b.tolist())

    def test_normalise_applied_once_per_text(self):
        with mock.patch("bqc.models.normalise", wraps=normalise) as spy:
            m = MLPScorer().fit(TEXTS, Y)
            self.assertEqual(spy.call_count, len(TEXTS))
            spy.reset_mock()
            m.score(["sofa", "  Drop TABLE "])
            self.assertEqual(spy.call_count, 2)

    def test_score_before_fit_raises(self):
        with self.assertRaises(NotFittedError):
            MLPScorer().score(["a"])

    def test_empty_score_before_fit_raises(self):
        with self.assertRaises(NotFittedError):
            MLPScorer().score([])

    def test_configuration(self):
        p = MLPScorer()._classifier.get_params()
        expected = dict(hidden_layer_sizes=(64,), activation="relu", solver="adam", alpha=1e-3,
                        early_stopping=True, validation_fraction=0.2, max_iter=500,
                        random_state=SEED)
        self.assertEqual({k: p[k] for k in expected}, expected)

    def test_uses_m3_vectoriser(self):
        v = MLPScorer()._vectoriser
        self.assertEqual(v.get_params(), make_vectoriser().get_params())

    def test_convergence_warning_not_suppressed(self):
        m = MLPScorer()
        m._classifier.set_params(max_iter=1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertIs(m.fit(TEXTS, Y), m)
        self.assertTrue(any(issubclass(w.category, ConvergenceWarning) for w in caught))
        self.assertEqual(m.score(Q).shape, (4,))

    def test_default_fit_does_not_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted()
        self.assertEqual([w.category for w in caught], [])

    def test_pickle_round_trip(self):
        m = fitted()
        m2 = pickle.loads(pickle.dumps(m))
        self.assertEqual(m.score(Q).tolist(), m2.score(Q).tolist())


if __name__ == "__main__":
    unittest.main()
