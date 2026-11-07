import pickle
import unittest
from unittest import mock

import numpy as np
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from bqc.features import make_vectoriser
from bqc.models import LRHandScorer, _hand_features
from bqc.preprocess import normalise

TEXTS = ["sofa", "sofa bed", "oak table", "arm chair", "drop table users", "1=1 or 2=2",
         "select star from x", "zzqx qzzx"] * 3
Y = [0, 0, 0, 0, 1, 1, 1, 1] * 3


def fitted():
    return LRHandScorer().fit(TEXTS, Y)


def parts(scorer):
    union = scorer._vectoriser
    return union.transformer_list[0][1], union.transformer_list[1][1]


class ConfigurationTest(unittest.TestCase):
    def test_classifier(self):
        self.assertEqual(LRHandScorer()._classifier.get_params(),
                         LogisticRegression(C=1.0, max_iter=1000).get_params())

    def test_tfidf_part(self):
        tfidf, _ = parts(LRHandScorer())
        self.assertEqual(tfidf.get_params(), make_vectoriser().get_params())

    def test_scaler(self):
        _, hand = parts(LRHandScorer())
        self.assertEqual(hand.steps[-1][1].get_params(), StandardScaler().get_params())
        self.assertIs(hand.steps[0][1].func, _hand_features)


class FeatureTest(unittest.TestCase):
    def test_coefficients_include_the_hand_columns(self):
        m = fitted()
        tfidf, _ = parts(m)
        self.assertEqual(m._classifier.coef_.shape, (1, len(tfidf.vocabulary_) + 3))

    def test_transformed_matrix_is_sparse(self):
        m = fitted()
        X = m._vectoriser.transform([normalise(t) for t in TEXTS])
        self.assertTrue(sparse.issparse(X))
        self.assertEqual(X.shape[0], len(TEXTS))

    def test_scaler_statistics_are_the_training_rows(self):
        m = fitted()
        scaler = parts(m)[1].steps[-1][1]
        hand = _hand_features([normalise(t) for t in TEXTS])
        np.testing.assert_allclose(scaler.mean_, hand.mean(axis=0))
        np.testing.assert_allclose(scaler.scale_, hand.std(axis=0))
        mean, scale = scaler.mean_.copy(), scaler.scale_.copy()
        m.score(["a very long query " * 20, "x"])
        self.assertEqual(scaler.mean_.tolist(), mean.tolist())
        self.assertEqual(scaler.scale_.tolist(), scale.tolist())

    def test_score_is_batch_independent(self):
        m = fitted()
        alone = m.score(["drop table"])[0]
        batch = m.score(["sofa", "a long query " * 30, "drop table", ""])
        self.assertEqual(batch[2], alone)

    def test_hand_features_use_normalised_text(self):
        m = fitted()
        a, b = m.score(["sofa ​ bed"]), m.score(["sofa bed"])
        self.assertEqual(a.tolist(), b.tolist())


class ScorerTest(unittest.TestCase):
    def test_fit_returns_self(self):
        s = LRHandScorer()
        self.assertIs(s.fit(TEXTS, Y), s)

    def test_normalise_applied_once_per_text(self):
        with mock.patch("bqc.models.normalise", wraps=normalise) as spy:
            m = LRHandScorer().fit(TEXTS, Y)
            self.assertEqual(spy.call_count, len(TEXTS))
            spy.reset_mock()
            m.score(["sofa", "  Drop TABLE "])
            self.assertEqual(spy.call_count, 2)

    def test_degenerate_inputs_score_finite(self):
        s = fitted().score(["", "☃☃", "zzzzqq", "x" * 10_000, "a b " * 2500])
        self.assertEqual(s.shape, (5,))
        self.assertTrue(np.isfinite(s).all())

    def test_range_and_alignment(self):
        m = fitted()
        s = m.score(["sofa bed", "drop table users", ""])
        self.assertTrue(((s >= 0) & (s <= 1)).all())
        self.assertLess(s[0], 0.5)
        self.assertGreater(s[1], 0.5)

    def test_empty_input(self):
        self.assertEqual(fitted().score([]).shape, (0,))

    def test_score_before_fit_raises(self):
        with self.assertRaises(NotFittedError):
            LRHandScorer().score(["a"])

    def test_empty_score_before_fit_raises(self):
        with self.assertRaises(NotFittedError):
            LRHandScorer().score([])

    def test_pickle_round_trip(self):
        m = fitted()
        m2 = pickle.loads(pickle.dumps(m))
        q = ["sofa", "drop table", "", "zzqx"]
        self.assertEqual(m.score(q).tolist(), m2.score(q).tolist())

    def test_refit_is_deterministic(self):
        q = ["sofa", "drop table"]
        self.assertEqual(fitted().score(q).tolist(), fitted().score(q).tolist())


if __name__ == "__main__":
    unittest.main()
