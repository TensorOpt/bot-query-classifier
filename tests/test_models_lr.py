import pickle
import unittest
from unittest import mock

import numpy as np
from sklearn.exceptions import NotFittedError

from bqc.features import make_vectoriser
from bqc.models import LogisticRegressionScorer
from bqc.preprocess import normalise

TEXTS = ["sofa", "sofa bed", "oak table", "arm chair", "drop table users", "1=1 or 2=2",
         "select star from x", "zzqx qzzx"] * 3
Y = [0, 0, 0, 0, 1, 1, 1, 1] * 3


def fitted():
    return LogisticRegressionScorer().fit(TEXTS, Y)


class LogisticRegressionScorerTest(unittest.TestCase):
    def test_fit_returns_self(self):
        s = LogisticRegressionScorer()
        self.assertIs(s.fit(TEXTS, Y), s)

    def test_scores_aligned_with_class_one(self):
        s = fitted().score(["sofa bed", "drop table users"])
        self.assertEqual(s.shape, (2,))
        self.assertLess(s[0], 0.5)
        self.assertGreater(s[1], 0.5)
        flipped = LogisticRegressionScorer().fit(TEXTS, [1 - a for a in Y]).score(["sofa bed"])
        self.assertGreater(flipped[0], 0.5)

    def test_range_and_dtype(self):
        s = fitted().score(["sofa", "", "drop table", "☃"])
        self.assertEqual(s.dtype.kind, "f")
        self.assertTrue(((s >= 0) & (s <= 1)).all())

    def test_score_normalises(self):
        m = fitted()
        a, b = m.score(["  SOFA "]), m.score(["sofa"])
        self.assertEqual(a.tolist(), b.tolist())

    def test_fit_normalises(self):
        raw = [f"  {t.upper()} " for t in TEXTS]
        a = LogisticRegressionScorer().fit(raw, Y).score(["sofa bed", "drop table"])
        b = fitted().score(["sofa bed", "drop table"])
        self.assertEqual(a.tolist(), b.tolist())

    def test_normalise_applied_once_per_text(self):
        with mock.patch("bqc.models.normalise", wraps=normalise) as spy:
            m = LogisticRegressionScorer().fit(TEXTS, Y)
            self.assertEqual(spy.call_count, len(TEXTS))
            spy.reset_mock()
            m.score(["sofa", "  Drop TABLE "])
            self.assertEqual(spy.call_count, 2)

    def test_empty_and_unknown_same_score(self):
        s = fitted().score(["", "☃☃", "   "])
        self.assertEqual(s[0], s[1])
        self.assertEqual(s[0], s[2])

    def test_empty_input(self):
        self.assertEqual(fitted().score([]).shape, (0,))

    def test_score_before_fit_raises(self):
        with self.assertRaises(NotFittedError):
            LogisticRegressionScorer().score(["a"])

    def test_empty_score_before_fit_raises(self):
        with self.assertRaises(NotFittedError):
            LogisticRegressionScorer().score([])

    def test_configuration(self):
        c = LogisticRegressionScorer()._classifier
        self.assertEqual((c.C, c.max_iter), (1.0, 1000))

    def test_uses_m3_vectoriser(self):
        v = LogisticRegressionScorer()._vectoriser
        self.assertEqual(v.get_params(), make_vectoriser().get_params())

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
