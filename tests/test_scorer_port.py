import inspect
import pickle
import unittest

import numpy as np

from bqc.models import Scorer
from tests.fakes import FakeScorer


class FakeScorerTest(unittest.TestCase):
    def test_meets_protocol(self):
        f = FakeScorer()
        for name, params in (("fit", ["texts", "y"]), ("score", ["texts"])):
            self.assertTrue(callable(getattr(f, name)))
            self.assertEqual(list(inspect.signature(getattr(f, name)).parameters), params)
        self.assertEqual(sorted(n for n in vars(Scorer) if n in ("fit", "score")), ["fit", "score"])
        self.assertIs(f.fit(["a"], [0]), f)

    def test_one_score_per_text(self):
        s = FakeScorer().fit(["a", "b"], [0, 1]).score(["abc", "12 ab", "", "x9"])
        self.assertIsInstance(s, np.ndarray)
        self.assertEqual(s.shape, (4,))
        self.assertTrue(((s >= 0) & (s <= 1)).all())

    def test_empty_input(self):
        self.assertEqual(FakeScorer().score([]).shape, (0,))

    def test_fixed_rule(self):
        s = FakeScorer().score(["abcd", "ab12", "1234", "", "a b"])
        self.assertEqual(s.tolist(), [0.0, 0.5, 1.0, 0.0, 0.0])

    def test_deterministic_and_independent_of_fit(self):
        texts = ["ab12", "x"]
        a = FakeScorer().score(texts)
        b = FakeScorer().fit(["1111"], [1]).score(texts)
        self.assertEqual(a.tolist(), b.tolist())

    def test_records_fit_input(self):
        f = FakeScorer()
        self.assertIsNone(f.fit_texts)
        texts, y = ("a", "b1"), (0, 1)
        f.fit(texts, y)
        self.assertEqual(f.fit_texts, ["a", "b1"])
        self.assertEqual(f.fit_y, [0, 1])

    def test_record_is_a_copy(self):
        texts = ["a"]
        f = FakeScorer().fit(texts, [0])
        texts.append("b")
        self.assertEqual(f.fit_texts, ["a"])

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            FakeScorer().fit(["a", "b"], [0])

    def test_picklable_fitted_and_unfitted(self):
        for f in (FakeScorer(), FakeScorer().fit(["a1"], [1])):
            g = pickle.loads(pickle.dumps(f))
            self.assertEqual(g.fit_texts, f.fit_texts)
            self.assertEqual(g.score(["a1"]).tolist(), f.score(["a1"]).tolist())
