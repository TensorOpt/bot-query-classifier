import pickle
import unittest
from unittest import mock

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression

from bqc.models import TrivialScorer, _hand_features
from bqc.preprocess import normalise

# Longer, punctuated strings are label 1 so the three features carry signal.
TEXTS = ["sofa", "sofa bed", "oak table", "arm chair", "drop table users; --", "1=1 or 2=2 #",
         "select * from x where 1=1", "zz!! qq?? ##"] * 3
Y = [0, 0, 0, 0, 1, 1, 1, 1] * 3


def fitted():
    return TrivialScorer().fit(TEXTS, Y)


class HandFeaturesTest(unittest.TestCase):
    def test_hand_computed(self):
        # Spaces count as non-alphanumeric (docstring rule); inputs are already normalised.
        rows = _hand_features(["", "sofa", "a b", "a-b c", "!!", "1=1 or 2=2"])
        expected = [
            [0, 0, 0.0],          # empty: no division by zero (11.1)
            [4, 1, 0.0],
            [3, 2, 1 / 3],        # one space
            [5, 2, 2 / 5],        # '-' and one space
            [2, 1, 1.0],
            [10, 3, 4 / 10],      # '=', '=', two spaces
        ]
        self.assertEqual(rows.shape, (6, 3))
        np.testing.assert_allclose(rows, expected)

    def test_no_texts_gives_zero_by_three(self):
        self.assertEqual(_hand_features([]).shape, (0, 3))

    def test_unicode_alphanumeric_not_counted(self):
        np.testing.assert_allclose(_hand_features(["é1"]), [[2, 1, 0.0]])


class TrivialScorerTest(unittest.TestCase):
    def test_fit_returns_self(self):
        s = TrivialScorer()
        self.assertIs(s.fit(TEXTS, Y), s)

    def test_scores_in_unit_interval(self):
        s = fitted().score(["sofa", "", "   ", "drop table;;;", "☃", "x" * 2000])
        self.assertEqual(s.shape, (6,))
        self.assertEqual(s.dtype.kind, "f")
        self.assertTrue(((s >= 0) & (s <= 1)).all())

    def test_scores_aligned_with_class_one(self):
        m = fitted()
        self.assertLess(m.score(["sofa"])[0], m.score(["1=1 or 2=2 #!!"])[0])
        flipped = TrivialScorer().fit(TEXTS, [1 - a for a in Y])
        self.assertGreater(flipped.score(["sofa"])[0], flipped.score(["1=1 or 2=2 #!!"])[0])

    def test_features_are_computed_on_normalised_text(self):
        m = fitted()
        self.assertEqual(m.score(["  SOFA   BED "]).tolist(), m.score(["sofa bed"]).tolist())
        # NFKC maps the full-width letters to ASCII, so the length is 4, not 4 wide chars.
        self.assertEqual(m.score(["ＳＯＦＡ"]).tolist(), m.score(["sofa"]).tolist())

    def test_fit_normalises(self):
        raw = [f"  {t.upper()}  " for t in TEXTS]
        a = TrivialScorer().fit(raw, Y).score(["sofa bed", "drop table;"])
        b = fitted().score(["sofa bed", "drop table;"])
        self.assertEqual(a.tolist(), b.tolist())

    def test_normalise_applied_once_per_text(self):
        with mock.patch("bqc.models.normalise", wraps=normalise) as spy:
            m = TrivialScorer().fit(TEXTS, Y)
            self.assertEqual(spy.call_count, len(TEXTS))
            spy.reset_mock()
            m.score(["sofa", "  Drop TABLE "])
            self.assertEqual(spy.call_count, 2)

    def test_empty_string_does_not_raise(self):
        s = fitted().score(["", "   "])
        self.assertEqual(s[0], s[1])

    def test_empty_input(self):
        self.assertEqual(fitted().score([]).shape, (0,))

    def test_score_before_fit_raises(self):
        with self.assertRaises(NotFittedError):
            TrivialScorer().score(["a"])

    def test_empty_score_before_fit_raises(self):
        with self.assertRaises(NotFittedError):
            TrivialScorer().score([])

    def test_configuration(self):
        self.assertEqual(TrivialScorer()._classifier.get_params(),
                         LogisticRegression().get_params())

    def test_pickle_round_trip(self):
        m = fitted()
        m2 = pickle.loads(pickle.dumps(m))
        q = ["sofa", "drop table;", "", "zz!!"]
        self.assertEqual(m.score(q).tolist(), m2.score(q).tolist())

    def test_refit_is_deterministic(self):
        q = ["sofa", "drop table;"]
        self.assertEqual(fitted().score(q).tolist(), fitted().score(q).tolist())


if __name__ == "__main__":
    unittest.main()
