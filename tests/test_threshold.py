import math
import unittest

from bqc.evaluate import Threshold, is_illegitimate, select_threshold


class SelectThresholdTest(unittest.TestCase):
    def test_384_scores_at_1_percent_give_third_highest(self):
        scores = [i / 384 for i in range(1, 385)]
        result = select_threshold(scores, 0.01)
        self.assertEqual(result.threshold, sorted(scores)[-3])
        self.assertEqual(result.fpr, 3 / 384)
        self.assertFalse(result.target_unmet)
        self.assertIsNone(result.warning)

    def test_order_of_input_does_not_matter(self):
        scores = [i / 384 for i in range(1, 385)]
        result = select_threshold(list(reversed(scores)), 0.01)
        self.assertEqual(result.threshold, sorted(scores)[-3])

    def test_ties_straddling_target_move_threshold_up(self):
        # 100 scores, target 5%: 5 flagged allowed. Scores 0.90 x3 and 0.80 x4
        # straddle the target: threshold 0.80 flags 7 (7%), so t is 0.90 (3%).
        scores = [0.1] * 93 + [0.80] * 4 + [0.90] * 3
        result = select_threshold(scores, 0.05)
        self.assertEqual(result.threshold, 0.90)
        self.assertEqual(result.fpr, 0.03)
        self.assertFalse(result.target_unmet)

    def test_ties_ending_exactly_at_target_are_included(self):
        # threshold 0.80 flags exactly 5 of 100 = 5% <= target
        scores = [0.1] * 95 + [0.80] * 2 + [0.90] * 3
        result = select_threshold(scores, 0.05)
        self.assertEqual(result.threshold, 0.80)
        self.assertEqual(result.fpr, 0.05)

    def test_threshold_is_one_of_the_scores(self):
        scores = [0.2, 0.4, 0.6, 0.8]
        result = select_threshold(scores, 0.5)
        self.assertIn(result.threshold, scores)
        self.assertEqual(result.threshold, 0.6)
        self.assertEqual(result.fpr, 0.5)

    def test_score_exactly_at_t_is_counted_illegitimate(self):
        scores = [0.2, 0.4, 0.6, 0.8]
        t = select_threshold(scores, 0.5).threshold
        flagged = [s for s in scores if is_illegitimate(s, t)]
        self.assertIn(t, flagged)
        self.assertEqual(len(flagged) / len(scores), 0.5)

    def test_target_unmet_returns_smallest_fpr_score(self):
        # the top score is tied 3 times: every candidate has FPR >= 30% > 10%
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.9, 0.9]
        result = select_threshold(scores, 0.1)
        self.assertEqual(result.threshold, 0.9)
        self.assertEqual(result.fpr, 0.3)
        self.assertTrue(result.target_unmet)

    def test_target_below_resolution_warns_without_rounding(self):
        scores = [i / 100 for i in range(1, 101)]
        result = select_threshold(scores, 0.005)
        self.assertIsNotNone(result.warning)
        # no silent rounding to zero flagged: one flagged is 1%, unmet at 0.5%
        self.assertEqual(result.threshold, 1.0)
        self.assertEqual(result.fpr, 0.01)
        self.assertTrue(result.target_unmet)

    def test_target_equal_to_resolution_does_not_warn(self):
        scores = [i / 100 for i in range(1, 101)]
        result = select_threshold(scores, 0.01)
        self.assertIsNone(result.warning)
        self.assertEqual(result.threshold, 1.0)
        self.assertFalse(result.target_unmet)

    def test_zero_target_warns_and_is_unmet(self):
        result = select_threshold([0.1, 0.2], 0.0)
        self.assertIsNotNone(result.warning)
        self.assertTrue(result.target_unmet)

    def test_all_scores_equal(self):
        result = select_threshold([0.5] * 10, 0.01)
        self.assertEqual(result.threshold, 0.5)
        self.assertEqual(result.fpr, 1.0)
        self.assertTrue(result.target_unmet)

    def test_target_of_one_picks_lowest_score(self):
        result = select_threshold([0.3, 0.1, 0.2], 1.0)
        self.assertEqual(result.threshold, 0.1)
        self.assertEqual(result.fpr, 1.0)
        self.assertFalse(result.target_unmet)

    def test_float_target_matches_exact_fraction(self):
        # 0.29 * 100 is 28.999999999999996: 29 flagged of 100 must still meet 29%
        scores = [i / 100 for i in range(1, 101)]
        result = select_threshold(scores, 0.29)
        self.assertEqual(result.fpr, 0.29)
        self.assertFalse(result.target_unmet)

    def test_no_scores_raises(self):
        with self.assertRaises(ValueError):
            select_threshold([], 0.01)

    def test_non_finite_score_raises(self):
        for bad in (math.nan, math.inf):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                select_threshold([0.1, bad], 0.01)

    def test_returns_threshold_tuple(self):
        self.assertIsInstance(select_threshold([0.1, 0.2], 0.5), Threshold)


class IsIllegitimateTest(unittest.TestCase):
    def test_boundary(self):
        self.assertTrue(is_illegitimate(0.5, 0.5))
        self.assertTrue(is_illegitimate(0.6, 0.5))
        self.assertFalse(is_illegitimate(0.4, 0.5))


if __name__ == "__main__":
    unittest.main()
