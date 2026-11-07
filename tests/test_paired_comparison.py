"""Paired comparisons (DESIGN.md 7, 8.2, 12.1 item 13): hand-computed on built fold entries,
then the two sections in a small real report. All numbers are dyadic so floats are exact."""
import math
import unittest
from unittest import mock

from fakes import FakeScorer
from test_run_report import LengthScorer, build

from bqc.evaluate import COMPARISON, PRIMARY, paired_comparison


def fold(repeat, fold_, recall=0.5, fpr=0.25, auc=0.5):
    return {"repeat": repeat, "fold": fold_, "recall": {"recall": recall},
            "fpr": {"point": fpr}, "roc_auc": auc}


def sides(first, second=None, metric="recall"):
    """Fold lists whose per-fold `first - second` of `metric` are the given numbers
    (second side is a constant 0.25 unless given)."""
    second = second if second is not None else [0.25] * len(first)
    keys = [(r, f) for r in range(3) for f in range(5)][:len(first)]
    a = [fold(r, f, **{metric: x + y}) for (r, f), x, y in zip(keys, first, second)]
    b = [fold(r, f, **{metric: y}) for (r, f), y in zip(keys, second)]
    return a, b


class StatisticsTest(unittest.TestCase):
    def test_differences_mean_sd_se_wins(self):
        a, b = sides([0.5, 0.25, 0.0, 0.25])  # mean 0.25, ss = 0.0625*2 = 0.125
        recall = paired_comparison(a, b)["recall"]
        self.assertEqual(recall["differences"], [0.5, 0.25, 0.0, 0.25])
        self.assertEqual(recall["mean"], 0.25)
        self.assertAlmostEqual(recall["sd"], math.sqrt(0.125 / 3))  # ddof=1, not /4
        self.assertAlmostEqual(recall["se"], math.sqrt(0.125 / 3) / 2)
        self.assertEqual(recall["wins"], 3)  # the 0.0 fold is a tie: no win

    def test_lower_fpr_is_a_win_and_higher_is_not(self):
        a, b = sides([-0.125, -0.0625, 0.125, 0.0], metric="fpr")
        fpr = paired_comparison(a, b)["fpr"]
        self.assertEqual(fpr["differences"], [-0.125, -0.0625, 0.125, 0.0])
        self.assertEqual(fpr["wins"], 2)

    def test_roc_auc_higher_is_a_win(self):
        a, b = sides([0.125, 0.0625, -0.125, 0.0], metric="auc")
        roc = paired_comparison(a, b)["roc_auc"]
        self.assertEqual(roc["differences"], [0.125, 0.0625, -0.125, 0.0])
        self.assertEqual(roc["wins"], 2)  # asymmetric: an inverted direction would give 1
        self.assertEqual(roc["mean"], 0.015625)

    def test_folds_are_aligned_by_repeat_and_fold_not_by_position(self):
        a, b = sides([0.5, 0.25, 0.0, 0.25], [0.0, 0.125, 0.25, 0.5])
        shuffled = [b[2], b[0], b[3], b[1]]
        self.assertEqual(paired_comparison(a, shuffled), paired_comparison(a, b))
        self.assertEqual(paired_comparison(a, shuffled)["recall"]["differences"],
                         [0.5, 0.25, 0.0, 0.25])
        self.assertEqual(paired_comparison(shuffled, a)["recall"]["differences"],
                         [-0.5, -0.25, 0.0, -0.25])

    def test_mismatched_or_duplicate_folds_raise(self):
        a, b = sides([0.5, 0.25, 0.0])
        with self.assertRaises(ValueError):
            paired_comparison(a, b[:2])
        with self.assertRaises(ValueError):
            paired_comparison(a, [b[0], b[1], fold(1, 4)])
        with self.assertRaises(ValueError):
            paired_comparison([a[0], a[0], a[1]], b)

    def test_fewer_than_two_folds_raise(self):
        a, b = sides([0.5])
        with self.assertRaises(ValueError):
            paired_comparison(a, b)
        with self.assertRaises(ValueError):
            paired_comparison([], [])


class VerdictTest(unittest.TestCase):
    def verdict(self, differences, fpr=None, auc=None):
        a, b = sides(differences)
        for key, values in (("fpr", fpr), ("roc_auc", auc)):
            if values is not None:
                for x, y, d in zip(a, b, values):
                    y[key] = {"point": 0.25} if key == "fpr" else 0.5
                    x[key] = {"point": 0.25 + d} if key == "fpr" else 0.5 + d
        return paired_comparison(a, b)

    def test_positive_beyond_noise_is_justified_not_worse(self):
        r = self.verdict([0.5, 0.5, 0.375, 0.625])
        self.assertTrue(r["justified"])
        self.assertFalse(r["worse"])

    def test_negative_beyond_noise_is_worse_not_justified(self):
        r = self.verdict([-0.5, -0.5, -0.375, -0.625])
        self.assertTrue(r["worse"])
        self.assertFalse(r["justified"])

    def test_within_noise_is_neither(self):
        r = self.verdict([0.5, -0.25, 0.125, -0.25])  # mean 0.0312, se 0.18
        self.assertFalse(r["justified"])
        self.assertFalse(r["worse"])

    def test_mean_exactly_at_two_standard_errors_is_neither(self):
        for sign in (1, -1):
            r = self.verdict([sign * x for x in (0.625, 0.125, 0.125, 0.125)])
            recall = r["recall"]
            self.assertEqual(recall["mean"], sign * 0.25)
            self.assertEqual(2 * recall["se"], 0.25)
            self.assertFalse(r["justified"])
            self.assertFalse(r["worse"])

    def test_ddof_zero_would_flip_justified(self):
        # n=2, diffs [0.5, 0.1]: mean 0.3, half-gap h = 0.2. With ddof=1, 2 se = 2h = 0.4 > 0.3:
        # not justified. With ddof=0, 2 se = sqrt(2) h = 0.283 < 0.3: it would be.
        r = self.verdict([0.5, 0.1])
        self.assertAlmostEqual(r["recall"]["sd"], 0.4 / math.sqrt(2))
        self.assertAlmostEqual(r["recall"]["se"], 0.2)
        self.assertFalse(r["justified"])

    def test_fpr_rise_beyond_noise_blocks_justified(self):
        r = self.verdict([0.5, 0.5, 0.375, 0.625], fpr=[0.125, 0.125, 0.125, 0.125 + 1 / 16])
        self.assertFalse(r["fpr_gate"]["holds"])
        self.assertEqual(r["fpr_gate"]["sign"], 1)
        self.assertFalse(r["justified"])
        self.assertFalse(r["worse"])

    def test_fpr_rise_within_noise_does_not_block_and_sign_is_reported(self):
        r = self.verdict([0.5, 0.5, 0.375, 0.625], fpr=[0.5, -0.25, 0.125, -0.25])
        gate = r["fpr_gate"]
        self.assertTrue(gate["holds"])
        self.assertEqual(gate["sign"], 1)
        self.assertEqual(gate["mean"], r["fpr"]["mean"])
        self.assertEqual(gate["sd"], r["fpr"]["sd"])
        self.assertTrue(r["justified"])

    def test_fpr_rise_exactly_at_two_standard_errors_does_not_block(self):
        r = self.verdict([0.5, 0.5, 0.375, 0.625], fpr=[0.625, 0.125, 0.125, 0.125])
        self.assertEqual(r["fpr"]["mean"], 0.25)
        self.assertEqual(2 * r["fpr"]["se"], 0.25)
        self.assertTrue(r["fpr_gate"]["holds"])
        self.assertEqual(r["fpr_gate"]["sign"], 1)
        self.assertTrue(r["justified"])

    def test_fpr_fall_and_zero_signs(self):
        r = self.verdict([0.5, 0.5, 0.375, 0.625], fpr=[-0.125] * 4)
        self.assertTrue(r["fpr_gate"]["holds"])
        self.assertEqual(r["fpr_gate"]["sign"], -1)
        r = self.verdict([0.5, 0.5, 0.375, 0.625], fpr=[0.0] * 4)
        self.assertEqual(r["fpr_gate"]["sign"], 0)

    def test_roc_auc_justified_does_not_override_a_failed_fpr_gate(self):
        r = self.verdict([0.5, 0.5, 0.375, 0.625], fpr=[0.125, 0.125, 0.125, 0.1875],
                         auc=[0.125, 0.125, 0.125, 0.1875])
        self.assertTrue(r["roc_auc"]["justified"])
        self.assertFalse(r["fpr_gate"]["holds"])
        self.assertFalse(r["justified"])

    def test_fpr_gate_does_not_make_worse(self):
        r = self.verdict([-0.5, -0.5, -0.375, -0.625], fpr=[0.125] * 3 + [0.1875])
        self.assertFalse(r["fpr_gate"]["holds"])
        self.assertTrue(r["worse"])

    def test_roc_auc_booleans_are_separate_and_do_not_change_justified(self):
        recall = [0.5, 0.5, 0.375, 0.625]
        up = self.verdict(recall, auc=[0.125, 0.125, 0.125, 0.1875])
        down = self.verdict(recall, auc=[-0.125, -0.125, -0.125, -0.1875])
        flat = self.verdict(recall, auc=[0.25, -0.25, 0.25, -0.25])
        self.assertEqual((up["roc_auc"]["justified"], up["roc_auc"]["worse"]), (True, False))
        self.assertEqual((down["roc_auc"]["justified"], down["roc_auc"]["worse"]), (False, True))
        self.assertEqual((flat["roc_auc"]["justified"], flat["roc_auc"]["worse"]), (False, False))
        for r in (up, down, flat):
            self.assertTrue(r["justified"])
            self.assertFalse(r["worse"])

    def test_sign_convention_is_first_minus_second(self):
        a, b = sides([0.5, 0.5, 0.375, 0.625])
        forward, backward = paired_comparison(a, b), paired_comparison(b, a)
        self.assertTrue(forward["justified"] and not forward["worse"])
        self.assertTrue(backward["worse"] and not backward["justified"])
        self.assertEqual(backward["recall"]["differences"],
                         [-d for d in forward["recall"]["differences"]])
        self.assertEqual(backward["recall"]["wins"], 0)


class HandSignTest(unittest.TestCase):
    """`model_comparison_hand` is `lr_hand` minus `lr`: hand-built fold lists, exact numbers."""

    def test_lr_hand_minus_lr_statistics_and_verdict(self):
        # Recall differences 0.25 x 4: sd 0, so the mean is beyond 2 se; FPR differences
        # -0.125 (lr_hand lower), so the gate holds.
        hand, lr = sides([0.25] * 4)
        hand_fpr, lr_fpr = sides([-0.125] * 4, metric="fpr")
        hand = [{**h, "fpr": f["fpr"]} for h, f in zip(hand, hand_fpr)]
        lr = [{**h, "fpr": f["fpr"]} for h, f in zip(lr, lr_fpr)]
        section = paired_comparison(hand, lr)
        self.assertEqual(section["recall"]["differences"], [0.25] * 4)
        self.assertEqual((section["recall"]["mean"], section["recall"]["sd"],
                          section["recall"]["se"], section["recall"]["wins"]), (0.25, 0.0, 0.0, 4))
        self.assertEqual(section["fpr"]["differences"], [-0.125] * 4)
        self.assertEqual(section["fpr"]["wins"], 4)
        self.assertTrue(section["justified"])
        self.assertFalse(section["worse"])
        swapped = paired_comparison(lr, hand)  # the sign flips: lr minus lr_hand
        self.assertEqual(swapped["recall"]["differences"], [-0.25] * 4)
        self.assertFalse(swapped["justified"])
        self.assertTrue(swapped["worse"])
        self.assertEqual(swapped["fpr"]["differences"], [0.125] * 4)  # lr minus lr_hand
        self.assertFalse(swapped["fpr_gate"]["holds"])  # a rise beyond noise: mean 0.125, se 0


class ReportSectionsTest(unittest.TestCase):
    """The sections in a small real report; models are LengthScorer variants."""

    class Low(LengthScorer):
        """Scores by length alone, so its folds differ from LengthScorer's."""
        def score(self, texts):
            return FakeScorer().score(texts) * 0 + [(len(t) % 7) / 7 for t in texts]

    class Mid(LengthScorer):
        """A third scorer, so its folds differ from both LengthScorer's and Low's."""
        def score(self, texts):
            return FakeScorer().score(texts) * 0 + [(len(t) % 5) / 5 for t in texts]

    BOTH = {"lr": LengthScorer, "mlp": Low}

    @staticmethod
    def keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield str(k)
                yield from ReportSectionsTest.keys(v)
        elif isinstance(node, list):
            for v in node:
                yield from ReportSectionsTest.keys(v)

    def test_arm_comparison_is_primary_minus_comparison_per_model(self):
        _, body = build(arms=[PRIMARY, COMPARISON], models=self.BOTH)
        self.assertEqual(set(body["arm_comparison"]), {"lr", "mlp"})
        for model, section in body["arm_comparison"].items():
            first, second = (body[arm]["models"][model]["folds"] for arm in (PRIMARY, COMPARISON))
            self.assertEqual(section, paired_comparison(first, second))
            self.assertEqual(section["recall"]["differences"], [
                a["recall"]["recall"] - b["recall"]["recall"] for a, b in zip(first, second)])
            self.assertEqual(section["fpr"]["differences"], [
                a["fpr"]["point"] - b["fpr"]["point"] for a, b in zip(first, second)])
            self.assertEqual(section["folds"], 6)  # 3 folds x 2 repeats

    def test_model_comparison_is_mlp_minus_lr_under_each_arm(self):
        _, body = build(arms=[PRIMARY, COMPARISON], models=self.BOTH)
        for arm in (PRIMARY, COMPARISON):
            mlp, lr = (body[arm]["models"][m]["folds"] for m in ("mlp", "lr"))
            self.assertEqual(body[arm]["model_comparison"], paired_comparison(mlp, lr))
            self.assertEqual(body[arm]["model_comparison"]["recall"]["differences"], [
                a["recall"]["recall"] - b["recall"]["recall"] for a, b in zip(mlp, lr)])

    def test_the_first_named_side_is_the_primary_arm_and_the_mlp(self):
        calls = []
        real = paired_comparison

        def spy(first, second):
            calls.append((first, second))
            return real(first, second)

        with mock.patch("bqc.evaluate.paired_comparison", spy):
            _, body = build(arms=[PRIMARY, COMPARISON], models=self.BOTH)
        folds = {(a, m): body[a]["models"][m]["folds"] for a in (PRIMARY, COMPARISON)
                 for m in self.BOTH}
        expected = [(folds[PRIMARY, m], folds[COMPARISON, m]) for m in self.BOTH]
        expected += [(folds[a, "mlp"], folds[a, "lr"]) for a in (PRIMARY, COMPARISON)]
        self.assertEqual(len(calls), 4)
        for pair in expected:
            self.assertTrue(any(f is pair[0] and s is pair[1] for f, s in calls), pair)

    def test_one_arm_has_no_arm_comparison(self):
        for arm in (PRIMARY, COMPARISON):
            _, body = build(arms=[arm], models=self.BOTH)
            self.assertNotIn("arm_comparison", body)
            self.assertIn("model_comparison", body[arm])

    def test_no_model_comparison_unless_both_models_ran(self):
        for models in ({"lr": LengthScorer}, {"mlp": LengthScorer},
                       {"lr": LengthScorer, "other": LengthScorer}):
            _, body = build(arms=[PRIMARY, COMPARISON], models=models)
            for arm in (PRIMARY, COMPARISON):
                self.assertNotIn("model_comparison", body[arm])
            self.assertEqual(set(body["arm_comparison"]), set(models))

    def test_model_comparison_hand_is_lr_hand_minus_lr_under_each_arm(self):
        models = {"lr": LengthScorer, "lr_hand": self.Mid, "mlp": self.Low}
        _, body = build(arms=[PRIMARY, COMPARISON], models=models)
        for arm in (PRIMARY, COMPARISON):
            self.assertNotEqual(body[arm]["model_comparison_hand"], body[arm]["model_comparison"])
            hand, lr = (body[arm]["models"][m]["folds"] for m in ("lr_hand", "lr"))
            self.assertEqual(body[arm]["model_comparison_hand"], paired_comparison(hand, lr))
            self.assertEqual(body[arm]["model_comparison_hand"]["recall"]["differences"], [
                a["recall"]["recall"] - b["recall"]["recall"] for a, b in zip(hand, lr)])
            mlp = body[arm]["models"]["mlp"]["folds"]  # model_comparison is unchanged
            self.assertEqual(body[arm]["model_comparison"], paired_comparison(mlp, lr))
        self.assertEqual(set(body["arm_comparison"]), set(models))  # covers lr_hand too

    def test_model_comparison_hand_needs_both_lr_and_lr_hand(self):
        for models in ({"lr": LengthScorer}, {"lr_hand": LengthScorer}, {"other": LengthScorer},
                       {"lr": LengthScorer, "mlp": self.Low}):
            _, body = build(arms=[PRIMARY, COMPARISON], models=models)
            for arm in (PRIMARY, COMPARISON):
                self.assertNotIn("model_comparison_hand", body[arm])
        _, body = build(arms=[PRIMARY], models={"lr": LengthScorer, "lr_hand": self.Low})
        self.assertIn("model_comparison_hand", body[PRIMARY])
        self.assertNotIn("model_comparison", body[PRIMARY])

    def test_no_smote_keys(self):
        _, body = build(arms=[PRIMARY, COMPARISON], models=self.BOTH)
        self.assertEqual([k for k in self.keys(body) if "smote" in k.lower()], [])


if __name__ == "__main__":
    unittest.main()
