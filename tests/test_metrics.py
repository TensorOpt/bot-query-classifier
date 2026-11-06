import json
import unittest
from unittest import mock

import numpy as np

from bqc.evaluate import FoldResult, Threshold, compute_metrics, wilson_interval
from bqc.synth.registry import EVAL_FAMILY

T = 0.5


def make_result(rows, repeat=0, fold=0, threshold=T, arm="product", model="lr"):
    """rows: (text, score, family, template_id); family None is a real query."""
    return FoldResult(
        repeat, fold, arm, model, Threshold(threshold, 0.0, False, None),
        np.array([r[1] for r in rows], dtype=float),
        [r[0] for r in rows],
        [0 if r[2] is None else 1 for r in rows],
        [r[2] for r in rows],
        [None if r[2] is None else (r[2], r[3]) for r in rows],
        [])


FOLD0 = [
    ("zen", 0.9, None, None), ("owl", 0.1, None, None), ("two  words", 0.5, None, None),
    ("a1", 0.8, "A", "t1"), ("a2", 0.2, "A", "t1"), ("b1", 0.5, "B", "t2"),
    ("e1", 0.7, EVAL_FAMILY, "e1"),
]
FOLD1 = [
    ("gold", 0.2, None, None), ("Zen ", 0.3, None, None),
    ("a3", 0.9, "A", "t1"), ("a4", 0.1, "A", "t3"), ("a5", 0.4, "A", "t3"),
    ("e2", 0.4, EVAL_FAMILY, "e1"),
]


def scenario():
    return [make_result(FOLD0, fold=0), make_result(FOLD1, fold=1)]


def fp_repeat(repeat, k, n=480):
    """One fold, `n` real queries of which the first `k` are flagged, plus two positives."""
    rows = [(f"q{i}", 1.0 if i < k else 0.0, None, None) for i in range(n)]
    rows += [("p1", 1.0, "A", "t1"), ("p2", 0.0, "A", "t1")]
    return make_result(rows, repeat=repeat)


def pooled_fpr(totals):
    return compute_metrics([fp_repeat(r, k) for r, k in enumerate(totals)])["pooled"]["fpr"]


class WilsonTest(unittest.TestCase):
    def check(self, k, n, low, high):
        got_low, got_high = wilson_interval(k, n)
        self.assertAlmostEqual(got_low, low, places=5)
        self.assertAlmostEqual(got_high, high, places=5)

    def test_reference_values(self):
        self.check(0, 480, 0, 0.00794)
        self.check(5, 480, 0.00446, 0.02415)
        self.check(1, 96, 0.00184, 0.05667)
        self.check(0, 30, 0, 0.11351)
        self.check(480, 480, 0.99206, 1)

    def test_bounds_are_clamped(self):
        self.assertEqual(wilson_interval(0, 480)[0], 0.0)
        self.assertEqual(wilson_interval(480, 480)[1], 1.0)

    def test_invalid_input_raises(self):
        for k, n in ((0, 0), (1, 0), (-1, 10), (11, 10)):
            with self.subTest(k=k, n=n), self.assertRaises(ValueError):
                wilson_interval(k, n)


class FprIntervalTest(unittest.TestCase):
    def test_repeat_interval_uses_its_own_count_at_n_480(self):
        metrics = compute_metrics([fp_repeat(0, 5), fp_repeat(1, 0), fp_repeat(2, 5)])
        first, second = metrics["repeats"][0]["fpr"], metrics["repeats"][1]["fpr"]
        self.assertEqual((first["false_positives"], first["n_negatives"]), (5, 480))
        self.assertAlmostEqual(first["point"], 5 / 480)
        self.assertAlmostEqual(first["ci_low"], 0.00446, places=5)
        self.assertAlmostEqual(first["ci_high"], 0.02415, places=5)
        self.assertAlmostEqual(second["ci_high"], 0.00794, places=5)

    def test_pooled_15_over_3_repeats_is_not_the_1440_row(self):
        fpr = pooled_fpr([5, 5, 5])
        self.assertEqual(fpr["false_positives"], 15)
        self.assertAlmostEqual(fpr["point"], 15 / 1440)
        self.assertAlmostEqual(fpr["ci_low"], 0.00446, places=5)
        self.assertAlmostEqual(fpr["ci_high"], 0.02415, places=5)
        self.assertGreater(abs(fpr["ci_high"] - 0.01712), 0.005)

    def test_pooled_total_0_has_upper_bound_00794(self):
        fpr = pooled_fpr([0, 0, 0])
        self.assertEqual(fpr["point"], 0.0)
        self.assertAlmostEqual(fpr["ci_high"], 0.00794, places=5)

    def test_non_integer_k_lies_between_neighbouring_counts(self):
        fpr = pooled_fpr([6, 5, 5])  # total 16, k = 16 / 3
        self.assertAlmostEqual(fpr["point"], 16 / 1440)
        for bound in ("ci_low", "ci_high"):
            five = wilson_interval(5, 480)[0 if bound == "ci_low" else 1]
            six = wilson_interval(6, 480)[0 if bound == "ci_low" else 1]
            self.assertGreater(fpr[bound], five)
            self.assertLess(fpr[bound], six)

    def test_n_and_repeat_count_come_from_the_run(self):
        metrics = compute_metrics([fp_repeat(0, 1, n=20), fp_repeat(1, 2, n=20)])
        fpr = metrics["pooled"]["fpr"]
        self.assertEqual((fpr["n_negatives"], fpr["repeats"]), (20, 2))
        self.assertAlmostEqual(fpr["point"], 3 / 40)
        low, high = wilson_interval(1.5, 20)
        self.assertAlmostEqual(fpr["ci_low"], low)
        self.assertAlmostEqual(fpr["ci_high"], high)

    def test_no_recall_interval(self):
        pooled = compute_metrics(scenario())["pooled"]
        self.assertEqual(set(pooled["recall"]), {"count", "caught", "recall"})

    def test_repeats_with_different_negative_counts_raise(self):
        with self.assertRaises(ValueError):
            compute_metrics([fp_repeat(0, 1, n=20), fp_repeat(1, 1, n=21)])


class FlagRuleTest(unittest.TestCase):
    def test_score_at_threshold_is_flagged_in_fpr_and_recall(self):
        rows = [("n1", 0.5, None, None), ("n2", 0.4, None, None),
                ("p1", 0.5, "A", "t1"), ("p2", 0.4, "A", "t1")]
        pooled = compute_metrics([make_result(rows)])["pooled"]
        self.assertEqual(pooled["fpr"]["false_positives"], 1)
        self.assertEqual((pooled["recall"]["caught"], pooled["recall"]["count"]), (1, 2))

    def test_every_flag_goes_through_is_illegitimate(self):
        rows = [("n1", 0.5, None, None), ("n2", 0.4, None, None),
                ("p1", 0.5, "A", "t1"), ("p2", 0.6, "A", "t1")]
        with mock.patch("bqc.evaluate.is_illegitimate", lambda s, t: s > t):
            pooled = compute_metrics([make_result(rows)])["pooled"]
        self.assertEqual(pooled["fpr"]["false_positives"], 0)
        self.assertEqual(pooled["recall"]["caught"], 1)
        self.assertEqual(pooled["confusion_matrix"], [[2, 0], [1, 1]])

    def test_each_fold_uses_its_own_threshold(self):
        rows = [("n1", 0.5, None, None), ("p1", 0.5, "A", "t1")]
        results = [make_result(rows, fold=0, threshold=0.5),
                   make_result(rows, fold=1, threshold=0.6)]
        pooled = compute_metrics(results)["pooled"]
        self.assertEqual(pooled["confusion_matrix"], [[1, 1], [1, 1]])


class ScenarioTest(unittest.TestCase):
    def setUp(self):
        self.metrics = compute_metrics(scenario())
        self.pooled = self.metrics["pooled"]

    def test_shape(self):
        self.assertEqual((self.metrics["arm"], self.metrics["model"]), ("product", "lr"))
        self.assertEqual(len(self.metrics["repeats"]), 1)
        self.assertEqual(self.metrics["repeats"][0]["repeat"], 0)
        self.assertEqual(len(self.metrics["folds"]), 2)

    def test_confusion_matrix_precision_f1_recall(self):
        self.assertEqual(self.pooled["confusion_matrix"], [[3, 2], [3, 3]])
        self.assertAlmostEqual(self.pooled["precision"], 0.6)
        self.assertAlmostEqual(self.pooled["macro_f1"], 2 * 0.6 * 0.5 / 1.1)
        self.assertEqual(self.pooled["recall"], {"count": 6, "caught": 3, "recall": 0.5})

    def test_repeat_block_equals_pooled_with_one_repeat(self):
        repeat = dict(self.metrics["repeats"][0])
        del repeat["repeat"]
        self.assertEqual(repeat, self.pooled)

    def test_test_side_ratio_is_stated(self):
        self.assertEqual(self.pooled["test_positive_to_negative_ratio"],
                         {"positives": 6, "negatives": 5, "ratio": 1.2})

    def test_roc_auc_pooled_and_over_folds(self):
        auc = self.pooled["roc_auc"]
        self.assertAlmostEqual(auc["pooled"], 17 / 30)
        self.assertAlmostEqual(auc["fold_mean"], (0.5 + 2 / 3) / 2)
        self.assertAlmostEqual(auc["fold_sd"], (2 / 3 - 0.5) / 2 ** 0.5)
        self.assertAlmostEqual(self.metrics["folds"][0]["roc_auc"], 0.5)
        self.assertAlmostEqual(self.metrics["folds"][1]["roc_auc"], 2 / 3)

    def test_per_family_recall_is_caught_over_total(self):
        families = self.pooled["families"]
        self.assertEqual(set(families), {"A", "B", "C", "D", EVAL_FAMILY})
        self.assertEqual((families["A"]["count"], families["A"]["caught"]), (5, 2))
        self.assertAlmostEqual(families["A"]["recall"], 0.4)
        self.assertEqual((families["B"]["count"], families["B"]["caught"]), (1, 1))
        self.assertEqual(families["C"], {"count": 0, "caught": 0, "recall": None,
                                          "evaluation_only": False})

    def test_unequal_family_counts_differ_from_mean_of_fold_recalls(self):
        folds = self.metrics["folds"]
        fold_recalls = [f["families"]["A"]["recall"] for f in folds]
        self.assertEqual(fold_recalls, [0.5, 1 / 3])
        self.assertNotAlmostEqual(self.pooled["families"]["A"]["recall"], sum(fold_recalls) / 2)

    def test_family_with_no_test_samples_in_a_fold(self):
        fold1 = self.metrics["folds"][1]["families"]["B"]
        self.assertEqual(fold1, {"count": 0, "caught": 0, "recall": None,
                                 "evaluation_only": False})
        fold0 = self.metrics["folds"][0]["families"]
        self.assertEqual((fold0["B"]["count"], fold0["B"]["caught"]), (1, 1))
        self.assertEqual(self.pooled["families"]["B"]["recall"], 1.0)

    def test_per_template_rows_leave_e_out(self):
        rows = {(r["family"], r["template_id"]): r for r in self.pooled["templates"]}
        self.assertEqual(set(rows), {("A", "t1"), ("A", "t3"), ("B", "t2")})
        self.assertEqual((rows[("A", "t1")]["count"], rows[("A", "t1")]["caught"]), (3, 2))
        self.assertEqual((rows[("A", "t3")]["count"], rows[("A", "t3")]["caught"]), (2, 0))
        self.assertEqual(rows[("A", "t3")]["recall"], 0.0)
        self.assertEqual(rows[("B", "t2")]["recall"], 1.0)

    def test_e_is_its_own_evaluation_only_row(self):
        e = self.pooled["families"][EVAL_FAMILY]
        self.assertEqual(e, {"count": 2, "caught": 1, "recall": 0.5, "evaluation_only": True})

    def test_false_positive_list_keeps_raw_strings(self):
        self.assertEqual(self.pooled["false_positives"], [
            {"query": "zen", "repeat": 0, "fold": 0},
            {"query": "two  words", "repeat": 0, "fold": 0}])

    def test_one_word_negatives_counted_once_after_normalise(self):
        # zen, owl, gold, and "Zen " normalising to zen again: 3 distinct words, zen flagged
        self.assertEqual(self.pooled["one_word"], {
            "total": 3, "flagged": 1, "flagged_mean_per_repeat": 1.0, "words": ["zen"]})

    def test_one_word_pooled_reports_union_and_mean_per_repeat(self):
        zen_hit = [("zen", 0.9, None, None), ("owl", 0.1, None, None),
                   ("p1", 0.9, "A", "t1"), ("p2", 0.1, "A", "t1")]
        owl_hit = [("zen", 0.1, None, None), ("owl", 0.9, None, None),
                   ("p1", 0.9, "A", "t1"), ("p2", 0.1, "A", "t1")]
        metrics = compute_metrics([make_result(zen_hit, repeat=0), make_result(owl_hit, repeat=1)])
        self.assertEqual(metrics["pooled"]["one_word"], {
            "total": 2, "flagged": 2, "flagged_mean_per_repeat": 1.0, "words": ["owl", "zen"]})
        self.assertEqual(metrics["repeats"][0]["one_word"]["flagged_mean_per_repeat"], 1.0)
        both = [("zen", 0.9, None, None), ("owl", 0.9, None, None),
                ("p1", 0.9, "A", "t1"), ("p2", 0.1, "A", "t1")]
        none = [("zen", 0.1, None, None), ("owl", 0.1, None, None),
                ("p1", 0.9, "A", "t1"), ("p2", 0.1, "A", "t1")]
        metrics = compute_metrics([make_result(both, repeat=0), make_result(none, repeat=1)])
        self.assertEqual(metrics["pooled"]["one_word"]["flagged"], 2)
        self.assertEqual(metrics["pooled"]["one_word"]["flagged_mean_per_repeat"], 1.0)
        self.assertEqual(metrics["repeats"][0]["one_word"]["flagged_mean_per_repeat"], 2.0)

    def test_one_word_flagged_in_both_repeats_counts_once_but_averages_one(self):
        # F7: with the union divided by the repeat count this would be 0.5
        hit = [("zen", 0.9, None, None), ("owl", 0.1, None, None),
               ("p1", 0.9, "A", "t1"), ("p2", 0.1, "A", "t1")]
        metrics = compute_metrics([make_result(hit, repeat=0), make_result(hit, repeat=1)])
        self.assertEqual(metrics["pooled"]["one_word"], {
            "total": 2, "flagged": 1, "flagged_mean_per_repeat": 1.0, "words": ["zen"]})

    def test_output_round_trips_through_json(self):
        self.assertEqual(json.loads(json.dumps(self.metrics)), self.metrics)

    def test_numpy_threshold_leaves_no_numpy_types(self):
        results = [make_result(rows, fold=i, threshold=np.float64(T))
                   for i, rows in enumerate((FOLD0, FOLD1))]
        metrics = compute_metrics(results)
        self.assertEqual(json.loads(json.dumps(metrics)), metrics)
        self.assertEqual(metrics["pooled"]["false_positives"], self.pooled["false_positives"])

    def test_macro_f1_differs_from_plain_f1_on_an_asymmetric_matrix(self):
        rows = [("n1", 0.9, None, None), ("n2", 0.1, None, None), ("n3", 0.1, None, None),
                ("n4", 0.1, None, None), ("p1", 0.9, "A", "t1"), ("p2", 0.1, "A", "t1")]
        pooled = compute_metrics([make_result(rows)])["pooled"]
        self.assertEqual(pooled["confusion_matrix"], [[3, 1], [1, 1]])
        self.assertAlmostEqual(pooled["macro_f1"], (0.75 + 0.5) / 2)


class TwoRepeatTest(unittest.TestCase):
    REPEAT1 = [
        [("n1", 0.1, None, None), ("n2", 0.2, None, None), ("p1", 0.9, "A", "t1"),
         ("e1", 0.3, EVAL_FAMILY, "e1")],
        [("n3", 0.1, None, None), ("n4", 0.2, None, None), ("n5", 0.3, None, None),
         ("p2", 0.9, "B", "t2")],
    ]

    def test_folds_and_repeat_blocks_do_not_mix_repeats(self):
        first = scenario()
        second = [make_result(rows, repeat=1, fold=i) for i, rows in enumerate(self.REPEAT1)]
        both = compute_metrics(first + second)
        alone = [compute_metrics(first), compute_metrics(second)]
        self.assertEqual(len(both["folds"]), 4)
        self.assertEqual(both["folds"], alone[0]["folds"] + alone[1]["folds"])
        for repeat in (0, 1):
            block = dict(both["repeats"][repeat])
            self.assertEqual(block.pop("repeat"), repeat)
            self.assertEqual(block, alone[repeat]["pooled"])
        self.assertEqual(both["pooled"]["roc_auc"]["fold_mean"],
                         float(np.mean([f["roc_auc"] for f in both["folds"]])))
        self.assertEqual(both["pooled"]["fpr"]["repeats"], 2)


class EvaluationOnlyFamilyTest(unittest.TestCase):
    def test_adding_e_samples_leaves_headline_numbers_unchanged(self):
        no_e = [[r for r in rows if r[2] != EVAL_FAMILY] for rows in (FOLD0, FOLD1)]
        more_e = [FOLD0 + [("e3", 0.99, EVAL_FAMILY, "e2"), ("e4", 0.0, EVAL_FAMILY, "e2")],
                  FOLD1]
        a = compute_metrics([make_result(no_e[0], fold=0), make_result(no_e[1], fold=1)])["pooled"]
        b = compute_metrics([make_result(more_e[0], fold=0),
                             make_result(more_e[1], fold=1)])["pooled"]
        for key in ("recall", "fpr", "confusion_matrix", "precision", "macro_f1", "roc_auc",
                    "test_positive_to_negative_ratio", "one_word", "false_positives"):
            self.assertEqual(a[key], b[key], key)
        self.assertEqual(b["families"][EVAL_FAMILY]["count"], 4)


class SmallCountTest(unittest.TestCase):
    def test_template_with_fewer_renders_counts_what_survives(self):
        rows = [("n1", 0.1, None, None), ("p1", 0.9, "A", "t1"), ("p2", 0.2, "A", "t1"),
                ("p3", 0.8, "A", "t1")]
        pooled = compute_metrics([make_result(rows)])["pooled"]
        row = pooled["templates"][0]
        self.assertEqual((row["count"], row["caught"]), (3, 2))
        self.assertEqual(pooled["families"]["A"]["count"], 3)

    def test_pooled_over_repeats_adds_counts(self):
        rows = [("n1", 0.1, None, None), ("p1", 0.9, "A", "t1"), ("p2", 0.2, "A", "t1")]
        metrics = compute_metrics([make_result(rows, repeat=0), make_result(rows, repeat=1)])
        self.assertEqual(len(metrics["repeats"]), 2)
        self.assertEqual(metrics["pooled"]["families"]["A"]["count"], 4)
        self.assertEqual(metrics["pooled"]["templates"][0]["caught"], 2)
        self.assertEqual(metrics["repeats"][1]["families"]["A"]["count"], 2)


class InvalidInputTest(unittest.TestCase):
    def test_single_class_roc_auc_raises(self):
        rows = [("n1", 0.1, None, None), ("n2", 0.9, None, None)]
        with self.assertRaises(ValueError):
            compute_metrics([make_result(rows)])

    def test_fold_with_only_e_positives_raises(self):
        rows = [("n1", 0.1, None, None), ("e1", 0.9, EVAL_FAMILY, "e1")]
        with self.assertRaises(ValueError):
            compute_metrics([make_result(rows)])

    def test_no_results_raises(self):
        with self.assertRaises(ValueError):
            compute_metrics([])

    def test_mixed_arms_or_models_raise(self):
        rows = FOLD0
        with self.assertRaises(ValueError):
            compute_metrics([make_result(rows, arm="product"), make_result(rows, arm="real")])
        with self.assertRaises(ValueError):
            compute_metrics([make_result(rows, model="lr"), make_result(rows, model="mlp")])


if __name__ == "__main__":
    unittest.main()
