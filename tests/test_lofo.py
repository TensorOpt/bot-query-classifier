import json
import unittest
from unittest import mock

import numpy as np
from fakes import FakeScorer
from synth_stubs import build_registry
from test_folds import CONFIG, PRODUCTS, REAL, run
from test_run_report import LengthScorer
from test_runner import Factory

from bqc.evaluate import (
    COMPARISON,
    PRIMARY,
    FoldError,
    FoldResult,
    NonFiniteScoreError,
    Threshold,
    assign_folds,
    build_run_report,
    check_fold,
    lofo_section,
    run_comparison_fold,
    run_primary_fold,
)
from bqc.synth.registry import EVAL_FAMILY, TRAINING_FAMILIES

RUNNERS = {PRIMARY: run_primary_fold, COMPARISON: run_comparison_fold}
# five outer folds, as in the real run: with three, an inner training fold can lose a family
SPLITS = dict(CONFIG, n_splits=5)
OPTIONS = dict(SPLITS, target_fpr=0.5)


def build(models=None, **kw):
    return build_run_report(**{
        **OPTIONS, "real_queries": REAL, "products": PRODUCTS, "part_numbers": [],
        "sanity": [], "models": models or {"lr": LengthScorer},
        "registry": build_registry(), **kw})


def lofo_data(family, arm):
    return run(arms=(arm,), leave_out=family, n_splits=5)


def main_data(arm):
    return run(arms=(arm,), n_splits=5)


def result(texts, scores, families, threshold, repeat=0, fold=0):
    return FoldResult(
        repeat, fold, PRIMARY, "lr", Threshold(threshold, 0.0, False, None),
        np.array(scores, dtype=float), texts, [0 if f is None else 1 for f in families],
        families, [None if f is None else (f, "t0") for f in families], [])


class DataTest(unittest.TestCase):
    def test_no_f_text_in_any_training_side_in_both_arms(self):
        for arm in (PRIMARY, COMPARISON):
            mains = main_data(arm)
            for family in TRAINING_FAMILIES:
                for data, main in zip(lofo_data(family, arm), mains):
                    f_texts = {t for t, f in zip(main.train_texts, main.train_families)
                               if f == family}
                    f_texts |= {t for t, f in zip(main.test_texts, main.test_families)
                                if f == family}
                    self.assertTrue(f_texts)
                    self.assertFalse(f_texts & set(data.train_texts))
                    self.assertNotIn(family, data.train_families)
                    self.assertEqual(set(data.train_families) - {None},
                                     set(TRAINING_FAMILIES) - {family})

    def test_training_side_stays_one_to_one_up_to_rounding(self):
        for arm in (PRIMARY, COMPARISON):
            for data in lofo_data("B", arm):
                positives = sum(data.train_y)
                negatives = len(data.train_y) - positives
                remaining = len({g for g in data.train_groups if g is not None})
                self.assertTrue(0 <= negatives - positives < remaining, (arm, positives, negatives))

    def test_negatives_are_unchanged(self):
        for arm in (PRIMARY, COMPARISON):
            for data, main in zip(lofo_data("C", arm), main_data(arm)):
                negatives = lambda d: [t for t, y in zip(d.train_texts, d.train_y) if y == 0]
                self.assertEqual(negatives(data), negatives(main))
                self.assertEqual(data.threshold_texts, main.threshold_texts)

    def test_test_side_is_the_main_runs_restricted_to_queries_and_f(self):
        for arm in (PRIMARY, COMPARISON):
            for family in TRAINING_FAMILIES:
                for data, main in zip(lofo_data(family, arm), main_data(arm)):
                    keep = [i for i, f in enumerate(main.test_families) if f in (None, family)]
                    self.assertEqual(data.test_texts, [main.test_texts[i] for i in keep])
                    self.assertEqual(data.test_y, [main.test_y[i] for i in keep])
                    self.assertEqual(data.test_groups, [main.test_groups[i] for i in keep])
                    self.assertNotIn(EVAL_FAMILY, data.test_families)

    def test_unknown_or_evaluation_family_is_refused(self):
        for family in (EVAL_FAMILY, "Z"):
            with self.assertRaises(ValueError):
                lofo_data(family, PRIMARY)

    def test_training_side_missing_one_of_the_other_three_raises(self):
        def without_b(*args, **kw):
            return [f._replace(train_templates=[t for t in f.train_templates if t[0] != "B"])
                    for f in assign_folds(*args, **kw)]

        with mock.patch("bqc.evaluate.assign_folds", without_b):
            with self.assertRaises(FoldError):
                run(leave_out="A", n_splits=5)
        with self.assertRaises(FoldError):
            check_fold([0, 1, 1], [None, "B", "C"], [0, 1], ["B", "C", "D"])
        check_fold([0, 1, 1], [None, "B", "C"], [0, 1], ["B", "C"])

    def test_inner_folds_assert_the_other_three_families(self):
        data = next(d for d in lofo_data("A", COMPARISON))
        keep = [i for i, f in enumerate(data.train_families) if f != "B"]
        broken = data._replace(
            train_texts=[data.train_texts[i] for i in keep],
            train_y=[data.train_y[i] for i in keep],
            train_families=[data.train_families[i] for i in keep],
            train_groups=[data.train_groups[i] for i in keep])
        with self.assertRaises(FoldError):
            run_comparison_fold(broken, Factory(), 0.5, "lr", leave_out="A")
        # without leave_out the same data lacks A as well and also raises
        with self.assertRaises(FoldError):
            run_comparison_fold(broken, Factory(), 0.5, "lr")

    def test_f_absent_from_the_test_side_does_not_raise(self):
        absent = [d for d in lofo_data("A", PRIMARY) if "A" not in d.test_families]
        self.assertTrue(absent)  # the stub configuration has such folds
        for data in absent:
            self.assertEqual(set(data.test_y), {0})
            run_primary_fold(data, Factory(), 0.5, "lr", leave_out="A")


class RunnerTest(unittest.TestCase):
    def test_nan_names_the_lofo_score_set(self):
        for arm, score_set in ((PRIMARY, "threshold queries"), (COMPARISON, "out-of-fold scores")):
            data = lofo_data("A", arm)[0]
            queries = (data.threshold_texts if arm == PRIMARY
                       else [t for t, y in zip(data.train_texts, data.train_y) if y == 0])
            with self.assertRaises(NonFiniteScoreError) as ctx:
                RUNNERS[arm](data, Factory({queries[0]: float("nan")}), 0.5, "lr", leave_out="A")
            self.assertIn(f"LOFO A {score_set}", str(ctx.exception))
            self.assertIn(f"arm {arm}", str(ctx.exception))
            test_query = next(t for t, f in zip(data.test_texts, data.test_families) if f is None)
            with self.assertRaises(NonFiniteScoreError) as ctx:
                RUNNERS[arm](data, Factory({test_query: float("inf")}), 0.5, "lr", leave_out="A")
            self.assertIn("LOFO A test queries", str(ctx.exception))

    def test_nan_in_f_rows_names_the_lofo_positives(self):
        data = next(d for d in lofo_data("A", PRIMARY) if "A" in d.test_families)
        positive = next(t for t, f in zip(data.test_texts, data.test_families) if f == "A")
        with self.assertRaises(NonFiniteScoreError) as ctx:
            run_primary_fold(data, Factory({positive: float("nan")}), 0.5, "lr", leave_out="A")
        self.assertIn("LOFO A test positives", str(ctx.exception))

    def test_main_run_names_are_unchanged(self):
        data = main_data(PRIMARY)[0]
        test_query = next(t for t, f in zip(data.test_texts, data.test_families) if f is None)
        with self.assertRaises(NonFiniteScoreError) as ctx:
            run_primary_fold(data, Factory({test_query: float("nan")}), 0.5, "lr")
        self.assertIn("score in test queries", str(ctx.exception))


class SectionTest(unittest.TestCase):
    def test_hand_computed_fpr_and_paired_difference(self):
        queries = ["q1", "q2", "q3", "q4"]
        texts = queries + ["a1", "a2", "a3", "a4"]
        families = [None] * 4 + ["A"] * 4
        main = [
            result(texts, [0, 0, 0, 0, .9, .9, .1, .1], families, .5),
            result(queries, [0, 0, 0, 0], [None] * 4, .5, fold=1)]
        lofo_a = [
            result(texts, [.4, .1, .1, .1, .9, .8, .7, .1], families, .3),
            result(queries, [.3, .1, .1, .1], [None] * 4, .3, fold=1)]
        section = lofo_section({f: lofo_a for f in TRAINING_FAMILIES}, main)
        row = section["A"]
        first, second = row["folds"]
        self.assertEqual(first["recall"], {"count": 4, "caught": 3, "recall": 0.75})
        self.assertEqual(first["fpr"], {"false_positives": 1, "n_negatives": 4, "point": 0.25})
        self.assertEqual((first["main_recall"], first["difference"]), (0.5, 0.25))
        self.assertEqual(first["threshold"], 0.3)
        self.assertEqual(second["recall"], {"count": 0, "caught": 0, "recall": None})
        self.assertEqual(second["fpr"]["false_positives"], 1)
        self.assertIsNone(second["main_recall"])
        self.assertIsNone(second["difference"])
        pooled = row["pooled"]
        self.assertEqual(pooled["recall"]["recall"], 0.75)
        self.assertEqual(pooled["fpr"],
                         {"false_positives": 2, "n_negatives": 8, "point": 0.25})
        self.assertEqual((pooled["main_recall"], pooled["difference"]), (0.5, 0.25))

    def test_unmatched_fold_raises(self):
        r = result(["q"], [0], [None], .5)
        with self.assertRaises(ValueError):
            lofo_section({f: [r._replace(fold=9)] for f in TRAINING_FAMILIES}, [r])


class ReportTest(unittest.TestCase):
    def test_nan_only_in_a_lofo_pass_names_the_lofo_score_set(self):
        def late_nan_factory(main_fits):
            made = []

            def factory():
                scorer = FakeScorer()
                nan = len(made) >= main_fits  # fits after the main run's are LOFO fits
                made.append(scorer)
                plain = scorer.score
                scorer.score = lambda texts: np.full(len(texts), np.nan) if nan else plain(texts)
                return scorer
            return factory

        for arm, main_fits in ((PRIMARY, 10), (COMPARISON, 60)):
            with self.assertRaises(NonFiniteScoreError) as ctx:
                build(models={"lr": late_nan_factory(main_fits)}, arms=(arm,), lofo=True)
            self.assertIn("LOFO A", str(ctx.exception))
            self.assertIn(f"arm {arm}", str(ctx.exception))
            build(models={"lr": late_nan_factory(main_fits)}, arms=(arm,))  # does not raise

    def test_parameters_record_lofo(self):
        self.assertIs(build(arms=(PRIMARY,))[0]["lofo"], False)
        self.assertIs(build(arms=(PRIMARY,), lofo=True)[0]["lofo"], True)

    def test_off_by_default(self):
        _, body = build()
        for arm in (PRIMARY, COMPARISON):
            self.assertNotIn("lofo", body[arm])

    def test_one_row_per_family_a_to_d_none_for_e_in_both_arms(self):
        parameters, body = build(lofo=True)
        for arm in (PRIMARY, COMPARISON):
            section = body[arm]["lofo"]["lr"]
            self.assertEqual(sorted(section), sorted(TRAINING_FAMILIES))
            self.assertNotIn("lofo", body[arm]["models"]["lr"])
            for row in section.values():
                self.assertEqual(len(row["folds"]), 10)
                self.assertEqual(
                    {"recall", "fpr", "main_recall", "difference"} - set(row["pooled"]), set())
                for fold in row["folds"]:
                    self.assertNotIn("roc_auc", fold)
        json.dumps(body)

    def test_rows_agree_with_the_main_run(self):
        _, body = build(lofo=True)
        for arm in (PRIMARY, COMPARISON):
            main = body[arm]["models"]["lr"]
            for family, row in body[arm]["lofo"]["lr"].items():
                for fold, main_fold in zip(row["folds"], main["folds"]):
                    self.assertEqual((fold["repeat"], fold["fold"]),
                                     (main_fold["repeat"], main_fold["fold"]))
                    expected = main_fold["families"][family]
                    self.assertEqual(fold["recall"]["count"], expected["count"])
                    self.assertEqual(fold["main_recall"], expected["recall"])
                    if fold["recall"]["recall"] is None:
                        self.assertIsNone(fold["difference"])
                    else:
                        self.assertAlmostEqual(
                            fold["difference"], fold["recall"]["recall"] - expected["recall"])

    def test_f_absent_from_a_test_side_gives_null_recall(self):
        _, body = build(lofo=True)
        recalls = [f["recall"]["recall"] for row in body[PRIMARY]["lofo"]["lr"].values()
                   for f in row["folds"]]
        self.assertIn(None, recalls)
        self.assertTrue(any(r is not None for r in recalls))

    def test_fit_counts(self):
        base, with_lofo = Factory(cls=FakeScorer), Factory(cls=FakeScorer)
        build(models={"lr": base}, arms=(PRIMARY,))
        build(models={"lr": with_lofo}, arms=(PRIMARY,), lofo=True)
        outer = len(base.made)
        self.assertEqual(outer, 10)
        self.assertEqual(len(with_lofo.made), outer + 4 * outer)  # 1 per family per fold
        base, with_lofo = Factory(cls=FakeScorer), Factory(cls=FakeScorer)
        build(models={"lr": base}, arms=(COMPARISON,))
        build(models={"lr": with_lofo}, arms=(COMPARISON,), lofo=True)
        self.assertEqual(len(base.made), 10 * 6)
        self.assertEqual(len(with_lofo.made), 10 * 6 + 4 * 10 * 6)  # inner folds + 1
