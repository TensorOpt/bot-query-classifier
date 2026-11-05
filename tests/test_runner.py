import ast
import unittest
import warnings
from pathlib import Path

import numpy as np
from fakes import FakeScorer
from sklearn.exceptions import ConvergenceWarning
from test_folds import REAL, run

from bqc.evaluate import (
    COMPARISON,
    PRIMARY,
    FoldData,
    NonFiniteScoreError,
    check_finite,
    is_illegitimate,
    run_primary_fold,
    select_threshold,
)
from bqc.synth.registry import EVAL_FAMILY

# FakeScorer scores: fraction of characters that are neither letters nor spaces.
THRESHOLD_TEXTS = ["ab", "a1", "a1b2", "a-1", "x9", "!!a", "q", "r-", "s1t", "u12v"]
TEST_QUERIES = ["tq1", "tq-", "tq"]
TEST_POSITIVES = ["pos 11", "pos-2"]
E_TEXTS = ["e!!", "e 99"]
CONTEXT = dict(repeat=2, fold=3, arm="product", model="lr")


def make_data(**changes):
    train_texts = ["span one", "span 2", "pos 1", "pos-3"]
    fields = dict(
        repeat=2, fold=3, arm=PRIMARY,
        train_texts=train_texts, train_y=[0, 0, 1, 1],
        train_families=[None, None, "A", "B"],
        train_groups=[None, None, ("A", "t1"), ("B", "t2")],
        threshold_texts=list(THRESHOLD_TEXTS),
        test_texts=TEST_QUERIES + TEST_POSITIVES + E_TEXTS,
        test_y=[0, 0, 0, 1, 1, 1, 1],
        test_families=[None] * 3 + ["C", "C", EVAL_FAMILY, EVAL_FAMILY],
        test_groups=[None] * 3 + [("C", "t3")] * 2 + [(EVAL_FAMILY, "e1")] * 2,
        report={})
    fields.update(changes)
    return FoldData(**fields)


class Factory:
    """Records every scorer it makes; scores come from `overrides` first, else FakeScorer."""

    def __init__(self, overrides=None, cls=FakeScorer):
        self.overrides = overrides or {}
        self.cls = cls
        self.made = []

    def __call__(self):
        outer = self

        class Scorer(self.cls):
            def score(self, texts):
                base = super().score(texts)
                return np.array([outer.overrides.get(t, b) for t, b in zip(texts, base)])

        scorer = Scorer()
        self.made.append(scorer)
        return scorer


def run_fold(data=None, factory=None, target_fpr=0.1):
    return run_primary_fold(data or make_data(), factory or Factory(), target_fpr, "lr")


class CheckFiniteTest(unittest.TestCase):
    def test_finite_array_passes_untouched(self):
        scores = np.array([0.0, 0.5, 1.0, -3.0])
        out = check_finite(scores, **CONTEXT, score_set="test queries")
        np.testing.assert_array_equal(out, scores)

    def test_empty_array_passes(self):
        self.assertEqual(len(check_finite([], **CONTEXT, score_set="E")), 0)

    def test_message_names_all_five(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(NonFiniteScoreError) as ctx:
                check_finite([0.1, bad], **CONTEXT, score_set="test queries")
            message = str(ctx.exception)
            for part in ("2", "3", "product", "lr", "test queries"):
                self.assertIn(part, message)
            self.assertIn("repeat 2", message)
            self.assertIn("fold 3", message)


class RunnerFiniteTest(unittest.TestCase):
    def assert_stops(self, text, score_set, value):
        with self.assertRaises(NonFiniteScoreError) as ctx:
            run_fold(factory=Factory({text: value}))
        message = str(ctx.exception)
        self.assertIn(score_set, message)
        for part in ("repeat 2", "fold 3", "product", "lr"):
            self.assertIn(part, message)

    def test_nan_in_test_queries(self):
        self.assert_stops("tq-", "test queries", float("nan"))

    def test_nan_in_e(self):
        self.assert_stops("e!!", "E", float("nan"))

    def test_nan_in_threshold_scores_is_not_the_select_threshold_error(self):
        with self.assertRaises(NonFiniteScoreError) as ctx:
            run_fold(factory=Factory({"a1": float("nan")}))
        self.assertIn("threshold queries", str(ctx.exception))

    def test_infinities_in_test_positives(self):
        self.assert_stops("pos 11", "test positives", float("inf"))
        self.assert_stops("pos-2", "test positives", float("-inf"))


class RunnerFitTest(unittest.TestCase):
    def test_one_fit_per_outer_fold_and_on_the_training_texts_only(self):
        factory = Factory()
        data = make_data()
        run_fold(data, factory)
        self.assertEqual(len(factory.made), 1)
        fitted = factory.made[0]
        self.assertEqual(fitted.fit_texts, data.train_texts)
        self.assertEqual(fitted.fit_y, data.train_y)

    def test_no_real_query_test_text_or_e_in_the_fit(self):
        factory = Factory()
        run_fold(make_data(), factory)
        forbidden = set(THRESHOLD_TEXTS + TEST_QUERIES + TEST_POSITIVES + E_TEXTS)
        self.assertFalse(forbidden & set(factory.made[0].fit_texts))

    def test_number_of_fits_equals_number_of_folds(self):
        factory = Factory()
        datas = run(arms=(PRIMARY,), repeats=1)
        for data in datas:
            run_primary_fold(data, factory, 0.1, "lr")
        self.assertEqual(len(factory.made), len(datas))
        self.assertEqual(len(datas), 3)
        for scorer, data in zip(factory.made, datas):
            self.assertEqual(scorer.fit_texts, data.train_texts)

    def test_comparison_arm_is_refused(self):
        with self.assertRaises(ValueError):
            run_fold(make_data(arm=COMPARISON))


class RunnerThresholdTest(unittest.TestCase):
    def test_threshold_is_the_rule_over_threshold_query_scores(self):
        scores = FakeScorer().score(THRESHOLD_TEXTS)
        expected = select_threshold(list(scores), 0.1)
        result = run_fold()
        self.assertEqual(result.threshold, expected)

    def test_test_side_scores_do_not_move_the_threshold(self):
        base = run_fold().threshold
        moved = run_fold(factory=Factory({
            t: 0.99 for t in TEST_QUERIES + TEST_POSITIVES + E_TEXTS})).threshold
        self.assertEqual(base, moved)

    def test_training_side_query_scores_move_the_threshold(self):
        base = run_fold().threshold
        moved = run_fold(factory=Factory({"a1": 0.99, "x9": 0.98})).threshold
        self.assertNotEqual(base, moved)

    def test_same_t_judges_every_test_side_score_set(self):
        # t is 1.0-ish when one threshold query scores 1.0 and the target allows one flag
        factory = Factory({"u12v": 0.7, "tq-": 0.7, "pos 11": 0.7, "e!!": 0.7, "e 99": 0.69})
        result = run_fold(factory=factory, target_fpr=0.1)
        t = result.threshold.threshold
        flagged = [is_illegitimate(s, t) for s in result.test_scores]
        self.assertEqual(t, 0.7)
        self.assertEqual(flagged, [False, True, False, True, False, True, False])


class RunnerShapeTest(unittest.TestCase):
    def test_test_side_rows_unchanged(self):
        data = make_data()
        result = run_fold(data)
        self.assertEqual(len(result.test_scores), len(data.test_texts))
        self.assertEqual(result.test_y, data.test_y)
        self.assertEqual(result.test_families, data.test_families)
        self.assertEqual(result.test_groups, data.test_groups)
        self.assertEqual((result.repeat, result.fold, result.arm, result.model),
                         (2, 3, PRIMARY, "lr"))

    def test_each_real_query_is_a_test_item_once_per_repeat(self):
        factory = Factory()
        datas = [d for d in run(arms=(PRIMARY,), repeats=2) if d.repeat == 0]
        seen = []
        for data in datas:
            result = run_primary_fold(data, factory, 0.5, "lr")
            seen += [t for t, f in zip(data.test_texts, data.test_families) if f is None]
            self.assertEqual(len(result.test_scores), len(data.test_texts))
        self.assertEqual(sorted(seen), sorted(REAL))


class ConvergenceWarningScorer(FakeScorer):
    def fit(self, texts, y):
        warnings.warn("did not converge", ConvergenceWarning)
        return super().fit(texts, y)


class RunnerWarningTest(unittest.TestCase):
    def test_fit_warning_is_captured_not_raised(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = run_fold(factory=Factory(cls=ConvergenceWarningScorer))
        self.assertEqual(result.fit_warnings, ["ConvergenceWarning: did not converge"])

    def test_no_warning_gives_an_empty_list(self):
        self.assertEqual(run_fold().fit_warnings, [])


class ImportTest(unittest.TestCase):
    def test_evaluate_imports_no_sklearn_estimator(self):
        source = (Path(__file__).parent.parent / "bqc" / "evaluate.py").read_text()
        modules = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
            elif isinstance(node, ast.Import):
                modules.update(a.name for a in node.names)
        self.assertEqual({m for m in modules if m.startswith("sklearn")},
                         {"sklearn.model_selection"})


if __name__ == "__main__":
    unittest.main()
