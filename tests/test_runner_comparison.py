import unittest
import warnings
from collections import Counter
from unittest import mock

import numpy as np
from fakes import FakeScorer
from sklearn.model_selection import StratifiedGroupKFold
from test_folds import run
from test_runner import ConvergenceWarningScorer, Factory

from bqc import SEED
from bqc.evaluate import (
    COMPARISON,
    PRIMARY,
    FoldData,
    FoldError,
    FoldResult,
    NonFiniteScoreError,
    run_comparison_fold,
    run_primary_fold,
    select_threshold,
)
from bqc.negatives import PART_NUMBER_SOURCE
from bqc.synth.partnumbers import SHAPE_IDS
from bqc.synth.registry import EVAL_FAMILY, TRAINING_FAMILIES, Registry

INNER = 5
REAL_TRAIN = [f"q{'-' * (i % 4)}{i}" for i in range(30)]
TEMPLATES = [(f, f"t{k}") for f in TRAINING_FAMILIES for k in range(5)]
POSITIVES = [(f"{f}{t} pos {j}", (f, t)) for f, t in TEMPLATES for j in range(4)]
TEST_QUERIES = ["tq1", "tq-", "tq"]
TEST_POSITIVES = ["tp 11", "tp-2"]
E_TEXTS = ["e!!", "e 99"]


def make_data(**changes):
    fields = dict(
        repeat=1, fold=2, arm=COMPARISON,
        train_texts=REAL_TRAIN + [t for t, _ in POSITIVES],
        train_y=[0] * len(REAL_TRAIN) + [1] * len(POSITIVES),
        train_families=[None] * len(REAL_TRAIN) + [g[0] for _, g in POSITIVES],
        train_groups=[None] * len(REAL_TRAIN) + [g for _, g in POSITIVES],
        threshold_texts=[],
        test_texts=TEST_QUERIES + TEST_POSITIVES + E_TEXTS,
        test_y=[0, 0, 0, 1, 1, 1, 1],
        test_families=[None] * 3 + ["C", "C", EVAL_FAMILY, EVAL_FAMILY],
        test_groups=[None] * 3 + [("C", "t9")] * 2 + [(EVAL_FAMILY, "e1")] * 2,
        report={})
    fields.update(changes)
    return FoldData(**fields)


PN_TEXTS = [f"zpn{i} {i}" for i in range(100)]
PN_SHAPES = [SHAPE_IDS[i // 25] for i in range(100)]


def make_pn_data(**changes):
    """`make_data()` with the 100 part-number rows between the real queries and the positives."""
    base = make_data()
    n = len(REAL_TRAIN)
    fields = dict(
        train_texts=REAL_TRAIN + PN_TEXTS + base.train_texts[n:],
        train_y=[0] * (n + 100) + base.train_y[n:],
        train_families=[None] * n + [PART_NUMBER_SOURCE] * 100 + base.train_families[n:],
        train_groups=[None] * n + [(PART_NUMBER_SOURCE, s) for s in PN_SHAPES]
        + base.train_groups[n:])
    fields.update(changes)
    return base._replace(**fields)


def run_fold(data=None, factory=None, target_fpr=0.1):
    return run_comparison_fold(data or make_data(), factory or Factory(), target_fpr, "lr")


def group_of(data):
    return dict(zip(data.train_texts, data.train_groups))


class ShapeTest(unittest.TestCase):
    def test_result_shape_and_test_side_unchanged(self):
        data = make_data()
        result = run_fold(data)
        self.assertIsInstance(result, FoldResult)
        self.assertEqual((result.repeat, result.fold, result.arm, result.model),
                         (1, 2, COMPARISON, "lr"))
        self.assertEqual(result.test_texts, data.test_texts)
        self.assertEqual(result.test_y, data.test_y)
        self.assertEqual(result.test_families, data.test_families)
        self.assertEqual(result.test_groups, data.test_groups)
        self.assertEqual(len(result.test_scores), len(data.test_texts))
        self.assertEqual(result.fit_warnings, [])

    def test_primary_arm_is_refused(self):
        with self.assertRaises(ValueError):
            run_fold(make_data(arm=PRIMARY))


class FitTest(unittest.TestCase):
    def test_number_of_fits_is_outer_folds_times_inner_plus_one(self):
        factory = Factory()
        datas = run(arms=(COMPARISON,), repeats=1)
        for data in datas:
            run_comparison_fold(data, factory, 0.5, "lr")
        self.assertEqual(len(datas), 3)
        self.assertEqual(len(factory.made), len(datas) * (INNER + 1))

    def test_every_fit_has_its_own_scorer_and_the_last_is_the_full_refit(self):
        factory = Factory()
        data = make_data()
        run_fold(data, factory)
        self.assertEqual(len(factory.made), INNER + 1)
        self.assertEqual(len({id(s) for s in factory.made}), INNER + 1)
        self.assertEqual(factory.made[-1].fit_texts, data.train_texts)
        self.assertEqual(factory.made[-1].fit_y, data.train_y)

    def test_no_test_side_or_e_text_in_any_fit(self):
        factory = Factory()
        run_fold(make_data(), factory)
        forbidden = set(TEST_QUERIES + TEST_POSITIVES + E_TEXTS)
        for scorer in factory.made:
            self.assertFalse(forbidden & set(scorer.fit_texts))

    def test_inner_folds_are_the_seeded_group_split_of_the_training_rows(self):
        data = make_data()
        factory = Factory()
        run_fold(data, factory)
        ids = {}
        groups = [g if g is not None else ("row", i) for i, g in enumerate(data.train_groups)]
        codes = [ids.setdefault(g, len(ids)) for g in groups]
        splitter = StratifiedGroupKFold(n_splits=INNER, shuffle=True, random_state=SEED + data.repeat)
        expected = [[data.train_texts[i] for i in train]
                    for train, _ in splitter.split(data.train_texts, data.train_y, codes)]
        self.assertEqual([s.fit_texts for s in factory.made[:INNER]], expected)

    def test_other_repeat_gives_other_inner_folds(self):
        fits = []
        for repeat in (0, 1):
            factory = Factory()
            run_fold(make_data(repeat=repeat), factory)
            fits.append([s.fit_texts for s in factory.made[:INNER]])
        self.assertNotEqual(fits[0], fits[1])

    def test_no_group_on_both_sides_of_an_inner_fold(self):
        data = make_data()
        factory = Factory()
        run_fold(data, factory)
        groups = group_of(data)
        seen = []
        for scorer in factory.made[:INNER]:
            fit = set(scorer.fit_texts)
            held_out = [t for t in data.train_texts if t not in fit]
            fit_groups = {groups[t] for t in fit if groups[t] is not None}
            held_groups = {groups[t] for t in held_out if groups[t] is not None}
            self.assertFalse(fit_groups & held_groups)
            seen += held_out
        # each training row is out of fold exactly once
        self.assertEqual(sorted(seen), sorted(data.train_texts))

    def test_inner_training_fold_missing_a_family_raises(self):
        # family D has one template, so the inner fold that holds it out lacks D
        keep = [i for i, g in enumerate(make_data().train_groups) if g is None or g[0] != "D"]
        base = make_data()
        rows = keep + [i for i, g in enumerate(base.train_groups) if g == ("D", "t0")]
        data = make_data(
            train_texts=[base.train_texts[i] for i in rows],
            train_y=[base.train_y[i] for i in rows],
            train_families=[base.train_families[i] for i in rows],
            train_groups=[base.train_groups[i] for i in rows])
        with self.assertRaisesRegex(FoldError, r"lacks families \['D'\]"):
            run_fold(data)


class PartNumberRowsTest(unittest.TestCase):
    def test_a_fake_scoring_the_rows_1_does_not_change_the_threshold(self):
        plain = run_comparison_fold(make_data(), Factory(), 0.1, "lr").threshold
        with_rows = run_comparison_fold(
            make_pn_data(), Factory({t: 1.0 for t in PN_TEXTS}), 0.1, "lr").threshold
        self.assertEqual(with_rows, plain)
        self.assertEqual(with_rows, select_threshold(list(FakeScorer().score(REAL_TRAIN)), 0.1))
        self.assertLess(max(FakeScorer().score(REAL_TRAIN)), 1.0)

    def test_the_rows_are_fitted_scored_out_of_fold_and_refitted(self):
        factory = Factory(cls=RecordingScorer)
        run_comparison_fold(make_pn_data(), factory, 0.1, "lr")
        scored = [t for scorer in factory.made[:INNER] for t in scorer.scored]
        self.assertTrue(set(PN_TEXTS) <= set(scored))
        self.assertTrue(set(PN_TEXTS) <= set(factory.made[-1].fit_texts))
        self.assertFalse(set(PN_TEXTS) & set(factory.made[-1].scored))  # never on the test side

    def test_the_inner_split_gives_the_rows_four_shape_groups_and_real_queries_their_own(self):
        seen = []

        class Spy(StratifiedGroupKFold):
            def split(self, x, y=None, groups=None):
                seen.append(list(groups))
                return super().split(x, y, groups)

        data = make_pn_data()
        with mock.patch("bqc.evaluate.StratifiedGroupKFold", Spy):
            run_comparison_fold(data, Factory(), 0.1, "lr")
        self.assertEqual(len(seen), 1)
        codes = seen[0]
        n = len(REAL_TRAIN)
        pn_codes = codes[n:n + 100]
        self.assertEqual(sorted(Counter(pn_codes).values()), [25, 25, 25, 25])
        self.assertEqual([len(set(pn_codes[i:i + 25])) for i in range(0, 100, 25)], [1] * 4)
        real_codes = codes[:n]
        self.assertEqual(len(set(real_codes)), n)  # one group per real query
        self.assertFalse(set(real_codes) & set(pn_codes))
        self.assertFalse(set(pn_codes) & set(codes[n + 100:]))  # not a template's group


class RecordingScorer(FakeScorer):
    def __init__(self):
        super().__init__()
        self.scored = []

    def score(self, texts):
        self.scored += list(texts)
        return super().score(texts)


class ScoringTest(unittest.TestCase):
    def setUp(self):
        self.data = make_data()
        self.factory = Factory(cls=RecordingScorer)
        run_fold(self.data, self.factory)
        self.inner = self.factory.made[:INNER]

    def test_each_inner_scorer_scores_nothing_it_fitted(self):
        for scorer in self.inner:
            self.assertTrue(scorer.scored)
            self.assertFalse(set(scorer.scored) & set(scorer.fit_texts))

    def test_inner_scorers_together_score_every_training_row_once(self):
        scored = [t for scorer in self.inner for t in scorer.scored]
        self.assertEqual(sorted(scored), sorted(self.data.train_texts))

    def test_the_refit_scores_exactly_the_test_texts(self):
        self.assertEqual(self.factory.made[-1].scored, self.data.test_texts)


class GenerationTest(unittest.TestCase):
    def test_generation_runs_once_per_outer_fold(self):
        with mock.patch.object(Registry, "render", autospec=True,
                               side_effect=Registry.render) as render:
            datas = run(arms=(COMPARISON,), repeats=1)
            after_generation = render.call_count
            for data in datas:
                run_comparison_fold(data, Factory(), 0.5, "lr")
        self.assertEqual(after_generation, 3 * len(datas))
        self.assertEqual(render.call_count, after_generation)


class ThresholdTest(unittest.TestCase):
    def test_threshold_is_the_rule_over_out_of_fold_real_query_scores(self):
        expected = select_threshold(list(FakeScorer().score(REAL_TRAIN)), 0.1)
        self.assertEqual(run_fold().threshold, expected)
        self.assertGreater(len(set(FakeScorer().score(REAL_TRAIN))), 2)

    def test_real_query_scores_move_the_threshold(self):
        base = run_fold().threshold
        moved = run_fold(factory=Factory({REAL_TRAIN[1]: 0.99, REAL_TRAIN[2]: 0.98})).threshold
        self.assertNotEqual(base, moved)

    def test_positive_scores_do_not_move_the_threshold(self):
        base = run_fold().threshold
        moved = run_fold(factory=Factory({t: 0.99 for t, _ in POSITIVES})).threshold
        self.assertEqual(base, moved)

    def test_test_side_scores_do_not_move_the_threshold(self):
        base = run_fold().threshold
        moved = run_fold(factory=Factory({
            t: 0.99 for t in TEST_QUERIES + TEST_POSITIVES + E_TEXTS})).threshold
        self.assertEqual(base, moved)


class FiniteTest(unittest.TestCase):
    def assert_stops(self, text, score_set, value):
        with self.assertRaises(NonFiniteScoreError) as ctx:
            run_fold(factory=Factory({text: value}))
        message = str(ctx.exception)
        self.assertIn(score_set, message)
        for part in ("repeat 1", "fold 2", COMPARISON, "lr"):
            self.assertIn(part, message)

    def test_nan_in_out_of_fold_scores(self):
        self.assert_stops(REAL_TRAIN[5], "out-of-fold", float("nan"))

    def test_nan_in_out_of_fold_positive_scores(self):
        self.assert_stops(POSITIVES[0][0], "out-of-fold", float("nan"))

    def test_nan_in_test_queries(self):
        self.assert_stops("tq-", "test queries", float("nan"))

    def test_inf_in_e(self):
        self.assert_stops("e!!", "E", float("inf"))

    def test_inf_in_test_positives(self):
        self.assert_stops("tp-2", "test positives", float("-inf"))


class WarningTest(unittest.TestCase):
    def test_warnings_from_inner_fits_and_refit_are_captured(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = run_fold(factory=Factory(cls=ConvergenceWarningScorer))
        self.assertEqual(result.fit_warnings,
                         ["ConvergenceWarning: did not converge"] * (INNER + 1))


class PairingTest(unittest.TestCase):
    def test_test_side_equals_the_primary_arm_for_each_repeat_and_fold(self):
        datas = run(arms=(PRIMARY, COMPARISON), repeats=2)
        by_key = {(d.repeat, d.fold, d.arm): d for d in datas}
        self.assertEqual(len(by_key), 12)
        for (repeat, fold, arm), data in by_key.items():
            if arm != COMPARISON:
                continue
            primary = run_primary_fold(by_key[(repeat, fold, PRIMARY)], Factory(), 0.5, "lr")
            comparison = run_comparison_fold(data, Factory(), 0.5, "lr")
            self.assertEqual(comparison.test_texts, primary.test_texts)
            self.assertEqual(comparison.test_y, primary.test_y)
            self.assertEqual(comparison.test_families, primary.test_families)
            self.assertEqual(comparison.test_groups, primary.test_groups)
            np.testing.assert_array_equal(comparison.test_scores, primary.test_scores)
            self.assertEqual((comparison.repeat, comparison.fold), (repeat, fold))


if __name__ == "__main__":
    unittest.main()
