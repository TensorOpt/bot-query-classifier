import unittest
from unittest import mock

import numpy as np
from fakes import FakeScorer
from synth_stubs import build_registry
from test_folds import PRODUCTS, REAL, run
from test_run_report import LengthScorer
from test_runner import Factory

from bqc import SEED
from bqc.evaluate import (
    COMPARISON,
    PRIMARY,
    FoldResult,
    NonFiniteScoreError,
    Threshold,
    assign_folds,
    build_run_report,
    catalogue_gap_section,
    iter_fold_data,
    run_comparison_fold,
    run_primary_fold,
    wilson_interval,
)
from bqc.negatives import PoolExhausted, build_spans
from bqc.wands import WandsProduct

RUNNERS = {PRIMARY: run_primary_fold, COMPARISON: run_comparison_fold}
CONFIG = dict(n=120, n_splits=3, repeats=1, test_per_template=5, eval_per_template=4)
OPTIONS = dict(CONFIG, target_fpr=0.5)
EMPTY_QUERIES = {3, 11, 20, 33}  # indices of the real queries with an empty class

FOLDS = assign_folds(REAL, build_registry(), n_splits=3, repeats=1)
CLASS_OF = {}
for _fold in FOLDS:
    for _j, _q in enumerate(_fold.test_queries):
        CLASS_OF[_q] = "abc"[(_fold.fold + _j % 2) % 3]  # fold 0: a, b; 1: b, c; 2: c, a
for _i in EMPTY_QUERIES:
    CLASS_OF[REAL[_i]] = ""
QUERY_CLASSES = [CLASS_OF[q] for q in REAL]
TEST_CLASSES = [{"a", "b"}, {"b", "c"}, {"c", "a"}]

# classes of the products by id range; the rest of the 300 carry "z"
PRODUCT_CLASSES = (
    [("a",)] * 10 + [("b",)] * 20 + [("c",)] * 30 + [("a", "c")] * 5 + [("b", "z")] * 5
    + [("z",)] * 190 + [()] * 40)
CLASSED = [WandsProduct(p.product_id, p.name, PRODUCT_CLASSES[p.product_id]) for p in PRODUCTS]
# hand-counted: fold 0 {a, b}: 10 + 20 + 5 (a, c) + 5 (b, z); fold 1 {b, c}: 20 + 30 + 5 + 5;
# fold 2 {c, a}: 30 + 10 + 5 (a, c); 40 products have no class and are never removed
REMOVED = [40, 60, 45]
UNCLASSIFIED = 40


def removed_ids(fold):
    return {p.product_id for p in CLASSED if set(p.classes) & TEST_CLASSES[fold]}


def gap_data(arms=(PRIMARY, COMPARISON), products=CLASSED, query_classes=QUERY_CLASSES, **kw):
    return list(iter_fold_data(
        REAL, products, [], [], arms=arms, registry=build_registry(),
        query_classes=query_classes, **{**CONFIG, **kw}))


def training_queries_kept(fold):
    """The fold's training-side real queries without those of the test side's classes."""
    return [q for q in FOLDS[fold].train_queries if CLASS_OF[q] not in TEST_CLASSES[fold]]


class FixtureTest(unittest.TestCase):
    def test_the_fixture_is_what_the_hand_counts_assume(self):
        for fold in range(3):
            test_classes = {CLASS_OF[q] for q in FOLDS[fold].test_queries} - {""}
            self.assertEqual(test_classes, TEST_CLASSES[fold])
            self.assertEqual(len(removed_ids(fold)), REMOVED[fold])
            removed_train = set(FOLDS[fold].train_queries) - set(training_queries_kept(fold))
            self.assertTrue(removed_train)
        self.assertEqual(sum(not p.classes for p in CLASSED), UNCLASSIFIED)
        self.assertTrue(any(CLASS_OF[q] == "" for f in FOLDS for q in f.train_queries))


class PrimaryRemovalTest(unittest.TestCase):
    def recorded_builds(self):
        calls = []

        def spy(products, n, seed, *args):
            spans, removals = build_spans(products, n, seed, *args)
            calls.append(([p.product_id for p in products], spans, seed))
            return spans, removals

        with mock.patch("bqc.evaluate.build_spans", side_effect=spy):
            data = gap_data(arms=(PRIMARY,))
        return calls, data

    def test_no_product_of_a_test_side_class_is_in_a_folds_pool_or_spans(self):
        calls, _ = self.recorded_builds()
        self.assertEqual(len(calls), 3)  # one build per fold
        for fold, (ids, spans, _) in enumerate(calls):
            gone = removed_ids(fold)
            self.assertFalse(gone & set(ids))
            self.assertEqual(set(ids), {p.product_id for p in CLASSED} - gone)
            self.assertFalse(gone & {s.product_id for s in spans})
            self.assertEqual(len(spans), 120)

    def test_multi_class_product_with_one_matching_class_is_removed(self):
        calls, _ = self.recorded_builds()
        ids_by_fold = [set(ids) for ids, _, _ in calls]
        for product_id in range(65, 70):  # classes ("b", "z"): fold 0 and 1 match b
            self.assertNotIn(product_id, ids_by_fold[0])
            self.assertNotIn(product_id, ids_by_fold[1])
            self.assertIn(product_id, ids_by_fold[2])
        for product_id in range(60, 65):  # ("a", "c"): every fold has a or c
            self.assertTrue(all(product_id not in ids for ids in ids_by_fold))

    def test_the_repeat_seed_is_kept(self):
        calls, _ = self.recorded_builds()
        self.assertEqual({seed for _, _, seed in calls}, {SEED})

    def test_removal_differs_per_fold_and_the_report_counts_it(self):
        calls, data = self.recorded_builds()
        self.assertEqual([d.report["products_removed"] for d in data], REMOVED)
        self.assertEqual(len({tuple(ids) for ids, _, _ in calls}), 3)
        self.assertEqual(len({tuple(s.text for s in spans) for _, spans, _ in calls}), 3)

    def test_empty_class_products_are_never_removed(self):
        calls, _ = self.recorded_builds()
        unclassified = {p.product_id for p in CLASSED if not p.classes}
        for ids, _, _ in calls:
            self.assertLessEqual(unclassified, set(ids))

    def test_threshold_set_drops_those_classes_but_keeps_empty_class_queries(self):
        for fold, data in enumerate(gap_data(arms=(PRIMARY,))):
            self.assertEqual(data.threshold_texts, training_queries_kept(fold))
            empties = [q for q in FOLDS[fold].train_queries if CLASS_OF[q] == ""]
            self.assertLessEqual(set(empties), set(data.threshold_texts))
            self.assertTrue(all(CLASS_OF[q] not in TEST_CLASSES[fold]
                                for q in data.threshold_texts))

    def test_the_test_side_keeps_every_test_query_with_empty_classes_too(self):
        for fold, data in enumerate(gap_data(arms=(PRIMARY,))):
            queries = [t for t, f in zip(data.test_texts, data.test_families) if f is None]
            self.assertEqual(queries, FOLDS[fold].test_queries)

    def test_no_real_query_is_in_the_fit(self):
        for data in gap_data(arms=(PRIMARY,)):
            self.assertFalse(set(data.train_texts) & set(REAL))

    def test_the_normal_runs_spans_are_unaffected(self):
        before = run(arms=(PRIMARY,), repeats=1, products=CLASSED)
        gap_data()
        after = run(arms=(PRIMARY,), repeats=1, products=CLASSED)
        self.assertEqual(before, after)
        gap = gap_data(arms=(PRIMARY,))
        self.assertNotEqual([d.train_texts for d in gap], [d.train_texts for d in before])


class AllEmptyClassesTest(unittest.TestCase):
    def test_no_class_means_no_removal_and_the_normal_runs_data(self):
        gap = gap_data(query_classes=[""] * len(REAL))
        normal = run(repeats=1, products=CLASSED)
        self.assertEqual([d.report["products_removed"] for d in gap if d.arm == PRIMARY],
                         [0, 0, 0])
        for g, n in zip(gap, normal):
            self.assertEqual(g.train_texts, n.train_texts)
            self.assertEqual(g.threshold_texts, n.threshold_texts)
            self.assertEqual(g.test_texts, n.test_texts)


class ComparisonRemovalTest(unittest.TestCase):
    def test_training_negatives_drop_the_classes_training_queries(self):
        for fold, data in enumerate(gap_data(arms=(COMPARISON,))):
            negatives = [t for t, y in zip(data.train_texts, data.train_y) if y == 0]
            self.assertEqual(negatives, training_queries_kept(fold))
            self.assertEqual(data.threshold_texts, [])
            self.assertNotIn("products_removed", data.report)

    def test_no_removed_query_is_in_any_fit_or_scored_for_the_threshold(self):
        for fold, data in enumerate(gap_data(arms=(COMPARISON,))):
            gone = set(FOLDS[fold].train_queries) - set(training_queries_kept(fold))
            factory = Factory()
            run_comparison_fold(data, factory, 0.5, "lr", gap=True)
            self.assertEqual(len(factory.made), 6)  # five inner fits and the refit
            for scorer in factory.made:
                self.assertFalse(gone & set(scorer.fit_texts))
            # the threshold set is the out-of-fold scores of the kept queries: none removed
            self.assertEqual(
                sum(1 for t, y in zip(data.train_texts, data.train_y) if y == 0),
                len(training_queries_kept(fold)))

    def test_empty_class_queries_stay_in_the_training_negatives(self):
        for fold, data in enumerate(gap_data(arms=(COMPARISON,))):
            kept = {t for t, y in zip(data.train_texts, data.train_y) if y == 0}
            empties = {q for q in FOLDS[fold].train_queries if CLASS_OF[q] == ""}
            self.assertLessEqual(empties, kept)

    def test_the_two_arms_share_the_test_side(self):
        data = gap_data()
        by_fold = {}
        for d in data:
            by_fold.setdefault(d.fold, {})[d.arm] = d
        for pair in by_fold.values():
            for field in ("test_texts", "test_y", "test_families", "test_groups"):
                self.assertEqual(getattr(pair[PRIMARY], field), getattr(pair[COMPARISON], field))


class ExhaustedPoolTest(unittest.TestCase):
    def test_too_few_products_left_raises_and_never_pads(self):
        everything = [WandsProduct(p.product_id, p.name, ("a", "b", "c")) for p in PRODUCTS]
        one_left = [WandsProduct(0, PRODUCTS[0].name, ())] + everything[1:]
        with self.assertRaises(PoolExhausted) as ctx:
            gap_data(arms=(PRIMARY,), products=one_left)
        self.assertLess(ctx.exception.reached, 120)

    def test_no_product_left_at_all_raises(self):
        everything = [WandsProduct(p.product_id, p.name, ("a", "b", "c")) for p in PRODUCTS]
        with self.assertRaises(PoolExhausted):
            gap_data(arms=(PRIMARY,), products=everything)

    def test_the_same_pool_without_a_gap_does_not_raise(self):
        everything = [WandsProduct(p.product_id, p.name, ("a", "b", "c")) for p in PRODUCTS]
        run(arms=(PRIMARY,), repeats=1, products=everything)


class QueryClassesTest(unittest.TestCase):
    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            gap_data(query_classes=QUERY_CLASSES[:-1])


class ScoreSetNamesTest(unittest.TestCase):
    def test_nan_names_the_gap_score_set_in_both_arms(self):
        for arm, score_set in ((PRIMARY, "threshold queries"), (COMPARISON, "out-of-fold scores")):
            data = gap_data(arms=(arm,))[0]
            queries = (data.threshold_texts if arm == PRIMARY
                       else [t for t, y in zip(data.train_texts, data.train_y) if y == 0])
            with self.assertRaises(NonFiniteScoreError) as ctx:
                RUNNERS[arm](data, Factory({queries[0]: float("nan")}), 0.5, "lr", gap=True)
            self.assertIn(f"catalogue gap {score_set}", str(ctx.exception))
            self.assertIn(f"arm {arm}", str(ctx.exception))

    def test_nan_in_each_test_score_set_names_it(self):
        for arm in (PRIMARY, COMPARISON):
            data = gap_data(arms=(arm,))[0]
            for score_set, wanted in (("test queries", None), ("E", "E"),
                                      ("test positives", "A")):
                text = next(t for t, f in zip(data.test_texts, data.test_families) if f == wanted)
                with self.assertRaises(NonFiniteScoreError) as ctx:
                    RUNNERS[arm](data, Factory({text: float("nan")}), 0.5, "lr", gap=True)
                self.assertIn(f"catalogue gap {score_set}", str(ctx.exception))

    def test_the_main_run_names_are_unchanged(self):
        data = gap_data(arms=(PRIMARY,))[0]
        with self.assertRaises(NonFiniteScoreError) as ctx:
            run_primary_fold(data, Factory({data.threshold_texts[0]: float("nan")}), 0.5, "lr")
        self.assertIn("score in threshold queries", str(ctx.exception))

    def test_nan_only_in_the_gap_pass_stops_the_run_naming_it(self):
        def late_nan_factory(main_fits):
            made = []

            def factory():
                scorer = FakeScorer()
                nan = len(made) >= main_fits  # fits after the main run's are gap fits
                made.append(scorer)
                plain = scorer.score
                scorer.score = lambda texts: np.full(len(texts), np.nan) if nan else plain(texts)
                return scorer
            return factory

        for arm, main_fits in ((PRIMARY, 3), (COMPARISON, 18)):
            with self.assertRaises(NonFiniteScoreError) as ctx:
                build(models={"lr": late_nan_factory(main_fits)}, arms=(arm,),
                      catalogue_gap=True)
            self.assertIn("catalogue gap", str(ctx.exception))
            self.assertIn(f"arm {arm}", str(ctx.exception))
            build(models={"lr": late_nan_factory(main_fits)}, arms=(arm,))  # does not raise


def build(models=None, **kw):
    return build_run_report(**{
        **OPTIONS, "real_queries": REAL, "products": CLASSED, "part_numbers": [],
        "sanity": [], "models": models or {"lr": LengthScorer},
        "registry": build_registry(), "query_classes": QUERY_CLASSES, **kw})


def result(texts, scores, families, threshold, repeat=0, fold=0, arm=PRIMARY):
    return FoldResult(
        repeat, fold, arm, "lr", Threshold(threshold, 0.0, False, None),
        np.array(scores, dtype=float), texts, [0 if f is None else 1 for f in families],
        families, [None if f is None else (f, "t0") for f in families], [])


def pooled_results(flagged_per_fold, arm=PRIMARY, negatives=96):
    """3 repeats of 5 folds with `negatives` queries each (n = 480 per repeat); the folds
    named in `flagged_per_fold` ({(repeat, fold): count}) have that many flagged queries."""
    results = []
    for repeat in range(3):
        for fold in range(5):
            hits = flagged_per_fold.get((repeat, fold), 0)
            texts = [f"q{repeat}-{fold}-{i}" for i in range(negatives)] + ["pos a", "pos b"]
            scores = [0.9 if i < hits else 0.1 for i in range(negatives)] + [0.9, 0.1]
            results.append(result(texts, scores, [None] * negatives + ["A", "A"], 0.5,
                                  repeat, fold, arm))
    return results


FLAGGED = {(0, 0): 3, (1, 2): 4, (2, 4): 5}  # 12 flagged in 1,440 queries


class SectionTest(unittest.TestCase):
    def test_hand_computed_pooled_fpr_uses_a_third_of_the_total_at_n_480(self):
        section = catalogue_gap_section(pooled_results(FLAGGED), 0.01)
        low, high = wilson_interval(12 / 3, 480)
        self.assertEqual(section["fpr"], {
            "false_positives": 12, "n_negatives": 480, "repeats": 3, "point": 12 / 1440,
            "ci_low": low, "ci_high": high})
        self.assertNotEqual((low, high), wilson_interval(12, 1440))

    def test_recall_and_flagged_queries(self):
        section = catalogue_gap_section(pooled_results(FLAGGED), 0.01)
        self.assertEqual(section["recall"], {"count": 30, "caught": 15, "recall": 0.5})
        self.assertEqual(len(section["flagged_queries"]), 12)
        self.assertEqual(section["flagged_queries"][0], {"query": "q0-0-0", "repeat": 0, "fold": 0})
        self.assertEqual({q["repeat"] for q in section["flagged_queries"]}, {0, 1, 2})

    def test_gap_failed_at_the_lower_bound_just_above_exactly_at_and_just_below_target(self):
        low = wilson_interval(4, 480)[0]
        results = pooled_results(FLAGGED)
        self.assertTrue(catalogue_gap_section(results, low - 1e-9)["gap_failed"])
        self.assertFalse(catalogue_gap_section(results, low)["gap_failed"])
        self.assertFalse(catalogue_gap_section(results, low + 1e-9)["gap_failed"])

    def test_a_pass_with_the_point_estimate_above_target_is_reported(self):
        results = pooled_results(FLAGGED)  # point 0.83%, lower bound 0.33%
        passed = catalogue_gap_section(results, 0.005)
        self.assertEqual((passed["gap_failed"], passed["point_estimate_above_target"]),
                         (False, True))
        failed = catalogue_gap_section(results, 0.001)
        self.assertEqual((failed["gap_failed"], failed["point_estimate_above_target"]),
                         (True, False))
        clean = catalogue_gap_section(results, 0.02)
        self.assertEqual((clean["gap_failed"], clean["point_estimate_above_target"]),
                         (False, False))

    def test_the_section_holds_the_target(self):
        self.assertEqual(catalogue_gap_section(pooled_results(FLAGGED), 0.01)["target_fpr"], 0.01)

    def test_values_are_plain_python_types(self):
        section = catalogue_gap_section(pooled_results(FLAGGED), 0.01)
        self.assertIs(type(section["gap_failed"]), bool)
        self.assertIs(type(section["point_estimate_above_target"]), bool)

    def test_comparison_results_work_the_same(self):
        section = catalogue_gap_section(pooled_results(FLAGGED, COMPARISON), 0.01)
        self.assertEqual(section["fpr"]["false_positives"], 12)


class ReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parameters, cls.body = build(catalogue_gap=True)

    def test_parameters_record_the_flag(self):
        self.assertIs(self.parameters["catalogue_gap"], True)
        self.assertIs(build(arms=(PRIMARY,))[0]["catalogue_gap"], False)

    def test_off_by_default(self):
        _, body = build()
        for arm in (PRIMARY, COMPARISON):
            self.assertNotIn("catalogue_gap", body[arm])

    def test_requires_the_classes(self):
        with self.assertRaises(ValueError):
            build(catalogue_gap=True, query_classes=None)

    def test_classes_alone_do_nothing(self):
        _, body = build(arms=(PRIMARY,))
        self.assertNotIn("catalogue_gap", body[PRIMARY])

    def test_a_section_per_arm_and_model_with_the_result_keys(self):
        for arm in (PRIMARY, COMPARISON):
            section = self.body[arm]["catalogue_gap"]["models"]["lr"]
            self.assertEqual(set(section), {
                "target_fpr", "fpr", "recall", "flagged_queries", "gap_failed",
                "point_estimate_above_target"})
            self.assertEqual(section["fpr"]["n_negatives"], len(REAL))

    def test_primary_records_products_removed_per_fold_and_the_unclassified_count(self):
        gap = self.body[PRIMARY]["catalogue_gap"]
        self.assertEqual(gap["folds"], [
            {"repeat": 0, "fold": k, "products_removed": REMOVED[k]} for k in range(3)])
        self.assertEqual(gap["unclassified_products"], UNCLASSIFIED)
        comparison = self.body[COMPARISON]["catalogue_gap"]
        self.assertNotIn("folds", comparison)
        self.assertNotIn("unclassified_products", comparison)

    def test_both_arms_are_paired_on_the_same_folds(self):
        sizes = {arm: self.body[arm]["catalogue_gap"]["models"]["lr"]["fpr"]["n_negatives"]
                 for arm in (PRIMARY, COMPARISON)}
        self.assertEqual(sizes[PRIMARY], sizes[COMPARISON])
        self.assertEqual(sizes[PRIMARY], len(REAL))

    def test_the_section_is_the_one_computed_from_the_gap_fold_results(self):
        results = []
        for data in gap_data(arms=(PRIMARY,)):
            results.append(run_primary_fold(data, LengthScorer, 0.5, "lr", gap=True))
        self.assertEqual(self.body[PRIMARY]["catalogue_gap"]["models"]["lr"],
                         catalogue_gap_section(results, 0.5))

    def test_the_main_sections_are_unchanged_by_the_gap_pass(self):
        _, plain = build()
        for arm in (PRIMARY, COMPARISON):
            self.assertEqual(plain[arm]["folds"], self.body[arm]["folds"])
            self.assertEqual(plain[arm]["models"], self.body[arm]["models"])

    def test_a_failed_gap_does_not_stop_the_run_or_change_anything_else(self):
        _, body = build(target_fpr=0.0001, catalogue_gap=True)
        for arm in (PRIMARY, COMPARISON):
            self.assertIn("gap_failed", body[arm]["catalogue_gap"]["models"]["lr"])

    def test_fit_counts(self):
        factory = Factory(cls=LengthScorer)
        build(models={"lr": factory}, arms=(PRIMARY,), catalogue_gap=True)
        self.assertEqual(len(factory.made), 3 + 3)
        factory = Factory(cls=LengthScorer)
        build(models={"lr": factory}, arms=(COMPARISON,), catalogue_gap=True)
        self.assertEqual(len(factory.made), 18 + 18)


if __name__ == "__main__":
    unittest.main()
