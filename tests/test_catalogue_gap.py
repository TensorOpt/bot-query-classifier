import json
import math
import unittest
from collections import Counter
from functools import lru_cache
from unittest import mock

import numpy as np
from fakes import FakeScorer
from synth_stubs import build_registry
from test_folds import CONFIG, PRODUCTS, REAL, run
from test_lofo import VariedScorer
from test_run_report import walk_types

import bqc.evaluate
from bqc import SEED
from bqc.evaluate import (
    COMPARISON,
    PRIMARY,
    FoldAssignment,
    FoldResult,
    NonFiniteScoreError,
    Threshold,
    assign_folds,
    build_run_report,
    catalogue_gap_removal,
    catalogue_gap_section,
    wilson_interval,
)
from bqc.negatives import PoolExhausted, build_spans
from bqc.wands import WandsProduct

INNER = 5
FOLDS = 8  # 2 repeats x 4 outer folds; 4 splits keep the comparison arm's inner folds valid
CLASSES = 20
# Queries 3 and 17 have no class; query i and i + 20 share one.
QUERY_CLASSES = ["" if i in (3, 17) else f"K{i % CLASSES}" for i in range(len(REAL))]
CLASS_OF = dict(zip(REAL, QUERY_CLASSES))


def product_classes(i):
    if i % 11 == 0:
        return ()  # 28 of the 300 products have no class
    if i % 7 == 0:
        return (f"K{i % CLASSES}", f"K{(i + 1) % CLASSES}")  # two classes
    return (f"K{i % CLASSES}",)


GAP_PRODUCTS = [p._replace(classes=product_classes(p.product_id)) for p in PRODUCTS]
UNCLASSIFIED = len(range(0, 300, 11))
CONFIG4 = dict(CONFIG, n_splits=4)
REPORT_CONFIG = dict(CONFIG4, target_fpr=0.5)
# every word of a stub product name is its own (`pw` + digits as letters), so a word names its product
OWNER = {w: p for p in GAP_PRODUCTS for w in p.name.split()}
SANITY = ["zzz qqq"]
ARMS_BOTH = (PRIMARY, COMPARISON)


class Recorder:
    """A scorer factory keeping every scorer it made, with the texts each one scored;
    `nan_at` lists the fits whose scorer returns NaN everywhere."""

    def __init__(self, nan_at=()):
        self.nan_at = set(nan_at)
        self.made = []

    def __call__(self):
        nan = len(self.made) in self.nan_at

        class Scorer(VariedScorer):
            def __init__(self):
                super().__init__()
                self.scored = []

            def score(self, texts):
                self.scored.append(list(texts))
                scores = super().score(texts)
                return np.full(len(scores), np.nan) if nan else scores

        scorer = Scorer()
        self.made.append(scorer)
        return scorer


class AllFlagged(FakeScorer):
    def score(self, texts):
        return np.ones(len(texts))


def build(**kw):
    kw.setdefault("query_classes", QUERY_CLASSES)
    return build_run_report(
        **REPORT_CONFIG, real_queries=REAL, products=GAP_PRODUCTS, part_numbers=[], sanity=[],
        models=kw.pop("models", {"lr": VariedScorer}), registry=build_registry(), **kw)


def gap_run(**kw):
    return run(products=GAP_PRODUCTS, **{**CONFIG4, "query_classes": QUERY_CLASSES, **kw})


@lru_cache
def gap_datas():
    return gap_run()


@lru_cache
def plain_datas():
    return run(products=GAP_PRODUCTS, **CONFIG4)


@lru_cache
def gap_build():
    return build(catalogue_gap=True)


def assignments():
    return {(a.repeat, a.fold): a
            for a in assign_folds(REAL, build_registry(), n_splits=4, repeats=2)}


def gap_of(fold):
    return {CLASS_OF[q] for q in fold.test_queries} - {""}


def removed_products(gap):
    return [p for p in GAP_PRODUCTS if gap & set(p.classes)]


def negatives(data):
    return [t for t, y in zip(data.train_texts, data.train_y) if y == 0]


class RemovalTest(unittest.TestCase):
    def test_catalogue_gap_removal_by_hand(self):
        fold = FoldAssignment(0, 0, ["t1", "t2", "t3", "t4"], ["x", "y", "z"], [], [])
        class_of = {"t1": "X", "t2": "Y", "t3": "", "t4": "W", "x": "X", "y": "", "z": "Y"}
        products = [WandsProduct(1, "a", ("X",)), WandsProduct(2, "b", ("Y", "Z")),
                    WandsProduct(3, "c", ("Z",)), WandsProduct(4, "d", ()),
                    WandsProduct(5, "e", ("W", "X")), WandsProduct(6, "f", ("W",))]
        queries, kept, counts = catalogue_gap_removal(fold, class_of, products)
        # gap classes are X and Y: the empty class of y defines nothing
        self.assertEqual(queries, ["t3", "t4"])
        self.assertEqual([p.product_id for p in kept], [3, 4, 6])
        self.assertEqual(counts, {"products_removed": 3, "unclassified_products": 1,
                                  "training_queries_removed": 2})

    def test_an_empty_class_product_is_never_removed_and_an_empty_class_query_defines_none(self):
        fold = FoldAssignment(0, 0, ["t1"], ["x"], [], [])
        queries, kept, counts = catalogue_gap_removal(
            fold, {"t1": "", "x": ""}, [WandsProduct(1, "a", ()), WandsProduct(2, "b", ("X",))])
        self.assertEqual(queries, ["t1"])
        self.assertEqual(len(kept), 2)
        self.assertEqual(counts["products_removed"], 0)


class DataTest(unittest.TestCase):
    def test_off_by_default(self):
        for d in plain_datas():
            self.assertNotIn("catalogue_gap", d.report)

    def test_one_data_per_fold_and_arm_and_arms_share_the_folds(self):
        datas = gap_datas()
        self.assertEqual(len(datas), 2 * FOLDS)
        for primary, comparison in zip(datas[0::2], datas[1::2]):
            self.assertEqual((primary.arm, comparison.arm), (PRIMARY, COMPARISON))
            self.assertEqual((primary.repeat, primary.fold), (comparison.repeat, comparison.fold))
            self.assertEqual(primary.test_texts, comparison.test_texts)

    def test_primary_no_span_from_a_product_of_a_test_side_class(self):
        multi_class_cases = 0
        for d in gap_datas()[0::2]:
            gap = gap_of(assignments()[(d.repeat, d.fold)])
            gone = {p.product_id for p in removed_products(gap)}
            multi_class_cases += sum(1 for p in removed_products(gap)
                                     if len(p.classes) > 1 and set(p.classes) - gap)
            spans = negatives(d)
            self.assertEqual(len(spans), CONFIG["n"])
            owners = {OWNER[w].product_id for s in spans for w in s.split()}
            self.assertTrue(owners)
            self.assertFalse(owners & gone)
        self.assertGreater(multi_class_cases, 0)  # a product with one matching class was removed

    def test_primary_spans_come_from_the_products_left_and_differ_per_fold(self):
        by_repeat = {}
        for d in gap_datas()[0::2]:
            by_repeat.setdefault(d.repeat, []).append(tuple(negatives(d)))
            self.assertEqual(d.report["span_lengths"], dict(Counter(
                len(s.split()) for s in negatives(d))))
        for spans in by_repeat.values():
            self.assertEqual(len(set(spans)), len(spans))

    def test_the_removed_count_differs_per_fold(self):
        counts = {d.report["catalogue_gap"]["products_removed"] for d in gap_datas()[0::2]}
        self.assertGreater(len(counts), 1)

    def test_primary_report_counts_products_removed_and_unclassified(self):
        for d in gap_datas()[0::2]:
            fold = assignments()[(d.repeat, d.fold)]
            gap = gap_of(fold)
            entry = d.report["catalogue_gap"]
            self.assertEqual(entry["products_removed"], len(removed_products(gap)))
            self.assertGreater(entry["products_removed"], 0)
            self.assertEqual(entry["unclassified_products"], UNCLASSIFIED)
            self.assertEqual(entry["training_queries_removed"],
                             len([q for q in fold.train_queries if CLASS_OF[q] in gap]))

    def test_comparison_report_counts_queries_only(self):
        for d in gap_datas()[1::2]:
            self.assertEqual(set(d.report["catalogue_gap"]), {"training_queries_removed"})

    def test_threshold_set_and_comparison_negatives_drop_those_classes_queries(self):
        for primary, comparison in zip(gap_datas()[0::2], gap_datas()[1::2]):
            fold = assignments()[(primary.repeat, primary.fold)]
            gap = gap_of(fold)
            kept = [q for q in fold.train_queries if CLASS_OF[q] not in gap]
            self.assertLess(len(kept), len(fold.train_queries))
            self.assertEqual(primary.threshold_texts, kept)
            self.assertEqual(comparison.threshold_texts, [])
            self.assertEqual(negatives(comparison), kept)
            self.assertEqual(comparison.report["train"]["negatives"], len(kept))

    def test_empty_class_queries_stay_in_folds_thresholds_and_negatives(self):
        empty = {q for q, c in CLASS_OF.items() if not c}
        self.assertEqual(len(empty), 2)
        seen = 0
        for primary, comparison in zip(gap_datas()[0::2], gap_datas()[1::2]):
            train = set(assignments()[(primary.repeat, primary.fold)].train_queries)
            for q in empty & train:
                seen += 1
                self.assertIn(q, primary.threshold_texts)
                self.assertIn(q, negatives(comparison))
        self.assertGreater(seen, 0)

    def test_test_side_queries_are_the_normal_runs(self):
        for gap, plain in zip(gap_datas(), plain_datas()):
            self.assertEqual((gap.repeat, gap.fold, gap.arm), (plain.repeat, plain.fold, plain.arm))
            queries = [t for t, f in zip(gap.test_texts, gap.test_families) if f is None]
            self.assertEqual(queries, assignments()[(gap.repeat, gap.fold)].test_queries)
            self.assertEqual(queries, [t for t, f in zip(plain.test_texts, plain.test_families)
                                       if f is None])
            self.assertTrue(any(f is not None for f in gap.test_families))  # and positives

    def test_the_normal_runs_spans_are_unaffected(self):
        gap_datas()
        for repeat in range(2):
            spans, _ = build_spans(GAP_PRODUCTS, CONFIG["n"], SEED + repeat, REAL, [], [])
            for d in plain_datas():
                if d.repeat == repeat and d.arm == PRIMARY:
                    self.assertEqual(negatives(d), [s.text for s in spans])
                self.assertNotIn("catalogue_gap", d.report)

    def test_query_classes_of_the_wrong_length_raise(self):
        with self.assertRaises(ValueError):
            gap_run(query_classes=QUERY_CLASSES[:-1])


class SpanBuildWiringTest(unittest.TestCase):
    def build_calls(self, arms):
        real = bqc.evaluate.build_spans
        with mock.patch("bqc.evaluate.build_spans", wraps=real) as spy:
            datas = gap_run(arms=arms, sanity=SANITY, part_numbers=["PN-1"])
        return spy.call_args_list, datas

    def test_primary_builds_spans_per_fold_from_the_products_left_by_keyword(self):
        calls, datas = self.build_calls((PRIMARY, COMPARISON))
        self.assertEqual(len(calls), 2 + FOLDS)  # one per repeat as usual, one per fold
        per_fold = [c for c in calls if not c.args]
        self.assertEqual(len(per_fold), FOLDS)
        for call, d in zip(per_fold, [d for d in datas if d.arm == PRIMARY]):
            gap = gap_of(assignments()[(d.repeat, d.fold)])
            kw = call.kwargs
            self.assertEqual(kw["products"], [p for p in GAP_PRODUCTS if not gap & set(p.classes)])
            self.assertEqual(kw["n"], CONFIG["n"])
            self.assertEqual(kw["seed"], SEED + d.repeat)
            self.assertEqual(list(kw["real_queries"]), REAL)
            self.assertEqual(list(kw["part_numbers"]), ["PN-1"])
            self.assertEqual(list(kw["sanity"]), SANITY)

    def test_comparison_only_builds_no_per_fold_spans(self):
        calls, _ = self.build_calls((COMPARISON,))
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(c.args for c in calls))

    def test_the_test_side_is_screened_against_the_folds_own_spans_too(self):
        real = bqc.evaluate._render_screened
        with mock.patch("bqc.evaluate._render_screened", wraps=real) as spy:
            datas = gap_run(arms=(PRIMARY,))
        # per fold: test templates, E, then the training positives
        for i, d in enumerate(datas):
            test_call = spy.call_args_list[3 * i]
            drop = test_call.args[-1]["training_negatives"]
            self.assertTrue(set(negatives(d)) <= set(drop))


class SectionTest(unittest.TestCase):
    def fold_result(self, repeat, fold, flagged, arm=PRIMARY):
        texts = [f"q{repeat}-{fold}-{i}" for i in range(96)] + ["p1", "p2"]
        scores = [1.0] * flagged + [0.0] * (96 - flagged) + [1.0, 0.0]
        return FoldResult(
            repeat, fold, arm, "lr", Threshold(0.5, 0.0, False, None), np.array(scores), texts,
            [0] * 96 + [1, 1], [None] * 96 + ["A", "A"],
            [None] * 96 + [("A", "t0")] * 2, [])

    def results(self, flagged_folds=12):
        return [self.fold_result(r, k, 1 if 5 * r + k < flagged_folds else 0)
                for r in range(3) for k in range(5)]

    def test_pooled_fpr_uses_total_over_three_at_n_480(self):
        section = catalogue_gap_section(self.results(), 0.01)
        fpr = section["fpr"]
        # 12 false positives in 3 repeats of 480 queries: k = 12 / 3 = 4, not 12
        self.assertEqual((fpr["false_positives"], fpr["n_negatives"], fpr["repeats"]), (12, 480, 3))
        self.assertAlmostEqual(fpr["point"], 12 / 1440)
        # Wilson by hand: p = 1/120, centre 0.012237, half-width 0.008992
        self.assertAlmostEqual(fpr["ci_low"], 0.003245, places=5)
        self.assertAlmostEqual(fpr["ci_high"], 0.021229, places=5)
        low, high = wilson_interval(4, 480)
        self.assertEqual((fpr["ci_low"], fpr["ci_high"]), (low, high))
        self.assertEqual(section["recall"]["recall"], 0.5)
        self.assertEqual(len(section["flagged"]), 12)
        self.assertEqual(section["target_fpr"], 0.01)

    def test_gap_failed_when_the_lower_bound_is_just_above_exactly_at_or_just_below(self):
        low = catalogue_gap_section(self.results(), 0.01)["fpr"]["ci_low"]
        failed = {
            "just above": (math.nextafter(low, 0.0), True),  # target just below the bound
            "exactly at": (low, False),
            "just below": (math.nextafter(low, 1.0), False),
        }
        for name, (target, expected) in failed.items():
            with self.subTest(name):
                self.assertIs(catalogue_gap_section(self.results(), target)["gap_failed"], expected)

    def test_a_pass_with_the_point_estimate_above_target_is_reported_as_such(self):
        # point 0.00833: lower bound 0.00325 is under 0.005, so a pass above target
        section = catalogue_gap_section(self.results(), 0.005)
        self.assertEqual((section["gap_failed"], section["point_estimate_above_target"]),
                         (False, True))

    def test_a_pass_under_target_and_a_failure_are_not_point_estimate_above_target(self):
        under = catalogue_gap_section(self.results(), 0.01)
        self.assertEqual((under["gap_failed"], under["point_estimate_above_target"]),
                         (False, False))
        failed = catalogue_gap_section(self.results(), 0.001)
        self.assertEqual((failed["gap_failed"], failed["point_estimate_above_target"]),
                         (True, False))

    def test_plain_python_types(self):
        section = catalogue_gap_section(self.results(), 0.01)
        bad = [(p, t) for p, t in walk_types(section)
               if t not in {dict, list, str, int, float, bool, type(None)}]
        self.assertEqual(bad, [])
        self.assertEqual(json.loads(json.dumps(section)), section)


class ReportTest(unittest.TestCase):
    def test_off_by_default_and_recorded_in_the_parameters(self):
        parameters, body = build(catalogue_gap=False)
        self.assertIs(parameters["catalogue_gap"], False)
        for arm in (PRIMARY, COMPARISON):
            self.assertNotIn("catalogue_gap", body[arm])
        self.assertIs(gap_build()[0]["catalogue_gap"], True)

    def test_section_per_arm_and_model_primary_alone_has_the_product_counts(self):
        _, body = gap_build()
        for arm in (PRIMARY, COMPARISON):
            section = body[arm]["catalogue_gap"]
            self.assertEqual(set(section["models"]), {"lr"})
            self.assertEqual(
                set(section["models"]["lr"]),
                {"target_fpr", "recall", "fpr", "flagged", "gap_failed",
                 "point_estimate_above_target"})
            self.assertEqual(section["models"]["lr"]["fpr"]["n_negatives"], len(REAL))
        self.assertNotIn("products", body[COMPARISON]["catalogue_gap"])
        products = body[PRIMARY]["catalogue_gap"]["products"]
        self.assertEqual(products["unclassified"], UNCLASSIFIED)
        self.assertEqual(
            products["removed_per_fold"],
            [{"repeat": d.repeat, "fold": d.fold,
              "removed": len(removed_products(gap_of(assignments()[(d.repeat, d.fold)])))}
             for d in gap_datas()[0::2]])

    def test_the_arms_pool_the_same_queries(self):
        _, body = gap_build()
        pooled = {arm: body[arm]["catalogue_gap"]["models"]["lr"]["fpr"] for arm in ARMS_BOTH}
        self.assertEqual(pooled[PRIMARY]["n_negatives"], pooled[COMPARISON]["n_negatives"])
        self.assertEqual(pooled[PRIMARY]["repeats"], 2)
        for arm in ARMS_BOTH:
            for hit in body[arm]["catalogue_gap"]["models"]["lr"]["flagged"]:
                self.assertIn(hit["query"], REAL)

    def test_the_headline_and_fold_reports_are_unchanged_by_the_check(self):
        _, plain = build(catalogue_gap=False)
        _, body = gap_build()
        for arm in ARMS_BOTH:
            self.assertEqual(body[arm]["models"], plain[arm]["models"])
            self.assertEqual(body[arm]["folds"], plain[arm]["folds"])
        self.assertEqual(body[PRIMARY]["span_lengths"], plain[PRIMARY]["span_lengths"])

    def test_plain_python_types_only(self):
        _, body = gap_build()
        bad = [(p, t) for p, t in walk_types(body)
               if t not in {dict, list, str, int, float, bool, type(None)}]
        self.assertEqual(bad, [])
        self.assertEqual(json.loads(json.dumps(body)), body)

    def test_a_failed_check_does_not_stop_the_run_or_change_anything(self):
        _, body = build(catalogue_gap=True, models={"flag": AllFlagged})
        for arm in ARMS_BOTH:
            section = body[arm]["catalogue_gap"]["models"]["flag"]
            self.assertIs(section["gap_failed"], True)
            self.assertEqual(section["fpr"]["point"], 1.0)
            self.assertIn("pooled", body[arm]["models"]["flag"])

    def test_the_gap_pass_gets_the_same_options_as_the_main_pass_without_lofo(self):
        real = bqc.evaluate.iter_fold_data
        with mock.patch("bqc.evaluate.iter_fold_data", wraps=real) as spy:
            build_run_report(
                **REPORT_CONFIG, real_queries=REAL, products=GAP_PRODUCTS,
                part_numbers=["PN-1"], sanity=SANITY, models={"lr": VariedScorer},
                registry=build_registry(), catalogue_gap=True, query_classes=QUERY_CLASSES,
                arms=[PRIMARY], vocabulary_cap=7)
        self.assertEqual(len(spy.call_args_list), 2)
        main, gap = (c.kwargs for c in spy.call_args_list)
        self.assertEqual(spy.call_args_list[0].args, ())
        self.assertEqual(spy.call_args_list[1].args, ())
        self.assertNotIn("query_classes", main)
        self.assertEqual(gap.pop("query_classes"), QUERY_CLASSES)
        self.assertEqual(gap, main)
        self.assertIs(main["lofo"], False)
        self.assertEqual(main["vocabulary_cap"], 7)
        self.assertEqual(main["sanity"], SANITY)
        self.assertEqual(list(main["arms"]), [PRIMARY])

    def test_catalogue_gap_without_query_classes_raises(self):
        with self.assertRaises(ValueError):
            build(catalogue_gap=True, query_classes=None)

    def test_query_classes_of_the_wrong_length_raise_before_any_fit(self):
        recorder = Recorder()
        with self.assertRaises(ValueError):
            build(catalogue_gap=True, query_classes=QUERY_CLASSES[:-1],
                  models={"lr": recorder})
        self.assertEqual(recorder.made, [])


class PoolTest(unittest.TestCase):
    def test_every_product_removed_raises_and_never_pads(self):
        every = tuple(f"K{i}" for i in range(CLASSES))
        products = [p._replace(classes=every) for p in PRODUCTS]
        with self.assertRaises(PoolExhausted):
            run(products=products, query_classes=QUERY_CLASSES, **CONFIG4)

    def test_too_few_products_left_raises_with_the_count_reached(self):
        every = tuple(f"K{i}" for i in range(CLASSES))
        # one unclassified product cannot give 120 distinct spans
        products = [p._replace(classes=every if p.product_id else ()) for p in PRODUCTS]
        with self.assertRaises(PoolExhausted) as ctx:
            run(products=products, query_classes=QUERY_CLASSES, **CONFIG4)
        self.assertLess(ctx.exception.reached, CONFIG["n"])

    def test_the_report_does_not_swallow_it(self):
        every = tuple(f"K{i}" for i in range(CLASSES))
        products = [p._replace(classes=every) for p in PRODUCTS]
        with self.assertRaises(PoolExhausted):
            build_run_report(
                **REPORT_CONFIG, real_queries=REAL, products=products, part_numbers=[],
                sanity=[], models={"lr": VariedScorer}, registry=build_registry(),
                catalogue_gap=True, query_classes=QUERY_CLASSES, arms=[PRIMARY])


class FitTest(unittest.TestCase):
    def fits(self, arm, catalogue_gap):
        recorder = Recorder()
        build(catalogue_gap=catalogue_gap, models={"lr": recorder}, arms=[arm])
        return recorder

    def test_gap_fit_counts(self):
        self.assertEqual(len(self.fits(PRIMARY, False).made), FOLDS)
        self.assertEqual(len(self.fits(PRIMARY, True).made), 2 * FOLDS)
        self.assertEqual(len(self.fits(COMPARISON, False).made), FOLDS * (INNER + 1))
        self.assertEqual(len(self.fits(COMPARISON, True).made), 2 * FOLDS * (INNER + 1))

    def test_a_new_scorer_per_fit(self):
        for arm in ARMS_BOTH:
            recorder = self.fits(arm, True)
            self.assertEqual(len({id(s) for s in recorder.made}), len(recorder.made))

    def test_comparison_no_removed_query_in_any_fit_or_scored_set(self):
        recorder = self.fits(COMPARISON, True)
        gap_fits = recorder.made[FOLDS * (INNER + 1):]
        datas = gap_datas()[1::2]
        self.assertEqual(len(gap_fits), len(datas) * (INNER + 1))
        for i, d in enumerate(datas):
            fold = assignments()[(d.repeat, d.fold)]
            gap = gap_of(fold)
            removed = {q for q in fold.train_queries if CLASS_OF[q] in gap}
            kept = [q for q in fold.train_queries if CLASS_OF[q] not in gap]
            self.assertTrue(removed)
            scorers = gap_fits[i * (INNER + 1):(i + 1) * (INNER + 1)]
            scored = set()
            for scorer in scorers:
                self.assertFalse(removed & set(scorer.fit_texts))
                self.assertTrue(scorer.fit_texts)
                for texts in scorer.scored:
                    self.assertFalse(removed & set(texts))
                    scored |= set(texts)
            self.assertTrue(set(kept) <= scored)  # each kept query is scored out of fold

    def test_primary_threshold_set_and_fits_leave_out_the_gap(self):
        recorder = self.fits(PRIMARY, True)
        gap_fits = recorder.made[FOLDS:]
        for scorer, d in zip(gap_fits, gap_datas()[0::2]):
            fold = assignments()[(d.repeat, d.fold)]
            gap = gap_of(fold)
            gone_words = {w for p in removed_products(gap) for w in p.name.split()}
            fit_negatives = [t for t, y in zip(scorer.fit_texts, scorer.fit_y) if y == 0]
            self.assertFalse(gone_words & {w for t in fit_negatives for w in t.split()})
            kept = [q for q in fold.train_queries if CLASS_OF[q] not in gap]
            self.assertEqual(scorer.scored[0], kept)  # the threshold queries come first
            self.assertFalse({q for q in fold.train_queries if CLASS_OF[q] in gap}
                             & set(scorer.scored[0]))


class NonFiniteTest(unittest.TestCase):
    def raises(self, arm, nan_at):
        with self.assertRaises(NonFiniteScoreError) as ctx:
            build(catalogue_gap=True, models={"lr": Recorder(nan_at=[nan_at])}, arms=[arm])
        return str(ctx.exception)

    def test_primary_nan_in_gap_scores_names_the_catalogue_gap_score_set(self):
        message = self.raises(PRIMARY, FOLDS + 1)  # the second gap fit
        self.assertIn("catalogue gap threshold queries", message)
        for part in ("repeat 0", "fold 1", "arm product", "model lr"):
            self.assertIn(part, message)

    def test_comparison_nan_in_a_gap_refit_names_the_catalogue_gap_score_set(self):
        message = self.raises(COMPARISON, FOLDS * (INNER + 1) + INNER)  # first gap refit
        self.assertIn("catalogue gap test queries", message)
        self.assertIn("arm real", message)

    def test_comparison_nan_in_a_gap_inner_fit_names_out_of_fold_scores(self):
        message = self.raises(COMPARISON, FOLDS * (INNER + 1) + 1)
        self.assertIn("catalogue gap out-of-fold scores", message)

    def test_the_main_run_keeps_its_own_names(self):
        for arm in ARMS_BOTH:
            message = self.raises(arm, 0)
            self.assertNotIn("catalogue gap", message)


if __name__ == "__main__":
    unittest.main()
