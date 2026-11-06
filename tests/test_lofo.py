import json
import unittest
from unittest import mock

import numpy as np
from fakes import FakeScorer
from synth_stubs import build_registry
from test_folds import CONFIG, PRODUCTS, REAL, run
from test_run_report import walk_types
from test_runner import Factory
from test_runner import make_data as make_primary_data
from test_runner_comparison import make_data as make_comparison_data

import bqc.evaluate
from bqc.evaluate import (
    COMPARISON,
    PRIMARY,
    FoldError,
    FoldResult,
    NonFiniteScoreError,
    Threshold,
    build_run_report,
    check_fold,
    lofo_pool,
    lofo_row,
    run_comparison_fold,
    run_primary_fold,
)
from bqc.negatives import token_occurrences
from bqc.synth.registry import EVAL_FAMILY, TRAINING_FAMILIES

INNER = 5


class VariedScorer(FakeScorer):
    """FakeScorer plus a length term, so the threshold queries have varied scores."""

    def score(self, texts):
        return super().score(texts) + np.array([(len(t) % 10) / 10 for t in texts])


class Recorder:
    """A scorer factory that keeps every scorer it made, in order; `nan_at` lists the
    indexes of the fits whose scorer returns NaN everywhere."""

    def __init__(self, nan_at=()):
        self.nan_at = set(nan_at)
        self.made = []

    def __call__(self):
        nan = len(self.made) in self.nan_at

        class Scorer(VariedScorer):
            def score(self, texts):
                scores = super().score(texts)
                return np.full(len(scores), np.nan) if nan else scores

        scorer = Scorer()
        self.made.append(scorer)
        return scorer


def remaining(family):
    return [f for f in TRAINING_FAMILIES if f != family]


def split(datas):
    """(main, {family: lofo}) per (repeat, fold, arm), in yield order."""
    out = []
    for d in datas:
        if d.lofo_family is None:
            out.append((d, {}))
        else:
            out[-1][1][d.lofo_family] = d
    return out


# 4 outer splits: with 3, an inner training fold of the comparison arm can lose a family
# once one is left out, on this tiny stub registry (the real data doesn't, M3.T13).
# 2 repeats x 4 folds = 8 outer folds per arm.
REPORT_CONFIG = dict(CONFIG, n_splits=4, target_fpr=0.5)
FOLDS = 8


def build(**kw):
    return build_run_report(
        **REPORT_CONFIG, real_queries=REAL, products=PRODUCTS,
        part_numbers=[], sanity=[], models=kw.pop("models", {"lr": VariedScorer}),
        registry=build_registry(), **kw)


def result(texts, scores, y, families, t, arm=PRIMARY, groups=None):
    groups = groups or [None if f is None else (f, "t0") for f in families]
    return FoldResult(0, 1, arm, "lr", Threshold(t, 0.0, False, None), np.array(scores),
                      list(texts), list(y), list(families), groups, [])


# main: 3 queries and 5 F positives (A) plus one B and one E row, which a row must ignore.
MAIN = result(
    ["q1", "q2", "q3", "a1", "a2", "a3", "a4", "a5", "b1", "e1"],
    [0.1, 0.3, 0.6, 0.9, 0.5, 0.4, 0.3, 0.1, 0.9, 0.9],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    [None] * 3 + ["A"] * 5 + ["B", EVAL_FAMILY], t=0.3)
# LOFO of A: the same queries and A strings, other scores, a higher threshold.
LOFO = result(
    ["q1", "q2", "q3", "a1", "a2", "a3", "a4", "a5"],
    [0.1, 0.5, 0.6, 0.9, 0.5, 0.4, 0.3, 0.1],
    [0, 0, 0, 1, 1, 1, 1, 1],
    [None] * 3 + ["A"] * 5, t=0.5)


class RowTest(unittest.TestCase):
    def test_hand_computed_recall_fpr_and_paired_difference(self):
        row = lofo_row(LOFO, "A", MAIN)
        # LOFO: scores >= 0.5 on A are 0.9 and 0.5, so 2 of 5; queries 0.5 and 0.6, so 2 of 3
        self.assertEqual(row["recall"], {"count": 5, "caught": 2, "recall": 0.4})
        self.assertEqual(row["fpr"]["false_positives"], 2)
        self.assertEqual(row["fpr"]["n_negatives"], 3)
        self.assertAlmostEqual(row["fpr"]["point"], 2 / 3)
        # main: scores >= 0.3 on A are 0.9, 0.5, 0.4, 0.3, so 4 of 5; B and E are ignored
        self.assertEqual(row["main_recall"], {"count": 5, "caught": 4, "recall": 0.8})
        self.assertAlmostEqual(row["recall_difference"], 0.4 - 0.8)
        self.assertEqual((row["repeat"], row["fold"], row["threshold"]), (0, 1, 0.5))
        self.assertIs(row["target_unmet"], False)
        self.assertEqual(row["fit_warnings"], [])

    def test_a_score_exactly_at_t_is_flagged(self):
        row = lofo_row(LOFO, "A", MAIN)
        self.assertEqual(row["recall"]["caught"], 2)  # a2 scores 0.5 = t
        self.assertEqual(row["fpr"]["false_positives"], 2)  # q2 scores 0.5 = t

    def test_family_absent_from_the_test_side_gives_null_recall_and_still_counts_fpr(self):
        lofo = result(["q1", "q2", "q3"], [0.1, 0.5, 0.6], [0, 0, 0], [None] * 3, t=0.5)
        main = result(["q1", "q2", "q3", "b1"], [0.1, 0.3, 0.6, 0.9], [0, 0, 0, 1],
                      [None] * 3 + ["B"], t=0.3)
        row = lofo_row(lofo, "A", main)
        self.assertEqual(row["recall"], {"count": 0, "caught": 0, "recall": None})
        self.assertEqual(row["main_recall"], {"count": 0, "caught": 0, "recall": None})
        self.assertIsNone(row["recall_difference"])
        self.assertEqual(row["fpr"]["false_positives"], 2)
        self.assertEqual(row["fpr"]["n_negatives"], 3)

    def test_plain_python_types(self):
        row = lofo_row(LOFO, "A", MAIN)
        bad = [(p, t) for p, t in walk_types(row)
               if t not in {dict, list, str, int, float, bool, type(None)}]
        self.assertEqual(bad, [])
        self.assertEqual(json.loads(json.dumps(row)), row)


class PoolTest(unittest.TestCase):
    def test_pooled_is_caught_over_total_not_a_mean_of_folds(self):
        small = lofo_row(LOFO, "A", MAIN)  # 2 of 5, main 4 of 5, fpr 2 of 3
        big_lofo = result(["q1", "a1", "a2"], [0.1, 0.9, 0.9], [0, 1, 1],
                          [None, "A", "A"], t=0.5)
        big_main = result(["q1", "a1", "a2"], [0.1, 0.9, 0.2], [0, 1, 1],
                          [None, "A", "A"], t=0.5)
        big = lofo_row(big_lofo, "A", big_main)  # 2 of 2, main 1 of 2, fpr 0 of 1
        pooled = lofo_pool([small, big])
        self.assertEqual(pooled["recall"], {"count": 7, "caught": 4, "recall": 4 / 7})
        self.assertEqual(pooled["main_recall"], {"count": 7, "caught": 5, "recall": 5 / 7})
        self.assertAlmostEqual(pooled["recall_difference"], 4 / 7 - 5 / 7)
        self.assertEqual(pooled["fpr"]["false_positives"], 2)
        self.assertEqual(pooled["fpr"]["n_negatives"], 4)
        self.assertEqual(pooled["fpr"]["point"], 0.5)

    def test_a_null_fold_counts_towards_fpr_only(self):
        full = lofo_row(LOFO, "A", MAIN)
        lofo = result(["q1", "q2"], [0.1, 0.6], [0, 0], [None] * 2, t=0.5)
        main = result(["q1", "q2"], [0.1, 0.6], [0, 0], [None] * 2, t=0.5)
        null = lofo_row(lofo, "A", main)
        pooled = lofo_pool([full, null])
        self.assertEqual(pooled["recall"], full["recall"])
        self.assertEqual(pooled["fpr"]["n_negatives"], 5)
        self.assertEqual(pooled["fpr"]["false_positives"], 3)

    def test_all_null_folds_give_null_recall_and_difference(self):
        lofo = result(["q1"], [0.6], [0], [None], t=0.5)
        null = lofo_row(lofo, "A", lofo)
        pooled = lofo_pool([null])
        self.assertIsNone(pooled["recall"]["recall"])
        self.assertIsNone(pooled["recall_difference"])
        self.assertEqual(pooled["fpr"]["point"], 1.0)

    def test_no_rows_raises(self):
        with self.assertRaises(ValueError):
            lofo_pool([])


class CheckFoldFamiliesTest(unittest.TestCase):
    Y = [0, 0, 1, 1, 1]

    def test_the_default_still_needs_all_four(self):
        with self.assertRaises(FoldError):
            check_fold(self.Y, [None, None, "B", "C", "D"], [0, 1])

    def test_remaining_three_pass_without_the_left_out_family(self):
        check_fold(self.Y, [None, None, "B", "C", "D"], [0, 1], families=remaining("A"))

    def test_a_remaining_family_missing_raises(self):
        with self.assertRaises(FoldError):
            check_fold(self.Y, [None, None, "B", "B", "D"], [0, 1], families=remaining("A"))

    def test_a_class_missing_raises(self):
        with self.assertRaises(FoldError):
            check_fold([1, 1, 1], ["B", "C", "D"], [0, 1], families=remaining("A"))


class DataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.datas = run(lofo=True)
        cls.groups = split(cls.datas)
        assert all(lofo for _, lofo in cls.groups), "lofo=True yielded no LOFO data"

    def test_off_by_default(self):
        self.assertTrue(all(d.lofo_family is None for d in run()))

    def test_four_families_per_fold_and_arm_in_order_none_for_e(self):
        self.assertEqual(len(self.groups), 12)  # 2 repeats x 3 folds x 2 arms
        for main, lofo in self.groups:
            self.assertEqual(list(lofo), list(TRAINING_FAMILIES))
            self.assertNotIn(EVAL_FAMILY, lofo)
            for d in lofo.values():
                self.assertEqual((d.repeat, d.fold, d.arm), (main.repeat, main.fold, main.arm))

    def test_main_data_is_the_same_with_lofo_on(self):
        plain = run()
        mains = [m for m, _ in self.groups]
        self.assertEqual([m[:-1] for m in mains], [p[:-1] for p in plain])

    def test_no_text_or_family_f_in_the_training_side(self):
        for main, lofo in self.groups:
            for family, d in lofo.items():
                self.assertEqual({f for f in d.train_families if f is not None},
                                 set(remaining(family)))
                self.assertNotIn(family, {g[0] for g in d.train_groups if g is not None})
                f_texts = {t for t, f in zip(main.train_texts, main.train_families)
                           if f == family}
                self.assertTrue(f_texts)
                self.assertFalse(f_texts & set(d.train_texts))

    def test_training_negatives_and_threshold_texts_are_the_arms_own_unchanged(self):
        for main, lofo in self.groups:
            negatives = [t for t, y in zip(main.train_texts, main.train_y) if y == 0]
            for d in lofo.values():
                self.assertEqual([t for t, y in zip(d.train_texts, d.train_y) if y == 0],
                                 negatives)
                self.assertEqual(d.threshold_texts, main.threshold_texts)

    def test_training_positives_equal_negatives_up_to_rounding(self):
        for main, lofo in self.groups:
            for family, d in lofo.items():
                report = d.report["train"]
                n_templates = len({g for g in d.train_groups if g is not None})
                per_template = report["negatives"] // n_templates
                self.assertEqual(report["requested_per_template"], per_template)
                lost = sum(report["dropped"].values()) + sum(report["shortfall"].values())
                self.assertEqual(report["positives"] + lost, per_template * n_templates)
                self.assertLess(report["negatives"] - per_template * n_templates, n_templates)
                self.assertGreaterEqual(report["negatives"], per_template * n_templates)

    def test_test_side_is_the_main_strings_restricted_to_queries_and_f(self):
        for main, lofo in self.groups:
            for family, d in lofo.items():
                keep = [i for i, f in enumerate(main.test_families) if f is None or f == family]
                self.assertEqual(d.test_texts, [main.test_texts[i] for i in keep])
                self.assertEqual(d.test_y, [main.test_y[i] for i in keep])
                self.assertEqual(d.test_families, [main.test_families[i] for i in keep])
                self.assertEqual(d.test_groups, [main.test_groups[i] for i in keep])
                self.assertEqual({f for f in d.test_families if f is not None} - {family}, set())

    def test_a_fold_without_f_on_the_test_side_still_yields_with_queries_only(self):
        # CONFIG repeat 0 fold 2 holds none of family A's templates on the test side
        main, lofo = next((m, l) for m, l in self.groups
                          if (m.repeat, m.fold, m.arm) == (0, 2, PRIMARY))
        self.assertEqual(main.report["test"]["families"]["A"], 0)
        self.assertEqual(set(lofo["A"].test_y), {0})
        self.assertEqual(set(lofo["A"].test_families), {None})
        self.assertEqual(lofo["A"].test_texts, [t for t, f in zip(main.test_texts,
                                                                   main.test_families) if f is None])

    def test_report_names_the_family_and_holds_the_train_counts(self):
        for _, lofo in self.groups:
            for family, d in lofo.items():
                self.assertEqual(d.report["lofo_family"], family)
                self.assertIn("train", d.report)


class CheckFoldWiringTest(unittest.TestCase):
    def test_lofo_training_sides_are_checked_with_the_remaining_families_by_keyword(self):
        real = bqc.evaluate.check_fold
        with mock.patch("bqc.evaluate.check_fold", wraps=real) as spy:
            datas = run(lofo=True, arms=(PRIMARY,), repeats=1)
        lofo = [d for d in datas if d.lofo_family]
        self.assertEqual(len(lofo), 12)
        calls = [c for c in spy.call_args_list if "families" in c.kwargs]
        self.assertEqual(len(calls), len(lofo))
        self.assertEqual(len(spy.call_args_list), len(datas))
        for call, d in zip(calls, lofo):
            self.assertEqual(list(call.kwargs["families"]), remaining(d.lofo_family))
            self.assertEqual(call.args[0], d.train_y)
            self.assertEqual(call.args[1], d.train_families)
            self.assertIn(set(call.args[2]), [{0, 1}])

    def test_a_training_side_missing_another_family_raises(self):
        real = bqc.evaluate.check_fold

        def drop_b(train_y, train_families, test_y, families=TRAINING_FAMILIES):
            if families is TRAINING_FAMILIES:  # the main run's own check is left alone
                return real(train_y, train_families, test_y)
            return real(train_y, ["C" if f == "B" else f for f in train_families], test_y,
                        families=families)

        with mock.patch("bqc.evaluate.check_fold", drop_b):
            with self.assertRaises(FoldError):
                run(lofo=True, arms=(PRIMARY,), repeats=1)


class RenderWiringTest(unittest.TestCase):
    def test_lofo_renders_with_the_main_vocabulary_seed_and_drop_lists(self):
        real = bqc.evaluate._render_screened
        with mock.patch("bqc.evaluate._render_screened", wraps=real) as spy:
            datas = run(lofo=True, arms=(PRIMARY, COMPARISON), repeats=1,
                        sanity=["zzz qqq"], part_numbers=["PN-1"])
        lofo_calls = [c for c in spy.call_args_list if not c.args]  # keyword calls only
        self.assertEqual(len(lofo_calls), 4 * 2 * 3)
        lofo_datas = [d for d in datas if d.lofo_family]
        for call, d in zip(lofo_calls, lofo_datas):
            kw = call.kwargs
            self.assertEqual(kw["sanity"], ["zzz qqq"])
            self.assertEqual(kw["drop_lists"]["part_numbers"], ["PN-1"])
            self.assertEqual(kw["drop_lists"]["real_queries"], REAL)
            self.assertEqual(kw["seed"], bqc.evaluate.SEED + d.repeat)
            self.assertEqual(kw["drop_lists"]["training_negatives"],
                             [t for t, y in zip(d.train_texts, d.train_y) if y == 0])
            self.assertEqual({f for f, _ in kw["templates"]}, set(remaining(d.lofo_family)))
            self.assertEqual(
                {g for g in d.train_groups if g is not None}, set(kw["templates"]))
            self.assertEqual(kw["per_template"], d.report["train"]["requested_per_template"])
        primary = [(c, d) for c, d in zip(lofo_calls, lofo_datas) if d.arm == PRIMARY]
        for call, d in primary:
            spans = [t for t, y in zip(d.train_texts, d.train_y) if y == 0]
            self.assertEqual(list(call.kwargs["vocabulary"]),
                             token_occurrences(spans, cap=bqc.evaluate.VOCABULARY_CAP))
        comparison = [(c, d) for c, d in zip(lofo_calls, lofo_datas) if d.arm == COMPARISON]
        for call, d in comparison:
            queries = [t for t, y in zip(d.train_texts, d.train_y) if y == 0]
            self.assertEqual(list(call.kwargs["vocabulary"]), token_occurrences(queries))


class RunnerPrefixTest(unittest.TestCase):
    """The runners' score-set names get the prefix, by keyword, in every score set."""

    def assert_named(self, run_fold, factory_values, score_set):
        with self.assertRaises(NonFiniteScoreError) as ctx:
            run_fold(Factory(factory_values))
        message = str(ctx.exception)
        self.assertIn(f"LOFO A {score_set}", message)

    def primary(self, factory):
        return run_primary_fold(make_primary_data(), factory, 0.1, "lr",
                                score_set_prefix="LOFO A")

    def comparison(self, factory):
        return run_comparison_fold(make_comparison_data(), factory, 0.1, "lr",
                                   score_set_prefix="LOFO A")

    def test_primary_threshold_queries(self):
        self.assert_named(self.primary, {"a1": float("nan")}, "threshold queries")

    def test_primary_test_queries_positives_and_e(self):
        self.assert_named(self.primary, {"tq-": float("nan")}, "test queries")
        self.assert_named(self.primary, {"pos 11": float("inf")}, "test positives")
        self.assert_named(self.primary, {"e!!": float("nan")}, "E")

    def test_comparison_out_of_fold_and_test_sets(self):
        self.assert_named(self.comparison, {"q-1": float("nan")}, "out-of-fold scores")
        self.assert_named(self.comparison, {"tq-": float("nan")}, "test queries")

    def test_no_prefix_leaves_the_names_alone(self):
        with self.assertRaises(NonFiniteScoreError) as ctx:
            run_primary_fold(make_primary_data(), Factory({"tq-": float("nan")}), 0.1, "lr")
        self.assertNotIn("LOFO", str(ctx.exception))
        self.assertTrue(str(ctx.exception).startswith("non-finite score in test queries"))


class ComparisonFamiliesTest(unittest.TestCase):
    def without(self, *families):
        data = make_comparison_data()
        keep = [i for i, f in enumerate(data.train_families) if f not in families]
        return data._replace(
            train_texts=[data.train_texts[i] for i in keep],
            train_y=[data.train_y[i] for i in keep],
            train_families=[data.train_families[i] for i in keep],
            train_groups=[data.train_groups[i] for i in keep])

    def test_the_left_out_family_is_not_expected_in_the_inner_folds(self):
        data = self.without("A")._replace(lofo_family="A")
        factory = Factory()
        run_comparison_fold(data, factory, 0.1, "lr")
        self.assertEqual(len(factory.made), INNER + 1)

    def test_without_lofo_family_a_missing_family_raises(self):
        with self.assertRaises(FoldError):
            run_comparison_fold(self.without("A"), Factory(), 0.1, "lr")

    def test_another_missing_family_raises_even_with_lofo(self):
        data = self.without("A", "B")._replace(lofo_family="A")
        with self.assertRaises(FoldError):
            run_comparison_fold(data, Factory(), 0.1, "lr")

    def test_a_one_class_test_side_does_not_raise_in_either_runner(self):
        queries_only = dict(test_texts=["tq1", "tq-", "tq"], test_y=[0, 0, 0],
                            test_families=[None] * 3, test_groups=[None] * 3)
        primary = run_primary_fold(
            make_primary_data(**queries_only, lofo_family="A"), Factory(), 0.1, "lr")
        comparison = run_comparison_fold(
            self.without("A")._replace(lofo_family="A", **queries_only), Factory(), 0.1, "lr")
        for r in (primary, comparison):
            self.assertEqual(len(r.test_scores), 3)


class ReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parameters, cls.body = build(lofo=True)

    def test_off_by_default_and_recorded_in_the_parameters(self):
        parameters, body = build()
        self.assertIs(parameters["lofo"], False)
        for arm in (PRIMARY, COMPARISON):
            self.assertNotIn("lofo", body[arm])
        self.assertIs(self.parameters["lofo"], True)

    def test_section_per_arm_model_and_family_a_to_d_only(self):
        for arm in (PRIMARY, COMPARISON):
            section = self.body[arm]["lofo"]
            self.assertEqual(set(section), {"lr"})
            self.assertEqual(list(section["lr"]), list(TRAINING_FAMILIES))
            for family, entry in section["lr"].items():
                self.assertEqual(set(entry), {"folds", "pooled"})
                self.assertEqual(len(entry["folds"]), FOLDS)
                for row in entry["folds"]:
                    for key in ("recall", "fpr", "main_recall", "recall_difference"):
                        self.assertIn(key, row)
                for key in ("recall", "fpr", "main_recall", "recall_difference"):
                    self.assertIn(key, entry["pooled"])

    def test_the_headline_is_separate_and_unchanged_by_lofo(self):
        _, plain = build()
        for arm in (PRIMARY, COMPARISON):
            self.assertEqual(self.body[arm]["models"], plain[arm]["models"])
            self.assertEqual(self.body[arm]["folds"], plain[arm]["folds"])

    def test_fake_ignores_fitting_so_lofo_matches_the_main_run_exactly(self):
        # the fake's scores don't depend on the fit: threshold and strings are the main run's
        for arm in (PRIMARY, COMPARISON):
            main = self.body[arm]["models"]["lr"]["pooled"]["families"]
            for family in TRAINING_FAMILIES:
                pooled = self.body[arm]["lofo"]["lr"][family]["pooled"]
                self.assertEqual(pooled["recall"]["recall"], main[family]["recall"])
                self.assertEqual(pooled["main_recall"]["recall"], main[family]["recall"])
                self.assertEqual(pooled["recall_difference"], 0.0)

    def test_null_recall_where_f_is_absent_from_the_test_side(self):
        # family B has no templates on the test side of repeat 1 fold 0 (checked below)
        main = next(f for f in self.body[PRIMARY]["folds"] if (f["repeat"], f["fold"]) == (1, 0))
        self.assertEqual(main["test"]["families"]["B"], 0)
        row = self.body[PRIMARY]["lofo"]["lr"]["B"]["folds"][4]
        self.assertEqual((row["repeat"], row["fold"]), (1, 0))
        self.assertIsNone(row["recall"]["recall"])
        self.assertEqual(row["recall"]["count"], 0)
        self.assertGreater(row["fpr"]["n_negatives"], 0)
        self.assertIsNone(row["recall_difference"])

    def test_plain_python_types_only(self):
        allowed = {dict, list, str, int, float, bool, type(None)}
        self.assertIn("lofo", self.body[PRIMARY])
        bad = [(p, t) for p, t in walk_types(self.body) if t not in allowed]
        self.assertEqual(bad, [])
        self.assertEqual(json.loads(json.dumps(self.body)), self.body)

    def test_two_models_each_get_their_rows(self):
        _, body = build(lofo=True, models={"lr": VariedScorer, "other": FakeScorer},
                        arms=[PRIMARY])
        self.assertEqual(set(body[PRIMARY]["lofo"]), {"lr", "other"})


class FitCountTest(unittest.TestCase):
    def fits(self, arm, lofo):
        recorder = Recorder()
        build(lofo=lofo, models={"lr": recorder}, arms=[arm])
        return len(recorder.made)

    def test_primary_arm_one_fit_per_family_per_fold(self):
        self.assertEqual(self.fits(PRIMARY, False), FOLDS)
        self.assertEqual(self.fits(PRIMARY, True), FOLDS + 4 * FOLDS)

    def test_comparison_arm_inner_folds_plus_one_per_family_per_fold(self):
        self.assertEqual(self.fits(COMPARISON, False), FOLDS * (INNER + 1))
        self.assertEqual(self.fits(COMPARISON, True),
                         FOLDS * (INNER + 1) + 4 * FOLDS * (INNER + 1))

    def test_no_f_text_in_any_lofo_fit(self):
        for arm in (PRIMARY, COMPARISON):
            recorder = Recorder()
            build(lofo=True, models={"lr": recorder}, arms=[arm])
            datas = split(run(lofo=True, arms=(arm,), n_splits=4))
            per_fold = 1 if arm == PRIMARY else INNER + 1
            index = 0
            for main, lofo in datas:
                index += per_fold  # the main fits
                for family in TRAINING_FAMILIES:
                    f_texts = {t for t, f in zip(main.train_texts, main.train_families)
                               if f == family}
                    for scorer in recorder.made[index:index + per_fold]:
                        self.assertFalse(f_texts & set(scorer.fit_texts))
                        self.assertTrue(scorer.fit_texts)
                    index += per_fold
            self.assertEqual(index, len(recorder.made))

    def test_a_new_scorer_per_fit(self):
        recorder = Recorder()
        build(lofo=True, models={"lr": recorder}, arms=[COMPARISON])
        self.assertEqual(len({id(s) for s in recorder.made}), len(recorder.made))


class NonFiniteLofoTest(unittest.TestCase):
    def raises(self, arm, nan_at):
        with self.assertRaises(NonFiniteScoreError) as ctx:
            build(lofo=True, models={"lr": Recorder(nan_at=[nan_at])}, arms=[arm])
        return str(ctx.exception)

    def test_primary_nan_in_lofo_scores_names_the_lofo_score_set(self):
        # fits per fold: main, then A, B, C, D; index 3 is family C of repeat 0 fold 0
        message = self.raises(PRIMARY, 3)
        self.assertIn("LOFO C threshold queries", message)
        for part in ("repeat 0", "fold 0", "arm product", "model lr"):
            self.assertIn(part, message)

    def test_comparison_nan_in_a_lofo_refit_names_the_lofo_score_set(self):
        # fits per fold: 6 main, then 6 each for A-D; index 11 is the refit of family A
        message = self.raises(COMPARISON, 11)
        self.assertIn("LOFO A test queries", message)
        self.assertIn("arm real", message)

    def test_comparison_nan_in_a_lofo_inner_fit_names_out_of_fold_scores(self):
        message = self.raises(COMPARISON, 6 + 6 + 6 + 1)  # family C, second inner fit
        self.assertIn("LOFO C out-of-fold scores", message)

    def test_the_main_run_keeps_its_own_names(self):
        for arm in (PRIMARY, COMPARISON):
            message = self.raises(arm, 0)
            self.assertNotIn("LOFO", message)


if __name__ == "__main__":
    unittest.main()
