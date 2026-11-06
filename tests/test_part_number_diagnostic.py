import copy
import json
import unittest

from fakes import FakeScorer
from synth_stubs import build_registry
from test_folds import PRODUCTS, REAL
from test_run_report import build
from test_runner import Factory

from bqc.evaluate import (
    COMPARISON,
    PRIMARY,
    FinalModel,
    NonFiniteScoreError,
    Threshold,
    fit_final_model,
    part_number_section,
)
from bqc.synth.partnumbers import SHAPE_IDS, generate_part_numbers
from bqc.synth.registry import Sample

THRESHOLD = Threshold(0.5, 0.0, False, None)
A, B = SHAPE_IDS[0], SHAPE_IDS[1]
# shape A scores 0.1 0.5 0.7 0.9, shape B scores 0.2 0.3 0.4 1.0 (hand-computed below)
SCORES = {"a1": 0.1, "a2": 0.5, "a3": 0.7, "a4": 0.9, "b1": 0.2, "b2": 0.3, "b3": 0.4, "b4": 1.0}
SAMPLES = [Sample(t, "P", A if t[0] == "a" else B, 0) for t in SCORES]


def final(arm=PRIMARY, model="lr", overrides=SCORES, threshold=THRESHOLD):
    return FinalModel(arm, model, Factory(overrides)(), threshold, [], {})


def entry(finals=None, samples=SAMPLES, arm=PRIMARY, model="lr"):
    section = part_number_section(samples, finals if finals is not None else [final()])
    return section["models"][arm][model]


class HandComputedTest(unittest.TestCase):
    def test_overall_fraction_min_median_max(self):
        e = entry()
        # flagged at >= 0.5: 0.5 0.7 0.9 1.0 -> 4 of 8
        self.assertEqual(e["n"], 8)
        self.assertEqual(e["flagged_fraction"], 0.5)
        self.assertEqual(e["min"], 0.1)
        self.assertEqual(e["max"], 1.0)
        # sorted: .1 .2 .3 .4 .5 .7 .9 1.0 -> median (0.4 + 0.5) / 2; the mean is 0.5125
        self.assertAlmostEqual(e["median"], 0.45)

    def test_per_shape_fraction(self):
        e = entry()
        self.assertEqual(e["per_shape"], {
            A: {"n": 4, "flagged_fraction": 0.75},
            B: {"n": 4, "flagged_fraction": 0.25}})

    def test_score_exactly_at_the_threshold_is_flagged(self):
        e = entry([final(overrides={**SCORES, "a3": 0.4, "a4": 0.4, "a2": 0.5, "b4": 0.4})])
        self.assertEqual(e["flagged_fraction"], 1 / 8)  # only a2 at 0.5
        self.assertEqual(e["per_shape"][A]["flagged_fraction"], 0.25)
        self.assertEqual(e["per_shape"][B]["flagged_fraction"], 0.0)

    def test_every_model_and_arm_uses_its_own_threshold(self):
        section = part_number_section(SAMPLES, [
            final(PRIMARY, "lr"),
            final(PRIMARY, "mlp", threshold=Threshold(0.95, 0.0, False, None)),
            final(COMPARISON, "lr", threshold=Threshold(0.0, 0.0, False, None))])
        models = section["models"]
        self.assertEqual({a: set(m) for a, m in models.items()},
                         {PRIMARY: {"lr", "mlp"}, COMPARISON: {"lr"}})
        self.assertEqual(models[PRIMARY]["lr"]["flagged_fraction"], 0.5)
        self.assertEqual(models[PRIMARY]["mlp"]["flagged_fraction"], 0.125)
        self.assertEqual(models[COMPARISON]["lr"]["flagged_fraction"], 1.0)
        self.assertEqual(models[PRIMARY]["mlp"]["threshold"], 0.95)
        self.assertEqual(models[COMPARISON]["lr"]["threshold"], 0.0)

    def test_odd_count_median_and_single_sample(self):
        e = entry(samples=SAMPLES[:3])  # 0.1 0.5 0.7
        self.assertEqual(e["median"], 0.5)
        one = entry(samples=SAMPLES[:1])
        self.assertEqual((one["min"], one["median"], one["max"]), (0.1, 0.1, 0.1))
        self.assertEqual(one["flagged_fraction"], 0.0)

    def test_no_samples_is_refused(self):
        with self.assertRaises(ValueError):
            part_number_section([], [final()])

    def test_result_is_json_serialisable(self):
        json.dumps(part_number_section(SAMPLES, [final()]))


class GeneratedSetTest(unittest.TestCase):
    def test_hundred_scores_for_each_model(self):
        samples = generate_part_numbers(["sofa", "rug", "oak"], 0, [], [])
        scored = []

        class Counting(FakeScorer):
            def score(self, texts):
                scored.append(list(texts))
                return super().score(texts)

        finals = [FinalModel(a, m, Counting(), THRESHOLD, [], {})
                  for a in (PRIMARY, COMPARISON) for m in ("lr", "mlp")]
        section = part_number_section(samples, finals)
        self.assertEqual(len(scored), 4)
        for texts in scored:
            self.assertEqual(texts, [s.text for s in samples])
        for models in section["models"].values():
            for e in models.values():
                self.assertEqual(e["n"], 100)
                self.assertEqual(set(e["per_shape"]), set(SHAPE_IDS))
                self.assertEqual({v["n"] for v in e["per_shape"].values()}, {25})


class NonFiniteTest(unittest.TestCase):
    def test_nan_raises_naming_the_score_set(self):
        for bad in (float("nan"), float("inf")):
            with self.subTest(bad):
                with self.assertRaises(NonFiniteScoreError) as ctx:
                    part_number_section(SAMPLES, [final(COMPARISON, "mlp", {**SCORES, "a2": bad})])
                message = str(ctx.exception)
                for part in ("part-number scores", "final model", COMPARISON, "mlp"):
                    self.assertIn(part, message)
                self.assertNotIn("a2", message.replace("part-number", ""))


class NoLeakTest(unittest.TestCase):
    def test_never_fits(self):
        class NoFit(FakeScorer):
            def fit(self, texts, y):
                raise AssertionError("fit called")

        scorer = NoFit()
        part_number_section(SAMPLES, [FinalModel(PRIMARY, "lr", scorer, THRESHOLD, [], {})])
        self.assertIsNone(scorer.fit_texts)

    def test_fit_inputs_contain_no_part_number_text(self):
        vocabulary = [w for q in REAL for w in q.split()]
        samples = generate_part_numbers(vocabulary, 0, [], [])
        texts = {s.text for s in samples}
        for arm in (PRIMARY, COMPARISON):
            with self.subTest(arm):
                factory = Factory()
                fitted = fit_final_model(arm, factory, 0.1, "lr", REAL, PRODUCTS, sorted(texts),
                                         [], n=120, registry=build_registry())
                part_number_section(samples, [fitted])
                self.assertTrue(factory.made)  # comparison: the inner folds' scorers too
                for scorer in factory.made:
                    self.assertFalse(texts & set(scorer.fit_texts))

    def test_scoring_does_not_change_the_final_model_or_other_sections(self):
        factory = Factory()
        fitted = fit_final_model(PRIMARY, factory, 0.1, "lr", REAL, PRODUCTS, [], [],
                                 n=120, registry=build_registry())
        before = (list(fitted.scorer.fit_texts), fitted.threshold)
        _, body = build()
        snapshot = copy.deepcopy(body)
        body["part_number_diagnostic"] = part_number_section(SAMPLES, [fitted])
        self.assertEqual((fitted.scorer.fit_texts, fitted.threshold), before)
        self.assertEqual(len(factory.made), 1)
        del body["part_number_diagnostic"]
        self.assertEqual(body, snapshot)
        json.dumps({**body, "part_number_diagnostic": part_number_section(SAMPLES, [fitted])})


if __name__ == "__main__":
    unittest.main()
