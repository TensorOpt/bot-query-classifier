import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import numpy as np
from fakes import FakeScorer
from sklearn.model_selection import StratifiedGroupKFold
from synth_stubs import build_registry
from test_folds import PRODUCTS, REAL, custom_registry
from test_runner import Factory

from bqc import SEED
from bqc.evaluate import (
    COMPARISON,
    PRIMARY,
    FinalModel,
    FoldError,
    NonFiniteScoreError,
    check_finite,
    fit_final_model,
    load_final_model,
    save_final_model,
    Threshold,
    select_threshold,
)
from bqc.negatives import VOCABULARY_CAP, build_spans, token_occurrences
from bqc.synth.registry import TRAINING_FAMILIES

N = 120
PER_TEMPLATE = N // 24
TARGET = 0.1
NOW = datetime(2026, 11, 6, tzinfo=timezone.utc)


class Memoriser(FakeScorer):
    """Scores 0.9 for a text it was fitted on and 0.1 otherwise: in-sample and
    out-of-fold scores of a real query differ."""

    def score(self, texts):
        seen = set(self.fit_texts or ())
        return np.array([0.9 if t in seen else 0.1 for t in texts])


def fit(arm, factory=None, registry=None, real=REAL, part_numbers=(), sanity=(), target=TARGET,
        n=N):
    return fit_final_model(
        arm, factory or Factory(), target, "lr", real, PRODUCTS, list(part_numbers),
        list(sanity), n=n, registry=registry or build_registry())


# Distinct scores, rising with the index: the last five real queries hold the top scores, so
# a threshold over any subset that drops them differs from the one over all 40.
RISING = {q: (i + 1) / len(REAL) for i, q in enumerate(REAL)}
RISING_THRESHOLD = Threshold(0.925, 0.1, False, None)  # 4 of 40 flagged at the 37th score


def positives(factory):
    scorer = factory.made[-1]
    return [t for t, y in zip(scorer.fit_texts, scorer.fit_y) if y == 1]


class CheckFiniteFinalTest(unittest.TestCase):
    def test_no_fold_in_message(self):
        with self.assertRaises(NonFiniteScoreError) as ctx:
            check_finite([float("nan")], None, None, "product", "lr", "threshold queries")
        message = str(ctx.exception)
        self.assertIn("final model", message)
        self.assertNotIn("repeat", message)
        self.assertNotIn("None", message)


class PrimaryTest(unittest.TestCase):
    def test_one_fit_on_spans_and_all_24_templates_without_e_or_real(self):
        factory = Factory()
        final = fit(PRIMARY, factory)
        self.assertEqual(len(factory.made), 1)
        scorer = factory.made[0]
        self.assertIs(final.scorer, scorer)
        self.assertEqual(scorer.fit_y.count(0), N)
        # 24 A-D templates x 5 each; family E would add 6 x 5 more
        self.assertEqual(scorer.fit_y.count(1), 24 * PER_TEMPLATE)
        self.assertEqual(final.report["families"]["E"], 0)
        self.assertEqual(
            {f: final.report["families"][f] for f in TRAINING_FAMILIES},
            {f: 6 * PER_TEMPLATE for f in TRAINING_FAMILIES})
        spans, _ = build_spans(PRODUCTS, N, SEED, REAL, [], [])
        self.assertEqual(scorer.fit_texts[:N], [s.text for s in spans])
        self.assertFalse(set(REAL) & set(scorer.fit_texts))
        self.assertFalse(any("quillon" in t for t in scorer.fit_texts))

    def test_quota_is_n_floor_24_per_template(self):
        # 299 // 23, // 24 and // 25 are 13, 12 and 11: only 12 passes
        factory = Factory()
        final = fit(PRIMARY, factory, n=299)
        self.assertEqual(factory.made[0].fit_y.count(0), 299)
        self.assertEqual(factory.made[0].fit_y.count(1), 24 * 12)
        self.assertEqual(
            {f: final.report["families"][f] for f in TRAINING_FAMILIES},
            {f: 6 * 12 for f in TRAINING_FAMILIES})

    def test_positives_are_rendered_with_seed_from_the_span_vocabulary(self):
        factory = Factory()
        registry = build_registry()
        fit(PRIMARY, factory, registry=registry)
        spans, _ = build_spans(PRODUCTS, N, SEED, REAL, [], [])
        vocabulary = token_occurrences([s.text for s in spans], cap=VOCABULARY_CAP)
        templates = [(f, t.template_id) for f in TRAINING_FAMILIES
                     for t in registry.templates(f)]
        samples, _ = registry.render(vocabulary, SEED, templates, PER_TEMPLATE)
        excluded = set(REAL) | {s.text for s in spans}
        expected = [x.text for x in samples if x.text not in excluded]
        self.assertGreater(len(expected), 0)
        self.assertEqual(positives(factory), expected)

    def test_threshold_covers_every_real_query_with_distinct_scores(self):
        final = fit(PRIMARY, Factory(RISING))
        self.assertEqual(final.threshold, RISING_THRESHOLD)

    def test_threshold_is_the_rule_over_all_real_queries(self):
        for target in (0.1, 0.5):
            final = fit(PRIMARY, target=target)
            self.assertEqual(
                final.threshold, select_threshold(FakeScorer().score(REAL), target))
        self.assertEqual((final.arm, final.model), (PRIMARY, "lr"))

    def test_threshold_scores_are_out_of_sample(self):
        final = fit(PRIMARY, Factory(cls=Memoriser), target=1.0)
        self.assertEqual(final.threshold.threshold, 0.1)

    def test_nan_in_threshold_scores_raises_naming_arm_model_set(self):
        with self.assertRaises(NonFiniteScoreError) as ctx:
            fit(PRIMARY, Factory({REAL[3]: float("nan")}))
        message = str(ctx.exception)
        for part in ("threshold queries", "product", "lr", "final model"):
            self.assertIn(part, message)
        self.assertNotIn(REAL[3], message)


class ComparisonTest(unittest.TestCase):
    def test_final_fit_on_960_equivalent_without_e(self):
        factory = Factory()
        final = fit(COMPARISON, factory)
        self.assertEqual(len(factory.made), 6)  # 5 inner + the final
        scorer = factory.made[-1]
        self.assertIs(final.scorer, scorer)
        self.assertEqual(scorer.fit_texts[:len(REAL)], REAL)
        self.assertEqual(scorer.fit_y.count(0), len(REAL))
        self.assertEqual(scorer.fit_y.count(1), 24 * 20)
        self.assertEqual(final.report["families"]["E"], 0)
        for inner in factory.made[:-1]:
            self.assertLess(len(inner.fit_texts), len(scorer.fit_texts))

    def test_threshold_comes_from_out_of_fold_scores(self):
        final = fit(COMPARISON, Factory(cls=Memoriser), target=1.0)
        self.assertEqual(final.threshold.threshold, 0.1)
        self.assertEqual(float(final.scorer.score([REAL[0]])[0]), 0.9)

    def test_threshold_covers_every_real_query_with_distinct_scores(self):
        final = fit(COMPARISON, Factory(RISING))
        self.assertEqual(final.threshold, RISING_THRESHOLD)

    def test_threshold_rule_runs_over_the_real_queries_only(self):
        final = fit(COMPARISON, target=0.3)  # FakeScorer ignores fitting: oof == direct
        self.assertEqual(final.threshold, select_threshold(FakeScorer().score(REAL), 0.3))

    def test_inner_splitter_is_five_shuffled_folds_seeded_with_seed(self):
        with mock.patch("bqc.evaluate.StratifiedGroupKFold", wraps=StratifiedGroupKFold) as m:
            fit(COMPARISON)
        m.assert_called_once_with(n_splits=5, shuffle=True, random_state=SEED)

    def test_nan_in_out_of_fold_scores_raises(self):
        with self.assertRaises(NonFiniteScoreError) as ctx:
            fit(COMPARISON, Factory({REAL[5]: float("nan")}))
        message = str(ctx.exception)
        for part in ("out-of-fold scores", "real", "lr", "final model"):
            self.assertIn(part, message)
        self.assertNotIn("repeat", message)

    def test_unknown_arm(self):
        with self.assertRaises(ValueError):
            fit("nope")


class RuleFourTest(unittest.TestCase):
    def test_sanity_match_fails_without_text(self):
        secret = "sanity secret 7"
        for arm in (PRIMARY, COMPARISON):
            with self.assertRaises(FoldError) as ctx:
                fit(arm, registry=custom_registry([secret]), sanity=[secret])
            self.assertNotIn(secret, str(ctx.exception))

    def test_other_matches_are_dropped_and_counted(self):
        span = build_spans(PRODUCTS, N, SEED, REAL, [], [])[0][0].text
        texts = [REAL[1], "pn-77x", span, "keepme 1"]
        for arm in (PRIMARY, COMPARISON):
            factory = Factory()
            final = fit(arm, factory, registry=custom_registry(texts), part_numbers=["pn-77x"])
            kept = set(positives(factory))
            self.assertFalse(kept & {REAL[1], "pn-77x"})
            self.assertIn("keepme 1", kept)
            dropped = final.report["dropped"]
            self.assertGreater(dropped["real_queries"] + dropped["training_negatives"], 0)
            self.assertGreater(dropped["part_numbers"], 0)
            if arm == PRIMARY:
                self.assertNotIn(span, kept)
                self.assertGreater(dropped["training_negatives"], 0)


class PickleTest(unittest.TestCase):
    def test_round_trip_scores_identically(self):
        probes = ["a b", "x1-2", "!!a", "tufted chair", ""]
        for arm in (PRIMARY, COMPARISON):
            final = fit(arm, FakeScorer)
            with tempfile.TemporaryDirectory() as tmp:
                path = save_final_model(final, Path(tmp) / "results", NOW)
                self.assertEqual(path.parent, Path(tmp) / "results")
                loaded = load_final_model(path)
            self.assertIsInstance(loaded, FinalModel)
            np.testing.assert_array_equal(
                loaded.scorer.score(probes), final.scorer.score(probes))
            self.assertEqual(loaded.threshold, final.threshold)
            self.assertEqual((loaded.arm, loaded.model), (arm, "lr"))

    def test_never_overwrites(self):
        final = fit(PRIMARY, FakeScorer)
        with tempfile.TemporaryDirectory() as tmp:
            first = save_final_model(final, Path(tmp), NOW)
            with self.assertRaises(FileExistsError):
                save_final_model(final, Path(tmp), NOW)
            self.assertTrue(first.exists())

    def test_file_names_differ_by_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = {save_final_model(fit(arm, FakeScorer), Path(tmp), NOW)
                     for arm in (PRIMARY, COMPARISON)}
            self.assertEqual(len(paths), 2)


if __name__ == "__main__":
    unittest.main()
