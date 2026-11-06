import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from fakes import FakeScorer
from synth_stubs import build_registry
from test_folds import PRODUCTS, REAL
from test_runner import Factory

from bqc.evaluate import (
    COMPARISON,
    PRIMARY,
    FinalModel,
    NonFiniteScoreError,
    Threshold,
    fit_final_model,
    sanity_section,
)
from bqc.report import write_report
from bqc.sanity import SanityError, load_sanity
from bqc.wands import WandsQuery

FIXTURES = Path(__file__).parent / "fixtures"
WANDS = [WandsQuery(0, "smoky mountain tray", ""), WandsQuery(3, "area rug", "")]
FORBIDDEN = ("macro_f1", "roc_auc", "ci_low", "ci_high", "precision", "recall", "fpr", "rate")
# sanity_full.csv: oak desk 1, "chair, red" 0, sofa 1, rug 0
SCORES = {"oak desk": 0.9, "chair, red": 0.8, "sofa": 0.2, "rug": 0.1}
THRESHOLD = Threshold(0.5, 0.0, False, None)


def final(arm=PRIMARY, model="lr", overrides=SCORES, threshold=THRESHOLD):
    return FinalModel(arm, model, Factory(overrides)(), threshold, [], {})


def section(name="sanity_full.csv", finals=None, wands=WANDS):
    with redirect_stdout(StringIO()):
        return sanity_section(FIXTURES / name, wands, finals if finals is not None else [final()])


def all_keys(value):
    if isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from all_keys(v)
    elif isinstance(value, list):
        for v in value:
            yield from all_keys(v)


class SkipTest(unittest.TestCase):
    def test_blank_partial_and_bad_label_files_give_the_skip_record(self):
        for name, n in (("sanity_blank.csv", 2), ("sanity_partial.csv", 1), ("sanity_bad.csv", 3)):
            with self.subTest(name):
                self.assertEqual(section(name), {"status": "skipped", "unlabelled": n})

    def test_message_is_printed(self):
        out = StringIO()
        with redirect_stdout(out):
            sanity_section(FIXTURES / "sanity_partial.csv", WANDS, [final()])
        self.assertEqual(out.getvalue(), "sanity set: 1 of 3 rows unlabelled, skipping\n")

    def test_scored_file_prints_nothing(self):
        out = StringIO()
        with redirect_stdout(out):
            sanity_section(FIXTURES / "sanity_full.csv", WANDS, [final()])
        self.assertEqual(out.getvalue(), "")

    def test_the_rest_of_the_report_is_present_and_the_run_json_has_the_record(self):
        from test_run_report import build

        _, body = build()
        body["sanity"] = section("sanity_blank.csv")
        with tempfile.TemporaryDirectory() as tmp:
            path = write_report(Path(tmp), "evaluate", {}, body)
            written = json.loads(path.read_text())
        self.assertEqual(written["sanity"], {"status": "skipped", "unlabelled": 2})
        self.assertIn(PRIMARY, written)
        self.assertIn(COMPARISON, written)
        self.assertTrue(written[PRIMARY]["models"])


class ScoredTest(unittest.TestCase):
    def test_counts_and_disagreements(self):
        result = section()
        self.assertEqual(result["status"], "scored")
        entry = result["models"][PRIMARY]["lr"]
        # rows: user label 0/1; columns: not flagged / flagged
        self.assertEqual(entry["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(entry["disagreements"], [
            {"id": 2, "query": "chair, red", "label": 0, "score": 0.8},
            {"id": 3, "query": "sofa", "label": 1, "score": 0.2},
        ])
        self.assertEqual(entry["threshold"], 0.5)

    def test_every_model_and_arm_has_its_own_entry_and_threshold(self):
        result = section(finals=[
            final(PRIMARY, "lr"),
            final(PRIMARY, "mlp", threshold=Threshold(0.15, 0.0, False, None)),
            final(COMPARISON, "lr", threshold=Threshold(0.85, 0.0, False, None))])
        models = result["models"]
        self.assertEqual({a: set(m) for a, m in models.items()},
                         {PRIMARY: {"lr", "mlp"}, COMPARISON: {"lr"}})
        self.assertEqual(models[PRIMARY]["lr"]["threshold"], 0.5)
        self.assertEqual(models[PRIMARY]["mlp"]["threshold"], 0.15)
        self.assertEqual(models[COMPARISON]["lr"]["threshold"], 0.85)
        self.assertEqual(models[PRIMARY]["mlp"]["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertEqual(models[COMPARISON]["lr"]["confusion_matrix"], [[2, 0], [1, 1]])
        self.assertEqual([d["id"] for d in models[PRIMARY]["mlp"]["disagreements"]], [2])
        self.assertEqual([d["id"] for d in models[COMPARISON]["lr"]["disagreements"]], [3])

    def test_score_equal_to_the_threshold_is_flagged(self):
        entry = section(finals=[final(overrides={**SCORES, "rug": 0.5})])["models"][PRIMARY]["lr"]
        self.assertEqual(entry["confusion_matrix"], [[0, 2], [1, 1]])
        self.assertIn(4, [d["id"] for d in entry["disagreements"]])

    def test_agreement_gives_an_empty_disagreement_list(self):
        entry = section(finals=[final(overrides={**SCORES, "chair, red": 0.1, "sofa": 0.9})])
        self.assertEqual(entry["models"][PRIMARY]["lr"]["disagreements"], [])
        self.assertEqual(entry["models"][PRIMARY]["lr"]["confusion_matrix"], [[2, 0], [0, 2]])

    def test_no_forbidden_metric_keys_and_no_float_rates(self):
        result = section(finals=[final(), final(COMPARISON)])
        keys = {str(k).lower() for k in all_keys(result)}
        for word in FORBIDDEN:
            self.assertFalse([k for k in keys if word in k], word)
        for arm in result["models"].values():
            for entry in arm.values():
                self.assertTrue(all(isinstance(c, int) for row in entry["confusion_matrix"]
                                    for c in row))

    def test_result_is_json_serialisable(self):
        json.dumps(section())


class NonFiniteTest(unittest.TestCase):
    def test_nan_raises_naming_the_score_set_and_no_query_text(self):
        for bad in (float("nan"), float("inf")):
            with self.subTest(bad):
                with self.assertRaises(NonFiniteScoreError) as ctx:
                    section(finals=[final(COMPARISON, "mlp", {"sofa": bad})])
                message = str(ctx.exception)
                for part in ("sanity scores", "final model", "comparison".replace(
                        "comparison", COMPARISON), "mlp"):
                    self.assertIn(part, message)
                for q in SCORES:
                    self.assertNotIn(q, message)


class DuplicateTest(unittest.TestCase):
    def test_duplicate_raises_with_the_id_and_without_the_text(self):
        with self.assertRaises(SanityError) as ctx:
            section("sanity_wands_dup.csv")
        self.assertEqual(ctx.exception.query_id, 7)
        message = str(ctx.exception)
        self.assertIn("id 7", message)
        self.assertNotIn("rug", message.lower())
        self.assertIsNone(ctx.exception.__cause__)
        self.assertTrue(ctx.exception.__suppress_context__)

    def test_duplicate_fails_even_when_the_labels_are_blank(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.csv"
            path.write_text("id,query,label\n9,Smoky  Mountain Tray,\n")
            with self.assertRaises(SanityError) as ctx:
                sanity_section(path, WANDS, [final()])
        self.assertEqual(ctx.exception.query_id, 9)

    def test_loader_error_carries_the_id_and_other_errors_have_none(self):
        with self.assertRaises(SanityError) as ctx:
            load_sanity(FIXTURES / "sanity_wands_dup.csv", WANDS)
        self.assertEqual(ctx.exception.query_id, 7)
        with self.assertRaises(SanityError) as ctx:
            load_sanity(FIXTURES / "sanity_badheader.csv", WANDS)
        self.assertIsNone(ctx.exception.query_id)

    def test_other_loader_errors_propagate_unchanged(self):
        with self.assertRaises(SanityError) as ctx:
            section("sanity_badheader.csv")
        self.assertIn("sanity_badheader.csv", str(ctx.exception))

    def test_no_wands_queries_is_refused(self):
        with self.assertRaises(ValueError):
            section(wands=[])


class NoLeakTest(unittest.TestCase):
    # "pwhhaj pwhhba" is a PRODUCTS span the primary fit (n=120) would otherwise contain
    SPAN = "pwhhaj pwhhba"
    SANITY = [q.query for q in load_sanity(FIXTURES / "sanity_full.csv", WANDS).queries] + [SPAN]

    def test_sanity_section_never_fits(self):
        class NoFit(FakeScorer):
            def fit(self, texts, y):
                raise AssertionError("fit called")

        scorer = NoFit()
        section(finals=[FinalModel(PRIMARY, "lr", scorer, THRESHOLD, [], {})])
        self.assertIsNone(scorer.fit_texts)

    def test_fit_inputs_contain_no_sanity_text(self):
        for arm in (PRIMARY, COMPARISON):
            with self.subTest(arm):
                factory = Factory()
                fitted = fit_final_model(arm, factory, 0.1, "lr", REAL, PRODUCTS, [],
                                         self.SANITY, n=120, registry=build_registry())
                self.assertTrue(factory.made)
                if arm == PRIMARY:
                    self.assertGreaterEqual(fitted.report["span_removals"]["sanity"], 1)
                for scorer in factory.made:
                    self.assertFalse(set(self.SANITY) & set(scorer.fit_texts))

    def test_scoring_does_not_change_the_final_model(self):
        factory = Factory()
        fitted = fit_final_model(PRIMARY, factory, 0.1, "lr", REAL, PRODUCTS, [], self.SANITY,
                                 n=120, registry=build_registry())
        before = (list(fitted.scorer.fit_texts), fitted.threshold)
        section(finals=[fitted])
        self.assertEqual((fitted.scorer.fit_texts, fitted.threshold), before)
        self.assertEqual(len(factory.made), 1)


if __name__ == "__main__":
    unittest.main()
