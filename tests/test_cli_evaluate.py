import contextlib
import io
import json
import unittest
from unittest import mock

import numpy as np
from fakes import FakeScorer
from synth_stubs import build_registry
from test_cli_generate import DataTestCase, query_texts

from bqc import cli
from bqc.cli import main
from bqc.evaluate import (
    COMPARISON,
    PRIMARY,
    FoldError,
    build_run_report,
    load_final_model,
)
from bqc.models import MLPScorer
from bqc.synth.partnumbers import SHAPES

# Tiny config: 40 fixture queries, 5 folds, one repeat, stub templates, 120 spans.
TINY = dict(n_splits=5, repeats=1, test_per_template=5, eval_per_template=4)
N = 120
SANITY = ["zzq plorf", "wux vil kam", "ab12cd34 xx", "plain words here"]
QUERIES = 40
ALL_MODELS = ["lr", "mlp", "trivial"]


class NanScorer(FakeScorer):
    def score(self, texts):
        return np.full(len(texts), np.nan)


class VariedScorer(FakeScorer):
    """FakeScorer plus a length term, so thresholds below 1 can be met."""

    def score(self, texts):
        return super().score(texts) + np.array([(len(t) % 10) / 10 for t in texts])


class Recording(VariedScorer):
    fits = []  # module level: the final model is pickled

    def fit(self, texts, y):
        Recording.fits.append(len(texts))
        return super().fit(texts, y)


class ClockedScorer(VariedScorer):
    clock = 0.0  # module level: the final model is pickled

    def fit(self, texts, y):
        ClockedScorer.clock += 5.0
        return super().fit(texts, y)

    def score(self, texts):
        ClockedScorer.clock += 3.0  # time outside the fit, inside `fit_final_model`
        return super().score(texts)


class EvaluateTestCase(DataTestCase):
    def setUp(self):
        super().setUp()
        self.queries = query_texts(QUERIES)
        self.write_data(self.queries, self.names)
        self.write_sanity_labels(SANITY, [0, 0, 1, 0])

    def write_sanity_labels(self, texts, labels):
        rows = "".join(f"{i + 1},{t},{l}\n" for i, (t, l) in enumerate(zip(texts, labels)))
        self.sanity_path.write_text("id,query,label\n" + rows, encoding="utf-8")

    def run_evaluate(self, negatives=(PRIMARY,), n=N, models=ALL_MODELS, target_fpr=0.5,
                     lofo=None, catalogue_gap=None, **kw):
        out, err = io.StringIO(), io.StringIO()
        config = dict(TINY, registry=build_registry(), **kw)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli._evaluate(
                list(negatives), n, list(models), target_fpr, lofo, catalogue_gap,
                query_path=self.query_path, product_path=self.product_path,
                checksums_path=self.checksums, sanity_path=self.sanity_path,
                results_dir=self.results, **config)
        return code, out.getvalue(), err.getvalue()

    @staticmethod
    def fake_mlp():
        """For tests that pin a name-based rule: the real MLP fit would only cost time."""
        return mock.patch.dict(cli.MODEL_FACTORIES, {"mlp": VariedScorer})

    def report(self, **kw):
        for path in self.results.glob("*") if self.results.exists() else ():
            path.unlink()  # one run per call: the glob below must find exactly one report
        code, out, err = self.run_evaluate(**kw)
        self.assertEqual(code, 0, err)
        files = sorted(self.results.glob("evaluate-*.json"))
        self.assertEqual(len(files), 1)
        return json.loads(files[0].read_text())

    def run_main(self, argv):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            main(["evaluate", *argv])
        return cm.exception.code, err.getvalue()


class EndToEndTest(EvaluateTestCase):
    def test_primary_arm_writes_one_file_with_every_section_per_model(self):
        r = self.report()
        self.assertEqual(r["config"]["command"], "evaluate")
        self.assertEqual(r["config"]["parameters"]["models"], ALL_MODELS)
        self.assertEqual(
            set(r) - {"config"}, {PRIMARY, "sanity", "part_number_diagnostic", "timing"})
        self.assertEqual(set(r[PRIMARY]["models"]), set(ALL_MODELS))
        self.assertEqual(r["sanity"]["status"], "scored")
        self.assertEqual(set(r["sanity"]["models"][PRIMARY]), set(ALL_MODELS))
        diagnostic = r["part_number_diagnostic"]["models"][PRIMARY]
        self.assertEqual(set(diagnostic), set(ALL_MODELS))
        self.assertEqual(diagnostic["lr"]["n"], 100)

    def test_product_and_real_write_one_file_with_both_arms(self):
        both = ["lr", "trivial"]  # the MLP's comparison-arm fits are the slow part
        r = self.report(negatives=(PRIMARY, COMPARISON), models=both)
        self.assertEqual(r["config"]["parameters"]["arms"], [PRIMARY, COMPARISON])
        for arm in (PRIMARY, COMPARISON):
            self.assertEqual(set(r[arm]["models"]), set(both))
            self.assertEqual(set(r["sanity"]["models"][arm]), set(both))
            self.assertEqual(set(r["part_number_diagnostic"]["models"][arm]), set(both))

    def test_comparison_arm_alone(self):
        r = self.report(negatives=(COMPARISON,), models=["lr", "trivial"])
        self.assertNotIn(PRIMARY, r)
        self.assertEqual(set(r["sanity"]["models"]), {COMPARISON})

    def test_models_limits_the_entries(self):
        r = self.report(negatives=(PRIMARY, COMPARISON), models=["trivial"])
        for arm in (PRIMARY, COMPARISON):
            self.assertEqual(set(r[arm]["models"]), {"trivial"})
            self.assertEqual(set(r["sanity"]["models"][arm]), {"trivial"})
            self.assertEqual(set(r["part_number_diagnostic"]["models"][arm]), {"trivial"})

    def test_the_trivial_baseline_row_is_present(self):
        r = self.report(models=["lr", "trivial"])
        self.assertIn("pooled", r[PRIMARY]["models"]["trivial"])

    def test_summary_output_names_the_file_and_the_fit_time(self):
        code, out, err = self.run_evaluate()
        self.assertEqual(code, 0)
        path = next(self.results.glob("evaluate-*.json"))
        self.assertIn(str(path), out)
        self.assertIn("MLP", out)

    def test_final_models_are_saved_for_every_arm_and_model(self):
        both = ["lr", "trivial"]
        self.report(negatives=(PRIMARY, COMPARISON), models=both)
        names = {p.name.rsplit("-", 1)[0] for p in self.results.glob("final-*.pkl")}
        self.assertEqual(names, {f"final-{arm}-{m}" for arm in (PRIMARY, COMPARISON)
                                 for m in both})


class FitTimeTest(EvaluateTestCase):
    def test_the_mlp_fit_time_is_in_the_json(self):
        r = self.report()
        seconds = r["timing"]["mlp_final_fit_seconds"]
        self.assertIsInstance(seconds, float)
        self.assertGreater(seconds, 0)

    def test_it_is_the_perf_counter_difference_around_the_primary_final_mlp_fit(self):
        real = cli.fit_final_model
        ticks = iter([100.0, 107.5])

        def counter():
            return next(ticks)

        calls = []

        def wrapped(arm, make_scorer, target_fpr, model, *a, **kw):
            calls.append((arm, model))
            return real(arm, make_scorer, target_fpr, model, *a, **kw)

        with mock.patch.object(cli.time, "perf_counter", counter), \
                mock.patch.object(cli, "fit_final_model", wrapped), self.fake_mlp():
            r = self.report(negatives=(PRIMARY, COMPARISON))
        self.assertEqual(r["timing"]["mlp_final_fit_seconds"], 7.5)
        self.assertIn((PRIMARY, "mlp"), calls)
        self.assertIn((COMPARISON, "mlp"), calls)

    def test_the_saved_final_mlp_is_the_plain_scorer_not_the_timer(self):
        self.report(models=["mlp"])
        path = next(self.results.glob("final-product-mlp-*.pkl"))
        self.assertIs(type(load_final_model(path).scorer), MLPScorer)

    def test_the_time_surrounds_the_fit_alone(self):
        ClockedScorer.clock = 0.0  # a fake clock: the fit adds 5 s, each `score` call 3 s
        with mock.patch.object(cli.time, "perf_counter", lambda: ClockedScorer.clock), \
                mock.patch.dict(cli.MODEL_FACTORIES, {"mlp": ClockedScorer}):
            r = self.report(models=["mlp"])
        self.assertGreater(ClockedScorer.clock, 5.0)  # the clock moved outside the fit too
        self.assertEqual(r["timing"]["mlp_final_fit_seconds"], 5.0)

    def test_no_time_without_the_mlp_or_without_the_primary_arm(self):
        with self.fake_mlp():
            self.assertIsNone(self.report(models=["lr"])["timing"]["mlp_final_fit_seconds"])
            self.assertIsNone(
                self.report(negatives=(COMPARISON,))["timing"]["mlp_final_fit_seconds"])

    def test_a_fit_over_five_minutes_is_said_in_the_output(self):
        ticks = iter([0.0, 301.0])
        with mock.patch.object(cli.time, "perf_counter", lambda: next(ticks)):
            code, out, err = self.run_evaluate(models=["mlp"])
        self.assertEqual(code, 0)
        self.assertIn("exceeds 5 minutes", out)

    def test_a_fast_fit_does_not_say_so(self):
        code, out, err = self.run_evaluate(models=["mlp"])
        self.assertNotIn("exceeds", out)


class SectionsTest(EvaluateTestCase):
    def test_lofo_adds_its_section_to_every_arm(self):
        r = self.report(negatives=(PRIMARY, COMPARISON), models=["lr", "trivial"],
                        lofo=["lr", "trivial"])
        for arm in (PRIMARY, COMPARISON):
            self.assertEqual(set(r[arm]["lofo"]), {"lr", "trivial"})
            self.assertEqual(set(r[arm]["lofo"]["lr"]), {"A", "B", "C", "D"})
        self.assertTrue(r["config"]["parameters"]["lofo"])

    def test_the_section_models_are_in_the_run_parameters(self):
        params = self.report(models=["lr", "trivial"])["config"]["parameters"]
        self.assertIsNone(params["lofo_models"])
        self.assertIsNone(params["catalogue_gap_models"])
        params = self.report(models=["lr", "trivial"], lofo=["lr"],
                             catalogue_gap=["trivial"])["config"]["parameters"]
        self.assertEqual(params["lofo_models"], ["lr"])
        self.assertEqual(params["catalogue_gap_models"], ["trivial"])

    def test_a_bare_flag_records_all_of_models_in_the_run_parameters(self):
        real = cli._evaluate

        def with_paths(*args):
            return real(*args, query_path=self.query_path, product_path=self.product_path,
                        checksums_path=self.checksums, sanity_path=self.sanity_path,
                        results_dir=self.results, registry=build_registry(), **TINY)

        with mock.patch.object(cli, "_evaluate", side_effect=with_paths), \
                contextlib.redirect_stdout(io.StringIO()):
            code = main(["evaluate", "--negatives-size", str(N), "--target-fpr", "0.5",
                         "--models", "lr,trivial", "--lofo", "--catalogue-gap"])
        self.assertEqual(code, 0)
        params = json.loads(next(self.results.glob("evaluate-*.json")).read_text())[
            "config"]["parameters"]
        self.assertEqual(params["lofo_models"], ["lr", "trivial"])
        self.assertEqual(params["catalogue_gap_models"], ["lr", "trivial"])

    def test_no_lofo_or_gap_section_by_default(self):
        r = self.report()
        self.assertNotIn("lofo", r[PRIMARY])
        self.assertNotIn("catalogue_gap", r[PRIMARY])
        self.assertFalse(r["config"]["parameters"]["lofo"])
        self.assertFalse(r["config"]["parameters"]["catalogue_gap"])

    def test_catalogue_gap_adds_its_section_in_either_arm(self):
        r = self.report(negatives=(PRIMARY, COMPARISON), models=["lr", "trivial"],
                        catalogue_gap=["lr", "trivial"])
        for arm in (PRIMARY, COMPARISON):
            self.assertEqual(set(r[arm]["catalogue_gap"]["models"]), {"lr", "trivial"})
            self.assertIn("fpr", r[arm]["catalogue_gap"]["models"]["lr"])
        self.assertIn("folds", r[PRIMARY]["catalogue_gap"])
        self.assertTrue(r["config"]["parameters"]["catalogue_gap"])

    def test_catalogue_gap_passes_the_loaders_classes(self):
        seen = {}

        def spy(**kw):
            seen.update(real_queries=kw["real_queries"], classes=kw["query_classes"])
            return build_run_report(**kw)

        with mock.patch.object(cli, "build_run_report", spy):
            self.report(models=["lr"], catalogue_gap=["lr"])
        self.assertEqual(seen["real_queries"], self.queries)
        self.assertEqual(seen["classes"], [f"C{i % 8}" for i in range(len(self.queries))])

    def test_catalogue_gap_needs_no_classes_when_off(self):
        seen = {}

        def spy(**kw):
            seen.update(kw)
            return build_run_report(**kw)

        with mock.patch.object(cli, "build_run_report", spy):
            self.report(models=["lr"])
        self.assertFalse(seen["lofo"])
        self.assertFalse(seen["catalogue_gap"])

    def test_the_sanity_skip_record_when_labels_are_blank(self):
        self.sanity_path.write_text("id,query,label\n1,zzq plorf,\n2,wux vil,\n")
        r = self.report(models=["lr"])
        self.assertEqual(r["sanity"], {"status": "skipped", "unlabelled": 2})
        self.assertIn("part_number_diagnostic", r)


class ModelSubsetTest(EvaluateTestCase):
    """Criterion 4: `--lofo` and `--catalogue-gap` model subsets and the MLP rule."""

    NOT_RUN = "not run"

    def assert_not_run(self, row, *words):
        self.assertEqual(row["status"], self.NOT_RUN)
        for word in words:
            self.assertIn(word, row["reason"])

    def assert_main_and_final_present(self, r, arms=(PRIMARY,)):
        for arm in arms:
            self.assertIn("pooled", r[arm]["models"]["mlp"])
            self.assertIn("mlp", r["sanity"]["models"][arm])
            self.assertIn("mlp", r["part_number_diagnostic"]["models"][arm])
        names = {p.name for p in self.results.glob("final-*-mlp-*.pkl")}
        self.assertEqual(len(names), len(arms))

    def test_lofo_never_runs_the_mlp_in_the_comparison_arm_even_when_named(self):
        with self.fake_mlp():
            r = self.report(negatives=(PRIMARY, COMPARISON), lofo=ALL_MODELS)
        self.assertEqual(set(r[PRIMARY]["lofo"]["mlp"]), {"A", "B", "C", "D"})
        self.assert_not_run(r[COMPARISON]["lofo"]["mlp"], "comparison", "8.6")
        for model in ("lr", "trivial"):
            self.assertEqual(set(r[COMPARISON]["lofo"][model]), {"A", "B", "C", "D"})
        self.assert_main_and_final_present(r, (PRIMARY, COMPARISON))

    def test_lofo_without_mlp_reports_its_rows_as_not_run_in_both_arms(self):
        with self.fake_mlp():
            r = self.report(negatives=(PRIMARY, COMPARISON), lofo=["lr", "trivial"])
        for arm in (PRIMARY, COMPARISON):
            self.assert_not_run(r[arm]["lofo"]["mlp"], "--lofo")
            self.assertEqual(set(r[arm]["lofo"]["lr"]), {"A", "B", "C", "D"})
        self.assert_main_and_final_present(r, (PRIMARY, COMPARISON))

    def test_catalogue_gap_without_mlp_reports_its_rows_as_not_run(self):
        with self.fake_mlp():
            r = self.report(negatives=(PRIMARY, COMPARISON), catalogue_gap=["lr", "trivial"])
        for arm in (PRIMARY, COMPARISON):
            gap = r[arm]["catalogue_gap"]["models"]
            self.assert_not_run(gap["mlp"], "--catalogue-gap")
            self.assertIn("fpr", gap["lr"])
            self.assertIn("fpr", gap["trivial"])
        self.assert_main_and_final_present(r, (PRIMARY, COMPARISON))

    def test_catalogue_gap_runs_the_mlp_in_both_arms_when_asked(self):
        with self.fake_mlp():
            r = self.report(negatives=(COMPARISON,), models=["mlp"], catalogue_gap=["mlp"])
            self.assertIn("fpr", r[COMPARISON]["catalogue_gap"]["models"]["mlp"])
            r = self.report(models=["mlp"], catalogue_gap=["mlp"])
        self.assertIn("fpr", r[PRIMARY]["catalogue_gap"]["models"]["mlp"])

    def test_the_two_sections_take_their_own_subsets(self):
        r = self.report(lofo=["lr"], catalogue_gap=["trivial"])
        self.assertEqual(set(r[PRIMARY]["lofo"]["lr"]), {"A", "B", "C", "D"})
        self.assert_not_run(r[PRIMARY]["lofo"]["trivial"], "--lofo")
        self.assertIn("fpr", r[PRIMARY]["catalogue_gap"]["models"]["trivial"])
        self.assert_not_run(r[PRIMARY]["catalogue_gap"]["models"]["lr"], "--catalogue-gap")

    def test_models_outside_the_run_have_no_rows_at_all(self):
        r = self.report(models=["lr", "trivial"], lofo=["lr", "trivial"])
        self.assertEqual(set(r[PRIMARY]["lofo"]), {"lr", "trivial"})

    def test_not_run_models_are_never_fitted_for_their_sections(self):
        Recording.fits.clear()
        with mock.patch.dict(cli.MODEL_FACTORIES, {"mlp": Recording}):
            self.report(models=["mlp", "lr"], lofo=["lr"], catalogue_gap=["lr"])
        # Skipped for LOFO and the gap: only the 5 main folds and the primary final fit.
        self.assertEqual(len(Recording.fits), 5 + 1)


class DeterminismTest(EvaluateTestCase):
    """11.18: the real models and the folds, end to end."""

    def run_twice(self, **kw):
        reports, finals = [], []
        for _ in range(2):
            r = self.report(**kw)
            r.pop("timing")
            reports.append(r)
            finals.append({p.name.split("-20")[0]: load_final_model(p)
                           for p in self.results.glob("final-*.pkl")})
        return reports, finals

    def test_same_inputs_give_the_same_json_and_the_same_final_models(self):
        (a, b), (fa, fb) = self.run_twice(
            negatives=(PRIMARY, COMPARISON), lofo=ALL_MODELS, catalogue_gap=["lr", "trivial"])
        self.assertEqual(a, b)
        self.assertEqual(set(fa), set(fb))
        self.assertEqual(len(fa), 6)
        probe = ["zzq plorf", "oak desk", "ab12cd34", ""]
        for name in fa:
            self.assertEqual(fa[name].threshold, fb[name].threshold)
            np.testing.assert_array_equal(
                fa[name].scorer.score(probe), fb[name].scorer.score(probe))


class UsageErrorTest(EvaluateTestCase):
    def assert_usage_error(self, argv):
        with mock.patch.object(cli, "_evaluate") as ev:  # never a real run if a check breaks
            code, err = self.run_main(argv)
        ev.assert_not_called()
        self.assertEqual(code, 2)
        self.assertIn("usage", err.lower())
        return err

    def test_target_fpr_outside_the_open_unit_interval(self):
        for value in ("nan", "inf", "-inf", "0", "1", "-0.1", "1.5", "abc", ""):
            with self.subTest(value=value):
                self.assert_usage_error(["--target-fpr", value])

    def test_target_fpr_inside_is_accepted(self):
        for value in ("0.01", "0.999", "1e-6"):
            with self.subTest(value=value), \
                    mock.patch.object(cli, "_evaluate", return_value=0) as ev:
                self.assertEqual(main(["evaluate", "--target-fpr", value]), 0)
                self.assertEqual(ev.call_args.args[3], float(value))

    def test_non_positive_negatives_size(self):
        for value in ("0", "-5", "x", "1.5"):
            with self.subTest(value=value):
                self.assert_usage_error(["--negatives-size", value])

    def test_fewer_than_one_render_per_template(self):
        self.assert_usage_error(["--negatives-size", "23"])

    def test_exactly_one_render_per_template_is_accepted(self):
        with mock.patch.object(cli, "_evaluate", return_value=0) as ev:
            self.assertEqual(main(["evaluate", "--negatives-size", "24"]), 0)
        self.assertEqual(ev.call_args.args[1], 24)

    def test_unknown_model_or_arm(self):
        for argv in (["--models", "svm"], ["--models", "lr,svm"], ["--models", ""],
                     ["--models", "lr,"], ["--negatives", "wands"], ["--negatives", "product,"],
                     ["--negatives", ""], ["--lofo", "svm"], ["--catalogue-gap", "svm"],
                     ["--lofo="], ["--catalogue-gap="]):
            with self.subTest(argv=argv):
                self.assert_usage_error(argv)

    def test_lofo_model_outside_models(self):
        err = self.assert_usage_error(["--models", "lr", "--lofo", "mlp"])
        self.assertIn("--lofo", err)
        self.assertIn("mlp", err)
        self.assert_usage_error(["--models", "lr,trivial", "--lofo", "lr,mlp"])

    def test_catalogue_gap_model_outside_models(self):
        err = self.assert_usage_error(["--models", "lr", "--catalogue-gap", "trivial"])
        self.assertIn("--catalogue-gap", err)
        self.assert_usage_error(["--models", "trivial", "--catalogue-gap", "lr"])

    def test_a_usage_error_runs_nothing(self):
        with mock.patch.object(cli, "_evaluate") as ev:
            self.run_main(["--target-fpr", "nan"])
            self.run_main(["--models", "lr", "--lofo", "mlp"])
        ev.assert_not_called()


class ParsingTest(EvaluateTestCase):
    def parsed(self, argv):
        with mock.patch.object(cli, "_evaluate", return_value=0) as ev:
            self.assertEqual(main(["evaluate", *argv]), 0)
        (negatives, n, models, target_fpr, lofo, gap), kw = ev.call_args
        self.assertEqual(kw, {})
        return negatives, n, models, target_fpr, lofo, gap

    def test_defaults(self):
        self.assertEqual(
            self.parsed([]), ([PRIMARY], 5000, ["lr", "mlp", "trivial"], 0.01, None, None))

    def test_repeated_names_are_kept_once(self):
        negatives, _, models, _, lofo, _ = self.parsed(
            ["--negatives", "product,product", "--models", "lr,lr,trivial", "--lofo", "lr,lr"])
        self.assertEqual((negatives, models, lofo), ([PRIMARY], ["lr", "trivial"], ["lr"]))

    def test_negatives_and_models_are_comma_lists(self):
        negatives, n, models, *_ = self.parsed(
            ["--negatives", "product,real", "--negatives-size", "240", "--models", "trivial,lr"])
        self.assertEqual((negatives, n, models), (["product", "real"], 240, ["trivial", "lr"]))

    def test_bare_flags_cover_all_of_models(self):
        *_, lofo, gap = self.parsed(["--lofo", "--catalogue-gap"])
        self.assertEqual((lofo, gap), (ALL_MODELS, ALL_MODELS))
        *_, lofo, gap = self.parsed(["--models", "lr,trivial", "--lofo", "--catalogue-gap"])
        self.assertEqual((lofo, gap), (["lr", "trivial"], ["lr", "trivial"]))

    def test_bare_flag_before_another_flag_and_at_the_end(self):
        *_, lofo, gap = self.parsed(["--lofo", "--models", "lr", "--catalogue-gap"])
        self.assertEqual((lofo, gap), (["lr"], ["lr"]))

    def test_a_list_names_the_models(self):
        *_, lofo, gap = self.parsed(["--lofo", "lr,trivial", "--catalogue-gap=trivial"])
        self.assertEqual((lofo, gap), (["lr", "trivial"], ["trivial"]))

    def test_the_list_may_be_given_before_models(self):
        *_, lofo, gap = self.parsed(["--lofo", "lr", "--models", "trivial,lr"])
        self.assertEqual(lofo, ["lr"])

    def test_the_failure_status_is_returned(self):
        with mock.patch.object(cli, "_evaluate", return_value=1):
            self.assertEqual(main(["evaluate"]), 1)

    def test_evaluate_does_not_call_fetch_or_generate(self):
        with mock.patch.object(cli, "_evaluate", return_value=0), \
                mock.patch.object(cli, "fetch") as fetch, \
                mock.patch.object(cli, "_generate") as gen:
            main(["evaluate"])
        fetch.assert_not_called()
        gen.assert_not_called()


class FailureTest(EvaluateTestCase):
    def assert_failed_cleanly(self, run=None, **kw):
        code, out, err = run or self.run_evaluate(**kw)
        self.assertEqual(code, 1)
        self.assertTrue(err)
        self.assertFalse(list(self.results.glob("*")) if self.results.exists() else [])
        return err

    def test_missing_raw_files_exit_non_zero_and_point_at_fetch(self):
        for path in (self.query_path, self.product_path):
            data = path.read_bytes()
            path.unlink()
            with self.subTest(path=path.name):
                err = self.assert_failed_cleanly()
                self.assertIn("fetch", err)
                self.assertIn(path.name, err)
            path.write_bytes(data)

    def test_exhausted_span_pool_exits_non_zero_with_its_count(self):
        self.write_data(self.queries, self.names[:1])
        err = self.assert_failed_cleanly()
        self.assertRegex(err, r"\d")
        self.assertIn("evaluate", err)

    def test_exhausted_part_number_pool_exits_non_zero_with_its_count(self):
        constant = [(shape, lambda rng, vocabulary: "same 1") for shape, _ in SHAPES]
        with mock.patch("bqc.negatives.SHAPES", constant):
            err = self.assert_failed_cleanly()
        self.assertIn("part-number", err)
        self.assertRegex(err, r"reached 1 of 100")
        self.assertIn("evaluate", err)
        self.assertNotIn("same 1", err)

    def test_hash_mismatch_exits_non_zero(self):
        self.query_path.write_text(self.query_path.read_text() + "99\textra\tC\n")
        self.assertIn("hash", self.assert_failed_cleanly())

    def test_sanity_error_message_is_withheld(self):
        self.write_sanity_labels([self.queries[0]], [0])
        err = self.assert_failed_cleanly()
        self.assertNotIn(self.queries[0], err)

    def test_a_non_finite_score_in_a_fold_exits_non_zero_and_writes_no_run_json(self):
        with mock.patch.dict(cli.MODEL_FACTORIES, {"lr": NanScorer}):
            err = self.assert_failed_cleanly(models=["lr"])
        self.assertIn("non-finite", err.lower())

    def test_a_non_finite_score_in_the_comparison_arm(self):
        with mock.patch.dict(cli.MODEL_FACTORIES, {"lr": NanScorer}):
            self.assert_failed_cleanly(negatives=(COMPARISON,), models=["lr"])

    def test_a_non_finite_score_only_on_the_sanity_queries_writes_no_run_json(self):
        class SanityNan(VariedScorer):
            def score(self, texts):
                scores = super().score(texts)
                return np.full(len(scores), np.nan) if "zzq plorf" in texts else scores

        with mock.patch.dict(cli.MODEL_FACTORIES, {"lr": SanityNan}):
            err = self.assert_failed_cleanly(models=["lr"])
        self.assertIn("non-finite", err.lower())
        self.assertIn("sanity", err)

    def test_a_save_failure_exits_non_zero_and_writes_no_run_json(self):
        with mock.patch.object(cli, "save_final_model", side_effect=OSError("disk full")):
            err = self.assert_failed_cleanly(models=["lr"])
        self.assertIn("disk full", err)

    def test_a_fold_error_exits_non_zero(self):
        with mock.patch.object(cli, "build_run_report", side_effect=FoldError("bad fold")):
            err = self.assert_failed_cleanly()
        self.assertIn("bad fold", err)


class BuildRunReportSubsetTest(unittest.TestCase):
    """The subset parameters of `build_run_report` itself, with fakes."""

    def build(self, **kw):
        from test_folds import CONFIG, PRODUCTS, REAL
        options = dict(CONFIG, n_splits=5, target_fpr=0.5, real_queries=REAL, products=PRODUCTS,
                       part_numbers=[], sanity=[], registry=build_registry(),
                       models={"a": VariedScorer, "b": VariedScorer}, **kw)
        return build_run_report(**options)

    def test_defaults_leave_the_section_rows_unchanged(self):
        _, body = self.build(lofo=True, catalogue_gap=True,
                             query_classes=[f"C{i % 8}" for i in range(40)])
        self.assertEqual(set(body[PRIMARY]["lofo"]["a"]), {"A", "B", "C", "D"})
        self.assertIn("fpr", body[COMPARISON]["catalogue_gap"]["models"]["b"])

    def test_not_run_rows_carry_the_given_reason_per_arm(self):
        _, body = self.build(
            lofo=True, catalogue_gap=True, query_classes=[f"C{i % 8}" for i in range(40)],
            lofo_not_run={COMPARISON: {"b": "because lofo"}},
            gap_not_run={PRIMARY: {"a": "because gap"}})
        self.assertEqual(body[COMPARISON]["lofo"]["b"],
                         {"status": "not run", "reason": "because lofo"})
        self.assertEqual(set(body[PRIMARY]["lofo"]["b"]), {"A", "B", "C", "D"})
        self.assertEqual(body[PRIMARY]["catalogue_gap"]["models"]["a"],
                         {"status": "not run", "reason": "because gap"})
        self.assertIn("fpr", body[COMPARISON]["catalogue_gap"]["models"]["a"])

    def test_a_comparison_arm_skip_is_not_fitted_there(self):
        from test_folds import CONFIG, PRODUCTS, REAL
        fits = []

        class Counting(VariedScorer):
            def fit(self, texts, y):
                fits.append(1)
                return super().fit(texts, y)

        def total(**kw):
            fits.clear()
            build_run_report(**dict(
                CONFIG, n_splits=5, target_fpr=0.5, real_queries=REAL, products=PRODUCTS,
                part_numbers=[], sanity=[], registry=build_registry(), arms=[COMPARISON],
                models={"b": Counting}, **kw))
            return len(fits)

        self.assertEqual(
            total(lofo=True, lofo_not_run={COMPARISON: {"b": "x"}}), total())
        self.assertGreater(total(lofo=True), total())

    def test_not_run_models_are_not_fitted_in_the_section_passes(self):
        fits = {"a": 0, "b": 0}

        def counting(name):
            class Counting(VariedScorer):
                def fit(self, texts, y):
                    fits[name] += 1
                    return super().fit(texts, y)
            return Counting

        from test_folds import CONFIG, PRODUCTS, REAL
        build_run_report(
            **dict(CONFIG, n_splits=5, target_fpr=0.5, real_queries=REAL, products=PRODUCTS,
                   part_numbers=[],
                   sanity=[], registry=build_registry(), arms=[PRIMARY],
                   models={"a": counting("a"), "b": counting("b")}, lofo=True,
                   lofo_not_run={PRIMARY: {"b": "x"}}))
        main_fits = 10  # 5 folds x 2 repeats
        self.assertEqual(fits["b"], main_fits)
        self.assertEqual(fits["a"], main_fits + 4 * main_fits)


if __name__ == "__main__":
    unittest.main()
