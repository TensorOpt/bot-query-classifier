import ast
import json
import tempfile
import unittest
import warnings
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from fakes import FakeScorer
from sklearn.exceptions import ConvergenceWarning
from synth_stubs import build_family, build_registry
from test_folds import CONFIG, PRODUCTS, REAL, custom_registry

import bqc.evaluate
import bqc.report
from bqc import SEED
from bqc.evaluate import (
    ARMS,
    COMPARISON,
    PRIMARY,
    FoldError,
    build_run_report,
    iter_fold_data,
    length_mix,
    run_report,
)
from bqc.negatives import (
    LENGTH_COUNTS,
    PART_NUMBER_REMOVAL_KINDS,
    PART_NUMBER_SOURCE,
    REMOVAL_KINDS,
    Span,
    build_spans,
)
from bqc.synth.partnumbers import SHAPE_IDS
from bqc.synth.registry import ALL_FAMILIES, EVAL_FAMILY, Registry, Template

ROOT = Path(bqc.evaluate.__file__).resolve().parent


class LengthScorer(FakeScorer):
    """FakeScorer whose scores vary on plain words, so a threshold below 1 can be met."""

    def score(self, texts):
        return super().score(texts) + [(len(t) % 10) / 10 for t in texts]


class NoisyScorer(LengthScorer):
    """Warns on every fit, as an MLP that does not converge does (11.12)."""

    def fit(self, texts, y):
        warnings.warn("did not converge", ConvergenceWarning)
        return super().fit(texts, y)


MODELS = {"lr": LengthScorer, "mlp": NoisyScorer}
OPTIONS = dict(CONFIG, target_fpr=0.5)


def options(**kw):
    return {**OPTIONS, "real_queries": REAL, "products": PRODUCTS, "part_numbers": [],
            "sanity": [], "models": MODELS, "registry": build_registry(), **kw}


def build(**kw):
    return build_run_report(**options(**kw))


def walk_types(value, path="body"):
    """Yield the exact type of every value, failing on anything json would coerce."""
    yield path, type(value)
    if isinstance(value, dict):
        for k, v in value.items():
            yield from walk_types(k, f"{path}.key")
            yield from walk_types(v, f"{path}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from walk_types(v, f"{path}[{i}]")


class StructureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parameters, cls.body = build()

    def test_plain_python_types_only(self):
        allowed = {dict, list, str, int, float, bool, type(None)}
        bad = [(p, t) for p, t in walk_types(self.body) if t not in allowed]
        self.assertEqual(bad, [])
        self.assertEqual(json.loads(json.dumps(self.body)), self.body)

    def test_both_arms_under_their_own_keys(self):
        self.assertEqual(set(self.body), {*ARMS, "arm_comparison"})
        for arm in ARMS:
            self.assertEqual(set(self.body[arm]["models"]), {"lr", "mlp"})
            self.assertEqual(self.body[arm]["models"]["lr"]["arm"], arm)

    def test_parameters_record_the_whole_config(self):
        self.assertEqual(self.parameters, {
            "arms": list(ARMS), "models": ["lr", "mlp"], "target_fpr": 0.5,
            "negatives_size": 120, "n_splits": 3, "repeats": 2, "test_per_template": 5,
            "eval_per_template": 4, "vocabulary_cap": 10_000, "lofo": False,
            "catalogue_gap": False, "part_number_count": 100,
            "part_number_seed_offset": 100_000})

    def test_part_number_counts_sit_next_to_the_span_removals_in_every_fold_report(self):
        for arm in ARMS:
            for f in self.body[arm]["folds"]:
                block = f["part_number_negatives"]
                self.assertEqual(block["count"], 100)
                self.assertEqual(block["per_shape"], {s: 25 for s in SHAPE_IDS})
                self.assertEqual(set(block["removals"]), set(PART_NUMBER_REMOVAL_KINDS))
                self.assertTrue(all(isinstance(v, int) for v in block["removals"].values()))
                if arm == PRIMARY:
                    self.assertEqual(f["train"]["negatives"], 120)
                self.assertIn("span_removals", f)

    def test_no_part_number_text_is_in_the_report(self):
        texts = {t for d in iter_fold_data(
            REAL, PRODUCTS, [], [], registry=build_registry(), **CONFIG)
            for t, fam in zip(d.train_texts, d.train_families) if fam == PART_NUMBER_SOURCE}
        blob = json.dumps([self.parameters, self.body])
        self.assertTrue(texts)
        for text in texts:
            self.assertNotIn(f'"{text}"', blob)

    def test_one_fold_report_per_repeat_fold_and_arm_with_counts(self):
        for arm in ARMS:
            folds = self.body[arm]["folds"]
            self.assertEqual([(f["repeat"], f["fold"]) for f in folds],
                             [(r, k) for r in range(2) for k in range(3)])
            for f in folds:
                self.assertEqual(f["arm"], arm)
                for side in ("train", "test"):
                    self.assertEqual(set(f[side]["families"]), set(ALL_FAMILIES))
                    self.assertIn("negatives", f[side])
                    self.assertIn("positives", f[side])
                    self.assertIn("dropped", f[side])
                    self.assertIn("shortfall", f[side])
                self.assertEqual(set(f["span_removals"]), set(REMOVAL_KINDS))

    def test_model_entries_carry_metrics_and_per_fold_fields(self):
        for arm in ARMS:
            lr = self.body[arm]["models"]["lr"]
            self.assertIn("pooled", lr)
            self.assertEqual(len(lr["folds"]), 6)
            for entry in lr["folds"]:
                self.assertIs(entry["target_unmet"], False)
                self.assertIsNone(entry["threshold_warning"])
                self.assertEqual(entry["fit_warnings"], [])
            self.assertIn("false_positives", lr["pooled"])
            self.assertIn("families", lr["pooled"])
            self.assertIn("templates", lr["pooled"])

    def test_convergence_warnings_are_per_fold(self):
        for arm in ARMS:
            mlp = self.body[arm]["models"]["mlp"]
            for entry in mlp["folds"]:
                self.assertTrue(entry["fit_warnings"])
                self.assertTrue(all(w == "ConvergenceWarning: did not converge"
                                    for w in entry["fit_warnings"]))
        primary = self.body[PRIMARY]["models"]["mlp"]["folds"][0]["fit_warnings"]
        comparison = self.body[COMPARISON]["models"]["mlp"]["folds"][0]["fit_warnings"]
        self.assertEqual(len(primary), 1)
        self.assertEqual(len(comparison), 6)  # five inner fits and the refit

    def test_fold_counts_come_from_the_data_once_not_per_model(self):
        for arm in ARMS:
            self.assertEqual(len(self.body[arm]["folds"]), 6)
        extras = {"target_unmet", "threshold_warning", "fit_warnings"}
        for entry in self.body[PRIMARY]["models"]["lr"]["folds"]:
            self.assertLessEqual(extras, set(entry))
            self.assertNotIn("train", entry)


class ThresholdFieldsTest(unittest.TestCase):
    def test_target_unmet_and_warning_are_reported(self):
        # FakeScorer scores every plain query 0, so no threshold reaches the target
        _, body = build(models={"fake": FakeScorer}, target_fpr=0.0001, arms=[PRIMARY])
        entries = body[PRIMARY]["models"]["fake"]["folds"]
        self.assertTrue(all(e["target_unmet"] is True for e in entries))
        self.assertTrue(all(isinstance(e["threshold_warning"], str) for e in entries))
        self.assertIn("resolution", entries[0]["threshold_warning"])

    def test_single_arm_run_has_only_that_key(self):
        _, body = build(arms=[COMPARISON])
        self.assertEqual(set(body), {COMPARISON})


class OneFoldWarnsScorer(LengthScorer):
    """Warns on the 4th fit of a counter shared by every scorer a factory makes."""
    fits = 0

    def fit(self, texts, y):
        type(self).fits += 1
        if type(self).fits == 4:
            warnings.warn("only here", ConvergenceWarning)
        return super().fit(texts, y)


class FitWarningPlacementTest(unittest.TestCase):
    def test_a_warning_sits_on_its_own_fold_entry_only(self):
        class Fresh(OneFoldWarnsScorer):
            fits = 0

        # primary arm, one fit per fold, repeat-major: the 4th fit is repeat 1, fold 0
        _, body = build(models={"w": Fresh}, arms=[PRIMARY])
        got = {(e["repeat"], e["fold"]): e["fit_warnings"]
               for e in body[PRIMARY]["models"]["w"]["folds"]}
        self.assertEqual(got.pop((1, 0)), ["ConvergenceWarning: only here"])
        self.assertTrue(all(w == [] for w in got.values()))
        self.assertEqual(len(got), 5)


def vocabulary_registry():
    """Stub registry whose last template per family renders one token of the vocabulary."""
    registry = Registry()
    for family in ALL_FAMILIES:
        build_family(registry, family)
        last = registry._templates[family].pop()
        registry.register(Template(
            family, last.template_id, last.hard, lambda rng, vocab: rng.choice(vocab)))
    return registry


def shortfalls(body, side="train"):
    return sum(sum(f[side]["shortfall"].values()) for f in body[PRIMARY]["folds"])


class ArgumentsReachTheFoldsTest(unittest.TestCase):
    """Every option of build_run_report must reach iter_fold_data (invented strings only)."""

    def test_a_positive_equal_to_a_sanity_query_raises(self):
        invented = "invented sanity probe"
        registry = custom_registry([invented])
        with self.assertRaises(FoldError):
            build(registry=registry, sanity=[invented])

    def test_part_number_strings_are_dropped_from_positives(self):
        invented = "zq-4417"
        registry = custom_registry([invented, "fresh one", "fresh two", "fresh three"])

        def dropped(part_numbers):
            _, body = build(registry=registry, part_numbers=part_numbers, arms=[PRIMARY])
            return sum(f[side]["dropped"].get("part_numbers", 0)
                       for f in body[PRIMARY]["folds"] for side in ("train", "test"))

        self.assertGreater(dropped([invented]), 0)
        self.assertEqual(dropped([]), 0)

    def test_vocabulary_cap_limits_the_primary_positives_vocabulary(self):
        registry = vocabulary_registry()
        _, capped = build(registry=registry, vocabulary_cap=2, arms=[PRIMARY])
        _, wide = build(registry=registry, vocabulary_cap=10_000, arms=[PRIMARY])
        self.assertGreater(shortfalls(capped), shortfalls(wide))

    def test_per_template_counts_are_not_swapped(self):
        _, body = build(arms=[PRIMARY])
        e_templates = len(build_registry().templates(EVAL_FAMILY))
        for f in body[PRIMARY]["folds"]:
            self.assertEqual(f["test"]["requested_per_template"], 5)
            self.assertEqual(f["test"]["families"][EVAL_FAMILY], 4 * e_templates)


class SpanMixTest(unittest.TestCase):
    @staticmethod
    def spans(seed, per_k):
        out, i = [], 0
        for k, count in per_k.items():
            for _ in range(count):
                words = [f"pw{chr(97 + i % 26)}{chr(97 + i // 26)}{chr(97 + j)}"
                         for j in range(k)]
                out.append(Span(" ".join(words), i, k, seed))
                i += 1
        return out

    def build_with(self, mixes):
        none = {kind: 0 for kind in REMOVAL_KINDS}

        def fake(products, n, seed, *args):
            return self.spans(seed, mixes[seed - SEED]), dict(none)

        with mock.patch("bqc.evaluate.build_spans", side_effect=fake):
            return build()[1]

    def test_hand_checked_mix_per_repeat_next_to_the_target(self):
        body = self.build_with([{1: 60, 2: 40, 3: 20}, {1: 30, 2: 60, 3: 30}])
        mix = body[PRIMARY]["span_lengths"]
        first, second = mix["repeats"]
        self.assertEqual((first["repeat"], second["repeat"]), (0, 1))
        self.assertEqual(first["counts"], {"1": 60, "2": 40, "3": 20})
        self.assertAlmostEqual(first["shares"]["1"], 0.5)
        self.assertAlmostEqual(first["shares"]["2"], 1 / 3)
        self.assertAlmostEqual(first["shares"]["3"], 1 / 6)
        self.assertAlmostEqual(first["mean"], 200 / 120)
        self.assertEqual(second["counts"], {"1": 30, "2": 60, "3": 30})
        self.assertAlmostEqual(second["mean"], 2.0)
        target = mix["target"]
        self.assertEqual(target["counts"], {str(k): v for k, v in LENGTH_COUNTS.items()})
        self.assertAlmostEqual(target["shares"]["1"], 23 / 480)
        self.assertAlmostEqual(target["shares"]["2"], 109 / 480)
        self.assertAlmostEqual(target["mean"], 1623 / 480)  # 3.38 words, DESIGN 3.2

    def test_comparison_arm_has_no_mix(self):
        body = self.build_with([{1: 60, 2: 40, 3: 20}, {1: 30, 2: 60, 3: 30}])
        self.assertNotIn("span_lengths", body[COMPARISON])
        for f in body[COMPARISON]["folds"]:
            self.assertNotIn("span_lengths", f)

    def test_length_mix_helper(self):
        mix = length_mix({1: 1, 3: 3})
        self.assertEqual(mix["counts"], {"1": 1, "3": 3})
        self.assertAlmostEqual(mix["shares"]["3"], 0.75)
        self.assertAlmostEqual(mix["mean"], 2.5)
        with self.assertRaises(ValueError):
            length_mix({})

    def test_mix_is_the_real_builders_output_on_a_real_build(self):
        _, body = build(arms=[PRIMARY])
        for repeat, entry in enumerate(body[PRIMARY]["span_lengths"]["repeats"]):
            spans, _ = build_spans(PRODUCTS, 120, SEED + repeat, REAL, [], [])
            self.assertEqual(sum(entry["counts"].values()), 120)
            self.assertEqual(entry["counts"],
                             {str(k): sum(s.k == k for s in spans)
                              for k in sorted({s.k for s in spans})})


class NoRemovedTextTest(unittest.TestCase):
    def test_removed_span_positive_part_number_and_sanity_text_is_absent(self):
        span_of = {r: build_spans(PRODUCTS, 120, SEED + r, REAL, [], [])[0][0].text
                   for r in (0, 1)}
        part = "kx-9931"
        sanity = ["zzsanity sentinel query"]
        registry = custom_registry([part, *span_of.values(), "fresh one", "fresh two"])
        # five outer folds, as in the real run: with three, the stub's two-template family A
        # can fall wholly into one inner held-out fold once the 100 part-number rows join the split
        _, body = build(registry=registry, part_numbers=[part], sanity=sanity, n_splits=5)
        blob = json.dumps(body)
        for text in [part, *span_of.values(), *sanity]:
            self.assertNotIn(text, blob)
        dropped = sum(sum(f[side]["dropped"].values())
                      for f in body[PRIMARY]["folds"] for side in ("train", "test"))
        self.assertGreater(dropped, 0)  # the texts were produced and removed, not never made


class DeterminismAndWriterTest(unittest.TestCase):
    def write(self, directory, second):
        stamp = datetime(2026, 11, 6, 12, 0, second, tzinfo=timezone.utc)
        return run_report(Path(directory), now=stamp, **options())

    def test_two_runs_give_identical_json_apart_from_the_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = self.write(directory, 1), self.write(directory, 2)
            self.assertNotEqual(a.name, b.name)
            self.assertTrue(a.name.endswith("-20261106T120001000000Z.json"), a.name)
            self.assertTrue(b.name.endswith("-20261106T120002000000Z.json"), b.name)
            self.assertEqual(a.read_text(), b.read_text())
            self.assertEqual(a.name.split("-")[:2], b.name.split("-")[:2])

    def test_file_holds_config_and_both_arms_and_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, 1)
            data = json.loads(path.read_text())
        config = data["config"]
        self.assertEqual(config["command"], "evaluate")
        self.assertEqual(config["seed"], SEED)
        for key in ("git_commit", "libraries", "config_hash"):
            self.assertIn(key, config)
        self.assertEqual(config["parameters"]["negatives_size"], 120)
        self.assertEqual(config["parameters"]["vocabulary_cap"], 10_000)
        self.assertEqual(set(data) - {"config"}, {*ARMS, "arm_comparison"})
        self.assertEqual(json.loads(json.dumps(data)), data)

    def test_written_through_the_one_writer(self):
        self.assertIs(bqc.evaluate.write_report, bqc.report.write_report)
        with mock.patch("bqc.evaluate.write_report", return_value=Path("x")) as writer:
            with tempfile.TemporaryDirectory() as directory:
                path = run_report(Path(directory), **options())
        self.assertEqual(path, Path("x"))
        args = writer.call_args.args
        self.assertEqual(args[1], "evaluate")
        self.assertEqual(set(args[3]), {*ARMS, "arm_comparison"})

    def test_evaluate_does_not_import_the_command_layer_or_write_json(self):
        tree = ast.parse((ROOT / "evaluate.py").read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported |= {f"{node.module}.{a.name}" for a in node.names}
                imported.add(node.module)
        self.assertFalse({"bqc.cli", "bqc.cli.cli", "bqc.cli"} & imported)
        self.assertNotIn("json", imported)


if __name__ == "__main__":
    unittest.main()
