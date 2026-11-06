import ast
import contextlib
import inspect
import io
import json
import unittest
from pathlib import Path
from unittest import mock

from synth_stubs import build_registry
from test_cli_generate import DataTestCase, query_texts

from bqc import SEED, cli, cluster
from bqc.cli import main
from bqc.negatives import token_occurrences
from bqc.synth.registry import QUOTA, REGISTRY, TRAINING_FAMILIES

QUERIES = 40
PER_TEMPLATE = 5  # 24 templates: 120 positives
POSITIVES = 24 * PER_TEMPLATE
SANITY = ["zzq plorf", "wux vil kam"]
GRID = (5, 10)  # n_components and min_cluster_size values of the tiny grid
CELLS = len(GRID) ** 2


def training_registry():
    return build_registry(TRAINING_FAMILIES)


class ClusterTestCase(DataTestCase):
    def setUp(self):
        super().setUp()
        self.queries = query_texts(QUERIES)
        self.write_data(self.queries, self.names)
        self.write_sanity(SANITY)
        self.registry = build_registry()  # includes E, which the run must not render
        # Tiny grid: the real one needs more rows and features than the fixtures have.
        for name in ("GRID_N_COMPONENTS", "GRID_MIN_CLUSTER_SIZE"):
            patcher = mock.patch.object(cluster, name, GRID)
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_cluster(self, **kw):
        out, err = io.StringIO(), io.StringIO()
        config = {"registry": self.registry, "per_template": PER_TEMPLATE, **kw}
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli._cluster(
                self.query_path, self.product_path, self.checksums, self.sanity_path,
                self.results, **config)
        return code, out.getvalue(), err.getvalue()

    def report(self, **kw):
        for path in self.results.glob("*") if self.results.exists() else ():
            path.unlink()  # one run per call
        code, out, err = self.run_cluster(**kw)
        self.assertEqual(code, 0, err)
        files = sorted(self.results.glob("cluster-*.json"))
        self.assertEqual(len(files), 1)
        return json.loads(files[0].read_text())

    def clustered(self, **kw):
        """The strings handed to run_grid, and the report of that run."""
        seen = []
        real_grid = cluster.run_grid

        def spy(texts):
            seen.append(list(texts))
            return real_grid(texts)

        with mock.patch.object(cluster, "run_grid", spy):
            report = self.report(**kw)
        self.assertEqual(len(seen), 1)
        return seen[0], report

    def part_numbers(self):
        sanity = [t for t in SANITY]
        return [s.text for s in cli._part_numbers(token_occurrences(self.queries),
                                                  self.queries, sanity)]

    def replace_first(self, texts):
        """Registry whose first rendered samples carry `texts` (for rule 4)."""
        real = self.registry.render

        def render(vocabulary, seed, templates, per_template):
            samples, shortfall = real(vocabulary, seed, templates, per_template)
            for i, text in enumerate(texts):
                samples[i] = samples[i]._replace(text=text)
            return samples, shortfall

        return mock.patch.object(self.registry, "render", render)


class EndToEndTest(ClusterTestCase):
    def test_writes_one_report_with_the_grid_scored(self):
        r = self.report()
        self.assertEqual(r["config"]["command"], "cluster")
        self.assertEqual(r["config"]["seed"], SEED)
        self.assertEqual(r["rows"], {"real": QUERIES, "positives": POSITIVES,
                                     "total": QUERIES + POSITIVES})
        self.assertEqual(len(r["grid"]), CELLS)
        pairs = [(c["n_components"], c["min_cluster_size"]) for c in r["grid"]]
        self.assertEqual(pairs, [(a, b) for a in GRID for b in GRID])
        for cell in r["grid"]:
            self.assertEqual(
                set(cell), {"n_components", "min_cluster_size", "n_clusters", "noise_fraction",
                            "oracle", "seed_rule"})
            self.assertEqual(cell["oracle"]["n_clusters"], cell["n_clusters"])
            self.assertEqual([row["k"] for row in cell["seed_rule"]], [10, 20, 40])
            self.assertIn("ari", cell["oracle"])

    def test_the_per_cell_labels_are_not_written(self):
        r = self.report()
        self.assertTrue(all("labels" not in cell for cell in r["grid"]))

    def test_the_report_round_trips_through_json(self):
        r = self.report()
        self.assertEqual(json.loads(json.dumps(r, allow_nan=False)), r)

    def test_summary_output_names_the_file(self):
        code, out, _ = self.run_cluster()
        self.assertEqual(code, 0)
        self.assertIn("wrote ", out)
        self.assertIn(".json", out)
        self.assertIn("rows: ", out)

    def test_parameters_are_recorded(self):
        p = self.report()["config"]["parameters"]
        self.assertEqual(p["per_template"], PER_TEMPLATE)
        self.assertEqual(p["families"], list(TRAINING_FAMILIES))
        self.assertEqual(p["grid"], {"n_components": list(GRID), "min_cluster_size": list(GRID)})


class StringsTest(ClusterTestCase):
    def test_real_queries_then_a_to_d_positives_from_the_real_vocabulary(self):
        texts, _ = self.clustered()
        templates = [(f, t.template_id) for f in TRAINING_FAMILIES
                     for t in self.registry.templates(f)]
        expected, _ = self.registry.render(
            token_occurrences(self.queries), SEED, templates, PER_TEMPLATE)
        self.assertEqual(texts[:QUERIES], self.queries)
        self.assertEqual(texts[QUERIES:], [s.text for s in expected])
        self.assertEqual(len(texts), QUERIES + POSITIVES)

    def test_no_family_e_no_part_number_and_no_product_span_text(self):
        texts, _ = self.clustered()
        e_templates = [("E", t.template_id) for t in self.registry.templates("E")]
        e_texts = {s.text for s in self.registry.render(
            token_occurrences(self.queries), SEED, e_templates, PER_TEMPLATE)[0]}
        self.assertFalse(e_texts & set(texts))
        self.assertFalse(set(self.part_numbers()) & set(texts))
        self.assertFalse([t for t in texts if "pwrd" in t])  # every product-name token has it
        self.assertFalse([t for t in texts if "zen" in t.split() or "chair" in t.split()])

    def test_the_defaults_are_the_real_registry_and_the_design_quota(self):
        for function in (cli._cluster, cli.build_cluster_report):
            defaults = inspect.signature(function).parameters
            self.assertIs(defaults["registry"].default, REGISTRY)
            self.assertEqual(defaults["per_template"].default, QUOTA)

    def test_the_render_call_uses_the_seed_the_quota_and_families_a_to_d(self):
        seen = {}

        def render(vocabulary, seed, templates, per_template):
            seen.update(seed=seed, per_template=per_template, templates=templates)
            return [], {}

        with mock.patch.object(self.registry, "render", render):
            self.run_cluster(per_template=QUOTA)
        self.assertEqual((seen["seed"], seen["per_template"]), (SEED, QUOTA))
        self.assertEqual(len(seen["templates"]), 24)
        self.assertEqual({f for f, _ in seen["templates"]}, set(TRAINING_FAMILIES))


class LabelsTest(ClusterTestCase):
    def test_truth_is_zero_for_real_and_one_for_positives_and_applied_after_the_grid(self):
        order, truths = [], []
        real_grid, real_oracle = cluster.run_grid, cluster.oracle_vote

        def grid(texts):
            order.append("grid")
            return real_grid(texts)

        def oracle(clusters, truth):
            order.append("oracle")
            truths.append(list(truth))
            return real_oracle(clusters, truth)

        with mock.patch.object(cluster, "run_grid", grid), \
                mock.patch.object(cluster, "oracle_vote", oracle):
            self.report()
        self.assertEqual(order[0], "grid")
        self.assertEqual(order.count("grid"), 1)
        self.assertEqual(len(truths), CELLS)
        for truth in truths:
            self.assertEqual(truth, [0] * QUERIES + [1] * POSITIVES)

    def test_the_seed_rule_gets_the_same_truth(self):
        truths = []
        real_rule = cluster.seed_rule

        def rule(clusters, truth):
            truths.append(list(truth))
            return real_rule(clusters, truth)

        with mock.patch.object(cluster, "seed_rule", rule):
            self.report()
        self.assertEqual(len(truths), CELLS)
        self.assertTrue(all(t == [0] * QUERIES + [1] * POSITIVES for t in truths))

    def test_clustering_takes_texts_only(self):
        # A grid that fails on anything but one argument would raise for a labels argument.
        def grid(*args, **kw):
            self.assertEqual((len(args), kw), (1, {}))
            return real(*args)

        real = cluster.run_grid
        with mock.patch.object(cluster, "run_grid", grid):
            self.report()


class RuleFourTest(ClusterTestCase):
    def test_drops_are_counted_and_the_texts_removed(self):
        p = self.part_numbers()
        with self.replace_first([self.queries[0], p[0], p[1], p[0]]):
            texts, r = self.clustered()
        self.assertEqual(r["positives_dropped"], {"real_queries": 1, "part_numbers": 3})
        self.assertEqual(r["rows"], {"real": QUERIES, "positives": POSITIVES - 4,
                                     "total": QUERIES + POSITIVES - 4})
        self.assertEqual(len(texts), QUERIES + POSITIVES - 4)
        self.assertEqual(texts.count(self.queries[0]), 1)
        self.assertFalse(set(p) & set(texts))

    def test_no_drops_reports_zero_counts(self):
        r = self.report()
        self.assertEqual(r["positives_dropped"], {"real_queries": 0, "part_numbers": 0})

    def test_a_dropped_string_matches_after_normalise(self):
        with self.replace_first([self.queries[1].upper()]):
            r = self.report()
        self.assertEqual(r["positives_dropped"]["real_queries"], 1)

    def test_a_sanity_collision_fails_the_run_without_the_text(self):
        with self.replace_first(["ok fine", SANITY[1].upper()]):
            code, out, err = self.run_cluster()
        self.assertEqual(code, 1)
        self.assertIn("sanity", err)
        for text in SANITY:
            self.assertNotIn(text, out + err)
            self.assertNotIn(text.upper(), out + err)
        self.assertFalse(self.results.exists() and list(self.results.glob("*")))


class FailureTest(ClusterTestCase):
    def test_a_missing_raw_file_points_at_fetch(self):
        for path in (self.query_path, self.product_path):
            with self.subTest(path=path.name):
                self.write_data(self.queries, self.names)
                path.unlink()
                code, out, err = self.run_cluster()
                self.assertEqual(code, 1)
                self.assertIn("fetch", err)
                self.assertIn(path.name, err)
                self.assertFalse(self.results.exists() and list(self.results.glob("*")))

    def test_an_invalid_sanity_file_withholds_its_text(self):
        self.sanity_path.write_text("id,query,label\n1,secret words\n")
        code, out, err = self.run_cluster()
        self.assertEqual(code, 1)
        self.assertNotIn("secret", out + err)

    def test_a_wrong_checksum_exits_non_zero(self):
        self.checksums.write_text("0" * 64 + "  query.tsv\n" + "0" * 64 + "  product.tsv\n")
        code, _, err = self.run_cluster()
        self.assertEqual(code, 1)
        self.assertTrue(err)


class DeterminismTest(ClusterTestCase):
    def test_two_runs_give_the_same_json_apart_from_the_timestamp(self):
        a = self.report()
        b = self.report()
        self.assertEqual(a, b)


class ParsingTest(unittest.TestCase):
    def test_the_command_runs_with_the_real_defaults(self):
        with mock.patch.object(cli, "_cluster", return_value=0) as run:
            self.assertEqual(main(["cluster"]), 0)
        run.assert_called_once_with()

    def test_the_failure_status_is_returned(self):
        with mock.patch.object(cli, "_cluster", return_value=1):
            self.assertEqual(main(["cluster"]), 1)

    def test_extra_arguments_are_a_usage_error_and_run_nothing(self):
        err = io.StringIO()
        with mock.patch.object(cli, "_cluster") as run, \
                contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            main(["cluster", "--negatives-size", "5"])
        run.assert_not_called()
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("usage", err.getvalue().lower())


class ModuleBoundaryTest(unittest.TestCase):
    def test_cluster_does_not_import_the_command_layer(self):
        tree = ast.parse(Path(cluster.__file__).read_text())
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                names.add(node.module or "")
                names |= {f"{node.module}.{a.name}" for a in node.names}
        self.assertFalse({n for n in names if n.startswith("bqc.cli")})


class GridReportTest(unittest.TestCase):
    def test_the_inputs_must_be_equally_long(self):
        with self.assertRaises(ValueError):
            cluster.grid_report(["a b", "c d"], [0])


if __name__ == "__main__":
    unittest.main()
