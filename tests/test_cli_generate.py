import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bqc import SEED, cli
from bqc.cli import main, one_off_share, sanity_overlaps
from bqc.negatives import VOCABULARY_CAP, build_spans, token_occurrences
from bqc.sanity import SanityQuery
from bqc.synth import registry
from bqc.synth.partnumbers import generate_part_numbers
from bqc.wands import PRODUCT_HEADER, WandsProduct

N = 240  # 10 renders per template
FAMILIES = {"A", "B", "C", "D", "E"}


def alpha(i):
    """Letters only, unique per i: a base-26 spelling."""
    out = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        out = chr(97 + r) + out
    return out


def product_names(count=400, words=12):
    """Twelve unique words each, led by `chair`; one product is led by `zen` (13.17 check)."""
    names = []
    for i in range(count):
        lead = "zen" if i == 0 else "chair"
        names.append(" ".join([lead] + [f"pwrd{alpha(i * words + j)}" for j in range(words - 1)]))
    return names


def query_texts(count=30):
    return [f"qwrd{alpha(3 * i)} qwrd{alpha(3 * i + 1)} qwrd{alpha(3 * i + 2)}" for i in range(count)]


class DataTestCase(unittest.TestCase):
    """Writes WANDS-shaped files with real hashes into a temporary directory."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.query_path = self.tmp / "query.tsv"
        self.product_path = self.tmp / "product.tsv"
        self.checksums = self.tmp / "CHECKSUMS"
        self.sanity_path = self.tmp / "sanity.csv"
        self.results = self.tmp / "results"
        self.queries = query_texts()
        self.names = product_names()
        self.write_data(self.queries, self.names)
        self.write_sanity([])

    def write_data(self, queries, names):
        q = "query_id\tquery\tquery_class\n" + "".join(
            f"{i}\t{t}\tC{i % 8}\n" for i, t in enumerate(queries))
        blank = ["x"] * (len(PRODUCT_HEADER) - 2)
        p = "\t".join(PRODUCT_HEADER) + "\n" + "".join(
            "\t".join([str(i), t, *blank]) + "\n" for i, t in enumerate(names))
        self.query_path.write_text(q, encoding="utf-8")
        self.product_path.write_text(p, encoding="utf-8")
        self.checksums.write_text("".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in (self.query_path, self.product_path)))

    def write_sanity(self, texts):
        rows = "".join(f"{i + 1},{t},0\n" for i, t in enumerate(texts))
        self.sanity_path.write_text("id,query,label\n" + rows, encoding="utf-8")

    def run_generate(self, n=N):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli._generate(
                n, self.query_path, self.product_path, self.checksums,
                self.sanity_path, self.results)
        return code, out.getvalue(), err.getvalue()

    def report(self, n=N):
        code, out, err = self.run_generate(n)
        self.assertEqual(code, 0, err)
        files = sorted(self.results.glob("generate-*.json"))
        self.assertEqual(len(files), 1)
        return json.loads(files[0].read_text())


class ReportContentTest(DataTestCase):
    def test_success_writes_spans_and_positives_with_metadata(self):
        r = self.report()
        self.assertEqual(len(r["spans"]), N)
        self.assertEqual(set(r["spans"][0]), {"text", "product_id", "k", "seed"})
        self.assertEqual({s["seed"] for s in r["spans"]}, {SEED})
        self.assertEqual(set(r["positives"][0]), {"text", "family", "template_id", "seed"})
        self.assertEqual({p["family"] for p in r["positives"]}, FAMILIES)
        self.assertEqual(sum(p["family"] == "E" for p in r["positives"]), 120)
        self.assertEqual(r["config"]["parameters"]["negatives_size"], N)
        self.assertEqual(r["config"]["parameters"]["vocabulary_cap"], VOCABULARY_CAP)

    def test_statistics_are_present(self):
        r = self.report()
        self.assertEqual(set(r["length_stats"]), FAMILIES | {"WANDS"})
        templates = r["templates"]
        self.assertEqual(len(templates), 30)
        first = templates[0]
        self.assertEqual(
            set(first),
            {"family", "template_id", "hard", "requested", "distinct", "distinct_share",
             "shortfall", "finding", "length", "dropped"},
        )
        self.assertEqual(first["requested"], N // 24)
        self.assertEqual(first["distinct"] + first["shortfall"], first["requested"])
        self.assertEqual(set(first["length"]), {"count", "min", "max", "mean", "median"})
        self.assertEqual(set(r["distinct_counts"]), FAMILIES | {"WANDS"})
        self.assertEqual(set(r["span_audit"]["removed"]),
                         {"empty", "drawn", "real_queries", "part_numbers", "sanity"})
        self.assertEqual(r["span_audit"]["real_queries_total"], len(self.queries))
        self.assertIn("real_queries_with_span_above_0.6", r["span_audit"])
        self.assertEqual(r["near_duplicates"]["threshold"], 0.6)
        self.assertIsInstance(r["near_duplicates"]["pairs"], list)
        shares = r["one_off_shares"]
        self.assertEqual(set(shares), {"hard_positive_word_slots", "span_token_occurrences"})
        for value in shares.values():
            self.assertTrue(0 <= value <= 1)
        self.assertEqual(r["vocabulary"]["cap"], VOCABULARY_CAP)
        self.assertEqual(r["sanity"]["count"], 0)
        self.assertEqual(r["sanity"]["above_0.5"], [])

    def test_template_under_half_distinct_is_a_finding(self):
        real = registry.render

        def short(vocabulary, seed, templates, per_template):
            samples, shortfall = real(vocabulary, seed, templates, per_template)
            first = templates[0]
            kept = [s for s in samples if (s.family, s.template_id) != first]
            kept += [s for s in samples if (s.family, s.template_id) == first][:3]
            shortfall = dict(shortfall)
            shortfall[first] = per_template - 3
            return kept, shortfall

        with mock.patch.object(cli, "render", short):
            r = self.report()
        flagged = [t for t in r["templates"] if t["finding"]]
        self.assertTrue(flagged)
        self.assertTrue(all(t["distinct_share"] < 0.5 for t in flagged))
        self.assertTrue(all(t["shortfall"] > 0 for t in flagged))

    def test_negatives_size_sets_spans_and_renders(self):
        r = self.report(480)
        self.assertEqual(len(r["spans"]), 480)
        self.assertEqual({t["requested"] for t in r["templates"] if t["family"] != "E"}, {20})

    def test_part_numbers_are_kept_apart(self):
        r = self.report()
        self.assertEqual(len(r["part_numbers"]), 100)
        self.assertEqual({p["family"] for p in r["part_numbers"]}, {"P"})
        self.assertNotIn("P", {p["family"] for p in r["positives"]})
        self.assertNotIn("P", r["length_stats"])
        self.assertNotIn("P", r["distinct_counts"])
        self.assertNotIn("P", {t["family"] for t in r["templates"]})
        positive_texts = {p["text"] for p in r["positives"]}
        self.assertFalse(positive_texts & {p["text"] for p in r["part_numbers"]})
        self.assertLessEqual(
            max((pair[0] for pair in r["near_duplicates"]["pairs"]), default=0),
            len(r["positives"]) - 1)

    def test_every_output_has_a_non_benign_token(self):
        r = self.report()
        self.assertEqual(r["benign_only_outputs"], {"training": 0, "E": 0})

    def test_summary_output_has_counts_and_the_path(self):
        code, out, _ = self.run_generate()
        self.assertEqual(code, 0)
        self.assertIn(str(self.results), out)


class VocabularyTest(DataTestCase):
    def tokens(self, r, families):
        return [t for p in r["positives"] if p["family"] in families for t in p["text"].split()]

    def test_common_word_is_drawn_far_more_often_than_a_rare_one(self):
        r = self.report(480)
        tokens = self.tokens(r, "ABCD")
        self.assertGreaterEqual(tokens.count("chair"), 5 * max(tokens.count("zen"), 1))

    def test_e_uses_query_words_and_a_to_d_use_span_words(self):
        r = self.report()
        training = self.tokens(r, "ABCD")
        e = self.tokens(r, "E")
        self.assertTrue(any(t.startswith("pwrd") for t in training))
        self.assertFalse(any("qwrd" in t for t in training))
        self.assertTrue(any(t.startswith("qwrd") for t in e))
        self.assertFalse(any("pwrd" in t or t == "chair" for t in e))


class CollisionTest(DataTestCase):
    def spans_and_part_numbers(self, n=N):
        products = [WandsProduct(i, name, ()) for i, name in enumerate(self.names)]
        p = generate_part_numbers(token_occurrences(self.queries), SEED, self.queries, [])
        p_texts = [s.text for s in p]
        spans, _ = build_spans(products, n, SEED, self.queries, p_texts, [])
        return [s.text for s in spans], p_texts

    def swap_first_texts(self, texts):
        """Patch render so the first samples of each call carry the given texts."""
        real = registry.render
        calls = []

        def swapped(vocabulary, seed, templates, per_template):
            samples, shortfall = real(vocabulary, seed, templates, per_template)
            calls.append(templates[0][0])
            samples = [s._replace(text=t) for s, t in zip(samples, texts)] + samples[len(texts):]
            return samples, shortfall

        return mock.patch.object(cli, "render", swapped), calls

    def test_sanity_collision_fails_without_text(self):
        secret = "kzqv hwxj 4417"
        self.write_sanity([secret])
        patch, _ = self.swap_first_texts([secret])
        with patch:
            code, out, err = self.run_generate()
        self.assertNotEqual(code, 0)
        self.assertNotIn(secret, out + err)
        self.assertIn("sanity", err)
        self.assertRegex(err, r"\b[A-E]\b")
        self.assertFalse(list(self.results.glob("*.json")))

    def test_sanity_collision_in_family_e_fails_too(self):
        secret = "kzqv hwxj 4417"
        self.write_sanity([secret])
        real = registry.render

        def only_e(vocabulary, seed, templates, per_template):
            samples, shortfall = real(vocabulary, seed, templates, per_template)
            if templates[0][0] == "E":
                samples = [samples[0]._replace(text=secret)] + samples[1:]
            return samples, shortfall

        with mock.patch.object(cli, "render", only_e):
            code, out, err = self.run_generate()
        self.assertNotEqual(code, 0)
        self.assertNotIn(secret, out + err)
        self.assertIn("E", err)

    def test_equal_to_span_real_query_or_p_string_is_dropped_and_counted(self):
        spans, p_texts = self.spans_and_part_numbers()
        texts = [spans[0], self.queries[0], p_texts[0]]
        patch, calls = self.swap_first_texts(texts)
        with patch:
            r = self.report()
        self.assertEqual(calls, ["A", "E"])  # first template of each call
        # the three swapped samples are the first three of each call's output
        for group in ("training", "E"):
            self.assertEqual(
                r["positives_dropped"][group],
                {"spans": 1, "real_queries": 1, "part_numbers": 1})
        positive_texts = [p["text"] for p in r["positives"]]
        self.assertNotIn(self.queries[0], positive_texts)
        self.assertNotIn(p_texts[0], positive_texts)
        self.assertNotIn(spans[0], positive_texts)
        self.assertNotIn(self.queries[0], json.dumps(r))
        self.assertEqual(
            sum(t["dropped"] for t in r["templates"]), 6)


class WiringTest(DataTestCase):
    def test_build_spans_gets_the_sanity_queries_and_the_p_strings(self):
        sanity = ["kzqv hwxj 4417", "plmn rtsy 9082"]
        self.write_sanity(sanity)
        with mock.patch.object(cli, "build_spans", wraps=cli.build_spans) as spans:
            r = self.report()
        args = spans.call_args.args
        self.assertEqual(args[4], [p["text"] for p in r["part_numbers"]])
        self.assertEqual(len(args[4]), 100)
        self.assertEqual(args[5], sanity)
        self.assertEqual(list(args[3]), self.queries)

    def test_near_duplicate_calls_never_see_p_strings(self):
        self.write_sanity(["kzqv hwxj 4417", "plmn rtsy 9082"])
        with mock.patch.object(cli, "near_duplicates", wraps=cli.near_duplicates) as near:
            r = self.report()
        p_texts = {p["text"] for p in r["part_numbers"]}
        span_texts = [s["text"] for s in r["spans"]]
        calls = [c.args for c in near.call_args_list]
        by_threshold = [c[2] for c in calls]
        # real queries vs spans, positives vs spans, sanity queries vs positives
        self.assertEqual(sorted(by_threshold), [0.5, 0.6, 0.6])
        for first, second, threshold in calls:
            self.assertFalse(p_texts & set(first))
            self.assertFalse(p_texts & set(second))
            if threshold == 0.6:
                self.assertEqual(list(second), span_texts)

    def test_vocabulary_cap_limits_the_span_occurrences(self):
        with mock.patch.object(cli, "VOCABULARY_CAP", 50):
            r = self.report()
        self.assertEqual(r["vocabulary"]["span_occurrences"], 50)
        self.assertEqual(r["vocabulary"]["cap"], 50)

    def test_sanity_error_message_is_withheld(self):
        secret = self.queries[0]  # a sanity query equal to a WANDS query is rejected
        self.write_sanity([secret])
        code, out, err = self.run_generate()
        self.assertEqual(code, 1)
        self.assertTrue(err)
        self.assertNotIn(secret, out + err)
        self.assertFalse(list(self.results.glob("*.json")))


class FailureTest(DataTestCase):
    def test_missing_raw_files_exit_non_zero_and_point_at_fetch(self):
        for path in (self.query_path, self.product_path):
            data = path.read_bytes()
            path.unlink()
            code, out, err = self.run_generate()
            self.assertNotEqual(code, 0)
            self.assertIn("fetch", err)
            self.assertIn(path.name, err)
            path.write_bytes(data)

    def test_exhausted_span_pool_exits_non_zero(self):
        self.write_data(self.queries, self.names[:1])
        code, out, err = self.run_generate(2400)
        self.assertNotEqual(code, 0)
        self.assertTrue(err)
        self.assertFalse(list(self.results.glob("*.json")))

    def test_hash_mismatch_exits_non_zero(self):
        self.query_path.write_text(self.query_path.read_text() + "99\textra\tC\n")
        code, out, err = self.run_generate()
        self.assertNotEqual(code, 0)
        self.assertIn("hash", err)

    def test_non_positive_negatives_size_is_a_usage_error(self):
        for value in ("0", "-5", "x"):
            err = io.StringIO()
            with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
                main(["generate", "--negatives-size", value])
            self.assertEqual(cm.exception.code, 2)
            self.assertIn("usage", err.getvalue().lower())

    def test_fewer_than_one_render_per_template_is_a_usage_error(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            main(["generate", "--negatives-size", "23"])
        self.assertEqual(cm.exception.code, 2)


class SanityOverlapTest(DataTestCase):
    def test_report_lists_a_query_just_above_half_and_still_exits_zero(self):
        first = self.report()
        base = next(p["text"] for p in first["positives"] if len(p["text"]) >= 12)
        near = base + "zz"
        self.write_sanity(["totally unrelated words here", near])
        for old in self.results.glob("*.json"):
            old.unlink()
        r = self.report()
        listed = r["sanity"]["above_0.5"]
        self.assertEqual([e["id"] for e in listed], [2])
        self.assertEqual(set(listed[0]), {"id", "max_jaccard"})
        self.assertGreater(listed[0]["max_jaccard"], 0.5)
        self.assertEqual(r["sanity"]["count"], 2)
        self.assertNotIn(near, json.dumps(r))

    def test_helper_threshold_is_strict(self):
        sanity = [SanityQuery(1, "abcdf"), SanityQuery(2, "abcdeg"), SanityQuery(3, "zzzz")]
        found = sanity_overlaps(sanity, ["abcde"], 0.5)
        # "abcdf" vs "abcde": 2 shared 3-grams of 4 = 0.5, not listed; "abcdeg": 3 of 4 = 0.75
        self.assertEqual(found, [{"id": 2, "max_jaccard": 0.75}])

    def test_helper_takes_the_maximum_over_positives(self):
        found = sanity_overlaps([SanityQuery(7, "abcdef")], ["abcdeg", "abcdef"], 0.5)
        self.assertEqual(found, [{"id": 7, "max_jaccard": 1.0}])


class OneOffShareTest(unittest.TestCase):
    def test_share_of_single_occurrence_tokens(self):
        vocabulary = ["chair", "chair", "zen", "oak"]
        self.assertEqual(one_off_share(["zen", "oak", "chair", "chair"], vocabulary), 0.5)
        self.assertEqual(one_off_share(vocabulary, vocabulary), 0.5)

    def test_empty_tokens_give_zero(self):
        self.assertEqual(one_off_share([], ["a"]), 0.0)


class DispatchTest(unittest.TestCase):
    def test_generate_does_not_call_fetch(self):
        with mock.patch.object(cli, "fetch") as fetch, \
                mock.patch.object(cli, "_generate", return_value=0) as gen:
            self.assertEqual(main(["generate", "--negatives-size", "48"]), 0)
        gen.assert_called_once_with(48)
        fetch.assert_not_called()

    def test_default_negatives_size_is_5000(self):
        with mock.patch.object(cli, "_generate", return_value=0) as gen:
            main(["generate"])
        gen.assert_called_once_with(5000)

    def test_fetch_does_not_call_generate(self):
        with mock.patch.object(cli, "fetch") as fetch, \
                mock.patch.object(cli, "_generate") as gen:
            self.assertEqual(main(["fetch"]), 0)
        self.assertEqual(fetch.call_count, 2)
        gen.assert_not_called()

    def test_generate_failure_status_is_returned(self):
        with mock.patch.object(cli, "_generate", return_value=1):
            self.assertEqual(main(["generate"]), 1)


if __name__ == "__main__":
    unittest.main()
