import ast
import collections
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from bqc import negatives
from bqc.negatives import (
    LENGTH_COUNTS,
    PART_NUMBER_CAP_FACTOR,
    PART_NUMBER_COUNT,
    PART_NUMBER_REMOVAL_KINDS,
    PART_NUMBER_SEED_OFFSET,
    PART_NUMBER_SOURCE,
    VOCABULARY_CAP,
    PoolExhausted,
    build_part_number_negatives,
    build_spans,
    fold_ascii,
    token_occurrences,
)
from bqc.preprocess import normalise
from bqc.synth.partnumbers import SHAPE_IDS, SHAPES
from bqc.wands import WandsProduct

ROOT = Path(__file__).resolve().parent.parent


def product(product_id, name):
    return WandsProduct(product_id, name, ())


def wide_pool(count=2000, words=12):
    """Products with unique words, so that spans rarely collide."""
    return [
        product(i, " ".join(f"w{i}x{j}" for j in range(words))) for i in range(count)
    ]


def only_k(k):
    return mock.patch.object(negatives, "LENGTH_COUNTS", {k: 1})


class LengthDistributionTest(unittest.TestCase):
    def test_constant_sums_to_480_and_matches_design(self):
        self.assertEqual(sum(LENGTH_COUNTS.values()), 480)
        self.assertEqual(
            LENGTH_COUNTS,
            {1: 23, 2: 109, 3: 169, 4: 99, 5: 41, 6: 16, 7: 11, 8: 8, 9: 3, 10: 1},
        )

    def test_share_of_each_k_is_close_to_the_constant(self):
        spans, _ = build_spans(wide_pool(), 4000, 1, [], [], [])
        seen = collections.Counter(s.k for s in spans)
        for k, count in LENGTH_COUNTS.items():
            self.assertAlmostEqual(seen[k] / 4000, count / 480, delta=0.03, msg=k)

    def test_short_product_redraws_the_product_not_the_length(self):
        pool = [product(1, "aa bb")] + [
            product(i, " ".join(f"w{i}x{j}" for j in range(6))) for i in range(2, 200)
        ]
        with mock.patch.object(negatives, "LENGTH_COUNTS", {4: 1}):
            spans, _ = build_spans(pool, 200, 3, [], [], [])
        self.assertEqual({s.k for s in spans}, {4})
        self.assertNotIn(1, {s.product_id for s in spans})


class FoldTest(unittest.TestCase):
    def test_non_ascii_folds(self):
        self.assertEqual(fold_ascii("Café  Müller chair"), "cafe muller chair")

    def test_dropped_word_recollapses_whitespace(self):
        self.assertEqual(fold_ascii("a 椅子 b"), "a b")

    def test_all_non_ascii_is_empty(self):
        self.assertEqual(fold_ascii("椅子"), "")

    def test_all_non_ascii_name_is_skipped(self):
        pool = [product(1, "椅子"), product(2, "chair")]
        with only_k(1):
            spans, _ = build_spans(pool, 1, 0, [], [], [])
            self.assertEqual([(s.text, s.product_id) for s in spans], [("chair", 2)])
            with self.assertRaises(PoolExhausted) as ctx:
                build_spans(pool, 2, 0, [], [], [])
        self.assertEqual(ctx.exception.reached, 1)

    def test_every_span_is_ascii_and_normalised(self):
        pool = [product(1, "Café Müller  Chair set"), product(2, "ÉCOLE bed frame")]
        pool += wide_pool(50, 10)
        spans, _ = build_spans(pool, 100, 5, [], [], [])
        for s in spans:
            self.assertTrue(s.text.isascii(), s.text)
            self.assertEqual(normalise(s.text), s.text)
            self.assertTrue(s.text)


class RemovalTest(unittest.TestCase):
    def setUp(self):
        self.pool = [product(1, "aa bb cc"), product(2, "dd ee ff")]

    def run_k3(self, **lists):
        args = {"real_queries": [], "part_numbers": [], "sanity": []}
        args.update(lists)
        with only_k(3):
            spans, counts = build_spans(self.pool, 1, 4, **args)
        self.assertEqual([s.text for s in spans], ["dd ee ff"])
        self.assertEqual(set(counts), set(negatives.REMOVAL_KINDS))
        self.assertNotIn("aa bb cc", repr((spans, counts)))
        return counts

    def test_real_query_removed_and_counted(self):
        counts = self.run_k3(real_queries=["AA  bb cc"])
        self.assertGreaterEqual(counts["real_queries"], 1)
        self.assertEqual(counts["part_numbers"] + counts["sanity"], 0)

    def test_part_number_string_removed_and_counted(self):
        counts = self.run_k3(part_numbers=["aa bb cc"])
        self.assertGreaterEqual(counts["part_numbers"], 1)
        self.assertEqual(counts["real_queries"] + counts["sanity"], 0)

    def test_sanity_query_removed_and_counted(self):
        counts = self.run_k3(sanity=["aa bb cc"])
        self.assertGreaterEqual(counts["sanity"], 1)
        self.assertEqual(counts["real_queries"] + counts["part_numbers"], 0)

    def test_duplicates_are_counted_as_drawn(self):
        with only_k(3):
            spans, counts = build_spans(self.pool, 2, 4, [], [], [])
        self.assertEqual(sorted(s.text for s in spans), ["aa bb cc", "dd ee ff"])
        self.assertGreaterEqual(counts["drawn"], 1)
        self.assertEqual(counts["empty"], 0)

    def test_empty_span_is_dropped_and_counted(self):
        real = negatives.normalise

        def fake(s):
            return "" if s == "bad" else real(s)

        pool = [product(1, "good bad")]
        with only_k(1), mock.patch.object(negatives, "normalise", fake):
            spans, counts = build_spans(pool, 1, 0, [], [], [])
        self.assertEqual([s.text for s in spans], ["good"])
        self.assertGreaterEqual(counts["empty"], 1)
        self.assertNotIn("bad", repr((spans, counts)))

    def test_no_span_equals_a_listed_string(self):
        pool = wide_pool(20, 3)
        banned = [p.name for p in pool[:10]]
        with only_k(3):
            spans, counts = build_spans(pool, 10, 2, banned, [], [])
        self.assertTrue(set(s.text for s in spans).isdisjoint(banned))
        self.assertEqual(counts["real_queries"] > 0, True)


class ExhaustionTest(unittest.TestCase):
    def test_n_above_distinct_spans_raises_with_count(self):
        pool = [product(1, "aa bb"), product(2, "cc")]
        with only_k(1):
            with self.assertRaises(PoolExhausted) as ctx:
                build_spans(pool, 10, 0, [], [], [])
        self.assertEqual(ctx.exception.reached, 3)
        self.assertIn("3", str(ctx.exception))

    def test_removals_can_exhaust_the_pool(self):
        pool = [product(1, "aa bb")]
        with only_k(1):
            with self.assertRaises(PoolExhausted) as ctx:
                build_spans(pool, 2, 0, ["aa"], [], [])
        self.assertEqual(ctx.exception.reached, 1)

    def test_no_product_long_enough_raises(self):
        pool = [product(1, "aa bb")]
        with only_k(5):
            with self.assertRaises(PoolExhausted) as ctx:
                build_spans(pool, 1, 0, [], [], [])
        self.assertEqual(ctx.exception.reached, 0)

    def test_empty_pool_raises(self):
        with self.assertRaises(PoolExhausted):
            build_spans([], 1, 0, [], [], [])


class DeterminismTest(unittest.TestCase):
    def test_same_seed_same_spans_and_different_seed_differs(self):
        pool = wide_pool(100, 10)
        a = build_spans(pool, 300, 7, ["x"], ["y"], ["z"])
        b = build_spans(pool, 300, 7, ["x"], ["y"], ["z"])
        c = build_spans(pool, 300, 8, ["x"], ["y"], ["z"])
        self.assertEqual(a, b)
        self.assertNotEqual(a[0], c[0])

    def test_identical_across_hash_seeds(self):
        code = (
            "from bqc.negatives import build_spans\n"
            "from bqc.wands import WandsProduct\n"
            "pool = [WandsProduct(i, ' '.join(f'w{i}x{j}' for j in range(10)), ())"
            " for i in range(100)]\n"
            "print(build_spans(pool, 200, 7, ['x'], ['y'], ['z']))\n"
        )
        outputs = set()
        for hash_seed in ("0", "1", "12345"):
            env = {**os.environ, "PYTHONHASHSEED": hash_seed}
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=ROOT, env=env, capture_output=True, text=True, check=True,
            )
            outputs.add(result.stdout)
        self.assertEqual(len(outputs), 1)


class SpanShapeTest(unittest.TestCase):
    def test_spans_are_contiguous_words_with_matching_metadata(self):
        pool = wide_pool(60, 10)
        by_id = {p.product_id: p.name.split() for p in pool}
        spans, _ = build_spans(pool, 200, 11, [], [], [])
        self.assertEqual(len(spans), 200)
        for s in spans:
            words = s.text.split()
            self.assertEqual(len(words), s.k)
            self.assertEqual(s.seed, 11)
            source = by_id[s.product_id]
            starts = [
                i for i in range(len(source) - s.k + 1) if source[i:i + s.k] == words
            ]
            self.assertTrue(starts, s)

    def test_default_n_is_5000(self):
        self.assertEqual(negatives.DEFAULT_N, 5000)


class TokenOccurrencesTest(unittest.TestCase):
    def test_repeats_and_order_kept_and_strings_normalised(self):
        self.assertEqual(
            token_occurrences(["Red  Chair", "chair red"]),
            ["red", "chair", "chair", "red"],
        )

    def test_cap_keeps_the_first_tokens(self):
        self.assertEqual(VOCABULARY_CAP, 10_000)
        strings = [f"t{i} t{i}" for i in range(6000)]
        full = token_occurrences(strings)
        capped = token_occurrences(strings, cap=VOCABULARY_CAP)
        self.assertEqual(len(full), 12000)
        self.assertEqual(capped, full[:10_000])

    def test_cap_below_the_token_count_returns_exactly_the_cap(self):
        # The cap is reached inside the first string, so the early return is taken (F6).
        self.assertEqual(token_occurrences(["a b c d", "e f"], cap=2), ["a", "b"])
        self.assertEqual(token_occurrences(["a b", "c d", "e f"], cap=3), ["a", "b", "c"])

    def test_empty_input(self):
        self.assertEqual(token_occurrences([]), [])


class ModuleRulesTest(unittest.TestCase):
    def test_no_sanity_import_and_no_file_access(self):
        tree = ast.parse((ROOT / "bqc" / "negatives.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [a.name for a in node.names]
            else:
                names = []
            for name in names:
                self.assertNotIn("sanity", name.split("."))
                self.assertNotIn(name, ("pathlib", "io", "csv"))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotEqual(node.func.id, "open")




VOCAB = ["leather", "sofa", "area", "rug", "accent", "chair", "oak", "desk"]
SEED_PN = 7 + PART_NUMBER_SEED_OFFSET


def pn(vocab=VOCAB, seed=SEED_PN, training=(), real=(), parts=(), sanity=()):
    return build_part_number_negatives(vocab, seed, training, real, parts, sanity)


def sequence_shape(prefix):
    """A shape that emits each string twice in a row: s0, s0, s1, s1, ..."""
    calls = []

    def render(rng, vocabulary):
        text = f"{prefix}{len(calls) // 2}"
        calls.append(text)
        return text

    render.calls = calls
    return render


class PartNumberNegativesTest(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(PART_NUMBER_SEED_OFFSET, 100_000)
        self.assertEqual(PART_NUMBER_COUNT, 100)
        self.assertEqual(PART_NUMBER_CAP_FACTOR, 20)
        self.assertEqual(PART_NUMBER_SOURCE, "part_number")

    def test_hundred_outputs_25_per_shape_in_order(self):
        out, counts = pn()
        self.assertEqual(len(out), 100)
        self.assertEqual([o.template_id for o in out[::25]], list(SHAPE_IDS))
        for i, shape in enumerate(SHAPE_IDS):
            self.assertEqual({o.template_id for o in out[i * 25:(i + 1) * 25]}, {shape})
        self.assertEqual({o.source for o in out}, {"part_number"})
        self.assertEqual({o.seed for o in out}, {SEED_PN})
        self.assertEqual(len({o.text for o in out}), 100)
        self.assertEqual(tuple(counts), PART_NUMBER_REMOVAL_KINDS)
        self.assertEqual(sum(counts.values()), 0)

    def test_same_seed_same_list_and_offset_seeds_differ(self):
        base = 42 + PART_NUMBER_SEED_OFFSET
        self.assertEqual(pn(seed=base), pn(seed=base))
        texts = [[o.text for o in pn(seed=base + r)[0]] for r in (0, 1)]
        self.assertNotEqual(texts[0], texts[1])

    def test_words_come_only_from_the_vocabulary(self):
        vocab = ["chair"] * 5 + ["zzsentinel"]
        out, _ = pn(vocab)
        rows = [o.text.split() for o in out if o.template_id == "words_part_number"]
        words = [w for row in rows for w in row[:-1]]
        self.assertTrue(set(words) <= set(vocab))
        self.assertIn("zzsentinel", words)
        others = [o.text for o in out if o.template_id != "words_part_number"]
        self.assertFalse(any("chair" in t or "zzsentinel" in t for t in others))

    def test_vocabulary_is_sampled_by_occurrence(self):
        out, _ = pn(["chair"] * 10 + ["zen"])
        words = [
            w
            for o in out
            if o.template_id == "words_part_number"
            for w in o.text.split()[:-1]
        ]
        self.assertGreater(words.count("chair"), 3 * words.count("zen"))

    def test_collisions_are_dropped_and_counted_without_text(self):
        first, _ = pn()
        # All four from shape one's first batch, which is drawn identically again.
        train, real, parts, sanity = (first[i].text for i in range(4))
        out, counts = pn(
            training=[train.upper()],
            real=[f"  {real}"],
            parts=[parts],
            sanity=[sanity],
        )
        self.assertEqual(len(out), 100)
        self.assertEqual(
            counts,
            {"training_negatives": 1, "real_queries": 1, "part_numbers": 1,
             "sanity": 1, "earlier": 0},
        )
        texts = {o.text for o in out}
        for removed in (train, real, parts, sanity):
            self.assertNotIn(removed, texts)
            self.assertNotIn(removed, repr(counts))
        self.assertEqual(len(texts), 100)

    def test_earlier_candidate_is_counted_not_kept(self):
        shapes = tuple((f"s{i}", sequence_shape(f"k{i}_")) for i in range(4))
        with mock.patch.object(negatives, "SHAPES", shapes):
            out, counts = pn()
        self.assertEqual(len(out), 100)
        self.assertEqual(len({o.text for o in out}), 100)
        self.assertEqual(counts["earlier"], 4 * 24)  # 49 draws per shape, 25 kept
        self.assertEqual([o.template_id for o in out[::25]], [f"s{i}" for i in range(4)])

    def test_constant_shape_exhausts_the_pool_at_the_cap(self):
        calls = []

        def constant(rng, vocabulary):
            calls.append(1)
            return "secretconst 123"

        shapes = (("c", constant),) + tuple(SHAPES[1:])
        with mock.patch.object(negatives, "SHAPES", shapes):
            with self.assertRaises(PoolExhausted) as ctx:
                pn()
        self.assertEqual(ctx.exception.reached, 1)
        self.assertEqual(len(calls), PART_NUMBER_CAP_FACTOR * PART_NUMBER_COUNT)
        self.assertNotIn("secretconst", str(ctx.exception))
        self.assertNotIn("123", str(ctx.exception))

    def test_cap_counts_candidates_over_the_whole_call(self):
        # Every shape drops half of its draws: 200 draws per shape stays under the cap.
        calls = []

        def twice(rng, vocabulary):
            calls.append(1)
            return f"x{len(calls) // 2}"

        with mock.patch.object(negatives, "SHAPES", tuple((f"s{i}", twice) for i in range(4))):
            out, _ = pn()
        self.assertEqual(len(out), 100)
        self.assertLessEqual(len(calls), 2000)

    def test_outputs_are_ascii_and_normalised(self):
        out, _ = pn()
        for o in out:
            self.assertTrue(o.text.isascii())
            self.assertEqual(o.text, normalise(o.text))

    def test_empty_vocabulary_raises(self):
        with self.assertRaises(ValueError):
            pn([])

    def test_opens_no_file(self):
        with mock.patch("builtins.open", side_effect=AssertionError("open")):
            pn()


if __name__ == "__main__":
    unittest.main()