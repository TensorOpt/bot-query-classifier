import os
import random
import subprocess
import sys
import unittest
from pathlib import Path

from bqc.preprocess import MAX_CODE_POINTS, normalise
from bqc.synth import registry as reg
from bqc.synth.registry import (
    ALL_FAMILIES,
    REGISTRY,
    TRAINING_FAMILIES,
    Registry,
    Sample,
    Template,
)
from synth_stubs import build_family, build_registry, gibberish, letters

ROOT = Path(__file__).resolve().parent.parent
# Families whose templates are registered; each of M2.T2-T6 adds its family here
# and imports its module in bqc/synth/__init__.py (M2.T1 criterion 8).
IMPLEMENTED_FAMILIES = ()
# Fold-sized, with the short WANDS words and number tokens that are vocabulary words.
FOLD_VOCAB = [
    "".join(chr(97 + (i // 26 ** p) % 26) for p in range(3)) for i in range(384)
] + ["a", "x", "or", '36"', "3-3/4"]
A_DIGIT_VOCAB = ["plainword", "w003096809", "18x18"]
SAMPLES_PER_TEMPLATE_CHECK = 300


def renders(template, vocab, n=SAMPLES_PER_TEMPLATE_CHECK):
    return [template.render(random.Random(i), vocab) for i in range(n)]


def check(tc, ok, rule, detail):
    """Assert with the rule's tag in the message, so tests can name the rule."""
    tc.assertTrue(ok, f"[{rule}] {detail!r}")


def assert_template_rules(tc, template, vocab=FOLD_VOCAB):
    """The per-template contract. Real templates are checked by RealTemplatesTest."""
    if template.hard:
        # First, so a stub that caches its first vocabulary is caught (rules 3, 8.3).
        for s in renders(template, ["otherword"], 20):
            check(tc, "otherword" in s, "sentinel", s)
        for s in renders(template, ["sentinelword"], 20):
            check(tc, "sentinelword" in s, "sentinel", s)
            check(tc, "otherword" not in s, "vocab-leak", s)
    outs = renders(template, vocab)
    for s in outs:
        check(tc, s.isascii(), "ascii", s)
        check(tc, s == s.lower(), "lowercase", s)
        check(tc, len(s) <= MAX_CODE_POINTS, "512", s[:40])  # before normalise truncates
        check(tc, normalise(s) == s, "normalise", s)
    check(tc, len(set(outs)) >= 50, "distinct", len(set(outs)))
    for s in outs:
        check(tc, reg.has_non_benign_token(s, vocab), "non-benign", s)
    if template.family == "A":
        for s in renders(template, A_DIGIT_VOCAB):
            if template.hard:
                check(tc, not any(c.isdigit() for c in s), "a-hard-digit", s)
            check(tc, reg.digit_tokens_below_block_rule(s) == [], "a-blocks", s)


class HelpersTest(unittest.TestCase):
    def test_block_count_examples_from_design(self):
        self.assertEqual(reg.block_count("xk9qz7vplm"), 5)
        self.assertEqual(reg.block_count("wf-2049x"), 3)
        self.assertEqual(reg.block_count("sofa"), 1)
        self.assertEqual(reg.block_count("8374651"), 1)

    def test_part_number_regex(self):
        for t in ["sku000123", "w003096809", "wf-2049x", "1234", "ab/123", "123ab"]:
            self.assertTrue(reg.PART_NUMBER_RE.match(t), t)
        for t in ["xk9qz7vplm", "12", "sofa", "abcde123", "a1b2c3"]:
            self.assertFalse(reg.PART_NUMBER_RE.match(t), t)

    def test_is_part_number_query_is_gone(self):
        self.assertFalse(hasattr(reg, "is_part_number_query"))

    def test_number_shaped_regex_matches_wands_forms(self):
        for t in ["72", "w003412", "sku000123", "wf-2049x", "18x18", "3/4", "72.5",
                  "3-3/4", '36"', "7qt", "e12/candelabra"]:
            self.assertTrue(reg.NUMBER_SHAPED_RE.match(t), t)
        for t in ["sofa", "/123", "page=17", "(555)", "a'1", ""]:
            self.assertFalse(reg.NUMBER_SHAPED_RE.match(t), t)

    def test_benign_token_number_shapes_with_at_most_three_blocks(self):
        for t in ["72", "w003412", "sku000123", "wf-2049x", "18x18", "3/4", "72.5",
                  "3-3/4", '36"', "7qt", "e12/candelabra"]:
            self.assertTrue(reg.is_benign_token(t, FOLD_VOCAB), t)

    def test_benign_token_vocabulary_word(self):
        self.assertTrue(reg.is_benign_token("sofa", ["sofa"]))
        self.assertFalse(reg.is_benign_token("sofa", ["rug"]))
        self.assertTrue(reg.is_benign_token("a", ["a"]))
        self.assertFalse(reg.is_benign_token("aa", ["a"]))

    def test_four_or_more_blocks_are_not_benign(self):
        for t in ["xk9qz7vplm", "a1b2c3", "3f2504e0-4f89-11d3-9a0c-0305e82c3301"]:
            self.assertFalse(reg.is_benign_token(t, FOLD_VOCAB), t)

    def test_punctuation_tokens_are_not_benign(self):
        for t in ["/123", "page=17", "(555)", "sofa'", "http://x.co", "&page=2"]:
            self.assertFalse(reg.is_benign_token(t, FOLD_VOCAB), t)

    def test_non_benign_check_on_design_table(self):
        vocab = ["leather", "sofa", "area", "rug", "cheap", "accent", "chair",
                 "call", "now", "a", "x", "or", "tv", "union", "select"]
        fails = ["leather sofa w003412", "area rug 18x18", "sofa", "w003412", "72",
                 "a", "x", "or", "tv"]
        passes = ["leather sofa' union select", "sofa' or 1=1--", "area rug&page=2",
                  "/search?q=accent+chair", "page=17", "cheap area rug http://x.co",
                  "call (555) 012-3456 now", "accent chair asdfgh", "xk9qz7vplm",
                  "aaaaaaaa", "aa", "ab", "what is the capital of france"]
        for q in fails:
            self.assertFalse(reg.has_non_benign_token(q, vocab), q)
        for q in passes:
            self.assertTrue(reg.has_non_benign_token(q, vocab), q)

    def test_empty_text_has_no_non_benign_token(self):
        self.assertFalse(reg.has_non_benign_token("", FOLD_VOCAB))

    def test_block_rule(self):
        self.assertEqual(reg.digit_tokens_below_block_rule("xk9qz7vplm abc"), [])
        self.assertEqual(
            reg.digit_tokens_below_block_rule("wf-2049x desk 18x18"),
            ["wf-2049x", "18x18"],
        )
        self.assertEqual(reg.digit_tokens_below_block_rule("qqqqwwwweeee 8374651"),
                         ["8374651"])


class StubContractTest(unittest.TestCase):
    def test_every_stub_template_passes(self):
        registry = build_registry()
        for family in ALL_FAMILIES:
            for t in registry.templates(family):
                with self.subTest(template=(t.family, t.template_id)):
                    assert_template_rules(self, t)

    def assert_fails_on(self, template, rule, vocab=FOLD_VOCAB):
        with self.assertRaises(AssertionError) as cm:
            assert_template_rules(self, template, vocab)
        self.assertIn(f"[{rule}]", str(cm.exception))

    def test_vocab_words_plus_part_number_fails_non_benign(self):
        t = Template("B", "bad", False, lambda rng, v:
                     f"{rng.choice(v)} {rng.choice(v)} w{rng.randint(1000, 99999)}")
        self.assert_fails_on(t, "non-benign")

    def test_stub_emitting_bare_number_fails_non_benign(self):
        t = Template("B", "bad", False, lambda rng, v: str(rng.randint(10, 99999)))
        self.assert_fails_on(t, "non-benign")

    def test_hard_stub_emitting_vocab_word_alone_fails_non_benign(self):
        t = Template("B", "bad", True, lambda rng, v: rng.choice(v))
        self.assert_fails_on(t, "non-benign")

    def test_non_vocabulary_letter_token_passes(self):
        t = Template("B", "ok", False, lambda rng, v: f"{rng.choice(v)} {letters(rng)}")
        assert_template_rules(self, t)

    def test_punctuation_bearing_token_passes(self):
        t = Template("B", "ok", False,
                     lambda rng, v: f"{rng.choice(v)}' {rng.choice(v)}")
        assert_template_rules(self, t)

    def test_five_block_gibberish_token_passes(self):
        t = Template("B", "ok", False, lambda rng, v: gibberish(rng))
        assert_template_rules(self, t)

    def test_vocabulary_word_a_fails_and_aa_passes(self):
        self.assertFalse(reg.has_non_benign_token("a", FOLD_VOCAB))
        self.assertTrue(reg.has_non_benign_token("aa", FOLD_VOCAB))
        bad = Template("B", "bad", False, lambda rng, v: "a " + rng.choice(v))
        self.assert_fails_on(bad, "non-benign")
        ok = Template("B", "ok", False, lambda rng, v: "aa " + rng.choice(v))
        assert_template_rules(self, ok)

    def test_family_a_stub_with_three_blocks_fails_block_rule(self):
        t = Template("A", "bad", False,
                     lambda rng, v: f"wf-{rng.randint(1000, 9999)}x {gibberish(rng)}")
        self.assert_fails_on(t, "a-blocks")

    def test_family_a_five_block_gibberish_passes(self):
        t = Template("A", "ok", False, lambda rng, v: gibberish(rng) + "ab")
        assert_template_rules(self, t)

    def test_family_a_hard_stub_taking_digit_word_fails(self):
        t = Template("A", "bad", True, lambda rng, v: f"{rng.choice(v)} {letters(rng)}")
        self.assert_fails_on(t, "a-hard-digit")

    def test_hard_stub_ignoring_vocabulary_fails_sentinel(self):
        t = Template("B", "bad", True, lambda rng, v: f"{letters(rng)} {gibberish(rng)}")
        self.assert_fails_on(t, "sentinel")

    def test_hard_stub_caching_first_vocabulary_fails_vocab_leak(self):
        cache = []

        def render(rng, v):
            cache.append(v[0]) if not cache else None
            return f"{v[0]} {cache[0]} {gibberish(rng)}"
        t = Template("B", "bad", True, render)
        self.assert_fails_on(t, "vocab-leak")

    def test_uppercase_nonascii_or_unnormalised_fail_their_rule(self):
        for out, rule in [("Sofa", "lowercase"), ("sofá", "ascii"),
                          ("a  b", "normalise"), (" a", "normalise")]:
            t = Template("B", "bad", False,
                         lambda rng, v, out=out: out + str(rng.random()))
            with self.subTest(out=out):
                self.assert_fails_on(t, rule)

    def test_stub_with_too_few_distinct_strings_fails(self):
        t = Template("B", "bad", False, lambda rng, v: f"abc {rng.randint(0, 9)}")
        self.assert_fails_on(t, "distinct")

    def test_over_512_code_points_fails_length_rule(self):
        t = Template("B", "bad", False, lambda rng, v: "ab " * 300 + str(rng.random()))
        self.assert_fails_on(t, "512")


class RealTemplatesTest(unittest.TestCase):
    """Runs the same contract on every registered template (M2.T2-T6 inherit it)."""

    def test_registered_families_are_the_implemented_ones(self):
        present = tuple(f for f in ALL_FAMILIES if REGISTRY.templates(f))
        self.assertEqual(present, tuple(IMPLEMENTED_FAMILIES))

    def test_registry_is_valid_for_implemented_families(self):
        REGISTRY.validate(IMPLEMENTED_FAMILIES)

    def test_registered_templates(self):
        for family in IMPLEMENTED_FAMILIES:
            for t in REGISTRY.templates(family):
                with self.subTest(template=(t.family, t.template_id)):
                    assert_template_rules(self, t)


class GenerateTest(unittest.TestCase):
    def setUp(self):
        self.registry = build_registry()

    def test_default_is_480_training_samples_without_e(self):
        samples = self.registry.generate(FOLD_VOCAB, seed=1)
        self.assertEqual(len(samples), 480)
        self.assertEqual({s.family for s in samples}, set(TRAINING_FAMILIES))
        for f in TRAINING_FAMILIES:
            self.assertEqual(sum(s.family == f for s in samples), 120)

    def test_family_e_requested_separately(self):
        samples = self.registry.generate(FOLD_VOCAB, seed=1, families=("E",))
        self.assertEqual(len(samples), 120)
        self.assertEqual({s.family for s in samples}, {"E"})

    def test_all_five_families(self):
        samples = self.registry.generate(FOLD_VOCAB, 1, ALL_FAMILIES)
        self.assertEqual(len(samples), 600)

    def test_metadata_on_every_sample(self):
        samples = self.registry.generate(FOLD_VOCAB, seed=77)
        for s in samples:
            self.assertIsInstance(s, Sample)
            self.assertTrue(s.text)
            self.assertIn(s.family, TRAINING_FAMILIES)
            self.assertTrue(s.template_id)
            self.assertEqual(s.seed, 77)

    def test_every_group_has_20(self):
        samples = self.registry.generate(FOLD_VOCAB, seed=1)
        counts = {}
        for s in samples:
            counts[(s.family, s.template_id)] = counts.get((s.family, s.template_id), 0) + 1
        self.assertEqual(len(counts), 24)
        self.assertEqual(set(counts.values()), {20})

    def test_same_seed_same_output_different_seed_differs(self):
        a = self.registry.generate(FOLD_VOCAB, seed=5)
        b = self.registry.generate(FOLD_VOCAB, seed=5)
        c = self.registry.generate(FOLD_VOCAB, seed=6)
        self.assertEqual(a, b)
        self.assertNotEqual([s.text for s in a], [s.text for s in c])

    def test_output_does_not_depend_on_other_families_requested(self):
        only_a = self.registry.generate(FOLD_VOCAB, 3, ("A",))
        both = self.registry.generate(FOLD_VOCAB, 3, ("A", "B"))
        self.assertEqual(both[:120], only_a)

    def test_same_output_across_hash_seeds(self):
        script = (
            "import sys, json; sys.path.insert(0, 'tests');"
            "from synth_stubs import build_registry;"
            "from bqc.synth.registry import REGISTRY;"
            "r = build_registry();"
            "v = ['sofa', 'rug', 'lamp', 'chair', 'desk'];"
            "print(json.dumps([tuple(s) for s in r.generate(v, 9, 'ABCDE')]));"
            f"print(json.dumps([tuple(s) for s in "
            f"REGISTRY.generate(v, 9, {tuple(IMPLEMENTED_FAMILIES)!r})]))"
        )
        outs = []
        for hs in ("1", "2", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=hs, PYTHONPATH=str(ROOT))
            p = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env,
                               capture_output=True, text=True, check=True)
            outs.append(p.stdout)
        self.assertEqual(outs[0], outs[1])
        self.assertEqual(outs[0], outs[2])

    def test_empty_vocabulary_raises(self):
        with self.assertRaises(ValueError):
            self.registry.generate([], seed=1)

    def test_unknown_family_raises(self):
        with self.assertRaises(ValueError):
            self.registry.generate(FOLD_VOCAB, 1, ("Z",))

    def test_vocabulary_not_mutated(self):
        v = list(FOLD_VOCAB)
        self.registry.generate(v, 1)
        self.assertEqual(v, FOLD_VOCAB)


class GroupSizesTest(unittest.TestCase):
    def test_sizes_without_generating(self):
        calls = []
        registry = Registry()
        for f in ALL_FAMILIES:
            for k in range(6):
                registry.register(Template(
                    f, f"{f}{k}", k < 2,
                    lambda rng, v: calls.append(1) or "x"))
        sizes = registry.group_sizes()
        self.assertEqual(calls, [])
        self.assertEqual(len(sizes), 24)
        self.assertEqual(set(sizes.values()), {20})
        self.assertNotIn(("E", "E0"), sizes)
        self.assertEqual(len(registry.group_sizes(("E",))), 6)

    def test_sizes_match_generated_groups(self):
        registry = build_registry()
        samples = registry.generate(FOLD_VOCAB, 1)
        counts = {}
        for s in samples:
            counts[(s.family, s.template_id)] = counts.get((s.family, s.template_id), 0) + 1
        self.assertEqual(registry.group_sizes(), counts)

    def test_invalid_registry_raises(self):
        with self.assertRaises(ValueError):
            Registry().group_sizes()


class ValidationTest(unittest.TestCase):
    def test_five_templates_raises(self):
        registry = build_registry(families="BCDE")
        for k in range(5):
            registry.register(Template("A", f"a{k}", k < 2, lambda rng, v: "x"))
        with self.assertRaises(ValueError):
            registry.generate(FOLD_VOCAB, 1)

    def test_seven_templates_raises(self):
        registry = build_registry()
        registry.register(Template("A", "extra", False, lambda rng, v: "x"))
        with self.assertRaises(ValueError):
            registry.generate(FOLD_VOCAB, 1)

    def test_quota_other_than_20_raises(self):
        registry = build_registry(families="BCDE")
        for k in range(6):
            registry.register(Template("A", f"a{k}", k < 2, lambda rng, v: "x",
                                       quota=10 if k == 3 else 20))
        with self.assertRaises(ValueError):
            registry.generate(FOLD_VOCAB, 1)

    def test_one_hard_template_raises(self):
        registry = build_registry(families="BCDE")
        for k in range(6):
            registry.register(Template("A", f"a{k}", k < 1, lambda rng, v: "x"))
        with self.assertRaises(ValueError):
            registry.generate(FOLD_VOCAB, 1)

    def test_two_hard_templates_ok(self):
        registry = build_registry()
        registry.validate(ALL_FAMILIES)

    def test_family_e_is_validated_too(self):
        registry = build_registry(families="ABCD")
        with self.assertRaises(ValueError):
            registry.generate(FOLD_VOCAB, 1, ("E",))
        registry.generate(FOLD_VOCAB, 1)  # training set unaffected

    def test_broken_family_outside_request_does_not_block(self):
        registry = build_registry(families="ABCD")
        registry.register(Template("E", "e0", True, lambda rng, v: "x"))
        self.assertEqual(len(registry.generate(FOLD_VOCAB, 1)), 480)

    def test_duplicate_template_id_raises(self):
        registry = Registry()
        registry.register(Template("A", "a0", True, lambda rng, v: "x"))
        with self.assertRaises(ValueError):
            registry.register(Template("A", "a0", False, lambda rng, v: "x"))

    def test_register_unknown_family_raises(self):
        with self.assertRaises(ValueError):
            Registry().register(Template("Z", "z0", True, lambda rng, v: "x"))

    def test_module_level_generate_uses_default_registry(self):
        # The default registry is empty until M2.T2-T6 register families.
        if not any(REGISTRY.templates(f) for f in TRAINING_FAMILIES):
            with self.assertRaises(ValueError):
                reg.generate(FOLD_VOCAB, 1)


if __name__ == "__main__":
    unittest.main()
