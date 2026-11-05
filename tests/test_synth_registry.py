import os
import random
import subprocess
import sys
import unittest
from pathlib import Path

from bqc.preprocess import normalise
from bqc.synth import registry as reg
from bqc.synth.registry import (
    ALL_FAMILIES,
    REGISTRY,
    TRAINING_FAMILIES,
    Registry,
    Sample,
    Template,
)
from synth_stubs import build_family, build_registry, gibberish

ROOT = Path(__file__).resolve().parent.parent
FOLD_VOCAB = [
    "".join(chr(97 + (i // 26 ** p) % 26) for p in range(3)) for i in range(384)
]
PART_VOCAB = FOLD_VOCAB + ["w003096809"]
SAMPLES_PER_TEMPLATE_CHECK = 300


def renders(template, vocab, n=SAMPLES_PER_TEMPLATE_CHECK):
    return [template.render(random.Random(i), vocab) for i in range(n)]


def assert_template_rules(tc, template, vocab=FOLD_VOCAB):
    """The per-template contract. Real templates are checked by RealTemplatesTest."""
    outs = renders(template, vocab)
    for s in outs:
        tc.assertTrue(s.isascii(), s)
        tc.assertEqual(s, s.lower(), s)
        tc.assertEqual(normalise(s), s, s)
        tc.assertLessEqual(len(normalise(s)), 512)
    tc.assertGreaterEqual(len(set(outs)), 50)
    for s in renders(template, PART_VOCAB):
        tc.assertFalse(reg.is_part_number_query(s), s)
    if template.family == "A":
        for s in outs:
            tc.assertEqual(reg.digit_tokens_below_block_rule(s), [], s)
    if template.hard:
        for s in renders(template, ["sentinelword"], 20):
            tc.assertIn("sentinelword", s)
        for s in outs:
            tc.assertNotIn("outsideword", s)


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

    def test_is_part_number_query(self):
        for q in ["w003096809", "sku000123", "wf-2049x", "8374651", "12", "7"]:
            self.assertTrue(reg.is_part_number_query(q), q)
        for q in ["wf-2049x desk", "ines shag rug w003096809", "xk9qz7vplm",
                  "aa", "", "qqqqwwwweeee 8374651"]:
            self.assertFalse(reg.is_part_number_query(q), q)

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

    def test_stub_emitting_part_number_shape_fails(self):
        t = Template("B", "bad", False, lambda rng, v: "wf-2049x")
        with self.assertRaises(AssertionError):
            assert_template_rules(self, t)

    def test_stub_emitting_bare_number_fails(self):
        t = Template("B", "bad", False, lambda rng, v: str(rng.randint(10, 99999)))
        with self.assertRaises(AssertionError):
            assert_template_rules(self, t)

    def test_hard_stub_emitting_vocab_word_alone_fails(self):
        t = Template("B", "bad", True, lambda rng, v: rng.choice(v))
        with self.assertRaises(AssertionError):
            assert_template_rules(self, t)

    def test_family_a_stub_with_three_blocks_fails(self):
        t = Template("A", "bad", False,
                     lambda rng, v: f"wf-{rng.randint(1000, 9999)}x {gibberish(rng)}")
        with self.assertRaises(AssertionError):
            assert_template_rules(self, t)

    def test_family_a_five_block_gibberish_passes(self):
        t = Template("A", "ok", False, lambda rng, v: gibberish(rng) + "ab")
        # five blocks, but "xk9qz7vplm"-style single tokens are not part numbers
        assert_template_rules(self, t)

    def test_stub_with_uppercase_nonascii_or_unnormalised_fails(self):
        for out in ["Sofa", "sofá", "a  b", " a"]:
            t = Template("B", "bad", False, lambda rng, v, out=out: out + str(rng.random()))
            with self.assertRaises(AssertionError):
                assert_template_rules(self, t)

    def test_stub_with_too_few_distinct_strings_fails(self):
        t = Template("B", "bad", False, lambda rng, v: f"abc {rng.randint(0, 9)}")
        with self.assertRaises(AssertionError):
            assert_template_rules(self, t)

    def test_over_512_code_points_fails(self):
        t = Template("B", "bad", False, lambda rng, v: "ab " * 300 + str(rng.random()))
        with self.assertRaises(AssertionError):
            assert_template_rules(self, t)


class RealTemplatesTest(unittest.TestCase):
    """Runs the same contract on every registered template (M2.T2-T6 inherit it)."""

    def test_registered_templates(self):
        for family in ALL_FAMILIES:
            for t in REGISTRY.templates(family):
                with self.subTest(template=(t.family, t.template_id)):
                    assert_template_rules(self, t)

    def test_registry_is_valid_for_registered_families(self):
        present = [f for f in ALL_FAMILIES if REGISTRY.templates(f)]
        if present:
            REGISTRY.validate(present)


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
            "r = build_registry();"
            "v = ['sofa', 'rug', 'lamp', 'chair', 'desk'];"
            "print(json.dumps([tuple(s) for s in r.generate(v, 9, 'ABCDE')]))"
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
