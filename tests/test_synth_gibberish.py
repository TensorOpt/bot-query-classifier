import csv
import random
import re
import unittest
from pathlib import Path

from bqc.preprocess import normalise
from bqc.synth import registry as reg
from bqc.synth.registry import REGISTRY, TEMPLATES_PER_FAMILY
from test_synth_registry import A_DIGIT_VOCAB, FOLD_VOCAB, renders

ROOT = Path(__file__).resolve().parent.parent
SEEDS = 2000
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
HEX_RE = re.compile(r"^[0-9a-f]+$")
ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
TEMPLATES = {t.template_id: t for t in REGISTRY.templates("A")}


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def outputs(template_id, vocab=FOLD_VOCAB, n=SEEDS):
    return renders(TEMPLATES[template_id], vocab, n)


class FamilyShapeTest(unittest.TestCase):
    def test_six_templates_two_hard_family_a(self):
        ts = REGISTRY.templates("A")
        self.assertEqual(len(ts), TEMPLATES_PER_FAMILY)
        self.assertEqual({t.template_id for t in ts if t.hard},
                         {"hard_mash", "hard_repeat"})
        self.assertEqual({t.family for t in ts}, {"A"})

    def test_registry_validates(self):
        REGISTRY.validate(("A",))


class BlockRuleTest(unittest.TestCase):
    def test_block_rule_over_many_seeds_for_every_template(self):
        for tid, t in TEMPLATES.items():
            for vocab in (FOLD_VOCAB, A_DIGIT_VOCAB):
                for s in renders(t, vocab, SEEDS):
                    with self.subTest(template=tid):
                        self.assertEqual(reg.digit_tokens_below_block_rule(s), [], s)

    def test_digit_bearing_tokens_have_at_least_four_blocks(self):
        for tid in ("alnum", "hex"):
            for s in outputs(tid):
                for tok in s.split():
                    self.assertGreaterEqual(reg.block_count(tok), 4, s)
                    self.assertTrue(any(c.isdigit() for c in tok), s)

    def test_digit_free_templates_have_no_digits(self):
        for tid in ("mash", "repeat", "hard_mash", "hard_repeat"):
            for s in outputs(tid):
                self.assertFalse(any(c.isdigit() for c in s), (tid, s))


class HardPositiveTest(unittest.TestCase):
    def test_vocab_with_digit_words_never_leaks_digit_words(self):
        vocab = ["plainword", "w003096809", "18x18"]
        for tid in ("hard_mash", "hard_repeat"):
            for s in outputs(tid, vocab):
                self.assertNotIn("w003096809", s.split())
                self.assertNotIn("18x18", s.split())
                self.assertIn("plainword", s.split())

    def test_hard_positive_contains_only_vocab_words_and_one_gibberish_token(self):
        for tid in ("hard_mash", "hard_repeat"):
            for s in outputs(tid):
                foreign = [w for w in s.split() if w not in FOLD_VOCAB]
                self.assertEqual(len(foreign), 1, s)
                self.assertGreater(len(s.split()), 1, s)

    def test_hard_positive_vocabulary_without_digit_free_word_raises(self):
        for tid in ("hard_mash", "hard_repeat"):
            with self.assertRaises(ValueError):
                TEMPLATES[tid].render(random.Random(0), ["18x18", "w003096809"])

    def test_gibberish_token_is_redrawn_when_it_collides_with_the_vocabulary(self):
        # Every 5-letter window of the mash rows: collisions are possible, never emitted.
        vocab = ["accent"] + [r[i:i + 5] for r in ROWS for i in range(len(r) - 4)]
        for tid in ("mash", "hard_mash"):
            for s in outputs(tid, vocab, 500):
                self.assertTrue(reg.has_non_benign_token(s, vocab), s)
        for s in outputs("hard_mash", vocab, 500):
            self.assertNotIn(s.split()[-1], vocab)

    def test_gibberish_token_is_not_a_perturbed_product_word(self):
        vocab = [w for w in FOLD_VOCAB if w.isalpha()]
        for tid in ("hard_mash", "hard_repeat"):
            for s in outputs(tid, n=300):
                for tok in (w for w in s.split() if w not in FOLD_VOCAB):
                    self.assertTrue(all(edit_distance(tok, w) > 1 for w in vocab), s)


class PatternShapeTest(unittest.TestCase):
    def test_mash_tokens_walk_a_keyboard_row_window(self):
        for s in outputs("mash"):
            toks = s.split()
            self.assertTrue(1 <= len(toks) <= 3, s)
            for tok in toks:
                self.assertTrue(5 <= len(tok) <= 12, s)
                self.assertTrue(any(set(tok) <= set(r) for r in ROWS), s)
                self.assertNotIn(tok, FOLD_VOCAB)

    def test_repeat_is_a_repeated_unit(self):
        for s in outputs("repeat"):
            self.assertEqual(len(s.split()), 1, s)
            self.assertRegex(s, r"^([a-z]{1,3})\1+$")
            self.assertTrue(6 <= len(s) <= 16, s)
            self.assertNotIn(s, FOLD_VOCAB)

    def test_repeat_includes_single_character_runs_and_multi_character_units(self):
        outs = outputs("repeat")
        self.assertTrue(any(len(set(s)) == 1 for s in outs))
        self.assertTrue(any(len(set(s)) > 1 for s in outs))

    def test_alnum_alternates_letters_and_digits(self):
        for s in outputs("alnum"):
            for tok in s.split():
                self.assertRegex(tok, r"^([a-z]+\d+){3,}[a-z]*$|^(\d+[a-z]+){3,}\d*$")
                self.assertGreaterEqual(len(tok), 11)

    def test_hex_is_uuid_or_hash_shaped(self):
        kinds = set()
        for s in outputs("hex"):
            self.assertEqual(len(s.split()), 1, s)
            if UUID_RE.match(s):
                kinds.add("uuid")
            else:
                self.assertRegex(s, HEX_RE)
                self.assertIn(len(s), (32, 40, 64), s)
                kinds.add(len(s))
        self.assertEqual(kinds, {"uuid", 32, 40, 64})


class IsolationTest(unittest.TestCase):
    def test_no_template_emits_a_sanity_query(self):
        with open(ROOT / "data" / "sanity.csv", newline="", encoding="utf-8") as f:
            sanity = {normalise(r["query"]) for r in csv.DictReader(f)}
        for tid, t in TEMPLATES.items():
            for s in renders(t, FOLD_VOCAB, 3000):
                self.assertNotIn(s, sanity, tid)

    def test_no_output_is_a_part_number_shape(self):
        for tid in TEMPLATES:
            for s in outputs(tid, n=500):
                self.assertFalse(reg.PART_NUMBER_RE.match(s), s)
                for tok in s.split():
                    self.assertFalse(reg.PART_NUMBER_RE.match(tok), s)


if __name__ == "__main__":
    unittest.main()
