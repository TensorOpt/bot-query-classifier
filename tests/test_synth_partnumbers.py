import unittest
from collections import Counter
from pathlib import Path

from bqc.preprocess import normalise
from bqc.sanity import load_sanity
import bqc.synth  # noqa: F401  (registers families A-E)
from bqc.synth.audit import AuditError
from bqc.synth.partnumbers import (
    FAMILY,
    LABEL,
    PER_SHAPE,
    SHAPE_IDS,
    generate_part_numbers,
)
from bqc.synth.registry import (
    ALL_FAMILIES,
    PART_NUMBER_RE,
    REGISTRY,
    Sample,
    is_benign_token,
)
from bqc.synth.registry import generate as generate_positives

FIXTURES = Path(__file__).parent / "fixtures"
SANITY = [q.query for q in load_sanity(FIXTURES / "sanity_blank.csv", []).queries]
VOCAB = ["leather", "sofa", "sofa", "area", "rug", "accent", "chair", "oak", "desk"]


def gen(vocab=VOCAB, seed=0, negatives=(), sanity=()):
    return generate_part_numbers(vocab, seed, negatives, sanity)


def by_shape(samples):
    return {s: [x for x in samples if x.template_id == s] for s in SHAPE_IDS}


class PartNumbersTest(unittest.TestCase):
    def test_hundred_distinct_strings(self):
        out = gen()
        self.assertEqual(len(out), 100)
        self.assertEqual(len({s.text for s in out}), 100)
        self.assertTrue(all(isinstance(s, Sample) for s in out))

    def test_twenty_five_per_shape(self):
        self.assertEqual(Counter(s.template_id for s in gen()),
                         {i: PER_SHAPE for i in SHAPE_IDS})
        self.assertEqual(PER_SHAPE, 25)

    def test_family_label_and_seed_metadata(self):
        self.assertEqual((FAMILY, LABEL), ("P", 0))
        for s in gen(seed=7):
            self.assertEqual((s.family, s.seed), ("P", 7))

    def test_first_three_shapes_match_regex_as_a_whole(self):
        shapes = by_shape(gen())
        for shape in SHAPE_IDS[:3]:
            for s in shapes[shape]:
                self.assertRegex(s.text, PART_NUMBER_RE, s.text)

    def test_shapes_differ_in_structure(self):
        shapes = by_shape(gen())
        for s in shapes["prefix_digits"]:
            self.assertRegex(s.text, r"^[a-z]{1,4}\d{3,}$")
        for s in shapes["digits"]:
            self.assertRegex(s.text, r"^\d{3,}$")
        for s in shapes["prefix_digits_suffix"]:
            self.assertRegex(s.text, r"^[a-z]{1,4}[-/]?\d{3,}[a-z0-9]{1,2}$")

    def test_fourth_shape_words_then_part_number(self):
        for s in by_shape(gen())["words_part_number"]:
            *words, last = s.text.split(" ")
            self.assertTrue(words, s.text)
            self.assertRegex(last, PART_NUMBER_RE, s.text)

    def test_every_token_is_benign_against_the_vocabulary(self):
        # Three blocks at most per token (2.2), over several seeds.
        for seed in range(20):
            for s in gen(seed=seed):
                for token in s.text.split(" "):
                    self.assertTrue(is_benign_token(token, set(VOCAB)), s.text)

    def test_ascii_lowercase_normalised(self):
        for s in gen():
            self.assertTrue(s.text.isascii())
            self.assertEqual(s.text, s.text.lower())
            self.assertEqual(normalise(s.text), s.text)

    def test_same_seed_identical_different_seed_differs(self):
        self.assertEqual(gen(seed=3), gen(seed=3))
        self.assertNotEqual([s.text for s in gen(seed=3)],
                            [s.text for s in gen(seed=4)])

    def test_product_words_only_from_vocabulary(self):
        vocab = {"leather", "sofa", "rug"}
        for s in by_shape(gen(vocab=sorted(vocab)))["words_part_number"]:
            self.assertLessEqual(set(s.text.split(" ")[:-1]), vocab, s.text)

    def test_vocabulary_change_is_honoured(self):
        for s in by_shape(gen(vocab=["onlyword"]))["words_part_number"]:
            self.assertEqual(s.text.split(" ")[:-1].count("onlyword"),
                             len(s.text.split(" ")) - 1)

    def test_empty_vocabulary_raises(self):
        with self.assertRaises(ValueError):
            gen(vocab=[])

    def test_single_word_vocabulary_still_gives_100_distinct(self):
        self.assertEqual(len({s.text for s in gen(vocab=["rug"])}), 100)

    def test_not_a_registered_family(self):
        self.assertNotIn("P", ALL_FAMILIES)
        self.assertEqual(REGISTRY.templates("A")[0].family, "A")
        with self.assertRaises(KeyError):
            REGISTRY.templates("P")

    def test_no_overlap_with_training_output_or_family_e(self):
        vocab = VOCAB * 50
        positives = {normalise(s.text) for s in
                     generate_positives(vocab, 0, ALL_FAMILIES)}
        parts = {normalise(s.text) for s in gen(vocab=vocab)}
        self.assertEqual(parts & positives, set())

    def test_collision_with_negative_raises(self):
        victim = gen()[40].text
        with self.assertRaises(AuditError) as cm:
            gen(negatives=["something else", victim.upper()])
        self.assertIn("negatives", str(cm.exception))

    def test_collision_with_sanity_raises(self):
        victim = gen()[90].text
        with self.assertRaises(AuditError) as cm:
            gen(sanity=SANITY + [f"  {victim} "])
        self.assertIn("sanity", str(cm.exception))

    def test_no_collision_with_fixture_sanity_by_default(self):
        self.assertEqual(len(gen(negatives=VOCAB, sanity=SANITY)), 100)


if __name__ == "__main__":
    unittest.main()
