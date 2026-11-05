import random
import re
import string
import unittest

from bqc.synth.registry import PART_NUMBER_RE, REGISTRY, TEMPLATES_PER_FAMILY
from test_synth_registry import FOLD_VOCAB, renders

SEEDS = 150
TEMPLATES = {t.template_id: t for t in REGISTRY.templates("C")}
PAGING_KEYS = {"page", "p", "offset", "limit", "start", "per_page", "rows", "skip",
               "size", "from", "pagesize"}
KV_RE = re.compile(r"([a-z_]+)=(\d+)")


def outputs(template_id, vocab=FOLD_VOCAB, n=SEEDS):
    return renders(TEMPLATES[template_id], vocab, n)


def pairs(text):
    return KV_RE.findall(text)


class FamilyShapeTest(unittest.TestCase):
    def test_six_templates_two_hard_family_c(self):
        ts = REGISTRY.templates("C")
        self.assertEqual(len(ts), TEMPLATES_PER_FAMILY)
        self.assertEqual({t.template_id for t in ts if t.hard},
                         {"hard_fragment", "hard_search_url"})
        self.assertEqual({t.family for t in ts}, {"C"})

    def test_registry_validates(self):
        REGISTRY.validate(("C",))


class NoPartNumberTest(unittest.TestCase):
    def test_no_output_is_a_part_number_or_bare_number(self):
        for tid in TEMPLATES:
            for s in outputs(tid, n=300):
                self.assertIsNone(PART_NUMBER_RE.match(s), (tid, s))
                self.assertFalse(s.isdigit(), (tid, s))

    def test_no_output_is_a_slash_number_path(self):
        for tid in TEMPLATES:
            for s in outputs(tid, n=300):
                self.assertNotRegex(s, r"^/\d+$")


class WalkTest(unittest.TestCase):
    def test_outputs_are_one_or_two_letters_and_reach_fifty_distinct(self):
        outs = outputs("walk", n=300)
        for s in outs:
            self.assertRegex(s, r"^[a-z]{1,2}$")
        self.assertGreaterEqual(len(set(outs)), 50)

    def test_never_emits_vocabulary_tokens(self):
        vocab = ["a", "x", "or", "tv"] + FOLD_VOCAB
        for s in outputs("walk", vocab, 400):
            self.assertNotIn(s, vocab)

    def test_vocabulary_covering_every_candidate_raises(self):
        letters = string.ascii_lowercase
        vocab = list(letters) + [a + b for a in letters for b in letters]
        with self.assertRaises(ValueError):
            TEMPLATES["walk"].render(random.Random(0), vocab)


class PatternKindTest(unittest.TestCase):
    def test_path_starts_with_slash_and_has_a_word_segment(self):
        for s in outputs("path"):
            self.assertRegex(s, r"^/[a-z]")
            self.assertNotIn(" ", s)
        outs = outputs("path", n=300)
        self.assertTrue(any("?" in s for s in outs))
        self.assertTrue(any("?" not in s for s in outs))

    def test_random_keys_vary_keys_with_fixed_form_numbers(self):
        keys = set()
        for s in outputs("random_keys"):
            found = pairs(s)
            self.assertTrue(found, s)
            for key, num in found:
                self.assertRegex(num, r"^[1-9]\d?$")  # one fixed form: 1-99
                self.assertNotIn(key, PAGING_KEYS)
                keys.add(key)
        self.assertGreater(len(keys), 50)

    def test_paging_uses_fixed_key_names_and_varying_numbers(self):
        keys, nums = set(), set()
        for s in outputs("paging", n=300):
            found = pairs(s)
            self.assertTrue(found, s)
            for key, num in found:
                keys.add(key)
                nums.add(num)
        self.assertTrue(keys <= PAGING_KEYS, keys - PAGING_KEYS)
        self.assertTrue({"page", "offset", "limit"} <= keys)
        self.assertGreater(len(nums), 50)


class HardPositiveTest(unittest.TestCase):
    def test_fragment_is_vocabulary_words_then_paging_fragment(self):
        vocab = ["leather", "sofa", "rug"]
        for s in outputs("hard_fragment", vocab):
            m = re.fullmatch(r"([a-z ]+)[&?](.+)", s)
            self.assertTrue(m, s)
            self.assertTrue(set(m.group(1).split()) <= set(vocab), s)
            self.assertTrue(pairs(m.group(2)), s)

    def test_search_url_wraps_vocabulary_words(self):
        vocab = ["leather", "sofa", "rug"]
        for s in outputs("hard_search_url", vocab):
            self.assertTrue(s.startswith(("/", "?")), s)
            self.assertIn("=", s)
            self.assertNotIn(" ", s)
            words = re.findall(r"[a-z]+", s.split("=", 1)[1])
            self.assertTrue(set(words) & set(vocab), s)

    def test_hard_positives_work_with_repeated_vocabulary_tokens(self):
        for tid in ("hard_fragment", "hard_search_url"):
            for s in outputs(tid, ["sofa"] * 5, 20):
                self.assertIn("sofa", s)

    def test_hard_positive_empty_vocabulary_raises(self):
        for tid in ("hard_fragment", "hard_search_url"):
            with self.assertRaises((ValueError, IndexError)):
                TEMPLATES[tid].render(random.Random(0), [])


if __name__ == "__main__":
    unittest.main()
