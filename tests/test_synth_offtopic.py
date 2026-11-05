import re
import unittest

from bqc.synth.registry import (
    REGISTRY,
    TEMPLATES_PER_FAMILY,
    TRAINING_FAMILIES,
    generate,
    group_sizes,
    has_non_benign_token,
)
from test_synth_registry import FOLD_VOCAB, renders

SEEDS = 150
TEMPLATES = {t.template_id: t for t in REGISTRY.templates("E")}
# Words of a product search; none may appear in a pattern template (2.1).
SHOPPING_WORDS = {"buy", "cheap", "price", "sale", "shop", "order", "shoes", "case",
                  "phone", "laptop", "size", "sofa", "rug", "chair", "table", "lamp"}
# Vocabulary holding every marker option of the pattern templates: nothing is left to pick.
EXHAUSTIVE_VOCAB = ["france", "japan", "brazil", "canada", "egypt", "norway", "kenya",
                    "peru", "india", "chile", "germany", "spain", "italy", "mexico",
                    "australia", "vietnam", "poland", "sweden", "turkey", "argentina"]


def outputs(template_id, vocab=FOLD_VOCAB, n=SEEDS):
    return renders(TEMPLATES[template_id], vocab, n)


class FamilyShapeTest(unittest.TestCase):
    def test_six_templates_two_hard_family_e(self):
        ts = REGISTRY.templates("E")
        self.assertEqual(len(ts), TEMPLATES_PER_FAMILY)
        self.assertEqual({t.template_id for t in ts if t.hard},
                         {"hard_question", "hard_chat"})
        self.assertEqual({t.family for t in ts}, {"E"})

    def test_registry_validates(self):
        REGISTRY.validate(("E",))

    def test_group_sizes_are_six_groups_of_twenty(self):
        sizes = group_sizes(("E",))
        self.assertEqual(len(sizes), 6)
        self.assertEqual(set(sizes.values()), {20})


class GenerationTest(unittest.TestCase):
    def test_requesting_e_returns_exactly_120(self):
        samples = generate(FOLD_VOCAB, 1, ("E",))
        self.assertEqual(len(samples), 120)
        self.assertEqual({s.family for s in samples}, {"E"})
        self.assertEqual(len({s.template_id for s in samples}), 6)

    def test_training_output_has_no_e_sample(self):
        samples = generate(FOLD_VOCAB, 1)
        self.assertEqual(len(samples), 480)
        self.assertNotIn("E", {s.family for s in samples})
        self.assertEqual(set(TRAINING_FAMILIES), {s.family for s in samples})


class NotAProductSearchTest(unittest.TestCase):
    def test_pattern_templates_use_no_shopping_words(self):
        for tid, t in TEMPLATES.items():
            if t.hard:
                continue
            for s in outputs(tid):
                self.assertFalse(SHOPPING_WORDS & set(re.findall(r"[a-z]+", s)), (tid, s))


class PatternTemplatesTest(unittest.TestCase):
    def test_trivia_is_a_question_or_request_about_a_place(self):
        for s in outputs("trivia"):
            self.assertRegex(
                s, r"^(what is|what's|tell me|do you know|how many people|when did)\b")

    def test_chat_starts_with_an_instruction(self):
        for s in outputs("chat"):
            self.assertRegex(s, r"^(write|compose|give me|explain|translate|can you) ")

    def test_python_snippets_look_like_python(self):
        for s in outputs("python"):
            self.assertRegex(
                s, r"^(for \w+ in range\(|def \w+\(|import \w+ as |\w+ = \[|while \w+ <)")

    def test_javascript_snippets_look_like_javascript(self):
        for s in outputs("javascript"):
            self.assertRegex(
                s, r"^(console\.log\(|const \w+ = \(|if \(|document\.|for \(let )")

    def test_non_vocabulary_marker_is_chosen_when_vocabulary_has_some_options(self):
        vocab = FOLD_VOCAB + EXHAUSTIVE_VOCAB[:-1]
        for s in outputs("trivia", vocab):
            self.assertIn("argentina", s)

    def test_marker_options_exhausted_raises_value_error(self):
        with self.assertRaises(ValueError):
            outputs("trivia", EXHAUSTIVE_VOCAB, 1)


class HardTemplatesTest(unittest.TestCase):
    def test_hard_outputs_contain_the_vocabulary_word(self):
        for tid in ("hard_question", "hard_chat"):
            for s in outputs(tid, ["sentinelword"], 30):
                self.assertIn("sentinelword", s)

    def test_hard_outputs_stay_non_benign_with_common_english_vocabulary(self):
        vocab = ["what", "is", "the", "of", "who", "why", "in", "write", "a", "about",
                 "tell", "me", "story", "history", "sofa", "rug"]
        for tid in ("hard_question", "hard_chat"):
            for s in outputs(tid, vocab, 100):
                self.assertTrue(has_non_benign_token(s, vocab), (tid, s))

    def test_hard_marker_exhausted_raises_value_error(self):
        vocab = ["history", "origin", "etymology", "symbolism", "philosophy"]
        # The first frame's markers are all vocabulary words, so 50 renders reach it.
        with self.assertRaises(ValueError):
            renders(TEMPLATES["hard_question"], vocab, 50)


if __name__ == "__main__":
    unittest.main()
