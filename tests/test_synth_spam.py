import random
import re
import unittest

from bqc.synth.registry import PART_NUMBER_RE, REGISTRY, TEMPLATES_PER_FAMILY
from test_synth_registry import FOLD_VOCAB, renders

SEEDS = 150
TEMPLATES = {t.template_id: t for t in REGISTRY.templates("D")}
URL_RE = re.compile(r"(https?://|www\.)\S+|\S+\.(example|test|invalid)\b\S*")
EMAIL_RE = re.compile(r"\b[a-z0-9._-]+@[a-z0-9-]+\.[a-z.]+")
# Number groups: (212) 555-0142, +1-212-555-0142, 1-212-555-0142.
PHONE_RE = re.compile(r"(?:\(\d{3}\)|\d{3}) 555-01\d\d\b|-555-01\d\d\b|\) 555-01\d\d\b")
RESERVED_DOMAIN_RE = re.compile(r"[a-z0-9-]+\.(example|test|invalid|example\.com|example\.net"
                                r"|example\.org)\b")
HARD = ("hard_url", "hard_contact")


def outputs(template_id, vocab=FOLD_VOCAB, n=SEEDS):
    return renders(TEMPLATES[template_id], vocab, n)


class FamilyShapeTest(unittest.TestCase):
    def test_six_templates_two_hard_family_d(self):
        ts = REGISTRY.templates("D")
        self.assertEqual(len(ts), TEMPLATES_PER_FAMILY)
        self.assertEqual({t.template_id for t in ts if t.hard}, set(HARD))
        self.assertEqual({t.family for t in ts}, {"D"})

    def test_registry_validates(self):
        REGISTRY.validate(("D",))


class KindsTest(unittest.TestCase):
    def test_url_template_emits_urls_only(self):
        for s in outputs("url"):
            self.assertRegex(s, URL_RE)
            self.assertNotIn(" ", s)
            self.assertNotIn("@", s)

    def test_email_template_emits_emails(self):
        for s in outputs("email"):
            self.assertRegex(s, EMAIL_RE)

    def test_phone_template_emits_555_01xx_numbers(self):
        for s in outputs("phone", n=300):
            self.assertRegex(s, PHONE_RE)

    def test_promo_covers_promotional_crypto_and_seo(self):
        outs = " | ".join(outputs("promo", n=300))
        for kind in ("buy cheap", "click here", "bitcoin", "seo", "nft", "forex"):
            self.assertIn(kind, outs)

    def test_each_kind_appears_across_the_family(self):
        everything = [s for tid in TEMPLATES for s in outputs(tid, n=300)]
        for pattern in (URL_RE, EMAIL_RE, PHONE_RE):
            self.assertTrue(any(pattern.search(s) for s in everything), pattern)


class InertTextTest(unittest.TestCase):
    def test_every_domain_is_reserved_or_fictional(self):
        for tid in TEMPLATES:
            for s in outputs(tid, n=300):
                for host in re.findall(r"@([a-z0-9.-]+)|(?:https?://|www\.)([a-z0-9.-]+)", s):
                    host = host[0] or host[1]
                    self.assertRegex(host, RESERVED_DOMAIN_RE, (tid, s))

    def test_phone_numbers_are_only_555_01xx(self):
        for tid in TEMPLATES:
            for s in outputs(tid, n=300):
                for line in re.findall(r"\b555-(\d{4})\b", s):
                    self.assertTrue(line.startswith("01"), (tid, s))
                if "555" in s:
                    self.assertRegex(s, PHONE_RE)


class NoPartNumberTest(unittest.TestCase):
    def test_no_output_or_token_is_a_part_number_or_bare_number(self):
        for tid in TEMPLATES:
            for s in outputs(tid, n=300):
                self.assertIsNone(PART_NUMBER_RE.match(s), (tid, s))
                self.assertFalse(s.isdigit(), (tid, s))

    def test_phone_output_never_matches_part_number_regex_by_token(self):
        for s in outputs("phone", n=300) + outputs("hard_contact", n=300):
            for token in s.split():
                self.assertIsNone(PART_NUMBER_RE.match(token), (token, s))
                self.assertFalse(token.isdigit(), (token, s))


class HardPositiveTest(unittest.TestCase):
    def test_hard_url_has_vocabulary_words_and_a_url(self):
        vocab = ["leather", "sofa", "rug"]
        for s in outputs("hard_url", vocab):
            self.assertRegex(s, URL_RE)
            self.assertTrue(set(s.split()) & set(vocab), s)

    def test_hard_contact_has_vocabulary_words_and_a_contact(self):
        vocab = ["leather", "sofa", "rug"]
        for s in outputs("hard_contact", vocab):
            self.assertTrue(EMAIL_RE.search(s) or PHONE_RE.search(s), s)
            self.assertTrue(set(s.split()) & set(vocab), s)
        outs = outputs("hard_contact", vocab, 300)
        self.assertTrue(any("@" in s for s in outs))
        self.assertTrue(any("555" in s for s in outs))

    def test_hard_positives_work_with_repeated_vocabulary_tokens(self):
        for tid in HARD:
            for s in outputs(tid, ["sofa"] * 5, 20):
                self.assertIn("sofa", s)

    def test_hard_positive_empty_vocabulary_raises(self):
        for tid in HARD:
            with self.assertRaises((ValueError, IndexError)):
                TEMPLATES[tid].render(random.Random(0), [])

    def test_non_hard_templates_ignore_the_vocabulary(self):
        for tid in set(TEMPLATES) - set(HARD):
            self.assertEqual(outputs(tid, ["zzz"], 20), outputs(tid, ["yyy"], 20))


if __name__ == "__main__":
    unittest.main()
