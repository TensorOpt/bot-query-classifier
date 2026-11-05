import random
import re
import unittest

from bqc.synth.registry import PART_NUMBER_RE, REGISTRY, TEMPLATES_PER_FAMILY
from test_synth_registry import FOLD_VOCAB, renders

SEEDS = 150
TEMPLATES = {t.template_id: t for t in REGISTRY.templates("D")}
TLDS = "biz|info|xyz|top|club|online|shop|site|com|net"
URL_RE = re.compile(rf"(https?://|www\.)\S+|\S+\.({TLDS})\b\S*")
EMAIL_RE = re.compile(r"\b[a-z0-9._-]+@[a-z0-9-]+\.[a-z.]+")
# Number groups: (212) 867-5309, +1-212-867-5309, 1-212-86-75-309 (random digits, three line shapes).
PHONE_RE = re.compile(r"(?:\(\d{3}\) |\b\d{3}-)(?:\d{3}-\d{4}|\d{2}-\d{2}-\d{3}|\d{3}-\d{2}-\d{2})\b")
HOST_RE = re.compile(rf"[a-z]+(?:-[a-z]+)?\.({TLDS})$")
BANNED = (".example", ".test", ".invalid", "555-01")
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

    def test_phone_template_emits_phone_numbers(self):
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


class InventedDetailsTest(unittest.TestCase):
    def test_no_output_uses_reserved_shapes(self):
        for tid in TEMPLATES:
            for s in outputs(tid, n=300):
                for banned in BANNED:
                    self.assertNotIn(banned, s, (tid, s))

    def test_every_host_is_label_parts_plus_a_spam_tld(self):
        for tid in TEMPLATES:
            for s in outputs(tid, n=300):
                for host in re.findall(r"@([a-z0-9.-]+)|(?:https?://|www\.)([a-z0-9.-]+)", s):
                    host = (host[0] or host[1]).rstrip(".")
                    self.assertRegex(host, HOST_RE, (tid, s))

    def test_every_spam_tld_and_varied_area_codes_appear(self):
        outs = [s for tid in TEMPLATES for s in outputs(tid, n=300)]
        text = " ".join(outs)
        for tld in TLDS.split("|"):
            self.assertRegex(text, rf"\.{tld}\b", tld)
        areas = {m for s in outputs("phone", n=300) for m in re.findall(r"\b(\d{3})-\d", s)}
        self.assertGreater(len(areas), 20)


class NoPartNumberTest(unittest.TestCase):
    def test_no_output_or_token_is_a_part_number_or_bare_number(self):
        for tid in TEMPLATES:
            for s in outputs(tid, n=300):
                self.assertIsNone(PART_NUMBER_RE.match(s), (tid, s))
                self.assertFalse(s.isdigit(), (tid, s))
                for token in s.split():
                    self.assertIsNone(re.fullmatch(r"\d+", token), (tid, s))

    def test_phone_output_never_matches_part_number_regex_by_token(self):
        for s in outputs("phone", n=300) + outputs("hard_contact", n=300):
            for token in s.split():
                self.assertIsNone(PART_NUMBER_RE.match(token), (token, s))
                self.assertIsNone(re.fullmatch(r"\d+", token), (token, s))


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
        self.assertTrue(any(PHONE_RE.search(s) for s in outs))

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
