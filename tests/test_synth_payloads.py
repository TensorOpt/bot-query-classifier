import random
import re
import unittest

from bqc.synth.registry import REGISTRY, TEMPLATES_PER_FAMILY
from test_synth_registry import FOLD_VOCAB, renders

SEEDS = 150
TEMPLATES = {t.template_id: t for t in REGISTRY.templates("B")}
SQL_RE = re.compile(r"union select|\bor\b.*=|sleep\(|--|;")
SCRIPT_RE = re.compile(r"<script>.*</script>|onerror=|onload=|javascript:")
TEMPLATE_RE = re.compile(r"\$\{.+\}|\{\{.+\}\}|<%=.+%>|#\{.+\}|\{% print .+ %\}")
TRAVERSAL_RE = re.compile(r"(\.\./|\.\.\\|\.\.%2f|\.\.\.\.//)")
NOSQL_RE = re.compile(r"\{\"\$(gt|ne|regex|where)\"|\[\$(gt|ne|regex)\]")


def outputs(template_id, vocab=FOLD_VOCAB, n=SEEDS):
    return renders(TEMPLATES[template_id], vocab, n)


class FamilyShapeTest(unittest.TestCase):
    def test_six_templates_two_hard_family_b(self):
        ts = REGISTRY.templates("B")
        self.assertEqual(len(ts), TEMPLATES_PER_FAMILY)
        self.assertEqual({t.template_id for t in ts if t.hard},
                         {"hard_sql", "hard_nosql"})
        self.assertEqual({t.family for t in ts}, {"B"})

    def test_registry_validates(self):
        REGISTRY.validate(("B",))


class PatternKindTest(unittest.TestCase):
    def assert_kind(self, tid, regex):
        for s in outputs(tid):
            self.assertRegex(s, regex)

    def test_sql(self):
        self.assert_kind("sql", SQL_RE)

    def test_script(self):
        self.assert_kind("script", SCRIPT_RE)

    def test_template(self):
        self.assert_kind("template", TEMPLATE_RE)

    def test_traversal(self):
        self.assert_kind("traversal", TRAVERSAL_RE)

    def test_hard_sql(self):
        self.assert_kind("hard_sql", SQL_RE)

    def test_hard_nosql(self):
        self.assert_kind("hard_nosql", NOSQL_RE)

    def test_each_template_has_varied_forms(self):
        # Sub-forms of a template all show up, not only one fixed payload.
        outs = outputs("sql", n=300)
        for marker in ("union select", "or "):
            self.assertTrue(any(marker in s for s in outs), marker)
        outs = outputs("template", n=300)
        for marker in ("{{", "<%="):
            self.assertTrue(any(marker in s for s in outs), marker)
        outs = outputs("traversal", n=300)
        self.assertTrue(any("passwd" in s for s in outs))
        self.assertTrue(any("passwd" not in s for s in outs))

    def test_sql_order_by_column_count_probe_appears(self):
        outs = outputs("sql", n=400)
        self.assertTrue(any(re.search(r"' order by \d+--", s) for s in outs))

    def test_template_statement_form_appears(self):
        outs = outputs("template", n=400)
        self.assertTrue(any(re.fullmatch(r"\{% print \d+\*\d+ %\}", s) for s in outs))

    def test_payload_characters_survive_lowercasing(self):
        specials = set("'\"<>${}[]()=;/\\%*&|-#")
        for tid in TEMPLATES:
            seen = set()
            for s in outputs(tid):
                self.assertEqual(s, s.lower())
                seen |= set(s) & specials
            self.assertTrue(seen, tid)


class HardPositiveTest(unittest.TestCase):
    def test_hard_positives_start_with_vocabulary_words_then_payload(self):
        vocab = ["leather", "sofa", "rug"]
        for tid in ("hard_sql", "hard_nosql"):
            for s in outputs(tid, vocab):
                toks = s.split()
                self.assertIn(re.match(r"[a-z]+", toks[0]).group(), vocab, s)
                self.assertTrue(any(not re.fullmatch(r"[a-z]+", t) for t in toks), s)

    def test_hard_positives_work_with_repeated_vocabulary_tokens(self):
        for tid in ("hard_sql", "hard_nosql"):
            for s in outputs(tid, ["sofa"] * 5, 20):
                self.assertIn("sofa", s)

    def test_hard_positive_empty_vocabulary_raises(self):
        for tid in ("hard_sql", "hard_nosql"):
            with self.assertRaises((ValueError, IndexError)):
                TEMPLATES[tid].render(random.Random(0), [])


if __name__ == "__main__":
    unittest.main()
