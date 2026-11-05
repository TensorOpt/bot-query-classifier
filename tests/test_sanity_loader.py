import hashlib
import tempfile
import unittest
from pathlib import Path

from bqc.preprocess import MAX_CODE_POINTS, normalise
from bqc.sanity import SANITY_PATH, SanityError, load_sanity
from bqc.wands import WandsQuery

FIXTURES = Path(__file__).parent / "fixtures"
WANDS = [WandsQuery(0, "smoky mountain tray"), WandsQuery(3, "area rug")]


def load(name):
    return load_sanity(FIXTURES / name, WANDS)


class LabelStateTest(unittest.TestCase):
    def test_all_blank(self):
        s = load("sanity_blank.csv")
        self.assertIsNone(s.labels)
        self.assertEqual(s.invalid, 2)
        self.assertEqual(s.message, "sanity set: 2 of 2 rows unlabelled, skipping")
        self.assertEqual([q.query for q in s.queries], ["oak desk", "chair"])

    def test_all_labelled_with_stray_spaces(self):
        s = load("sanity_full.csv")
        self.assertEqual(s.labels, [1, 0, 1, 0])
        self.assertEqual(s.invalid, 0)
        self.assertIsNone(s.message)
        self.assertEqual([q.id for q in s.queries], [1, 2, 3, 4])

    def test_partly_labelled_is_not_scored(self):
        s = load("sanity_partial.csv")
        self.assertIsNone(s.labels)
        self.assertEqual(s.invalid, 1)
        self.assertEqual(s.message, "sanity set: 1 of 3 rows unlabelled, skipping")
        self.assertEqual(len(s.queries), 3)

    def test_whitespace_only_labels_count_as_blank(self):
        s = load("sanity_spaces.csv")
        self.assertIsNone(s.labels)
        self.assertEqual(s.invalid, 2)
        self.assertEqual(len(s.queries), 2)

    def test_invalid_labels_make_set_not_scorable(self):
        s = load("sanity_bad.csv")
        self.assertIsNone(s.labels)
        self.assertEqual(s.invalid, 3)
        self.assertEqual(s.message, "sanity set: 3 of 4 rows unlabelled, skipping")
        self.assertEqual(len(s.queries), 4)


class QueryHandlingTest(unittest.TestCase):
    def test_queries_come_back_raw(self):
        s = load("sanity_partial.csv")
        self.assertEqual(s.queries[0].query, "Oak  Desk")

    def test_quoted_commas_and_crlf(self):
        # Written here: core.autocrlf=input would turn a committed fixture's CRLF into LF.
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sanity.csv"
            p.write_bytes(b'id,query,label\r\n1,"chair, red",1\r\n2,sofa,0\r\n')
            s = load_sanity(p, WANDS)
        self.assertEqual([q.query for q in s.queries], ["chair, red", "sofa"])
        self.assertEqual(s.labels, [1, 0])

    def test_wands_duplicate_after_normalise_names_the_id(self):
        with self.assertRaises(SanityError) as cm:
            load("sanity_wands_dup.csv")
        self.assertIn("id 7", str(cm.exception))

    def test_wands_duplicate_raises_even_when_unlabelled(self):
        wands = [WandsQuery(0, "OAK  DESK")]
        with self.assertRaises(SanityError) as cm:
            load_sanity(FIXTURES / "sanity_blank.csv", wands)
        self.assertIn("id 1", str(cm.exception))

    def test_wrong_header_names_the_file(self):
        with self.assertRaises(SanityError) as cm:
            load("sanity_badheader.csv")
        self.assertIn("sanity_badheader.csv", str(cm.exception))

    def test_missing_file_names_the_file(self):
        with self.assertRaises(SanityError) as cm:
            load("nope.csv")
        self.assertIn("nope.csv", str(cm.exception))

    def test_does_not_write_to_the_file(self):
        p = FIXTURES / "sanity_full.csv"
        before = (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
        load_sanity(p, WANDS)
        after = (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
        self.assertEqual(before, after)


class RealSanityFileTest(unittest.TestCase):
    """Structure only; asserts nothing about label values. Needs no WANDS file."""

    def test_structure(self):
        s = load_sanity(SANITY_PATH, [])
        self.assertEqual([q.id for q in s.queries], list(range(1, 51)))
        normalised = [normalise(q.query) for q in s.queries]
        self.assertEqual(len(set(normalised)), 50)
        for n in normalised:
            self.assertLessEqual(len(n), MAX_CODE_POINTS)


if __name__ == "__main__":
    unittest.main()
