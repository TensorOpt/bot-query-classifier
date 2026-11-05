import unittest
from pathlib import Path

from bqc.sanity import load_sanity
from bqc.synth.audit import (
    NEGATIVES_THRESHOLD,
    SANITY_THRESHOLD,
    WANDS_KEY,
    AuditError,
    LengthStats,
    NearDuplicate,
    check_exact,
    length_stats,
    near_duplicates,
)
from bqc.synth.registry import Sample

FIXTURES = Path(__file__).parent / "fixtures"
# Unlabelled fixture: the loader still exposes its queries (8.6.1).
SANITY = [q.query for q in load_sanity(FIXTURES / "sanity_blank.csv", []).queries]
NEGATIVES = ["smoky mountain tray", "area rug"]


class CheckExactTest(unittest.TestCase):
    def test_distinct_passes(self):
        check_exact(["zzqx 1", "kk77"], NEGATIVES, SANITY)

    def test_identical_negative_raises_naming_both_and_list(self):
        with self.assertRaises(AuditError) as cm:
            check_exact(["area rug"], NEGATIVES, SANITY)
        msg = str(cm.exception)
        self.assertEqual(msg.count("area rug"), 2)
        self.assertIn("negatives", msg)
        self.assertNotIn("sanity", msg)

    def test_case_and_whitespace_negative(self):
        with self.assertRaises(AuditError) as cm:
            check_exact(["  AREA \t Rug "], NEGATIVES, SANITY)
        self.assertIn("AREA", str(cm.exception))
        self.assertIn("area rug", str(cm.exception))

    def test_identical_sanity_raises_naming_both_and_list(self):
        self.assertIn("oak desk", SANITY)
        with self.assertRaises(AuditError) as cm:
            check_exact(["oak desk"], NEGATIVES, SANITY)
        msg = str(cm.exception)
        self.assertEqual(msg.count("oak desk"), 2)
        self.assertIn("sanity", msg)
        self.assertNotIn("negatives", msg)

    def test_case_and_whitespace_sanity(self):
        with self.assertRaises(AuditError) as cm:
            check_exact(["x", "OAK   Desk"], NEGATIVES, SANITY)
        self.assertIn("OAK   Desk", str(cm.exception))
        self.assertIn("oak desk", str(cm.exception))

    def test_unicode_equivalent_after_normalise(self):
        with self.assertRaises(AuditError):
            check_exact(["ＡＲＥＡ rug"], NEGATIVES, SANITY)

    def test_stored_string_is_normalised_too(self):
        with self.assertRaises(AuditError):
            check_exact(["area rug"], ["Area   RUG"], [])

    def test_callable_with_only_positives_and_accepts_iterators(self):
        check_exact(iter(["a"]))
        with self.assertRaises(AuditError):
            check_exact(iter(["oak desk"]), iter([]), iter(SANITY))

    def test_empty_positive_does_not_match_nonempty(self):
        check_exact([""], NEGATIVES, SANITY)


# Sets of 3-grams: "abcd" has 2, "abcde" 3, "abcdef" 4, "abcdefg" 5, "abcdefgh" 6.
class NearDuplicatesTest(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(NEGATIVES_THRESHOLD, 0.6)
        self.assertEqual(SANITY_THRESHOLD, 0.5)

    def test_at_0_6_exactly_not_reported(self):
        # 3 shared of 5 in the union.
        self.assertEqual(near_duplicates(["abcde"], ["abcdefg"], 0.6), [])

    def test_just_above_0_6_reported(self):
        # 4 shared of 6.
        got = near_duplicates(["abcdef"], ["abcdefgh"], 0.6)
        self.assertEqual(len(got), 1)
        self.assertIsInstance(got[0], NearDuplicate)
        self.assertEqual(got[0][:2], (0, 0))
        self.assertAlmostEqual(got[0].jaccard, 4 / 6)

    def test_at_0_5_exactly_not_reported(self):
        # 2 shared of 4.
        self.assertEqual(near_duplicates(["abcd"], ["abcdef"], 0.5), [])

    def test_just_above_0_5_reported(self):
        # 3 shared of 5.
        got = near_duplicates(["abcde"], ["abcdefg"], 0.5)
        self.assertEqual([(d.positive_index, d.other_index) for d in got], [(0, 0)])
        self.assertAlmostEqual(got[0].jaccard, 3 / 5)

    def test_reports_every_pair_with_indices(self):
        got = near_duplicates(
            ["abcdef", "zzzzzz", "abcdefg"], ["qqqqqq", "abcdefg", "abcdef"], 0.6)
        self.assertEqual(
            sorted((d.positive_index, d.other_index) for d in got),
            [(0, 1), (0, 2), (2, 1), (2, 2)])

    def test_identical_strings_score_one(self):
        got = near_duplicates(["abcdef"], ["abcdef"], 0.6)
        self.assertEqual(got[0].jaccard, 1.0)

    def test_normalised_before_comparison(self):
        got = near_duplicates(["ABC   DEF"], ["abc def"], 0.6)
        self.assertEqual(len(got), 1)

    def test_short_strings_do_not_raise_or_report(self):
        for short in ("", "a", "ab"):
            self.assertEqual(near_duplicates([short], [short, "abcdef"], 0.0), [])
            self.assertEqual(near_duplicates(["abcdef"], [short], 0.0), [])

    def test_empty_inputs(self):
        self.assertEqual(near_duplicates([], ["abc"], 0.5), [])
        self.assertEqual(near_duplicates(["abc"], [], 0.5), [])


class LengthStatsTest(unittest.TestCase):
    def test_per_family_and_wands(self):
        samples = [
            Sample("aa", "A", "t", 0), Sample("aaaa", "A", "t", 0),
            Sample("aaaaaa", "A", "t", 0), Sample("bbb", "B", "t", 0),
        ]
        got = length_stats(samples, ["x", "yyy", "zzzzz", "wwwwwwwwww"])
        self.assertEqual(set(got), {"A", "B", WANDS_KEY})
        self.assertEqual(got["A"], LengthStats(3, 2, 6, 4.0, 4))
        self.assertEqual(got["B"], LengthStats(1, 3, 3, 3.0, 3))
        self.assertEqual(got[WANDS_KEY], LengthStats(4, 1, 10, 4.75, 4.0))

    def test_length_is_of_raw_text(self):
        got = length_stats([Sample("a  b", "C", "t", 0)], [])
        self.assertEqual(got["C"].max, 4)

    def test_empty_inputs(self):
        got = length_stats([], [])
        self.assertEqual(got, {WANDS_KEY: LengthStats(0, 0, 0, 0.0, 0.0)})

    def test_wands_accepts_iterator(self):
        got = length_stats([], iter(["ab"]))
        self.assertEqual(got[WANDS_KEY].count, 1)


if __name__ == "__main__":
    unittest.main()
