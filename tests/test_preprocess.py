import unittest

from bqc.preprocess import MAX_CODE_POINTS, normalise


class NormaliseTest(unittest.TestCase):
    def test_empty_and_whitespace(self):
        for s in ["", " ", " \t\n\r  ", "  "]:
            self.assertEqual(normalise(s), "")

    def test_tab_and_newline_become_space(self):
        self.assertEqual(normalise("a\tb"), "a b")
        self.assertEqual(normalise("a\nb"), "a b")
        self.assertEqual(normalise("  a \t\n b  "), "a b")

    def test_invisible_characters_removed_and_neighbours_joined(self):
        self.assertEqual(normalise("so​fa"), "sofa")
        self.assertEqual(normalise("so﻿fa"), "sofa")
        self.assertEqual(normalise("so\x00fa"), "sofa")

    def test_fullwidth_maps_to_ascii(self):
        self.assertEqual(normalise("ＳＯＦＡ"), "sofa")

    def test_keeps_marks_emoji_and_non_ascii_letters(self):
        self.assertEqual(normalise("café 🛋 é"), "café 🛋 é")
        self.assertEqual(normalise("Ünïcödé"), "ünïcödé")

    def test_mixed_case(self):
        self.assertEqual(normalise("Grey SOFA"), "grey sofa")

    def test_length_boundaries(self):
        for n in (511, 512, 513):
            out = normalise("a" * n)
            self.assertEqual(len(out), min(n, MAX_CODE_POINTS))

    def test_cut_inside_space_leaves_no_trailing_space(self):
        out = normalise("a" * 511 + " b")
        self.assertEqual(out, "a" * 511)
        self.assertFalse(out.endswith(" "))

    def test_one_megabyte_input(self):
        out = normalise("ab " * 400_000)
        self.assertLessEqual(len(out), 512)
        self.assertFalse(out.endswith(" "))

    def test_nfkc_lengthening_is_truncated(self):
        out = normalise("ﬁ" * 300)
        self.assertEqual(out, "fi" * 256)
        self.assertEqual(len(out), 512)

    def test_idempotent(self):
        samples = [
            "", " ", "a\tb", "so​fa", "so\x00fa", "ＳＯＦＡ",
            "café 🛋 é", "Grey SOFA", "a" * 511, "a" * 513, "a" * 511 + " b",
            "ﬁ" * 300, "ab " * 400_000, "İ" * 600, "x y",
        ]
        for s in samples:
            once = normalise(s)
            self.assertEqual(normalise(once), once)


if __name__ == "__main__":
    unittest.main()
