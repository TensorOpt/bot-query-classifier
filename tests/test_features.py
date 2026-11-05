import unittest

from bqc.features import make_vectoriser


class MakeVectoriserTest(unittest.TestCase):
    def test_parameters(self):
        v = make_vectoriser()
        self.assertEqual(v.analyzer, "char")
        self.assertEqual(v.ngram_range, (1, 4))
        self.assertTrue(v.lowercase)
        self.assertTrue(v.sublinear_tf)
        self.assertEqual(v.min_df, 2)

    def test_unfitted_and_independent(self):
        a, b = make_vectoriser(), make_vectoriser()
        self.assertIsNot(a, b)
        self.assertFalse(hasattr(a, "vocabulary_"))

    def test_empty_and_unknown_strings_give_zero_row(self):
        v = make_vectoriser().fit(["sofa", "sofa", "lamp", "lamp"])
        x = v.transform(["", "zzz", "   "])
        self.assertEqual(x.shape[0], 3)
        self.assertEqual(x.nnz, 0)

    def test_cross_token_ngrams_and_punctuation_kept(self):
        v = make_vectoriser().fit(["sofa' or 1=1--", "chair' or 2=2--"])
        vocab = v.vocabulary_
        self.assertIn("' or", vocab)
        self.assertIn("=", vocab)
        self.assertIn("--", vocab)

    def test_min_df_drops_one_document_ngram(self):
        v = make_vectoriser().fit(["abcx", "abcy"])
        self.assertIn("abc", v.vocabulary_)
        self.assertNotIn("abcx", v.vocabulary_)
        self.assertNotIn("abcy", v.vocabulary_)

    def test_no_test_only_ngram_in_vocabulary(self):
        train = ["sofa", "sofa bed", "lamp"]
        test = ["qzqz", "qzqz"]  # "qz" occurs twice, but only in the test split
        v = make_vectoriser().fit(train)
        self.assertNotIn("qz", v.vocabulary_)
        self.assertEqual(v.transform(test).nnz, 0)


if __name__ == "__main__":
    unittest.main()
