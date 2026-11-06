import hashlib
import tempfile
import unittest
from pathlib import Path

from bqc.wands import WandsError, load_queries

FIXTURES = Path(__file__).parent / "fixtures"
TSV = FIXTURES / "query.tsv"
CHECKSUMS = FIXTURES / "CHECKSUMS"
GOOD_HASH = CHECKSUMS.read_text().split()[0]


class LoadQueriesTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def checksums_for(self, name, digest):
        p = self.tmp / "CHECKSUMS"
        p.write_text(f"{digest}  {name}\n")
        return p

    def test_loads_ten_rows_in_file_order_with_ids(self):
        rows = load_queries(TSV, CHECKSUMS)
        self.assertEqual([r.query_id for r in rows], list(range(10)))
        self.assertEqual(rows[0].query, "smoky mountain tray")
        self.assertEqual(rows[9].query, "minnestrista")

    def test_quoted_rows_are_unquoted(self):
        queries = [r.query for r in load_queries(TSV, CHECKSUMS)]
        self.assertEqual(queries[1], 'fawkes 36" blue vanity')
        self.assertEqual(queries[4], 'writing desk 48"')

    def test_empty_query_class_is_accepted(self):
        queries = [r.query for r in load_queries(TSV, CHECKSUMS)]
        self.assertIn("drudge report", queries)
        self.assertIn("kisner", queries)

    def test_double_space_kept_and_nothing_normalised(self):
        queries = [r.query for r in load_queries(TSV, CHECKSUMS)]
        self.assertEqual(queries[3], "area  rug")

    def test_fields_include_query_class_last(self):
        self.assertEqual(
            load_queries(TSV, CHECKSUMS)[0]._fields, ("query_id", "query", "query_class")
        )

    def test_query_classes_of_all_ten_rows_in_file_order(self):
        classes = [r.query_class for r in load_queries(TSV, CHECKSUMS)]
        self.assertEqual(
            classes,
            [
                "Wall Art",
                "Bathroom Vanities",
                "",
                "Rugs",
                "Desks",
                "",
                "Sectionals",
                "Cookware",
                "Doors",
                "Outdoor",
            ],
        )

    def test_empty_query_class_cells_give_empty_string(self):
        by_query = {r.query: r.query_class for r in load_queries(TSV, CHECKSUMS)}
        self.assertEqual(by_query["drudge report"], "")
        self.assertEqual(by_query["kisner"], "")

    def test_query_class_with_space_is_kept_whole(self):
        self.assertEqual(load_queries(TSV, CHECKSUMS)[0].query_class, "Wall Art")

    def test_wrong_hash_raises_naming_path(self):
        bad = self.checksums_for("query.tsv", "0" * 64)
        with self.assertRaises(WandsError) as cm:
            load_queries(TSV, bad)
        self.assertIn(str(TSV), str(cm.exception))

    def test_hash_checked_before_parsing(self):
        # Bad header AND wrong hash: the error must be the hash one.
        p = self.tmp / "query.tsv"
        p.write_bytes(b"nonsense\n")
        with self.assertRaises(WandsError) as cm:
            load_queries(p, self.checksums_for("query.tsv", GOOD_HASH))
        self.assertIn("hash", str(cm.exception).lower())

    def test_missing_file_raises_naming_path(self):
        missing = self.tmp / "nope.tsv"
        with self.assertRaises(WandsError) as cm:
            load_queries(missing, CHECKSUMS)
        self.assertIn(str(missing), str(cm.exception))

    def test_missing_checksums_file_raises(self):
        with self.assertRaises(WandsError) as cm:
            load_queries(TSV, self.tmp / "absent")
        self.assertIn("absent", str(cm.exception))

    def test_no_checksum_entry_for_file_raises(self):
        other = self.checksums_for("other.csv", GOOD_HASH)
        with self.assertRaises(WandsError) as cm:
            load_queries(TSV, other)
        self.assertIn(str(TSV), str(cm.exception))

    def test_wrong_header_raises_naming_path(self):
        p = self.tmp / "query.tsv"
        p.write_bytes(TSV.read_bytes().replace(b"query_class", b"category", 1))
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        with self.assertRaises(WandsError) as cm:
            load_queries(p, self.checksums_for("query.tsv", digest))
        self.assertIn(str(p), str(cm.exception))
        self.assertIn("header", str(cm.exception).lower())

    def test_malformed_row_raises_naming_path(self):
        p = self.tmp / "query.tsv"
        p.write_bytes(b"query_id\tquery\tquery_class\nx\tfoo\t\n")
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        with self.assertRaises(WandsError) as cm:
            load_queries(p, self.checksums_for("query.tsv", digest))
        self.assertIn(str(p), str(cm.exception))


if __name__ == "__main__":
    unittest.main()
