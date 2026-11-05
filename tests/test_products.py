import csv
import tempfile
import unittest
from pathlib import Path

from bqc.wands import WandsError, WandsProduct, load_products

FIXTURES = Path(__file__).parent / "fixtures"
TSV = FIXTURES / "product.tsv"
CHECKSUMS = FIXTURES / "CHECKSUMS"
HEADER = (
    "product_id\tproduct_name\tproduct_class\tcategory hierarchy\tproduct_description"
    "\tproduct_features\trating_count\taverage_rating\treview_count"
)


class LoadProductsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        # Start every test from the default limit, so the long-field test proves the
        # loader raises it, whatever ran before.
        original = csv.field_size_limit()
        self.addCleanup(csv.field_size_limit, original)
        csv.field_size_limit(131072)

    def write(self, text, name="product.tsv"):
        """Write `text` and a matching checksums file; return (path, checksums)."""
        import hashlib

        path = self.tmp / name
        data = text.encode("utf-8")
        path.write_bytes(data)
        checksums = self.tmp / "CHECKSUMS"
        checksums.write_text(f"{hashlib.sha256(data).hexdigest()}  {name}\n")
        return path, checksums

    def test_rows_in_file_order_with_three_fields(self):
        rows = load_products(TSV, CHECKSUMS)
        self.assertEqual([r.product_id for r in rows], [0, 1, 2])
        self.assertEqual(rows[0].name, "solid wood platform bed")
        self.assertEqual(rows[2].name, "all-clad 7 qt . slow cooker")
        self.assertEqual(WandsProduct._fields, ("product_id", "name", "classes"))

    def test_description_longer_than_default_csv_limit_loads(self):
        self.assertGreater(max(len(f) for f in TSV.read_text(encoding="utf-8").split("\t")), 131072)
        self.assertEqual(len(load_products(TSV, CHECKSUMS)), 3)

    def test_quoted_field_and_non_ascii_name_returned_raw(self):
        self.assertEqual(load_products(TSV, CHECKSUMS)[1].name, 'café 36" vanity')

    def test_empty_class_is_empty_tuple(self):
        self.assertEqual(load_products(TSV, CHECKSUMS)[1].classes, ())

    def test_multi_class_is_split_into_tuple(self):
        rows = load_products(TSV, CHECKSUMS)
        self.assertEqual(rows[0].classes, ("Beds", "Bedroom Furniture"))
        self.assertEqual(rows[2].classes, ("Slow Cookers",))

    def test_columns_matched_by_name_not_position(self):
        header = "\t".join(
            ["rating_count", "product_class", "category hierarchy", "product_name",
             "product_id", "product_description", "product_features",
             "average_rating", "review_count"]
        )
        row = "9\tRugs\tHome\tblue rug\t7\td\tf\t1\t2"
        path, checksums = self.write(f"{header}\n{row}\n")
        self.assertEqual(
            load_products(path, checksums), [WandsProduct(7, "blue rug", ("Rugs",))]
        )

    def test_wrong_hash_raises_naming_path(self):
        bad = self.tmp / "CHECKSUMS"
        bad.write_text("0" * 64 + "  product.tsv\n")
        with self.assertRaises(WandsError) as cm:
            load_products(TSV, bad)
        self.assertIn(str(TSV), str(cm.exception))
        self.assertIn("hash", str(cm.exception).lower())

    def test_missing_file_raises_naming_path(self):
        missing = self.tmp / "nope.tsv"
        with self.assertRaises(WandsError) as cm:
            load_products(missing, CHECKSUMS)
        self.assertIn(str(missing), str(cm.exception))

    def test_missing_checksum_line_raises(self):
        empty = self.tmp / "CHECKSUMS"
        empty.write_text("")
        with self.assertRaises(WandsError):
            load_products(TSV, empty)

    def test_wrong_header_raises_naming_path(self):
        path, checksums = self.write("a\tb\tc\n1\t2\t3\n")
        with self.assertRaises(WandsError) as cm:
            load_products(path, checksums)
        self.assertIn(str(path), str(cm.exception))

    def test_header_with_underscore_hierarchy_raises(self):
        path, checksums = self.write(HEADER.replace("category hierarchy", "category_hierarchy") + "\n")
        with self.assertRaises(WandsError) as cm:
            load_products(path, checksums)
        self.assertIn(str(path), str(cm.exception))

    def test_empty_file_raises(self):
        path, checksums = self.write("")
        with self.assertRaises(WandsError) as cm:
            load_products(path, checksums)
        self.assertIn(str(path), str(cm.exception))

    def test_wrong_column_count_names_row_number(self):
        good = "1\ta\tRugs\th\td\tf\t1\t2\t3"
        path, checksums = self.write(f"{HEADER}\n{good}\n2\tshort\tRugs\n{good}\n")
        with self.assertRaises(WandsError) as cm:
            load_products(path, checksums)
        message = str(cm.exception)
        self.assertIn(str(path), message)
        self.assertIn("row 2", message)

    def test_non_integer_id_raises_naming_path(self):
        path, checksums = self.write(f"{HEADER}\nx\ta\tRugs\th\td\tf\t1\t2\t3\n")
        with self.assertRaises(WandsError) as cm:
            load_products(path, checksums)
        self.assertIn(str(path), str(cm.exception))


if __name__ == "__main__":
    unittest.main()
