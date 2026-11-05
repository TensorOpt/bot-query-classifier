import contextlib
import hashlib
import http.client
import io
import runpy
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from bqc import cli
from bqc.wands import WandsError, fetch, load_queries

FIXTURES = Path(__file__).parent / "fixtures"
TSV = FIXTURES / "query.tsv"
GOOD = TSV.read_bytes()


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def stub(data):
    return lambda url, *a, **k: Response(data)


def failing(url, *a, **k):
    raise urllib.error.URLError("no network")


def midstream_failure(url, *a, **k):
    class Broken(Response):
        def read(self, *a):
            raise ConnectionResetError("reset")

    return Broken(b"x")


def truncated(url, *a, **k):
    class Short(Response):
        def read(self, *a):
            raise http.client.IncompleteRead(b"partial", 100)

    return Short(b"x")


class FetchTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.target = self.tmp / "raw" / "query.csv"
        self.checksums = self.tmp / "CHECKSUMS"
        digest = hashlib.sha256(GOOD).hexdigest()
        self.checksums.write_text(f"{digest}  query.csv\n")

    def leftovers(self):
        d = self.target.parent
        return sorted(p.name for p in d.iterdir()) if d.exists() else []

    def test_success_writes_exact_bytes_and_loads(self):
        fetch(self.target, self.checksums, opener=stub(GOOD))
        self.assertEqual(self.target.read_bytes(), GOOD)
        self.assertEqual(self.leftovers(), ["query.csv"])
        self.assertEqual(len(load_queries(self.target, self.checksums)), 10)

    def test_network_error_leaves_nothing(self):
        with self.assertRaises(WandsError):
            fetch(self.target, self.checksums, opener=failing)
        self.assertEqual(self.leftovers(), [])

    def test_failure_mid_download_leaves_nothing(self):
        with self.assertRaises(WandsError):
            fetch(self.target, self.checksums, opener=midstream_failure)
        self.assertEqual(self.leftovers(), [])

    def test_truncated_download_leaves_nothing(self):
        with self.assertRaises(WandsError):
            fetch(self.target, self.checksums, opener=truncated)
        self.assertEqual(self.leftovers(), [])

    def test_opener_receives_timeout_60(self):
        opener = mock.Mock(side_effect=stub(GOOD))
        fetch(self.target, self.checksums, opener=opener)
        self.assertEqual(opener.call_args.kwargs.get("timeout"), 60)

    def test_wrong_bytes_leave_nothing(self):
        with self.assertRaisesRegex(WandsError, "hash mismatch"):
            fetch(self.target, self.checksums, opener=stub(GOOD + b"x"))
        self.assertEqual(self.leftovers(), [])

    def test_failed_fetch_keeps_existing_good_file(self):
        fetch(self.target, self.checksums, opener=stub(GOOD))
        for opener in (failing, stub(b"bad")):
            with self.assertRaises(WandsError):
                fetch(self.target, self.checksums, opener=opener)
            self.assertEqual(self.target.read_bytes(), GOOD)
            self.assertEqual(self.leftovers(), ["query.csv"])

    def test_missing_checksum_line_leaves_nothing(self):
        self.checksums.write_text("")
        with self.assertRaises(WandsError):
            fetch(self.target, self.checksums, opener=stub(GOOD))
        self.assertEqual(self.leftovers(), [])

    def test_missing_checksums_file_leaves_nothing(self):
        self.checksums.unlink()
        with self.assertRaises(WandsError):
            fetch(self.target, self.checksums, opener=stub(GOOD))
        self.assertEqual(self.leftovers(), [])


class FetchCommandTest(unittest.TestCase):
    def test_success_fetches_both_files_and_returns_zero(self):
        with mock.patch.object(cli, "fetch") as f:
            self.assertEqual(cli.main(["fetch"]), 0)
        self.assertEqual(
            [(c.args[0], c.args[1], c.args[2]) for c in f.call_args_list],
            [
                (cli.RAW_PATH, cli.CHECKSUMS_PATH, cli.URL),
                (cli.PRODUCT_PATH, cli.CHECKSUMS_PATH, cli.PRODUCT_URL),
            ],
        )
        self.assertEqual(cli.RAW_PATH.name, "query.csv")
        self.assertEqual(cli.PRODUCT_PATH.name, "product.csv")

    def test_error_returns_non_zero_and_reports(self):
        err = io.StringIO()
        with (
            mock.patch.object(cli, "fetch", side_effect=WandsError("boom")),
            contextlib.redirect_stderr(err),
        ):
            self.assertNotEqual(cli.main(["fetch"]), 0)
        self.assertIn("boom", err.getvalue())

    def test_failure_on_second_file_returns_non_zero_and_keeps_first(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        raw = Path(tmp.name) / "raw"
        checksums = Path(tmp.name) / "CHECKSUMS"
        checksums.write_text(
            f"{hashlib.sha256(GOOD).hexdigest()}  query.csv\n"
            f"{hashlib.sha256(b'product').hexdigest()}  product.csv\n"
        )

        def fake_fetch(target, checksums_path, url):
            opener = stub(GOOD) if target.name == "query.csv" else failing
            fetch(target, checksums_path, url, opener=opener)

        err = io.StringIO()
        with (
            mock.patch.object(cli, "RAW_PATH", raw / "query.csv"),
            mock.patch.object(cli, "PRODUCT_PATH", raw / "product.csv"),
            mock.patch.object(cli, "CHECKSUMS_PATH", checksums),
            mock.patch.object(cli, "fetch", side_effect=fake_fetch),
            contextlib.redirect_stderr(err),
        ):
            self.assertNotEqual(cli.main(["fetch"]), 0)
        self.assertEqual(sorted(p.name for p in raw.iterdir()), ["query.csv"])
        self.assertEqual((raw / "query.csv").read_bytes(), GOOD)
        self.assertIn("no network", err.getvalue())


class ExitStatusTest(unittest.TestCase):
    def test_main_return_value_becomes_exit_status(self):
        with mock.patch("bqc.cli.main", return_value=3):
            with self.assertRaises(SystemExit) as cm:
                runpy.run_module("bqc", run_name="__main__")
        self.assertEqual(cm.exception.code, 3)


if __name__ == "__main__":
    unittest.main()
