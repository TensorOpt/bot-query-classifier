import contextlib
import io
import subprocess
import sys
import unittest

from bqc.cli import main


class CliTest(unittest.TestCase):
    def run_main(self, argv):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            main(argv)
        return cm.exception.code, err.getvalue()

    def test_unknown_command_exits_non_zero(self):
        code, err = self.run_main(["nonsense"])
        self.assertNotEqual(code, 0)
        self.assertIn("usage", err.lower())

    def test_no_command_exits_non_zero(self):
        code, err = self.run_main([])
        self.assertNotEqual(code, 0)
        self.assertIn("usage", err.lower())

    def test_python_dash_m_without_command(self):
        proc = subprocess.run(
            [sys.executable, "-m", "bqc"], capture_output=True, text=True
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("usage", proc.stderr.lower())


if __name__ == "__main__":
    unittest.main()
