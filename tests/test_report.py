import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from bqc import SEED
from bqc.report import write_report

NOW = datetime(2026, 11, 5, 12, 30, 15, tzinfo=timezone.utc)


class WriteReportTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name) / "results"  # does not exist yet

    def write(self, parameters=None, body=None, now=NOW):
        return write_report(
            self.dir, "generate", parameters or {"negatives_size": 10}, body or {"x": [1]}, now
        )

    def test_config_fields_and_body(self):
        data = json.loads(self.write().read_text())
        config = data["config"]
        self.assertEqual(config["command"], "generate")
        self.assertEqual(config["parameters"], {"negatives_size": 10})
        self.assertEqual(config["seed"], SEED)
        self.assertIsInstance(config["git_commit"], str)
        self.assertTrue(config["git_commit"])
        self.assertEqual(set(config["libraries"]), {"python", "numpy", "scikit-learn"})
        self.assertEqual(len(config["config_hash"]), 64)
        self.assertEqual(data["x"], [1])

    def test_name_has_command_hash_and_timestamp(self):
        path = self.write()
        data = json.loads(path.read_text())
        self.assertEqual(path.parent, self.dir)
        self.assertEqual(
            path.name,
            f"generate-{data['config']['config_hash'][:12]}-20261105T123015000000Z.json",
        )

    def test_hash_follows_parameters_not_body_or_time(self):
        def digest(path):
            return json.loads(path.read_text())["config"]["config_hash"]

        a = digest(self.write({"negatives_size": 10}, {"x": 1}))
        b = digest(self.write({"negatives_size": 10}, {"x": 2}, NOW.replace(second=16)))
        c = digest(self.write({"negatives_size": 11}, {"x": 1}, NOW.replace(second=17)))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_collision_does_not_overwrite(self):
        path = self.write(body={"x": "first"})
        with self.assertRaises(FileExistsError):
            self.write(body={"x": "second"})
        self.assertEqual(json.loads(path.read_text())["x"], "first")


if __name__ == "__main__":
    unittest.main()
