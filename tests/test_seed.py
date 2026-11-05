import unittest

import bqc


class SeedTest(unittest.TestCase):
    def test_seed_value(self):
        self.assertEqual(bqc.SEED, 20261105)


if __name__ == "__main__":
    unittest.main()
