import inspect
import json
import random
import unittest
from unittest import mock

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
)

from bqc import SEED
from bqc.cluster import N_DRAWS, SEED_KS, draw_seeds, oracle_vote, seed_draw, seed_rule


def all_rows(n):
    """Patch target: every row is a seed, so a draw is deterministic and hand-computable."""
    return mock.patch("bqc.cluster.draw_seeds", return_value=list(range(n)))


class SignatureAndGridTest(unittest.TestCase):
    def test_signatures_pinned(self):
        def names(f):
            return list(inspect.signature(f).parameters)
        self.assertEqual(names(oracle_vote), ["clusters", "truth"])
        self.assertEqual(names(draw_seeds), ["n_rows", "k", "draw_index"])
        self.assertEqual(names(seed_draw), ["clusters", "truth", "k", "draw_index"])
        self.assertEqual(names(seed_rule), ["clusters", "truth"])

    def test_grid_pinned(self):
        self.assertEqual(SEED_KS, (10, 20, 40))
        self.assertEqual(N_DRAWS, 20)


class OracleVoteTest(unittest.TestCase):
    def test_purity_counts_and_ari_by_hand(self):
        # cluster 0: truth 0,0,1 -> label 0, purity 2/3; cluster 1: 1,1,1 -> purity 1; two noise.
        r = oracle_vote([0, 0, 0, 1, 1, 1, -1, -1], [0, 0, 1, 1, 1, 1, 1, 0])
        self.assertEqual(r["clusters"], [
            {"cluster": 0, "size": 3, "purity": 2 / 3, "label": 0},
            {"cluster": 1, "size": 3, "purity": 1.0, "label": 1},
        ])
        self.assertEqual((r["n_clusters"], r["non_noise"], r["noise_fraction"]), (2, 6, 0.25))
        # contingency [[2,1],[0,3]]: (4 - 2.8) / (6.5 - 2.8) = 12/37
        self.assertAlmostEqual(r["ari"], 12 / 37, places=12)
        c, t = [0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 1]
        self.assertAlmostEqual(r["ami"], adjusted_mutual_info_score(t, c), places=12)
        self.assertAlmostEqual(r["homogeneity"], homogeneity_score(t, c), places=12)
        self.assertAlmostEqual(r["completeness"], completeness_score(t, c), places=12)

    def test_negative_ari_by_hand(self):
        r = oracle_vote([0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 0])
        self.assertAlmostEqual(r["ari"], -1 / 9, places=12)

    def test_half_and_half_cluster_is_legitimate(self):
        r = oracle_vote([0, 0, 1, 1, 1], [0, 1, 1, 1, 0])
        self.assertEqual(r["clusters"][0], {"cluster": 0, "size": 2, "purity": 0.5, "label": 0})
        self.assertEqual(r["clusters"][1]["label"], 1)

    def test_clusters_listed_in_ascending_id_order(self):
        # HDBSCAN ids don't appear in order: id 1 shows up first here.
        r = oracle_vote([1, 1, 0, 0, 0], [1, 1, 0, 0, 1])
        self.assertEqual([c["cluster"] for c in r["clusters"]], [0, 1])
        self.assertEqual([c["size"] for c in r["clusters"]], [3, 2])
        self.assertEqual([c["label"] for c in r["clusters"]], [0, 1])

    def test_strict_majority_of_illegitimate_is_one(self):
        r = oracle_vote([0, 0, 0], [1, 1, 0])
        self.assertEqual(r["clusters"][0]["label"], 1)

    def test_many_clusters_per_class(self):
        clusters = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        truth = [0] * 6 + [1] * 6
        r = oracle_vote(clusters, truth)
        self.assertEqual(r["n_clusters"], 6)
        self.assertEqual([c["purity"] for c in r["clusters"]], [1.0] * 6)
        self.assertEqual([c["label"] for c in r["clusters"]], [0, 0, 0, 1, 1, 1])
        self.assertAlmostEqual(r["ari"], 3 / 14, places=12)
        self.assertAlmostEqual(r["homogeneity"], 1.0, places=12)
        self.assertLess(r["completeness"], 1.0)

    def test_metrics_exclude_noise_rows(self):
        # Without the noise rows the clustering is perfect; noise as a cluster would not be.
        r = oracle_vote([0, 0, 1, 1, -1, -1], [0, 0, 1, 1, 0, 1])
        for key in ("ari", "ami", "homogeneity", "completeness"):
            self.assertEqual(r[key], 1.0, key)
        self.assertAlmostEqual(r["noise_fraction"], 1 / 3)
        self.assertEqual(r["non_noise"], 4)

    def test_noise_true_labels_do_not_change_ari(self):
        clusters = [0] * 5 + [1] * 5 + [-1] * 4
        a = oracle_vote(clusters, [0, 0, 0, 0, 1] + [1, 1, 1, 1, 0] + [0] * 4)
        b = oracle_vote(clusters, [0, 0, 0, 0, 1] + [1, 1, 1, 1, 0] + [1] * 4)
        self.assertEqual(a, b)

    def test_all_noise(self):
        r = oracle_vote([-1] * 6, [0, 0, 1, 1, 0, 1])
        self.assertEqual(r["non_noise"], 0)
        self.assertEqual(r["n_clusters"], 0)
        self.assertEqual(r["noise_fraction"], 1.0)
        self.assertEqual(r["clusters"], [])
        for key in ("ari", "ami", "homogeneity", "completeness"):
            self.assertIsNone(r[key], key)

    def test_degenerate_one_cluster_and_one_class(self):
        one_cluster = oracle_vote([0] * 4, [0, 0, 1, 1])
        self.assertEqual([one_cluster[k] for k in ("ari", "ami", "homogeneity", "completeness")],
                         [0.0, 0.0, 0.0, 1.0])
        one_class = oracle_vote([0, 0, 1, 1], [1] * 4)
        self.assertEqual([one_class[k] for k in ("ari", "ami", "homogeneity", "completeness")],
                         [0.0, 0.0, 1.0, 0.0])

    def test_json_ready_and_plain_types(self):
        r = oracle_vote(np.array([0, 0, 1, 1, -1]), np.array([0, 1, 1, 1, 0]))
        self.assertEqual(json.loads(json.dumps(r)), r)
        for c in r["clusters"]:
            self.assertIs(type(c["cluster"]), int)
            self.assertIs(type(c["label"]), int)

    def test_bad_input_raises(self):
        with self.assertRaises(ValueError):
            oracle_vote([], [])
        with self.assertRaises(ValueError):
            oracle_vote([0, 0], [0])


class DrawSeedsTest(unittest.TestCase):
    def test_mechanics_pinned(self):
        self.assertEqual(draw_seeds(60, 10, 3), random.Random(SEED + 3).sample(range(60), 10))
        self.assertEqual(draw_seeds(60, 10, 3), [20, 28, 14, 19, 24, 35, 0, 54, 48, 42])

    def test_same_index_same_draw_and_different_index_differs(self):
        self.assertEqual(draw_seeds(960, 20, 0), draw_seeds(960, 20, 0))
        self.assertNotEqual(draw_seeds(960, 20, 0), draw_seeds(960, 20, 1))

    def test_draw_is_over_all_rows_without_replacement(self):
        d = draw_seeds(60, 40, 0)
        self.assertEqual(len(d), 40)
        self.assertEqual(len(set(d)), 40)
        self.assertTrue(all(0 <= i < 60 for i in d))

    def test_k_equal_to_rows_covers_every_row(self):
        self.assertEqual(sorted(draw_seeds(5, 5, 0)), [0, 1, 2, 3, 4])

    def test_k_larger_than_rows_raises(self):
        with self.assertRaises(ValueError):
            draw_seeds(5, 6, 0)
        with self.assertRaises(ValueError):
            seed_draw([0] * 5, [0] * 5, 6, 0)


class SeedDrawTest(unittest.TestCase):
    def test_accuracy_fpr_recall_by_hand(self):
        # Every row a seed. Cluster 0 truth 1,0,0,0,0 -> legit; cluster 1 truth 0,1,1,1,1 -> 1;
        # noise truth 0,0,0,0 -> legit. TP 4, FN 1, FP 1, TN 8.
        clusters = [0] * 5 + [1] * 5 + [-1] * 4
        truth = [1, 0, 0, 0, 0] + [0, 1, 1, 1, 1] + [0] * 4
        with all_rows(14):
            r = seed_draw(clusters, truth, 14, 0)
        self.assertEqual(r, {"accuracy": 12 / 14, "fpr": 1 / 9, "recall": 4 / 5})

    def test_moving_noise_truth_changes_recall_not_ari(self):
        clusters = [0] * 5 + [1] * 5 + [-1] * 4
        base = [0, 0, 0, 0, 1] + [1, 1, 1, 1, 0]
        with all_rows(14):
            a = seed_draw(clusters, base + [0] * 4, 14, 0)
            b = seed_draw(clusters, base + [1] * 4, 14, 0)
        self.assertEqual(a["recall"], 4 / 5)
        self.assertEqual(b["recall"], 4 / 9)
        self.assertEqual(a["fpr"], 1 / 9)
        self.assertEqual(b["fpr"], 1 / 5)
        self.assertEqual(b["accuracy"], 8 / 14)

    def test_seed_tie_is_legitimate(self):
        clusters = [0] * 4 + [1] * 4
        truth = [1, 0, 1, 0, 1, 1, 1, 1]
        with mock.patch("bqc.cluster.draw_seeds", return_value=[0, 1]):
            r = seed_draw(clusters, truth, 2, 0)
        # Cluster 0 tied 1-1 and cluster 1 has no seed: everything legitimate.
        self.assertEqual(r, {"accuracy": 2 / 8, "fpr": 0.0, "recall": 0.0})

    def test_strict_seed_majority_is_illegitimate(self):
        clusters = [0] * 4 + [1] * 4
        truth = [1, 0, 1, 0, 1, 1, 1, 1]
        with mock.patch("bqc.cluster.draw_seeds", return_value=[0, 1, 2]):
            r = seed_draw(clusters, truth, 3, 0)
        # Cluster 0 predicted 1: TP 2 of 6, both negatives are false positives.
        self.assertEqual(r, {"accuracy": 2 / 8, "fpr": 1.0, "recall": 2 / 6})

    def test_cluster_without_seed_and_noise_fail_open(self):
        clusters = [0, 0, 0, 1, 1, 1, -1, -1]
        truth = [1, 1, 1, 1, 1, 1, 1, 1]
        with mock.patch("bqc.cluster.draw_seeds", return_value=[0, 1, 6]):
            r = seed_draw(clusters, truth, 3, 0)
        # Only cluster 0 is labelled (1); cluster 1 has no seed, noise labels nothing.
        self.assertEqual(r["recall"], 3 / 8)
        self.assertEqual(r["accuracy"], 3 / 8)
        self.assertIsNone(r["fpr"])

    def test_seed_on_noise_point_labels_nothing(self):
        clusters = [0, 0, 0, -1, -1]
        truth = [1, 1, 1, 1, 1]
        with mock.patch("bqc.cluster.draw_seeds", return_value=[3, 4]):
            r = seed_draw(clusters, truth, 2, 0)
        self.assertEqual(r, {"accuracy": 0.0, "fpr": None, "recall": 0.0})

    def test_real_draw_matches_reference_loop(self):
        rng = random.Random(1)
        n = 60
        clusters = [rng.choice([-1, 0, 1, 2]) for _ in range(n)]
        truth = [rng.choice([0, 1]) for _ in range(n)]
        seeds = random.Random(SEED + 5).sample(range(n), 10)
        votes = {}
        for i in seeds:
            if clusters[i] != -1:
                votes.setdefault(clusters[i], []).append(truth[i])
        pred = [1 if clusters[i] in votes and sum(votes[clusters[i]]) * 2 > len(votes[clusters[i]])
                else 0 for i in range(n)]
        tp = sum(p and t for p, t in zip(pred, truth))
        fp = sum(p and not t for p, t in zip(pred, truth))
        r = seed_draw(clusters, truth, 10, 5)
        self.assertEqual(r["accuracy"], sum(p == t for p, t in zip(pred, truth)) / n)
        self.assertEqual(r["recall"], tp / sum(truth))
        self.assertEqual(r["fpr"], fp / (n - sum(truth)))

    def test_many_clusters_per_class(self):
        clusters = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        truth = [0] * 6 + [1] * 6
        with all_rows(12):
            self.assertEqual(seed_draw(clusters, truth, 12, 0),
                             {"accuracy": 1.0, "fpr": 0.0, "recall": 1.0})

    def test_no_positives_recall_none(self):
        with all_rows(4):
            r = seed_draw([0, 0, 1, 1], [0, 0, 0, 0], 4, 0)
        self.assertEqual(r, {"accuracy": 1.0, "fpr": 0.0, "recall": None})


class SeedRuleTest(unittest.TestCase):
    def test_draw_indices_and_k_grid(self):
        calls = []

        def fake(clusters, truth, k, draw_index):
            calls.append((k, draw_index))
            return {"accuracy": float(draw_index ** 2), "fpr": 0.5, "recall": None}

        with mock.patch("bqc.cluster.seed_draw", side_effect=fake):
            r = seed_rule([0] * 50, [0] * 50)
        self.assertEqual(calls, [(k, i) for k in (10, 20, 40) for i in range(20)])
        self.assertEqual([e["k"] for e in r], [10, 20, 40])
        for e in r:
            self.assertEqual(e["n_draws"], 20)
            # mean of i**2 for 0..19 is 123.5; the median is 90.5
            self.assertEqual(e["accuracy"], 123.5)
            self.assertEqual(e["fpr"], 0.5)
            self.assertIsNone(e["recall"])
            self.assertEqual([d["accuracy"] for d in e["draws"]],
                             [float(i ** 2) for i in range(20)])

    def test_draw_seeds_receives_seed_plus_index(self):
        with mock.patch("bqc.cluster.draw_seeds", wraps=draw_seeds) as spy:
            seed_rule([0] * 50, [0] * 50)
        self.assertEqual(spy.call_args_list[0].args, (50, 10, 0))
        self.assertEqual(spy.call_args_list[-1].args, (50, 40, 19))
        self.assertEqual(len(spy.call_args_list), 60)

    def test_json_ready(self):
        rng = random.Random(2)
        clusters = [rng.choice([-1, 0, 1]) for _ in range(60)]
        truth = [rng.choice([0, 1]) for _ in range(60)]
        r = seed_rule(clusters, truth)
        self.assertEqual(json.loads(json.dumps(r)), r)

    def test_all_noise(self):
        truth = [0, 1] * 30
        for e in seed_rule([-1] * 60, truth):
            self.assertEqual((e["fpr"], e["recall"], e["accuracy"]), (0.0, 0.0, 0.5))

    def test_too_few_rows_for_largest_k_raises(self):
        with self.assertRaises(ValueError):
            seed_rule([0] * 30, [0] * 30)


if __name__ == "__main__":
    unittest.main()
