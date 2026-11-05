import json
import unittest
from collections import Counter

from synth_stubs import build_family, build_registry

from bqc import SEED
from bqc.evaluate import (
    ARMS,
    COMPARISON,
    PRIMARY,
    FoldError,
    assign_folds,
    check_fold,
    iter_fold_data,
)
from bqc.negatives import build_spans
from bqc.synth.registry import (
    ALL_FAMILIES,
    EVAL_FAMILY,
    TEMPLATES_PER_FAMILY,
    TRAINING_FAMILIES,
    Registry,
    Template,
)
from bqc.wands import WandsProduct

WORDS = ["oak", "desk", "lamp", "rug", "sofa", "bed", "shelf", "vase"]
REAL = [f"{a} {b}" for a in WORDS for b in WORDS if a != b][:40]
REAL[0] = "oak quillon"  # sentinel word `quillon` appears in one real query only


def alpha(i):
    """A digit-free word for each number (the stub family A hard template needs one)."""
    return "pw" + "".join(chr(97 + int(c)) for c in str(i))


PRODUCTS = [
    WandsProduct(i, " ".join(alpha(i * 100 + j) for j in range(12)), ()) for i in range(300)]
PRODUCTS[0] = WandsProduct(0, "spansentinel " + " ".join(alpha(j) for j in range(11)), ())
CONFIG = dict(n=120, n_splits=3, repeats=2, test_per_template=5, eval_per_template=4)


def run(registry=None, real=REAL, products=PRODUCTS, part_numbers=(), sanity=(), **kw):
    options = {**CONFIG, **kw}
    registry = registry or build_registry()
    return list(iter_fold_data(
        real, products, list(part_numbers), list(sanity), registry=registry, **options))


def assignments(**kw):
    return assign_folds(REAL, build_registry(), **{**dict(n_splits=3, repeats=2), **kw})


def custom_registry(texts):
    """Stub registry whose last template per family renders only strings from `texts`."""
    registry = Registry()
    for family in ALL_FAMILIES:
        build_family(registry, family)
        last = registry._templates[family].pop()
        registry.register(Template(
            family, last.template_id, last.hard, lambda rng, vocab: rng.choice(texts)))
    return registry


class AssignFoldsTest(unittest.TestCase):
    def test_counts_and_no_group_on_both_sides(self):
        folds = assignments()
        self.assertEqual(len(folds), 6)
        all_templates = {(f, t.template_id) for f in TRAINING_FAMILIES
                         for t in build_registry().templates(f)}
        for a in folds:
            self.assertFalse(set(a.train_queries) & set(a.test_queries))
            self.assertFalse(set(a.train_templates) & set(a.test_templates))
            self.assertEqual(set(a.train_templates) | set(a.test_templates), all_templates)
            self.assertEqual(len(a.train_queries) + len(a.test_queries), len(REAL))

    def test_each_group_in_one_test_fold_per_repeat(self):
        folds = assignments()
        for repeat in (0, 1):
            mine = [a for a in folds if a.repeat == repeat]
            self.assertEqual([a.fold for a in mine], [0, 1, 2])
            self.assertEqual(sorted(q for a in mine for q in a.test_queries), sorted(REAL))
            self.assertEqual(
                Counter(t for a in mine for t in a.test_templates),
                Counter({(f, t.template_id): 1 for f in TRAINING_FAMILIES
                         for t in build_registry().templates(f)}))

    def test_spans_and_e_are_not_fold_units(self):
        for a in assignments():
            self.assertTrue(all(f in TRAINING_FAMILIES
                                for f, _ in a.train_templates + a.test_templates))

    def test_same_seed_same_folds_and_repeats_differ(self):
        self.assertEqual(assignments(), assignments())
        by_repeat = {r: [a.test_queries for a in assignments() if a.repeat == r] for r in (0, 1)}
        self.assertNotEqual(by_repeat[0], by_repeat[1])
        self.assertNotEqual(assignments(seed=SEED + 7), assignments())

    def test_defaults_are_five_splits_three_repeats(self):
        self.assertEqual(len(assign_folds(REAL, build_registry())), 15)

    def test_group_sizes_come_from_the_registry(self):
        registry = build_registry()
        self.assertEqual(set(registry.group_sizes().values()), {20})
        self.assertEqual(TEMPLATES_PER_FAMILY, 6)


class CheckFoldTest(unittest.TestCase):
    FAMILIES = [None, None, "A", "B", "C", "D"]
    Y = [0, 0, 1, 1, 1, 1]

    def test_complete_fold_passes(self):
        check_fold(self.Y, self.FAMILIES, [0, 1])

    def test_training_missing_class_raises(self):
        with self.assertRaises(FoldError):
            check_fold([1, 1, 1, 1], ["A", "B", "C", "D"], [0, 1])
        with self.assertRaises(FoldError):
            check_fold([0, 0], [None, None], [0, 1])

    def test_training_missing_family_raises(self):
        with self.assertRaises(FoldError):
            check_fold([0, 1, 1, 1], [None, "A", "B", "C"], [0, 1])

    def test_e_does_not_stand_in_for_a_training_family(self):
        with self.assertRaises(FoldError):
            check_fold([0, 1, 1, 1, 1], [None, "A", "B", "C", "E"], [0, 1])

    def test_test_missing_class_raises(self):
        with self.assertRaises(FoldError):
            check_fold(self.Y, self.FAMILIES, [1, 1])
        with self.assertRaises(FoldError):
            check_fold(self.Y, self.FAMILIES, [0, 0])


class FoldDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = run()
        cls.by_key = {(d.repeat, d.fold, d.arm): d for d in cls.data}

    def test_one_item_per_repeat_fold_arm(self):
        self.assertEqual(len(self.data), 2 * 3 * 2)
        self.assertEqual(len(self.by_key), len(self.data))
        self.assertEqual(ARMS, (PRIMARY, COMPARISON))

    def test_no_e_in_training_and_test_has_e(self):
        for d in self.data:
            self.assertNotIn(EVAL_FAMILY, d.train_families)
            e_count = d.test_families.count(EVAL_FAMILY)
            self.assertEqual(d.report["test"]["families"][EVAL_FAMILY], e_count)
            self.assertGreater(e_count, 0)
            self.assertLessEqual(e_count, 6 * CONFIG["eval_per_template"])

    def test_test_side_identical_across_arms(self):
        for r in (0, 1):
            for k in (0, 1, 2):
                p, c = self.by_key[(r, k, PRIMARY)], self.by_key[(r, k, COMPARISON)]
                self.assertEqual(p.test_texts, c.test_texts)
                self.assertEqual(p.test_y, c.test_y)
                self.assertEqual(p.test_families, c.test_families)

    def test_test_side_composition(self):
        folds = {(a.repeat, a.fold): a for a in assignments()}
        for d in self.data:
            a = folds[(d.repeat, d.fold)]
            n_real = len(a.test_queries)
            self.assertEqual(d.test_texts[:n_real], a.test_queries)
            self.assertEqual(d.test_y[:n_real], [0] * n_real)
            self.assertTrue(all(f is None for f in d.test_families[:n_real]))
            counts = Counter(d.test_families[n_real:])
            self.assertTrue(set(counts) <= set(ALL_FAMILIES))
            self.assertEqual(len(d.test_texts), len(d.test_y))
            self.assertEqual(len(d.test_texts), len(d.test_families))

    def test_primary_training_side(self):
        folds = {(a.repeat, a.fold): a for a in assignments()}
        for r in (0, 1):
            spans, _ = build_spans(PRODUCTS, CONFIG["n"], SEED + r, REAL, [], [])
            span_texts = [s.text for s in spans]
            for k in (0, 1, 2):
                d = self.by_key[(r, k, PRIMARY)]
                a = folds[(r, k)]
                self.assertEqual(d.train_texts[:len(span_texts)], span_texts)
                self.assertEqual(d.train_y[:len(span_texts)], [0] * len(span_texts))
                self.assertEqual(d.threshold_texts, a.train_queries)
                self.assertFalse(set(d.train_texts) & set(REAL))
                self.assertEqual(d.train_y.count(0), len(span_texts))

    def test_spans_shared_across_folds_and_differ_across_repeats(self):
        def spans(r, k):
            d = self.by_key[(r, k, PRIMARY)]
            return d.train_texts[:d.train_y.index(1)]
        self.assertEqual(spans(0, 0), spans(0, 1))
        self.assertEqual(spans(0, 0), spans(0, 2))
        self.assertNotEqual(spans(0, 0), spans(1, 0))

    def test_comparison_training_side(self):
        folds = {(a.repeat, a.fold): a for a in assignments()}
        for d in self.data:
            if d.arm != COMPARISON:
                continue
            a = folds[(d.repeat, d.fold)]
            n_real = len(a.train_queries)
            self.assertEqual(d.train_texts[:n_real], a.train_queries)
            self.assertEqual(d.train_y.count(0), n_real)
            self.assertEqual(d.threshold_texts, [])
            self.assertEqual(d.train_y.count(1), 20 * len(a.train_templates) - sum(
                d.report["train"]["shortfall"].values()) - sum(
                    d.report["train"]["dropped"].values()))

    def test_primary_positives_per_template(self):
        folds = {(a.repeat, a.fold): a for a in assignments()}
        for d in self.data:
            if d.arm == PRIMARY:
                a = folds[(d.repeat, d.fold)]
                per = CONFIG["n"] // len(a.train_templates)
                self.assertEqual(d.report["train"]["requested_per_template"], per)

    def test_every_fold_passes_the_check(self):
        for d in self.data:
            check_fold(d.train_y, d.train_families, d.test_y)

    def test_report_counts_and_zero_test_families(self):
        for d in self.data:
            rep = d.report
            self.assertEqual((rep["repeat"], rep["fold"], rep["arm"]),
                             (d.repeat, d.fold, d.arm))
            for side, y, fams in (("train", d.train_y, d.train_families),
                                  ("test", d.test_y, d.test_families)):
                self.assertEqual(rep[side]["negatives"], y.count(0))
                self.assertEqual(rep[side]["positives"], y.count(1))
                self.assertEqual(set(rep[side]["families"]), set(ALL_FAMILIES))
                for f in ALL_FAMILIES:
                    self.assertEqual(rep[side]["families"][f], fams.count(f))
            self.assertEqual(set(rep["span_removals"]),
                             {"empty", "drawn", "real_queries", "part_numbers", "sanity"})

    def test_report_is_json_serialisable(self):
        json.dumps([d.report for d in self.data])


class VocabularyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = run()

    def words(self, d, side):
        texts = d.train_texts if side == "train" else d.test_texts
        y = d.train_y if side == "train" else d.test_y
        return [w for t, label in zip(texts, y) if label for w in t.split()]

    def test_primary_training_positives_have_no_real_query_token(self):
        real_tokens = {w for q in REAL for w in q.split()}
        for d in self.data:
            if d.arm == PRIMARY:
                self.assertFalse(real_tokens & set(self.words(d, "train")))
        # positive control: catalogue words do reach them
        self.assertTrue(any("pw" in w for d in self.data if d.arm == PRIMARY
                            for w in self.words(d, "train")))

    def test_span_only_word_never_reaches_test_side(self):
        # A sentinel word only in a span: build the spans, find one word that is in no real query.
        for d in self.data:
            if d.arm != PRIMARY:
                continue
            n_spans = d.train_y.index(1)
            span_words = {w for t in d.train_texts[:n_spans] for w in t.split()}
            real_words = {w for q in REAL for w in q.split()}
            only_span = span_words - real_words
            self.assertTrue(only_span)
            self.assertFalse(only_span & set(self.words(d, "test")))

    def test_comparison_uses_training_side_queries_only(self):
        folds = {(a.repeat, a.fold): a for a in assignments()}
        reached = False
        for d in self.data:
            if d.arm != COMPARISON:
                continue
            a = folds[(d.repeat, d.fold)]
            train_words = {w for q in a.train_queries for w in q.split()}
            test_only = {w for q in a.test_queries for w in q.split()} - train_words
            self.assertFalse(test_only & set(self.words(d, "train")))
            reached |= bool(train_words & set(self.words(d, "train")))
        self.assertTrue(reached)

    def test_sentinel_in_test_query_reaches_test_positives_not_training(self):
        folds = {(a.repeat, a.fold): a for a in assignments()}
        in_test = False
        for d in self.data:
            a = folds[(d.repeat, d.fold)]
            if any("quillon" in q for q in a.test_queries):
                self.assertNotIn("quillon", self.words(d, "train"))
                in_test |= "quillon" in self.words(d, "test")
        self.assertTrue(in_test)

    def test_frequent_word_dominates_rare_word(self):
        # A side whose real queries hold `chair` x10 and `zen` x1 (8.3, 13.17).
        real = [f"chair q{i}" for i in range(10)] + ["zen q10"] + [f"chair r{i}" for i in range(10)]
        real += [f"item s{i}" for i in range(20)]
        data = run(real=real, n_splits=2, repeats=1, arms=(COMPARISON,))
        words = Counter()
        for d in data:
            words.update(w for t, y in zip(d.train_texts, d.train_y) if y for w in t.split())
        self.assertGreater(words["chair"], 3 * max(words["zen"], 1))


class RuleFourTest(unittest.TestCase):
    def first_positive(self, side, arm=PRIMARY):
        d = next(d for d in run() if d.arm == arm)
        y = d.test_y if side == "test" else d.train_y
        texts = d.test_texts if side == "test" else d.train_texts
        return next(t for t, label in zip(texts, y) if label)

    def test_collision_with_sanity_raises_on_training_side(self):
        with self.assertRaises(FoldError) as cm:
            run(sanity=[self.first_positive("train")])
        self.assertNotIn(self.first_positive("train"), str(cm.exception))

    def test_collision_with_sanity_raises_on_test_side(self):
        with self.assertRaises(FoldError):
            run(sanity=[self.first_positive("test")])

    def test_collision_with_sanity_raises_in_the_comparison_arm(self):
        with self.assertRaises(FoldError):
            run(sanity=[self.first_positive("train", COMPARISON)], arms=(COMPARISON,))

    def test_collisions_are_dropped_counted_and_withheld(self):
        span_of = {r: build_spans(PRODUCTS, CONFIG["n"], SEED + r, REAL, [], [])[0][0].text
                   for r in (0, 1)}
        always = [REAL[5], "kx-9931"]
        keep = ["fresh one", "fresh two", "fresh three"]
        registry = custom_registry(always + list(span_of.values()) + keep)
        data = run(registry=registry, part_numbers=["kx-9931"])
        for d in data:
            forbidden = {*always, span_of[d.repeat]}
            # The comparison arm's training negatives are queries, so a span may be a positive there.
            train_forbidden = forbidden if d.arm == PRIMARY else set(always)
            self.assertFalse(train_forbidden & set(d.train_texts[d.train_y.index(1):]))
            self.assertFalse(forbidden & {t for t, y in zip(d.test_texts, d.test_y) if y})
            blob = json.dumps(d.report)
            for text in forbidden:
                self.assertNotIn(text, blob)
        total = Counter()
        for d in data:
            for side in ("train", "test"):
                total.update(d.report[side]["dropped"])
        self.assertGreater(total["training_negatives"], 0)
        self.assertGreater(total["real_queries"], 0)
        self.assertGreater(total["part_numbers"], 0)


class DegenerateFoldTest(unittest.TestCase):
    def test_a_fold_missing_a_training_family_raises(self):
        registry = Registry()
        for family in ALL_FAMILIES:
            build_family(registry, family)
        # Every template of family D renders the same string, which a real query equals:
        # all of D is dropped, so no training fold has family D.
        d_text = REAL[5]
        for t in list(registry._templates["D"]):
            registry._templates["D"].remove(t)
            registry.register(Template("D", t.template_id, t.hard, lambda rng, v: d_text))
        with self.assertRaises(FoldError):
            run(registry=registry)


if __name__ == "__main__":
    unittest.main()
