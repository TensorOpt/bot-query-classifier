import inspect
import json
import unittest
from collections import Counter
from unittest import mock

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
from bqc.negatives import (
    PART_NUMBER_COUNT,
    PART_NUMBER_SEED_OFFSET,
    PART_NUMBER_SOURCE,
    PartNumberNegative,
    PoolExhausted,
    build_part_number_negatives,
    build_spans,
    token_occurrences,
)
from bqc.synth.partnumbers import SHAPE_IDS
from bqc.synth.registry import (
    ALL_FAMILIES,
    EVAL_FAMILY,
    QUOTA,
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
                self.assertEqual(d.train_y.count(0), len(span_texts) + PART_NUMBER_COUNT)

    def test_spans_shared_across_folds_and_differ_across_repeats(self):
        def spans(r, k):
            d = self.by_key[(r, k, PRIMARY)]
            return d.train_texts[:CONFIG["n"]]
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
            self.assertEqual(d.train_y.count(0), n_real + PART_NUMBER_COUNT)
            self.assertEqual(d.threshold_texts, [])
            self.assertEqual(d.train_y.count(1), 20 * len(a.train_templates) - sum(
                d.report["train"]["shortfall"].values()) - sum(
                    d.report["train"]["dropped"].values()))

    def test_primary_positives_per_template(self):
        folds = {(a.repeat, a.fold): a for a in assignments()}
        for d in self.data:
            if d.arm == PRIMARY:
                a = folds[(d.repeat, d.fold)]
                per = (CONFIG["n"] + PART_NUMBER_COUNT) // len(a.train_templates)
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
                hard = PART_NUMBER_COUNT if side == "train" else 0
                self.assertEqual(rep[side]["negatives"], y.count(0) - hard)
                self.assertEqual(fams.count(PART_NUMBER_SOURCE), hard)
                self.assertEqual(rep[side]["positives"], y.count(1))
                self.assertEqual(set(rep[side]["families"]), set(ALL_FAMILIES))
                for f in ALL_FAMILIES:
                    self.assertEqual(rep[side]["families"][f], fams.count(f))
            self.assertEqual(set(rep["span_removals"]),
                             {"empty", "drawn", "real_queries", "part_numbers", "sanity"})

    def test_family_absent_from_a_test_fold_does_not_raise_and_reports_zero(self):
        # Fixture folds: family A has no test sample in (0, 2) and (1, 0), yet it trains.
        for key in ((0, 2), (1, 0)):
            for arm in ARMS:
                d = self.by_key[(*key, arm)]
                self.assertEqual(d.test_families.count("A"), 0)
                self.assertEqual(d.report["test"]["families"]["A"], 0)
                self.assertGreater(d.report["train"]["families"]["A"], 0)
                check_fold(d.train_y, d.train_families, d.test_y)

    def test_groups_give_the_template_of_each_row(self):
        folds = {(a.repeat, a.fold): a for a in assignments()}
        for d in self.data:
            a = folds[(d.repeat, d.fold)]
            for side, groups, families, y, allowed in (
                    ("train", d.train_groups, d.train_families, d.train_y, a.train_templates),
                    ("test", d.test_groups, d.test_families, d.test_y, a.test_templates)):
                self.assertEqual(len(groups), len(y))
                for group, family, label in zip(groups, families, y):
                    if family == PART_NUMBER_SOURCE:
                        self.assertEqual((label, group[0]), (0, PART_NUMBER_SOURCE))
                        self.assertIn(group[1], SHAPE_IDS)
                    elif label == 0:
                        self.assertIsNone(group)
                    else:
                        self.assertEqual(group[0], family)
                        if group[0] != EVAL_FAMILY:
                            self.assertIn(group, allowed)
                if side == "test":
                    e_ids = {g for g in groups if g and g[0] == EVAL_FAMILY}
                    self.assertEqual(
                        e_ids, {(EVAL_FAMILY, t.template_id)
                                for t in build_registry().templates(EVAL_FAMILY)})
                    self.assertEqual(
                        {g for g in groups if g and g[0] != EVAL_FAMILY}, set(a.test_templates))

    def test_report_is_json_serialisable(self):
        json.dumps([d.report for d in self.data])


class SpanInputsTest(unittest.TestCase):
    """build_spans must get every list the repeat needs, each with `seed + repeat`."""

    def calls(self):
        with mock.patch("bqc.evaluate.build_spans", wraps=build_spans) as spy:
            run(part_numbers=["kx-9931"], sanity=["unrelated sanity query"])
        return spy.call_args_list

    def test_build_spans_receives_all_three_lists_and_the_repeat_seed(self):
        calls = self.calls()
        self.assertEqual(len(calls), CONFIG["repeats"])
        for repeat, call in enumerate(calls):
            products, n, seed, real, part_numbers, sanity = call.args
            self.assertEqual((n, seed), (CONFIG["n"], SEED + repeat))
            self.assertIs(products, PRODUCTS)
            self.assertEqual(list(real), REAL)
            self.assertEqual(list(part_numbers), ["kx-9931"])
            self.assertEqual(list(sanity), ["unrelated sanity query"])


class VocabularyCapTest(unittest.TestCase):
    def test_primary_training_positives_draw_only_on_the_first_tokens(self):
        cap = 6
        data = [d for d in run(vocabulary_cap=cap) if d.arm == PRIMARY]
        for d in data:
            n_spans = CONFIG["n"]
            span_texts = d.train_texts[:n_spans]
            allowed = set(token_occurrences(span_texts, cap=cap))
            span_words = {w for t in span_texts for w in t.split()}
            positives = [t for t, y in zip(d.train_texts, d.train_y) if y]
            used = {w for t in positives for w in t.split()} & span_words
            self.assertTrue(used)
            self.assertLessEqual(used, allowed)
            self.assertGreater(len(span_words), len(allowed))


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
            n_spans = CONFIG["n"]
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


BUILDER = "bqc.evaluate.build_part_number_negatives"
REMOVAL_KINDS_PN = ("training_negatives", "real_queries", "part_numbers", "sanity", "earlier")


def recording_builder():
    """Patch the builder with a wrapper that records each call's arguments."""
    return mock.patch(BUILDER, wraps=build_part_number_negatives)


def named(call):
    """The arguments of a recorded builder call by parameter name."""
    return dict(inspect.signature(build_part_number_negatives).bind(
        *call.args, **call.kwargs).arguments)


def fixed_builder(prefix="zqxtok"):
    """A builder stand-in: 100 distinct rows, 25 per shape, every token starting `prefix`."""
    def build(vocabulary, seed, *lists):
        rows = [PartNumberNegative(f"{prefix}{i} 55", PART_NUMBER_SOURCE, SHAPE_IDS[i // 25], seed)
                for i in range(PART_NUMBER_COUNT)]
        return rows, dict.fromkeys(REMOVAL_KINDS_PN, 0)
    return build


def pn_texts(d):
    return [t for t, f in zip(d.train_texts, d.train_families) if f == PART_NUMBER_SOURCE]


def pn_shapes(d):
    return [g[1] for g, f in zip(d.train_groups, d.train_families) if f == PART_NUMBER_SOURCE]


class PartNumberNegativesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with recording_builder() as spy:
            cls.data = run()
        cls.calls = [named(c) for c in spy.call_args_list]
        cls.by_key = {(d.repeat, d.fold, d.arm): d for d in cls.data}
        cls.folds = {(a.repeat, a.fold): a for a in assignments()}

    def test_every_training_side_holds_100_tagged_rows_25_per_shape_label_0(self):
        for d in self.data:
            rows = [i for i, f in enumerate(d.train_families) if f == PART_NUMBER_SOURCE]
            self.assertEqual(len(rows), PART_NUMBER_COUNT, (d.repeat, d.fold, d.arm))
            self.assertEqual({d.train_y[i] for i in rows}, {0})
            self.assertEqual(Counter(pn_shapes(d)), {s: 25 for s in SHAPE_IDS})
            self.assertEqual(
                {d.train_groups[i] for i in rows}, {(PART_NUMBER_SOURCE, s) for s in SHAPE_IDS})
            self.assertEqual(len(set(pn_texts(d))), PART_NUMBER_COUNT)

    def test_rows_never_reach_the_threshold_set_the_test_side_or_e(self):
        for d in self.data:
            texts = set(pn_texts(d))
            self.assertFalse(texts & set(d.threshold_texts))
            self.assertFalse(texts & set(d.test_texts))
            self.assertNotIn(PART_NUMBER_SOURCE, d.test_families)
            self.assertNotIn(PART_NUMBER_SOURCE, {g[0] for g in d.test_groups if g})
            self.assertEqual(d.report["test"]["negatives"], d.test_y.count(0))

    def test_rows_are_ascii_in_both_arms(self):
        for d in self.data:
            self.assertTrue(all(t.isascii() for t in pn_texts(d)))

    def test_the_reported_negatives_stay_the_span_or_query_count(self):
        for d in self.data:
            expected = (CONFIG["n"] if d.arm == PRIMARY
                        else len(self.folds[(d.repeat, d.fold)].train_queries))
            self.assertEqual(d.report["train"]["negatives"], expected)
            self.assertEqual(d.train_y.count(0), expected + PART_NUMBER_COUNT)

    def test_primary_arm_same_within_a_repeat_different_across_repeats(self):
        def rows(r, k):
            return pn_texts(self.by_key[(r, k, PRIMARY)])
        self.assertEqual(rows(0, 0), rows(0, 1))
        self.assertEqual(rows(0, 0), rows(0, 2))
        self.assertNotEqual(rows(0, 0), rows(1, 0))

    def test_seed_is_seed_plus_offset_plus_repeat_for_the_primary_arm(self):
        self.assertEqual(PART_NUMBER_SEED_OFFSET, 100_000)
        primary = [c for c in self.calls if len(c["training_negatives"]) == CONFIG["n"]]
        self.assertEqual([c["seed"] for c in primary], [SEED + 100_000, SEED + 100_001])

    def test_primary_builds_once_per_repeat_and_comparison_once_per_fold(self):
        sizes = Counter(len(c["training_negatives"]) == CONFIG["n"] for c in self.calls)
        self.assertEqual(sizes, {True: 2, False: 6})

    def test_comparison_folds_of_a_repeat_share_the_seed_not_the_vocabulary(self):
        comparison = [c for c in self.calls if len(c["training_negatives"]) != CONFIG["n"]]
        ordered = sorted(self.folds.values(), key=lambda a: (a.repeat, a.fold))
        self.assertEqual(len(comparison), len(ordered))
        for c, a in zip(comparison, ordered):
            self.assertEqual(c["seed"], SEED + 100_000 + a.repeat)
            self.assertEqual(list(c["training_negatives"]), a.train_queries)
            self.assertEqual(c["vocabulary"], token_occurrences(a.train_queries))
            self.assertEqual("quillon" in c["vocabulary"], REAL[0] in a.train_queries)
        self.assertTrue(any("quillon" in c["vocabulary"] for c in comparison))
        self.assertTrue(any("quillon" not in c["vocabulary"] for c in comparison))

    def test_primary_vocabulary_is_the_spans_capped_occurrences(self):
        primary = [c for c in self.calls if len(c["training_negatives"]) == CONFIG["n"]]
        for c in primary:
            self.assertEqual(
                c["vocabulary"], token_occurrences(c["training_negatives"], cap=10_000))
            self.assertNotIn("quillon", c["vocabulary"])  # a real-query token, not a span token

    def test_a_test_side_sentinel_never_reaches_a_negative_in_either_arm(self):
        checked = 0
        for d in self.data:
            if REAL[0] in self.folds[(d.repeat, d.fold)].test_queries:
                checked += 1
                self.assertFalse(any("quillon" in t for t in pn_texts(d)))
        self.assertGreater(checked, 0)

    def test_a_training_side_sentinel_reaches_negatives_only_in_the_comparison_arm(self):
        reached = {PRIMARY: False, COMPARISON: False}
        for d in self.data:
            if any("quillon" in t for t in pn_texts(d)):
                reached[d.arm] = True
                self.assertIn(REAL[0], self.folds[(d.repeat, d.fold)].train_queries)
        self.assertEqual(reached, {PRIMARY: False, COMPARISON: True})

    def test_product_words_in_primary_negatives_come_from_the_spans(self):
        found = False
        for d in self.data:
            if d.arm == PRIMARY:
                vocabulary = set(token_occurrences(d.train_texts[:CONFIG["n"]]))
                words = {w for t in pn_texts(d) for w in t.split() if w.startswith("pw")}
                found |= bool(words)
                self.assertLessEqual(words, vocabulary)
        self.assertTrue(found)

    def test_drop_lists_are_the_arms_negatives_all_real_queries_p_strings_and_sanity(self):
        with recording_builder() as spy:
            run(part_numbers=["kx-9931"], sanity=["not a bot"])
        calls = [named(c) for c in spy.call_args_list]
        self.assertEqual(len(calls), 8)
        train_queries = [a.train_queries for a in self.folds.values()]
        for c in calls:
            self.assertEqual(list(c["real_queries"]), REAL)
            self.assertEqual(list(c["part_numbers"]), ["kx-9931"])
            self.assertEqual(list(c["sanity"]), ["not a bot"])
            negatives = list(c["training_negatives"])
            if len(negatives) == CONFIG["n"]:
                self.assertFalse(set(negatives) & set(REAL))  # the spans
            else:
                self.assertIn(negatives, train_queries)

    def test_a_p_string_equal_to_a_candidate_is_dropped_and_counted_in_both_arms(self):
        for arm in ARMS:
            victim = pn_texts(self.by_key[(0, 0, arm)])[0]
            data = run(part_numbers=[victim], arms=(arm,))
            counted = 0
            for d in data:
                self.assertNotIn(victim, pn_texts(d))
                self.assertEqual(len(pn_texts(d)), PART_NUMBER_COUNT)
                counted += d.report["part_number_negatives"]["removals"]["part_numbers"]
            self.assertGreaterEqual(counted, 1)

    def test_a_sanity_query_equal_to_a_candidate_is_dropped_and_counted_in_both_arms(self):
        for arm in ARMS:
            victim = pn_texts(self.by_key[(0, 0, arm)])[0]
            data = run(sanity=[victim], arms=(arm,))
            counted = 0
            for d in data:
                self.assertNotIn(victim, pn_texts(d))
                counted += d.report["part_number_negatives"]["removals"]["sanity"]
            self.assertGreaterEqual(counted, 1)

    def removals_with_a_frequent_candidate(self, victim, arm):
        """Run with every shape emitting `victim` about 30% of the time; the removal counts."""
        def render(rng, vocabulary):
            return victim if rng.random() < 0.3 else f"free{rng.randrange(10**6)} 1"
        total = Counter()
        with mock.patch("bqc.negatives.SHAPES", [(s, render) for s in SHAPE_IDS]):
            for d in run(arms=(arm,), repeats=1):
                self.assertNotIn(victim, pn_texts(d))
                self.assertEqual(len(pn_texts(d)), PART_NUMBER_COUNT)
                total.update(d.report["part_number_negatives"]["removals"])
        return total

    def test_a_query_of_either_side_is_dropped_and_counted_in_the_comparison_arm(self):
        # REAL[3] is a training-side query in some folds of repeat 0 and a test-side one in others
        total = self.removals_with_a_frequent_candidate(REAL[3], COMPARISON)
        self.assertGreaterEqual(total["training_negatives"], 1)
        self.assertGreaterEqual(total["real_queries"], 1)

    def test_a_span_is_dropped_and_counted_in_the_primary_arm(self):
        span = build_spans(PRODUCTS, CONFIG["n"], SEED, REAL, [], [])[0][0].text
        total = self.removals_with_a_frequent_candidate(span, PRIMARY)
        self.assertGreaterEqual(total["training_negatives"], 1)

    def test_a_test_side_real_query_is_dropped_and_counted_in_the_primary_arm(self):
        total = self.removals_with_a_frequent_candidate(REAL[3], PRIMARY)
        self.assertGreaterEqual(total["real_queries"], 1)

    def test_report_key_next_to_span_removals_with_counts_and_no_text(self):
        for d in self.data:
            block = d.report["part_number_negatives"]
            self.assertEqual(set(block), {"count", "per_shape", "removals"})
            self.assertEqual(block["count"], PART_NUMBER_COUNT)
            self.assertEqual(block["per_shape"], {s: 25 for s in SHAPE_IDS})
            self.assertEqual(set(block["removals"]), set(REMOVAL_KINDS_PN))
            blob = json.dumps(d.report)
            for text in pn_texts(d):
                self.assertNotIn(f'"{text}"', blob)
                self.assertNotIn(text, blob)
            self.assertIn("span_removals", d.report)

    def test_check_fold_and_the_family_counts_ignore_the_tag(self):
        for d in self.data:
            check_fold(d.train_y, d.train_families, d.test_y)
            families = d.report["train"]["families"]
            self.assertEqual(set(families), set(ALL_FAMILIES))
            self.assertEqual(sum(families.values()), d.train_y.count(1))
            self.assertEqual(d.report["train"]["positives"], d.train_y.count(1))

    def test_primary_quota_counts_the_100_as_negatives(self):
        for d in self.data:
            if d.arm == PRIMARY:
                templates = len(self.folds[(d.repeat, d.fold)].train_templates)
                per = d.report["train"]["requested_per_template"]
                self.assertEqual(per, (CONFIG["n"] + 100) // templates)
                self.assertNotEqual(per, CONFIG["n"] // templates)  # the old formula

    def test_comparison_quota_in_the_main_run_stays_the_fixed_quota(self):
        for d in self.data:
            if d.arm == COMPARISON:
                self.assertEqual(d.report["train"]["requested_per_template"], QUOTA)


class PartNumberPositivesTest(unittest.TestCase):
    def test_a_sentinel_token_in_the_negatives_is_in_no_training_positive(self):
        with mock.patch(BUILDER, side_effect=fixed_builder()):
            data = run()
        for d in data:
            self.assertEqual(sum(f == PART_NUMBER_SOURCE for f in d.train_families), 100)
            for t, y in zip(d.train_texts, d.train_y):
                if y == 1:
                    self.assertNotIn("zqxtok", t)

    def test_a_positive_equal_to_a_negative_is_dropped_counted_and_the_test_side_unscreened(self):
        victims = sorted({t for d in run() for t in pn_texts(d)})
        registry = custom_registry(victims + ["fresh one", "fresh two"])
        patched = run(registry=registry)
        with mock.patch(BUILDER, side_effect=fixed_builder("other")):
            unpatched = run(registry=registry)  # same templates, different negatives
        self.assertEqual(len(patched), len(unpatched))
        for d, u in zip(patched, unpatched):
            where = (d.repeat, d.fold, d.arm)
            positives = {t for t, y in zip(d.train_texts, d.train_y) if y}
            self.assertFalse(set(pn_texts(d)) & positives, where)
            self.assertGreater(d.report["train"]["dropped"]["part_number_negatives"], 0, where)
            self.assertEqual(d.test_texts, u.test_texts, where)
            self.assertEqual(d.test_y, u.test_y)
            self.assertEqual(d.test_families, u.test_families)
            self.assertEqual(d.test_groups, u.test_groups)
            self.assertEqual(d.report["test"], u.report["test"])
        self.assertTrue(any(set(d.test_texts) & set(victims) for d in patched))


class PartNumberExhaustionTest(unittest.TestCase):
    def test_pool_exhausted_from_the_builder_stops_iteration(self):
        for arm in ARMS:
            with mock.patch(BUILDER, side_effect=PoolExhausted("reached 7 of 100", 7)):
                with self.assertRaises(PoolExhausted) as cm:
                    run(arms=(arm,))
            self.assertEqual(cm.exception.reached, 7)


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
