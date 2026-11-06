"""Outer folds, per-fold data, thresholds and the primary-arm runner (DESIGN.md 8.1, 8.2,
8.3, 8.4, 11.16, 11.30).

Fold units are the real queries (one group each) and the training templates
(one group each, nominal size QUOTA). Spans are never fold units. `select_threshold`
picks the threshold from the real queries' scores, `check_finite` stops the run on a
non-finite score, and `run_primary_fold` cross-fits one fold of the primary arm.
`compute_metrics` turns fold results into the numbers of 8.5.
"""
import math
import pickle
import warnings
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from bqc import SEED
from bqc.models import Scorer
from bqc.negatives import (
    DEFAULT_N,
    LENGTH_COUNTS,
    VOCABULARY_CAP,
    build_spans,
    token_occurrences,
)
from bqc.preprocess import normalise
from bqc.report import write_report
from bqc.synth.audit import SanityCollision, screen_samples
from bqc.synth.registry import (
    ALL_FAMILIES,
    EVAL_FAMILY,
    QUOTA,
    REGISTRY,
    TRAINING_FAMILIES,
    Registry,
    Sample,
)
from bqc.wands import WandsProduct

PRIMARY = "product"
COMPARISON = "real"
ARMS = (PRIMARY, COMPARISON)
N_SPLITS = 5
REPEATS = 3
TEST_PER_TEMPLATE = 100
EVAL_PER_TEMPLATE = 20  # family E: 6 templates x 20 = 120


class FoldError(ValueError):
    """A fold lacks a class or a family, or a positive equals a sanity query."""


class FoldAssignment(NamedTuple):
    """One outer fold: real queries (original order) and (family, template_id) pairs per side."""
    repeat: int
    fold: int
    train_queries: list[str]
    test_queries: list[str]
    train_templates: list[tuple[str, str]]
    test_templates: list[tuple[str, str]]


class FoldData(NamedTuple):
    """Texts and labels for one (repeat, fold, arm); `report` is for the run JSON (8.1).

    `*_families` hold the family of each positive and None for a real query.
    `*_groups` hold the `(family, template_id)` of each positive and None for a real
    query: per-template recall (M3.T7) and inner-fold groups (M3.T13) use them, a real
    query being its own group by position.
    `threshold_texts` are the training-side real queries in the primary arm and
    empty in the comparison arm (its threshold comes from an inner split, later).
    """
    repeat: int
    fold: int
    arm: str
    train_texts: list[str]
    train_y: list[int]
    train_families: list[str | None]
    train_groups: list[tuple[str, str] | None]
    threshold_texts: list[str]
    test_texts: list[str]
    test_y: list[int]
    test_families: list[str | None]
    test_groups: list[tuple[str, str] | None]
    report: dict[str, Any]


def assign_folds(
    real_queries: Sequence[str],
    registry: Registry = REGISTRY,
    n_splits: int = N_SPLITS,
    repeats: int = REPEATS,
    seed: int = SEED,
) -> list[FoldAssignment]:
    """StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed + repeat) per repeat (8.1).

    Each real query is a group of 1; each A-D template of `registry` is a group of its
    `group_sizes()` size. Returns repeats * n_splits assignments, repeat-major.
    """
    sizes = registry.group_sizes()
    templates = list(sizes)
    n_queries = len(real_queries)
    groups = list(range(n_queries))
    y = [0] * n_queries
    for j, key in enumerate(templates):
        groups += [n_queries + j] * sizes[key]
        y += [1] * sizes[key]
    x = np.zeros((len(y), 1))
    folds = []
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed + repeat)
        for fold, (_, test_rows) in enumerate(splitter.split(x, y, groups)):
            test_groups = {groups[i] for i in test_rows}
            folds.append(FoldAssignment(
                repeat, fold,
                [q for i, q in enumerate(real_queries) if i not in test_groups],
                [q for i, q in enumerate(real_queries) if i in test_groups],
                [t for j, t in enumerate(templates) if n_queries + j not in test_groups],
                [t for j, t in enumerate(templates) if n_queries + j in test_groups],
            ))
    return folds


def check_fold(
    train_y: Sequence[int],
    train_families: Sequence[str | None],
    test_y: Sequence[int],
    families: Sequence[str] = TRAINING_FAMILIES,
) -> None:
    """Raise FoldError if a class or a family of `families` (A-D by default; the other three
    in LOFO, 8.6) is missing from the training side, or a class from the test side.
    Families are never asserted on the test side (11.16).
    """
    if set(train_y) != {0, 1}:
        raise FoldError(f"training fold classes are {sorted(set(train_y))}, need 0 and 1")
    missing = [f for f in families if f not in set(train_families)]
    if missing:
        raise FoldError(f"training fold lacks families {missing}")
    if set(test_y) != {0, 1}:
        raise FoldError(f"test fold classes are {sorted(set(test_y))}, need 0 and 1")


def _render_screened(
    registry: Registry,
    vocabulary: Sequence[str],
    seed: int,
    templates: Sequence[tuple[str, str]],
    per_template: int,
    sanity: Sequence[str],
    drop_lists: dict[str, Sequence[str]],
) -> tuple[list[Sample], dict[str, int], dict[str, int]]:
    """Render, fail on a sanity match, drop other matches (rule 4).

    Returns the kept samples, per-template shortfalls (keys "family/template_id") and the
    removals per list in `drop_lists`. No message or count holds text.
    """
    samples, shortfall = registry.render(vocabulary, seed, templates, per_template)
    try:
        kept, dropped = screen_samples(samples, drop_lists, sanity)
    except SanityCollision:
        raise FoldError("a positive equals a sanity query (text withheld)") from None
    return kept, {f"{f}/{t}": v for (f, t), v in shortfall.items()}, dropped


class _Side(NamedTuple):
    texts: list[str]
    y: list[int]
    families: list[str | None]
    groups: list[tuple[str, str] | None]
    report: dict[str, Any]


def _side(
    negatives: Sequence[str],
    renders: Sequence[tuple[list[Sample], dict[str, int], dict[str, int]]],
    requested_per_template: int,
) -> _Side:
    samples = [s for kept, _, _ in renders for s in kept]
    dropped = Counter()
    shortfall: dict[str, int] = {}
    for _, short, drop in renders:
        dropped.update(drop)
        shortfall.update(short)
    families = Counter(s.family for s in samples)
    return _Side(
        list(negatives) + [s.text for s in samples],
        [0] * len(negatives) + [1] * len(samples),
        [None] * len(negatives) + [s.family for s in samples],
        [None] * len(negatives) + [(s.family, s.template_id) for s in samples],
        {
            "negatives": len(negatives),
            "positives": len(samples),
            "families": {f: families[f] for f in ALL_FAMILIES},
            "dropped": dict(dropped),
            "shortfall": shortfall,
            "requested_per_template": requested_per_template,
        },
    )


def iter_fold_data(
    real_queries: Sequence[str],
    products: Sequence[WandsProduct],
    part_numbers: Sequence[str],
    sanity: Sequence[str],
    arms: Sequence[str] = ARMS,
    n: int = DEFAULT_N,
    registry: Registry = REGISTRY,
    n_splits: int = N_SPLITS,
    repeats: int = REPEATS,
    seed: int = SEED,
    test_per_template: int = TEST_PER_TEMPLATE,
    eval_per_template: int = EVAL_PER_TEMPLATE,
    vocabulary_cap: int = VOCABULARY_CAP,
    leave_out: str | None = None,
    query_classes: Sequence[str] | None = None,
) -> Iterator[FoldData]:
    """Yield a FoldData per (repeat, fold, arm) (8.2, 8.3).

    With `leave_out` a family of A-D (LOFO, 8.6): its templates leave the training side,
    the others render at `n_negatives // n_remaining_templates` each (1:1; `n_negatives`
    is `n` in the primary arm), and the test side keeps only the test-side queries and that
    family's rows. The fold check still sees the full test side, so a fold with none of the
    family's templates on the test side doesn't raise, and asserts the other three families.

    With `query_classes` (the `query_class` of each real query; "" for none) this is the
    catalogue-gap pass (8.6 item 5): per fold, the products carrying a class of a test-side
    query leave the span pool, spans are built from what is left (so once per fold, still with
    seed `seed + repeat`; PoolExhausted if too few remain), the training-side queries of those
    classes leave the primary arm's threshold set and the comparison arm's training negatives,
    and the primary arm's report gains `products_removed`. An empty class removes nothing.

    Spans are built once per repeat with seed `seed + repeat`. The test side (test-side
    queries, test-template positives, family E) is built once per fold and shared by
    the arms. Positives equal to a sanity query raise FoldError; positives equal to a
    training negative, a real query or a part-number string are dropped and counted.
    Raises FoldError from check_fold on a degenerate fold.
    """
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise ValueError(f"unknown arms {unknown}")
    if leave_out is not None and leave_out not in TRAINING_FAMILIES:
        raise ValueError(f"leave_out must be one of {list(TRAINING_FAMILIES)}, got {leave_out!r}")
    kept_families = [f for f in TRAINING_FAMILIES if f != leave_out]
    if query_classes is not None and len(query_classes) != len(real_queries):
        raise ValueError(
            f"{len(query_classes)} query classes for {len(real_queries)} real queries")
    class_of = dict(zip(real_queries, query_classes or ()))
    folds = assign_folds(real_queries, registry, n_splits, repeats, seed)
    eval_templates = [(EVAL_FAMILY, t.template_id) for t in registry.templates(EVAL_FAMILY)]
    for repeat in range(repeats):
        render_seed = seed + repeat
        if query_classes is None:
            spans, removals = build_spans(
                products, n, render_seed, real_queries, part_numbers, sanity)
            span_texts = [s.text for s in spans]
            span_lengths = Counter(s.k for s in spans)
        for fold in (f for f in folds if f.repeat == repeat):
            train_queries = fold.train_queries
            if query_classes is not None:
                gone = {class_of[q] for q in fold.test_queries} - {""}
                kept_products = [p for p in products if not gone & set(p.classes)]
                spans, removals = build_spans(
                    kept_products, n, render_seed, real_queries, part_numbers, sanity)
                span_texts = [s.text for s in spans]
                span_lengths = Counter(s.k for s in spans)
                train_queries = [q for q in train_queries if class_of[q] not in gone]
            test_vocabulary = token_occurrences(fold.test_queries)
            test_drop = {"training_negatives": span_texts, "real_queries": real_queries,
                         "part_numbers": part_numbers}
            test = _side(fold.test_queries, [
                _render_screened(registry, test_vocabulary, render_seed, templates, per,
                                 sanity, test_drop)
                for templates, per in ((fold.test_templates, test_per_template),
                                       (eval_templates, eval_per_template))
            ], test_per_template)
            for arm in arms:
                train_templates = [t for t in fold.train_templates if t[0] != leave_out]
                if arm == PRIMARY:
                    negatives = span_texts
                    vocabulary = token_occurrences(span_texts, cap=vocabulary_cap)
                    per_template = n // len(train_templates)
                else:
                    negatives = train_queries
                    vocabulary = token_occurrences(train_queries)
                    per_template = (len(negatives) // len(train_templates)
                                    if leave_out else QUOTA)
                drop = {"training_negatives": negatives, "real_queries": real_queries,
                        "part_numbers": part_numbers}
                train = _side(negatives, [_render_screened(
                    registry, vocabulary, render_seed, train_templates, per_template,
                    sanity, drop)], per_template)
                check_fold(train.y, train.families, test.y, kept_families)
                rows = [i for i, f in enumerate(test.families)
                        if leave_out is None or f is None or f == leave_out]
                yield FoldData(
                    repeat, fold.fold, arm, train.texts, train.y, train.families, train.groups,
                    list(train_queries) if arm == PRIMARY else [],
                    [test.texts[i] for i in rows], [test.y[i] for i in rows],
                    [test.families[i] for i in rows], [test.groups[i] for i in rows],
                    {"repeat": repeat, "fold": fold.fold, "arm": arm,
                     **({"leave_out": leave_out} if leave_out else {}),
                     "span_removals": removals, "train": train.report, "test": test.report,
                     **({"span_lengths": dict(span_lengths)} if arm == PRIMARY else {}),
                     **({"products_removed": len(products) - len(kept_products)}
                        if arm == PRIMARY and query_classes is not None else {})},
                )


class Threshold(NamedTuple):
    """A chosen threshold over the real queries' scores (8.4, 11.10, 11.11).

    `fpr` is the share of those scores flagged at `threshold`. `target_unmet` is set
    when no score reaches the target. `warning` is None, or says that the target is
    below the resolution 1/n of the scores.
    """
    threshold: float
    fpr: float
    target_unmet: bool
    warning: str | None


def is_illegitimate(score: float, threshold: float) -> bool:
    """The one decision rule (11.19): a score at the threshold is illegitimate."""
    return score >= threshold


def select_threshold(scores: Sequence[float], target_fpr: float) -> Threshold:
    """Smallest score in `scores` whose FPR is at or below `target_fpr` (8.4).

    `scores` are the real queries' scores only (8.2). If no score reaches the target,
    returns the score with the smallest FPR and `target_unmet=True` (11.10). Raises
    ValueError on no scores or a non-finite score.
    """
    if len(scores) == 0:
        raise ValueError("no scores to select a threshold from")
    if not all(math.isfinite(s) for s in scores):
        raise ValueError("scores must be finite")
    n = len(scores)
    ordered = sorted(scores, reverse=True)
    best: tuple[float, float] | None = None  # (threshold, fpr), lowest qualifying score
    for flagged, score in enumerate(ordered, start=1):
        if flagged < n and ordered[flagged] == score:
            continue  # flagged counts every score tied at this one
        fpr = flagged / n
        if fpr > target_fpr:
            break
        best = (score, fpr)
    unmet = best is None
    if best is None:
        top = ordered[0]
        best = (top, sum(1 for s in ordered if is_illegitimate(s, top)) / n)
    warning = None
    if target_fpr < 1 / n:
        warning = f"target_fpr {target_fpr} is below the resolution 1/{n} of the scores"
    return Threshold(best[0], best[1], unmet, warning)


class NonFiniteScoreError(ValueError):
    """A score array held NaN or an infinity (11.30)."""


def check_finite(
    scores: Sequence[float],
    repeat: int | None,
    fold: int | None,
    arm: str,
    model: str,
    score_set: str,
) -> np.ndarray:
    """Return `scores` as an array; raise NonFiniteScoreError on any NaN or inf (11.30).

    The message names the repeat, fold, arm, model and score set. A final model (8.7) has
    no outer fold: `repeat=None` makes the message say "final model" instead.
    """
    array = np.asarray(scores, dtype=float)
    if not np.isfinite(array).all():
        where = "final model" if repeat is None else f"repeat {repeat}, fold {fold}"
        raise NonFiniteScoreError(
            f"non-finite score in {score_set} ({where}, arm {arm}, model {model})")
    return array


class FoldResult(NamedTuple):
    """One fold's scores at one arm, with the threshold they were judged against.

    `test_scores` line up with the fold's `test_texts`, `test_y`, `test_families` and
    `test_groups`, all copied here unchanged. `fit_warnings` are the warnings raised by
    `fit`, as "Category: message" (11.12).
    """
    repeat: int
    fold: int
    arm: str
    model: str
    threshold: Threshold
    test_scores: np.ndarray
    test_texts: list[str]
    test_y: list[int]
    test_families: list[str | None]
    test_groups: list[tuple[str, str] | None]
    fit_warnings: list[str]


def run_primary_fold(
    data: FoldData,
    make_scorer: Callable[[], Scorer],
    target_fpr: float,
    model: str,
    leave_out: str | None = None,
    gap: bool = False,
) -> FoldResult:
    """Cross-fit one primary-arm fold (8.2): fit once on `data.train_texts`, set `t` from
    the scores of `data.threshold_texts` (training-side real queries, not in the fit),
    and score the test side, which includes family E. Every score array goes through
    `check_finite`. Warnings raised by `fit` are returned, not raised (11.12).
    `leave_out` marks a LOFO fold (8.6): it prefixes every score-set name ("LOFO A test
    queries"). `gap` marks a catalogue-gap fold (8.6 item 5): the prefix is "catalogue gap ".
    Raises ValueError if `data` is not the primary arm.
    """
    if data.arm != PRIMARY:
        raise ValueError(f"run_primary_fold needs the {PRIMARY!r} arm, got {data.arm!r}")
    context = dict(repeat=data.repeat, fold=data.fold, arm=data.arm, model=model)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scorer = make_scorer().fit(data.train_texts, data.train_y)
    fit_warnings = [f"{w.category.__name__}: {w.message}" for w in caught]
    threshold = select_threshold(
        check_finite(scorer.score(data.threshold_texts), **context,
                     score_set=_score_set("threshold queries", leave_out, gap)),
        target_fpr)
    return _test_result(data, scorer, threshold, model, fit_warnings, leave_out, gap)


def _score_set(name: str, leave_out: str | None, gap: bool = False) -> str:
    """The score-set name for `check_finite`, prefixed with "LOFO <family> " in a LOFO run
    and with "catalogue gap " in the catalogue-gap pass."""
    if gap:
        return f"catalogue gap {name}"
    return name if leave_out is None else f"LOFO {leave_out} {name}"


def _test_result(
    data: FoldData,
    scorer: Scorer,
    threshold: Threshold,
    model: str,
    fit_warnings: list[str],
    leave_out: str | None = None,
    gap: bool = False,
) -> FoldResult:
    """Score the test side with a fitted `scorer`, check every score set is finite, and
    return the fold's FoldResult. Shared by both arms' runners."""
    context = dict(repeat=data.repeat, fold=data.fold, arm=data.arm, model=model)
    scores = np.asarray(scorer.score(data.test_texts), dtype=float)
    for score_set, family_matches in (
            ("test queries", lambda f: f is None),
            ("E", lambda f: f == EVAL_FAMILY),
            ("test positives", lambda f: f is not None and f != EVAL_FAMILY)):
        rows = [family_matches(f) for f in data.test_families]
        check_finite(scores[rows], **context, score_set=_score_set(score_set, leave_out, gap))
    return FoldResult(
        data.repeat, data.fold, data.arm, model, threshold, scores,
        list(data.test_texts), list(data.test_y), list(data.test_families),
        list(data.test_groups), fit_warnings)


def _inner_out_of_fold(
    texts: Sequence[str],
    y: Sequence[int],
    families: Sequence[str | None],
    groups: Sequence[tuple[str, str] | None],
    make_scorer: Callable[[], Scorer],
    seed: int,
    leave_out: str | None = None,
) -> np.ndarray:
    """Out-of-fold scores of every row from the inner `StratifiedGroupKFold` (8.2): a real
    query (None in `groups`) is its own group, a template's rows share one. Each inner
    training fold passes `check_fold` for the families left after `leave_out`."""
    group_ids: dict[Any, int] = {}
    codes = [group_ids.setdefault(g if g is not None else ("row", i), len(group_ids))
             for i, g in enumerate(groups)]
    expected = [f for f in TRAINING_FAMILIES if f != leave_out]
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    out_of_fold = np.full(len(texts), np.nan)
    for train, held_out in splitter.split(texts, y, codes):
        check_fold([y[i] for i in train], [families[i] for i in train],
                   [y[i] for i in held_out], expected)
        scorer = make_scorer().fit([texts[i] for i in train], [y[i] for i in train])
        out_of_fold[held_out] = scorer.score([texts[i] for i in held_out])
    return out_of_fold


def run_comparison_fold(
    data: FoldData,
    make_scorer: Callable[[], Scorer],
    target_fpr: float,
    model: str,
    leave_out: str | None = None,
    gap: bool = False,
) -> FoldResult:
    """Nested-split one comparison-arm fold (8.2): an inner `StratifiedGroupKFold` over the
    training rows gives out-of-fold scores for the training-side real queries, `t` is the 8.4
    rule over them, then a fresh scorer is refitted on all training rows and scores the test
    side as `run_primary_fold` does. `make_scorer` is called once per fit. Every inner
    training fold passes `check_fold`; every score array goes through `check_finite`.
    Warnings from the inner fits and the refit are returned, not raised (11.12).
    `leave_out` marks a LOFO fold (8.6): the inner check asks for the other three
    families only, and score-set names are prefixed as in `run_primary_fold`, also for `gap`.
    Raises ValueError if `data` is not the comparison arm.
    """
    if data.arm != COMPARISON:
        raise ValueError(f"run_comparison_fold needs the {COMPARISON!r} arm, got {data.arm!r}")
    context = dict(repeat=data.repeat, fold=data.fold, arm=data.arm, model=model)
    texts, y, families = data.train_texts, data.train_y, data.train_families
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out_of_fold = _inner_out_of_fold(
            texts, y, families, data.train_groups, make_scorer, SEED + data.repeat, leave_out)
        scorer = make_scorer().fit(texts, y)
    fit_warnings = [f"{w.category.__name__}: {w.message}" for w in caught]
    out_of_fold = check_finite(
        out_of_fold, **context, score_set=_score_set("out-of-fold scores", leave_out, gap))
    real = [i for i, f in enumerate(families) if f is None]
    threshold = select_threshold(out_of_fold[real], target_fpr)
    return _test_result(data, scorer, threshold, model, fit_warnings, leave_out, gap)


def wilson_interval(k: float, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval `(low, high)` for `k` successes in `n` trials, clamped to
    [0, 1]. `k` need not be an integer (8.5); the default `z` is the two-sided 95% point.
    Raises ValueError if `n < 1` or `k` is outside [0, n]. Standard library only.
    """
    if n < 1 or not 0 <= k <= n:
        raise ValueError(f"need n >= 1 and 0 <= k <= n, got k={k}, n={n}")
    p = k / n
    scale = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / scale
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / scale
    # k = 0 and k = n pin one end exactly; rounding would leave a 1e-18 residue there
    return (0.0 if k == 0 else max(0.0, centre - half),
            1.0 if k == n else min(1.0, centre + half))


def compute_metrics(results: Sequence[FoldResult]) -> dict[str, Any]:
    """Every number of 8.5 for one (arm, model), from its folds' results, as plain
    Python types (JSON-ready).

    Every flag comes from `is_illegitimate` at the result's own threshold. Headline
    numbers (recall, FPR, confusion matrix, precision, macro F1, ROC-AUC) use the test
    rows outside family E; E has its own evaluation-only row in `families`.
    Returns `{"arm", "model", "repeats": [block per repeat], "pooled": block,
    "folds": [one entry per fold]}`. Raises ValueError on no results, mixed arms or
    models, repeats with different numbers of test negatives, or a single-class
    ROC-AUC input.
    """
    if not results:
        raise ValueError("no results to compute metrics from")
    if len({(r.arm, r.model) for r in results}) != 1:
        raise ValueError("results must share one arm and one model")
    rows = [row for result in sorted(results, key=lambda r: (r.repeat, r.fold))
            for row in _rows(result)]
    repeats = sorted({row.repeat for row in rows})
    negatives = {len([row for row in rows if row.repeat == repeat and row.y == 0])
                 for repeat in repeats}
    if len(negatives) != 1:
        raise ValueError(f"repeats have different numbers of test negatives: {sorted(negatives)}")
    n = negatives.pop()
    folds = [_fold_entry(result, [row for row in rows
                                  if (row.repeat, row.fold) == (result.repeat, result.fold)])
             for result in sorted(results, key=lambda r: (r.repeat, r.fold))]
    return {
        "arm": results[0].arm,
        "model": results[0].model,
        "repeats": [
            {"repeat": repeat,
             **_block([row for row in rows if row.repeat == repeat],
                      [f for f in folds if f["repeat"] == repeat], 1, n)}
            for repeat in repeats],
        "pooled": _block(rows, folds, len(repeats), n),
        "folds": folds,
    }


class _Row(NamedTuple):
    """One test-side row with its flag, decided by `is_illegitimate` at the fold's threshold."""
    repeat: int
    fold: int
    text: str
    score: float
    y: int
    family: str | None
    group: tuple[str, str] | None
    flagged: bool


def _rows(result: FoldResult) -> list[_Row]:
    t = float(result.threshold.threshold)  # runners give numpy.float64; keep flags Python bools
    return [
        _Row(result.repeat, result.fold, text, float(score), y, family, group,
             is_illegitimate(float(score), t))
        for text, score, y, family, group in zip(
            result.test_texts, result.test_scores, result.test_y,
            result.test_families, result.test_groups)]


def _recall(positives: list[_Row]) -> dict[str, Any]:
    caught = sum(row.flagged for row in positives)
    return {"count": len(positives), "caught": caught,
            "recall": caught / len(positives) if positives else None}


def _families(rows: list[_Row]) -> dict[str, dict[str, Any]]:
    return {family: {**_recall([row for row in rows if row.family == family]),
                     "evaluation_only": family == EVAL_FAMILY}
            for family in ALL_FAMILIES}


def _roc_auc(headline: list[_Row]) -> float:
    y = [row.y for row in headline]
    if len(set(y)) < 2:
        raise ValueError("ROC-AUC needs both classes among the headline test rows")
    return float(roc_auc_score(y, [row.score for row in headline]))


def _fold_entry(result: FoldResult, rows: list[_Row]) -> dict[str, Any]:
    headline = [row for row in rows if row.family != EVAL_FAMILY]
    negatives = [row for row in headline if row.y == 0]
    false_positives = sum(row.flagged for row in negatives)
    return {
        "repeat": result.repeat,
        "fold": result.fold,
        "threshold": float(result.threshold.threshold),
        "recall": _recall([row for row in headline if row.y == 1]),
        "fpr": {"false_positives": false_positives, "n_negatives": len(negatives),
                "point": false_positives / len(negatives) if negatives else None},
        "roc_auc": _roc_auc(headline),
        "families": _families(rows),
    }


def _block(
    rows: list[_Row],
    folds: list[dict[str, Any]],
    repeats: int,
    n: int,
) -> dict[str, Any]:
    """The 8.5 numbers over `rows` spanning `repeats` repeats of `n` test negatives each.
    Headline numbers and `templates` leave family E out; `families` keeps it as its own row."""
    headline = [row for row in rows if row.family != EVAL_FAMILY]
    negatives = [row for row in headline if row.y == 0]
    positives = [row for row in headline if row.y == 1]
    false_positives = sum(row.flagged for row in negatives)
    ci_low, ci_high = wilson_interval(false_positives / repeats, n)
    y = [row.y for row in headline]
    flagged = [int(row.flagged) for row in headline]
    fold_aucs = [f["roc_auc"] for f in folds]
    word_flagged: dict[str, bool] = {}
    flagged_per_repeat: dict[int, set[str]] = {}
    for row in negatives:
        word = normalise(row.text)
        if len(word.split()) == 1:
            word_flagged[word] = word_flagged.get(word, False) or row.flagged
            if row.flagged:
                flagged_per_repeat.setdefault(row.repeat, set()).add(word)
    group_rows: dict[tuple[str, str], list[_Row]] = {}
    for row in headline:  # E has its own row in `families`; 8.5 asks for 24 template rows
        if row.group is not None:
            group_rows.setdefault(row.group, []).append(row)
    return {
        "recall": _recall(positives),
        "fpr": {"false_positives": false_positives, "n_negatives": n, "repeats": repeats,
                "point": false_positives / (n * repeats), "ci_low": ci_low, "ci_high": ci_high},
        "confusion_matrix": confusion_matrix(y, flagged, labels=[0, 1]).tolist(),
        "precision": float(precision_score(y, flagged, zero_division=0.0)),
        "macro_f1": float(f1_score(y, flagged, average="macro", zero_division=0.0)),
        "test_positive_to_negative_ratio": {
            "positives": len(positives), "negatives": len(negatives),
            "ratio": len(positives) / len(negatives) if negatives else None},
        "roc_auc": {
            "pooled": _roc_auc(headline),
            "fold_mean": float(np.mean(fold_aucs)),
            "fold_sd": float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else None},
        "families": _families(rows),
        "templates": [
            {"family": family, "template_id": template_id, **_recall(members)}
            for (family, template_id), members in sorted(group_rows.items())],
        "one_word": {
            "total": len(word_flagged),
            "flagged": sum(word_flagged.values()),
            "flagged_mean_per_repeat": sum(map(len, flagged_per_repeat.values())) / repeats,
            "words": sorted(w for w, hit in word_flagged.items() if hit)},
        "false_positives": [{"query": row.text, "repeat": row.repeat, "fold": row.fold}
                            for row in negatives if row.flagged],
    }


def length_mix(counts: Mapping[int, int]) -> dict[str, Any]:
    """Span length mix of 3.2 as JSON-ready data: `{"counts", "shares", "mean"}`, keyed by
    the word count as a string. Raises ValueError on no spans."""
    total = sum(counts.values())
    if total < 1:
        raise ValueError("no spans to take a length mix of")
    return {
        "counts": {str(k): counts[k] for k in sorted(counts)},
        "shares": {str(k): counts[k] / total for k in sorted(counts)},
        "mean": sum(k * c for k, c in counts.items()) / total,
    }


def build_run_report(
    real_queries: Sequence[str],
    products: Sequence[WandsProduct],
    part_numbers: Sequence[str],
    sanity: Sequence[str],
    models: Mapping[str, Callable[[], Scorer]],
    target_fpr: float,
    arms: Sequence[str] = ARMS,
    n: int = DEFAULT_N,
    registry: Registry = REGISTRY,
    n_splits: int = N_SPLITS,
    repeats: int = REPEATS,
    test_per_template: int = TEST_PER_TEMPLATE,
    eval_per_template: int = EVAL_PER_TEMPLATE,
    vocabulary_cap: int = VOCABULARY_CAP,
    lofo: bool = False,
    catalogue_gap: bool = False,
    query_classes: Sequence[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run every arm and model over the outer folds; return `(parameters, body)` for
    `write_report` (8.1, 8.2, 11.12).

    `body` has one key per arm. Each holds `folds` (the `FoldData.report` of every
    (repeat, fold), once per arm), `models` (per model: `compute_metrics` plus, on each
    fold entry, `target_unmet`, `threshold_warning` and `fit_warnings`) and, in the
    primary arm only, `span_lengths`: the realised mix per repeat next to the target.
    With `lofo` (off by default; 8.6 item 1) each arm also has `lofo`: per model, the
    `lofo_section` rows, run on the same folds with each family of A-D left out in turn.
    With `catalogue_gap` (off by default; 8.6 item 5; needs `query_classes`, the `query_class`
    of each real query) each arm also has `catalogue_gap`: per model, the
    `catalogue_gap_section` of a pass on the same folds with the test side's classes removed;
    the primary arm adds `folds` (`products_removed` per fold) and `unclassified_products`.
    A failed gap never stops the run.
    No removed span, positive or sanity text is in it.
    """
    if catalogue_gap and query_classes is None:
        raise ValueError("catalogue_gap needs query_classes")
    parameters = {
        "arms": list(arms), "models": list(models), "target_fpr": target_fpr,
        "negatives_size": n, "n_splits": n_splits, "repeats": repeats,
        "test_per_template": test_per_template, "eval_per_template": eval_per_template,
        "vocabulary_cap": vocabulary_cap, "lofo": lofo, "catalogue_gap": catalogue_gap,
    }
    runners = {PRIMARY: run_primary_fold, COMPARISON: run_comparison_fold}
    fold_reports: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms}
    results: dict[str, dict[str, list[FoldResult]]] = {
        arm: {model: [] for model in models} for arm in arms}
    mixes: dict[int, dict[str, Any]] = {}
    fold_data = partial(
        iter_fold_data, real_queries=real_queries, products=products, part_numbers=part_numbers,
        sanity=sanity, arms=arms, n=n, registry=registry, n_splits=n_splits, repeats=repeats,
        seed=SEED, test_per_template=test_per_template, eval_per_template=eval_per_template,
        vocabulary_cap=vocabulary_cap)
    for data in fold_data():
        report = dict(data.report)
        if "span_lengths" in report:  # the same for every fold of a repeat: kept once
            mixes[data.repeat] = length_mix(report.pop("span_lengths"))
        fold_reports[data.arm].append(report)
        for model, make_scorer in models.items():
            results[data.arm][model].append(
                runners[data.arm](data, make_scorer, target_fpr, model))
    lofo_results: dict[str, dict[str, dict[str, list[FoldResult]]]] = {
        arm: {family: {model: [] for model in models} for family in TRAINING_FAMILIES}
        for arm in arms}
    for family in TRAINING_FAMILIES if lofo else ():
        for data in fold_data(leave_out=family):
            for model, make_scorer in models.items():
                lofo_results[data.arm][family][model].append(
                    runners[data.arm](data, make_scorer, target_fpr, model, leave_out=family))
    gap_results: dict[str, dict[str, list[FoldResult]]] = {
        arm: {model: [] for model in models} for arm in arms}
    products_removed: list[dict[str, int]] = []
    for data in fold_data(query_classes=query_classes) if catalogue_gap else ():
        if "products_removed" in data.report:
            products_removed.append({"repeat": data.repeat, "fold": data.fold,
                                     "products_removed": data.report["products_removed"]})
        for model, make_scorer in models.items():
            gap_results[data.arm][model].append(
                runners[data.arm](data, make_scorer, target_fpr, model, gap=True))
    body: dict[str, Any] = {}
    for arm in arms:
        body[arm] = {"folds": fold_reports[arm],
                     "models": {model: _model_entry(rs) for model, rs in results[arm].items()}}
    if lofo:
        for arm in arms:
            body[arm]["lofo"] = {
                model: lofo_section(
                    {family: lofo_results[arm][family][model] for family in TRAINING_FAMILIES},
                    results[arm][model])
                for model in models}
    if catalogue_gap:
        for arm in arms:
            body[arm]["catalogue_gap"] = {"models": {
                model: catalogue_gap_section(rs, target_fpr)
                for model, rs in gap_results[arm].items()}}
        if PRIMARY in arms:
            body[PRIMARY]["catalogue_gap"].update(
                folds=products_removed,
                unclassified_products=sum(1 for p in products if not p.classes))
    if PRIMARY in arms:
        body[PRIMARY]["span_lengths"] = {
            "target": length_mix(LENGTH_COUNTS),
            "repeats": [{"repeat": r, **mixes[r]} for r in sorted(mixes)]}
    return parameters, body


def catalogue_gap_section(results: Sequence[FoldResult], target_fpr: float) -> dict[str, Any]:
    """The catalogue-gap numbers of one (arm, model) from its gap folds (8.6 item 5): the
    pooled `fpr` (with the Wilson interval of `compute_metrics`), `recall`, the
    `flagged_queries`, `gap_failed` (the interval's lower bound is above `target_fpr`: the
    lenient rule of 12.1 item 14) and `point_estimate_above_target` (a pass whose point
    estimate is above the target). Nothing acts on it."""
    pooled = compute_metrics(results)["pooled"]
    fpr = pooled["fpr"]
    failed = bool(fpr["ci_low"] > target_fpr)
    return {
        "target_fpr": target_fpr,
        "fpr": fpr,
        "recall": pooled["recall"],
        "flagged_queries": pooled["false_positives"],
        "gap_failed": failed,
        "point_estimate_above_target": bool(not failed and fpr["point"] > target_fpr),
    }


def lofo_section(
    lofo: Mapping[str, Sequence[FoldResult]],
    main: Sequence[FoldResult],
) -> dict[str, Any]:
    """One row per family A-D (8.6 item 1) for one (arm, model), without ROC-AUC.

    `lofo` maps each family F to its LOFO fold results (test side: the queries and F's
    rows); `main` holds the main run's results of the same arm and model. Each row has
    `folds` (per fold: `threshold`, F's `recall`, achieved `fpr`, the main run's
    `main_recall` on F and `difference` = LOFO minus main, `None` when F has no test rows
    in that fold) and `pooled` (the same over all folds). Raises ValueError on a LOFO
    fold with no matching main fold.
    """
    main_by_fold = {(r.repeat, r.fold): r for r in main}
    section = {}
    for family in TRAINING_FAMILIES:
        folds = []
        pooled_f: list[_Row] = []
        pooled_main: list[_Row] = []
        queries: list[_Row] = []
        for result in sorted(lofo[family], key=lambda r: (r.repeat, r.fold)):
            if (result.repeat, result.fold) not in main_by_fold:
                raise ValueError(f"no main fold for LOFO fold {(result.repeat, result.fold)}")
            rows = _rows(result)
            f_rows = [row for row in rows if row.family == family]
            main_rows = [row for row in _rows(main_by_fold[(result.repeat, result.fold)])
                         if row.family == family]
            negatives = [row for row in rows if row.y == 0]
            folds.append({
                "repeat": result.repeat, "fold": result.fold,
                "threshold": float(result.threshold.threshold),
                **_lofo_numbers(f_rows, main_rows, negatives)})
            pooled_f += f_rows
            pooled_main += main_rows
            queries += negatives
        section[family] = {"folds": folds,
                           "pooled": _lofo_numbers(pooled_f, pooled_main, queries)}
    return section


def _lofo_numbers(
    f_rows: list[_Row],
    main_rows: list[_Row],
    negatives: list[_Row],
) -> dict[str, Any]:
    recall, main_recall = _recall(f_rows), _recall(main_rows)["recall"]
    false_positives = sum(row.flagged for row in negatives)
    return {
        "recall": recall,
        "fpr": {"false_positives": false_positives, "n_negatives": len(negatives),
                "point": false_positives / len(negatives) if negatives else None},
        "main_recall": main_recall,
        "difference": (None if recall["recall"] is None or main_recall is None
                       else recall["recall"] - main_recall),
    }


def _model_entry(results: list[FoldResult]) -> dict[str, Any]:
    """`compute_metrics` with each fold entry extended by what the threshold and the fit
    report (`target_unmet`, `threshold_warning`, `fit_warnings`; 8.4, 11.10, 11.12)."""
    entry = compute_metrics(results)
    by_fold = {(r.repeat, r.fold): r for r in results}
    for fold in entry["folds"]:
        result = by_fold[(fold["repeat"], fold["fold"])]
        fold["target_unmet"] = bool(result.threshold.target_unmet)
        fold["threshold_warning"] = result.threshold.warning
        fold["fit_warnings"] = list(result.fit_warnings)
    return entry


def run_report(
    results_dir: Path,
    now: datetime | None = None,
    **options: Any,
) -> Path:
    """`build_run_report(**options)`, written through `write_report` as command
    `evaluate`; returns the path. `now` is for tests."""
    parameters, body = build_run_report(**options)
    return write_report(results_dir, "evaluate", parameters, body, now)


class FinalModel(NamedTuple):
    """A model fitted for inspection (8.7): the scorer, its threshold, and the run's counts.

    `report` holds the fit's negatives, positives per family, rule-4 drops and shortfalls
    (counts only). `fit_warnings` are as in `FoldResult`.
    """
    arm: str
    model: str
    scorer: Scorer
    threshold: Threshold
    fit_warnings: list[str]
    report: dict[str, Any]


def fit_final_model(
    arm: str,
    make_scorer: Callable[[], Scorer],
    target_fpr: float,
    model: str,
    real_queries: Sequence[str],
    products: Sequence[WandsProduct],
    part_numbers: Sequence[str],
    sanity: Sequence[str],
    n: int = DEFAULT_N,
    registry: Registry = REGISTRY,
    vocabulary_cap: int = VOCABULARY_CAP,
) -> FinalModel:
    """Fit the final model of one arm (8.7); family E is not rendered.

    Primary: `n` spans with seed `SEED`, positives from all A-D templates at `n // 24`
    each from the spans' token occurrences, one fit, and `t` from the 8.4 rule over all
    the real queries (none was trained on). Comparison: the real queries as negatives,
    `QUOTA` positives per template from their token occurrences, `t` from the inner
    out-of-fold pass (as `run_comparison_fold`, seed `SEED`), then a fit on every row.
    Rule 4 as in `iter_fold_data`: a positive equal to a sanity query raises FoldError,
    one equal to a training negative, a real query or a part-number string is dropped and
    counted in `report["dropped"]`. Every score array goes through `check_finite` with no
    fold. `FoldError` and `ValueError` as there; an unknown `arm` raises ValueError.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    context = dict(repeat=None, fold=None, arm=arm, model=model)
    templates = [(f, t.template_id) for f in TRAINING_FAMILIES for t in registry.templates(f)]
    if arm == PRIMARY:
        spans, removals = build_spans(products, n, SEED, real_queries, part_numbers, sanity)
        negatives = [s.text for s in spans]
        vocabulary = token_occurrences(negatives, cap=vocabulary_cap)
        per_template = n // len(templates)
    else:
        removals = None
        negatives = list(real_queries)
        vocabulary = token_occurrences(negatives)
        per_template = QUOTA
    drop = {"training_negatives": negatives, "real_queries": real_queries,
            "part_numbers": part_numbers}
    train = _side(negatives, [_render_screened(
        registry, vocabulary, SEED, templates, per_template, sanity, drop)], per_template)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if arm == PRIMARY:
            scorer = make_scorer().fit(train.texts, train.y)
            real_scores = check_finite(
                scorer.score(real_queries), **context,
                score_set=_score_set("threshold queries", None))
        else:
            out_of_fold = _inner_out_of_fold(
                train.texts, train.y, train.families, train.groups, make_scorer, SEED)
            real_scores = check_finite(
                out_of_fold, **context, score_set=_score_set("out-of-fold scores", None)
            )[:len(negatives)]
            scorer = make_scorer().fit(train.texts, train.y)
    fit_warnings = [f"{w.category.__name__}: {w.message}" for w in caught]
    report = {**train.report, **({"span_removals": removals} if removals else {})}
    return FinalModel(
        arm, model, scorer, select_threshold(real_scores, target_fpr), fit_warnings, report)


def save_final_model(final: FinalModel, results_dir: Path, now: datetime | None = None) -> Path:
    """Pickle `final` (scorer and threshold included) to
    `<results_dir>/final-<arm>-<model>-<UTC timestamp>.pkl`, creating `results_dir`, and return
    the path. An existing file is never overwritten: FileExistsError. `now` is for tests.
    Not a deployment artefact (8.7); load only pickles you wrote. The scorer's class must be
    importable at module level (probed 2026-11-06, Python stdlib pickle: `FakeScorer` and its
    numpy state round-trip, a class defined inside a function does not pickle)."""
    stamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"final-{final.arm}-{final.model}-{stamp.strftime('%Y%m%dT%H%M%S%fZ')}.pkl"
    with path.open("xb") as f:  # "x": never overwrite
        pickle.dump(final, f)
    return path


def load_final_model(path: Path) -> FinalModel:
    """Load a pickle written by `save_final_model`."""
    with path.open("rb") as f:
        return pickle.load(f)
