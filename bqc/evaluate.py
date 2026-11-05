"""Outer folds, per-fold data, thresholds and the primary-arm runner (DESIGN.md 8.1, 8.2,
8.3, 8.4, 11.16, 11.30).

Fold units are the real queries (one group each) and the training templates
(one group each, nominal size QUOTA). Spans are never fold units. `select_threshold`
picks the threshold from the real queries' scores, `check_finite` stops the run on a
non-finite score, and `run_primary_fold` cross-fits one fold of the primary arm.
"""
import math
import warnings
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from typing import Any, NamedTuple

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from bqc import SEED
from bqc.models import Scorer
from bqc.negatives import DEFAULT_N, VOCABULARY_CAP, build_spans, token_occurrences
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
) -> None:
    """Raise FoldError if a class or a training family (A-D) is missing from the training
    side, or a class from the test side. Families are never asserted on the test side (11.16).
    """
    if set(train_y) != {0, 1}:
        raise FoldError(f"training fold classes are {sorted(set(train_y))}, need 0 and 1")
    missing = [f for f in TRAINING_FAMILIES if f not in set(train_families)]
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
) -> Iterator[FoldData]:
    """Yield a FoldData per (repeat, fold, arm) (8.2, 8.3).

    Spans are built once per repeat with seed `seed + repeat`. The test side (test-side
    queries, test-template positives, family E) is built once per fold and shared by
    the arms. Positives equal to a sanity query raise FoldError; positives equal to a
    training negative, a real query or a part-number string are dropped and counted.
    Raises FoldError from check_fold on a degenerate fold.
    """
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise ValueError(f"unknown arms {unknown}")
    folds = assign_folds(real_queries, registry, n_splits, repeats, seed)
    eval_templates = [(EVAL_FAMILY, t.template_id) for t in registry.templates(EVAL_FAMILY)]
    for repeat in range(repeats):
        render_seed = seed + repeat
        spans, removals = build_spans(
            products, n, render_seed, real_queries, part_numbers, sanity)
        span_texts = [s.text for s in spans]
        for fold in (f for f in folds if f.repeat == repeat):
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
                if arm == PRIMARY:
                    negatives = span_texts
                    vocabulary = token_occurrences(span_texts, cap=vocabulary_cap)
                    per_template = n // len(fold.train_templates)
                else:
                    negatives = fold.train_queries
                    vocabulary = token_occurrences(fold.train_queries)
                    per_template = QUOTA
                drop = {"training_negatives": negatives, "real_queries": real_queries,
                        "part_numbers": part_numbers}
                train = _side(negatives, [_render_screened(
                    registry, vocabulary, render_seed, fold.train_templates, per_template,
                    sanity, drop)], per_template)
                check_fold(train.y, train.families, test.y)
                yield FoldData(
                    repeat, fold.fold, arm, train.texts, train.y, train.families, train.groups,
                    list(fold.train_queries) if arm == PRIMARY else [],
                    test.texts, test.y, test.families, test.groups,
                    {"repeat": repeat, "fold": fold.fold, "arm": arm,
                     "span_removals": removals, "train": train.report, "test": test.report},
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
    repeat: int,
    fold: int,
    arm: str,
    model: str,
    score_set: str,
) -> np.ndarray:
    """Return `scores` as an array; raise NonFiniteScoreError on any NaN or inf (11.30).

    The message names the repeat, fold, arm, model and score set.
    """
    array = np.asarray(scores, dtype=float)
    if not np.isfinite(array).all():
        raise NonFiniteScoreError(
            f"non-finite score in {score_set} (repeat {repeat}, fold {fold}, arm {arm}, "
            f"model {model})")
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
) -> FoldResult:
    """Cross-fit one primary-arm fold (8.2): fit once on `data.train_texts`, set `t` from
    the scores of `data.threshold_texts` (training-side real queries, not in the fit),
    and score the test side, which includes family E. Every score array goes through
    `check_finite`. Warnings raised by `fit` are returned, not raised (11.12).
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
                     score_set="threshold queries"),
        target_fpr)
    return _test_result(data, scorer, threshold, model, fit_warnings)


def _test_result(
    data: FoldData,
    scorer: Scorer,
    threshold: Threshold,
    model: str,
    fit_warnings: list[str],
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
        check_finite(scores[rows], **context, score_set=score_set)
    return FoldResult(
        data.repeat, data.fold, data.arm, model, threshold, scores,
        list(data.test_texts), list(data.test_y), list(data.test_families),
        list(data.test_groups), fit_warnings)


def run_comparison_fold(
    data: FoldData,
    make_scorer: Callable[[], Scorer],
    target_fpr: float,
    model: str,
) -> FoldResult:
    """Nested-split one comparison-arm fold (8.2): an inner `StratifiedGroupKFold` over the
    training rows gives out-of-fold scores for the training-side real queries, `t` is the 8.4
    rule over them, then a fresh scorer is refitted on all training rows and scores the test
    side as `run_primary_fold` does. `make_scorer` is called once per fit. Every inner
    training fold passes `check_fold`; every score array goes through `check_finite`.
    Warnings from the inner fits and the refit are returned, not raised (11.12).
    Raises ValueError if `data` is not the comparison arm.
    """
    if data.arm != COMPARISON:
        raise ValueError(f"run_comparison_fold needs the {COMPARISON!r} arm, got {data.arm!r}")
    context = dict(repeat=data.repeat, fold=data.fold, arm=data.arm, model=model)
    # a real query is its own group (None in train_groups); a template's rows share one
    group_ids: dict[Any, int] = {}
    codes = [group_ids.setdefault(g if g is not None else ("row", i), len(group_ids))
             for i, g in enumerate(data.train_groups)]
    texts, y, families = data.train_texts, data.train_y, data.train_families
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=SEED + data.repeat)
    out_of_fold = np.full(len(texts), np.nan)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for train, held_out in splitter.split(texts, y, codes):
            check_fold([y[i] for i in train], [families[i] for i in train],
                       [y[i] for i in held_out])
            scorer = make_scorer().fit([texts[i] for i in train], [y[i] for i in train])
            out_of_fold[held_out] = scorer.score([texts[i] for i in held_out])
        scorer = make_scorer().fit(texts, y)
    fit_warnings = [f"{w.category.__name__}: {w.message}" for w in caught]
    real = [i for i, f in enumerate(families) if f is None]
    threshold = select_threshold(
        check_finite(out_of_fold[real], **context, score_set="out-of-fold queries"),
        target_fpr)
    return _test_result(data, scorer, threshold, model, fit_warnings)
