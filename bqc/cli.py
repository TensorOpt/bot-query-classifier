"""Command line entry point and command implementations."""

import argparse
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from bqc import SEED
from bqc.evaluate import (
    ARMS,
    COMPARISON,
    LR,
    MLP,
    PRIMARY,
    FoldError,
    NonFiniteScoreError,
    build_run_report,
    fit_final_model,
    part_number_section,
    sanity_section,
    save_final_model,
)
from bqc.models import LogisticRegressionScorer, MLPScorer, Scorer, TrivialScorer
from bqc.negatives import (
    DEFAULT_N,
    VOCABULARY_CAP,
    PoolExhausted,
    build_spans,
    token_occurrences,
)
from bqc.preprocess import normalise
from bqc.report import RESULTS_DIR, write_report
from bqc.sanity import SANITY_PATH, SanityError, SanityQuery, SanitySet, load_sanity
import bqc.synth  # noqa: F401  (registers families A-E)
from bqc.synth.audit import (
    NEGATIVES_THRESHOLD,
    SANITY_THRESHOLD,
    WANDS_KEY,
    AuditError,
    SanityCollision,
    length_stats,
    near_duplicates,
    screen_samples,
)
from bqc.synth.partnumbers import generate_part_numbers
from bqc.synth.registry import (
    ALL_FAMILIES,
    EVAL_FAMILY,
    QUOTA,
    REGISTRY,
    TRAINING_FAMILIES,
    Registry,
    Sample,
    has_non_benign_token,
    render,
)
from bqc.wands import (
    CHECKSUMS_PATH,
    PRODUCT_PATH,
    PRODUCT_URL,
    RAW_PATH,
    URL,
    WandsError,
    WandsProduct,
    WandsQuery,
    fetch,
    load_products,
    load_queries,
)

MIN_DISTINCT_SHARE = 0.5  # a template under this share of distinct strings is a finding (4)

TRAINING_TEMPLATE_COUNT = 24  # templates of A-D: positives per template are n // 24

MODEL_FACTORIES: dict[str, Callable[[], Scorer]] = {
    LR: LogisticRegressionScorer, MLP: MLPScorer, "trivial": TrivialScorer}
DEFAULT_TARGET_FPR = 0.01
# Section 10: above this per fit, the M6 invocation leaves the MLP out of --lofo and
# --catalogue-gap. `evaluate` only reports a fit over it; it applies no rule.
MLP_FIT_LIMIT_SECONDS = 300


class CommandError(Exception):
    """`generate` or `evaluate` can't finish; the message holds no sanity or removed text."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bqc")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    commands.add_parser("fetch", help="download WANDS query.csv and product.csv into data/raw/")
    generate = commands.add_parser(
        "generate", help="build the default training set and write a report to results/")
    generate.add_argument(
        "--negatives-size", type=_positive_int, default=DEFAULT_N, metavar="N",
        help=f"number of product-name spans (default {DEFAULT_N})")
    evaluate = commands.add_parser(
        "evaluate", help="run the models over the outer folds and write a run report to results/")
    evaluate.add_argument(
        "--negatives", type=_arm_list, default=[PRIMARY], metavar="ARMS",
        help=f"comma list of arms, from {', '.join(ARMS)} (default {PRIMARY})")
    evaluate.add_argument(
        "--negatives-size", type=_positive_int, default=DEFAULT_N, metavar="N",
        help=f"number of product-name spans (default {DEFAULT_N})")
    evaluate.add_argument(
        "--models", type=_model_list, default=list(MODEL_FACTORIES), metavar="MODELS",
        help=f"comma list of models, from {', '.join(MODEL_FACTORIES)} (default all)")
    evaluate.add_argument(
        "--target-fpr", type=_open_unit_float, default=DEFAULT_TARGET_FPR, metavar="P",
        help=f"target false-positive rate, in (0, 1) (default {DEFAULT_TARGET_FPR})")
    # A non-string const bypasses `type`, so a bare flag gives [] (probed 2026-11-06,
    # Python 3.13.18); `--lofo=` with an empty value reaches `type` and is an unknown model.
    sections = (("--lofo", "leave-one-family-out rows"), ("--catalogue-gap", "catalogue-gap check"))
    for flag, what in sections:
        evaluate.add_argument(
            flag, nargs="?", const=[], default=None, type=_model_list, metavar="MODELS",
            help=f"also run the {what} for a comma list of models (default all of --models)")
    return parser


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be positive, got {number}")
    return number


def _name_list(value: str, known: Sequence[str], what: str) -> list[str]:
    names = value.split(",")
    for name in names:
        if name not in known:
            raise argparse.ArgumentTypeError(
                f"unknown {what} {name!r}; choose from {', '.join(known)}")
    return list(dict.fromkeys(names))


def _arm_list(value: str) -> list[str]:
    return _name_list(value, ARMS, "arm")


def _model_list(value: str) -> list[str]:
    return _name_list(value, list(MODEL_FACTORIES), "model")


def _open_unit_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {value!r}") from None
    if not 0 < number < 1:  # false for nan, which select_threshold would accept silently
        raise argparse.ArgumentTypeError(f"must be in (0, 1), got {value}")
    return number


def sanity_overlaps(
    sanity: Sequence[SanityQuery], positives: Sequence[str], threshold: float
) -> list[dict[str, Any]]:
    """`{"id", "max_jaccard"}` of every sanity query whose character-3-gram Jaccard to some
    positive is strictly above `threshold`; no query text (4, 8.6.1)."""
    best: dict[int, float] = {}
    for pair in near_duplicates([q.query for q in sanity], positives, threshold):
        best[pair.positive_index] = max(best.get(pair.positive_index, 0.0), pair.jaccard)
    return [{"id": sanity[i].id, "max_jaccard": best[i]} for i in sorted(best)]


def one_off_share(tokens: Sequence[str], vocabulary: Sequence[str]) -> float:
    """Share of `tokens` whose count in the occurrence list `vocabulary` is exactly 1 (13.17)."""
    if not tokens:
        return 0.0
    counts = Counter(vocabulary)
    return sum(counts[t] == 1 for t in tokens) / len(tokens)


def _screen(
    samples: Sequence[Sample], others: dict[str, Sequence[str]], sanity: Sequence[str]
) -> tuple[list[Sample], dict[str, int]]:
    """Rule 4 for rendered positives: fail on a sanity match, drop the other matches.

    Returns the kept samples (original order) and the removals per list in `others`.
    """
    try:
        return screen_samples(samples, others, sanity)
    except SanityCollision as e:
        raise CommandError(
            f"a positive of template {e.family}/{e.template_id} equals a sanity query "
            "(text withheld); fix the template"
        ) from None


def _template_report(
    family: str, template, requested: int, rendered: Sequence[Sample],
    kept: Sequence[Sample], shortfall: int,
) -> dict[str, Any]:
    distinct = len({s.text for s in rendered})
    stats = length_stats(kept, [])
    return {
        "family": family,
        "template_id": template.template_id,
        "hard": template.hard,
        "requested": requested,
        "distinct": distinct,
        "distinct_share": distinct / requested,
        "shortfall": shortfall,
        "finding": distinct / requested < MIN_DISTINCT_SHARE,
        "length": stats.get(family, stats[WANDS_KEY])._asdict(),
        "dropped": len(rendered) - len(kept),
    }


def _load_inputs(
    query_path: Path, product_path: Path, checksums_path: Path, sanity_path: Path
) -> tuple[list[WandsQuery], list[WandsProduct], SanitySet]:
    """The WANDS queries and products and the sanity set, for `generate` and `evaluate`.

    Raises CommandError (a raw file is missing; the message points at `fetch`), WandsError or
    SanityError."""
    for path in (query_path, product_path):
        if not path.exists():
            raise CommandError(f"missing {path}; run `python -m bqc fetch` first")
    queries = load_queries(query_path, checksums_path)
    products = load_products(product_path, checksums_path)
    return queries, products, load_sanity(sanity_path, queries)


def _part_numbers(
    real_vocabulary: Sequence[str], real: Sequence[str], sanity_texts: Sequence[str]
) -> list[Sample]:
    """The 100 part-number strings (8.6 item 4); CommandError, text withheld, on a collision."""
    try:
        return generate_part_numbers(real_vocabulary, SEED, real, sanity_texts)
    except AuditError:
        raise CommandError(
            "a part-number string equals a real or sanity query (text withheld)") from None


def build_generate_report(
    n: int,
    query_path: Path,
    product_path: Path,
    checksums_path: Path,
    sanity_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build everything of `generate` (3.2, 4, 10) and return (parameters, report body).

    Raises CommandError, WandsError, SanityError or PoolExhausted; none of the messages
    holds sanity text.
    """
    queries, products, sanity = _load_inputs(query_path, product_path, checksums_path, sanity_path)
    real = [q.query for q in queries]
    sanity_texts = [q.query for q in sanity.queries]

    real_vocabulary = token_occurrences(real)
    part_numbers = _part_numbers(real_vocabulary, real, sanity_texts)
    p_texts = [s.text for s in part_numbers]
    spans, removed = build_spans(products, n, SEED, real, p_texts, sanity_texts)
    span_texts = [s.text for s in spans]
    vocabulary = token_occurrences(span_texts, cap=VOCABULARY_CAP)

    per_template = n // TRAINING_TEMPLATE_COUNT
    others = {"spans": span_texts, "real_queries": real, "part_numbers": p_texts}
    groups = {
        "training": (TRAINING_FAMILIES, vocabulary, per_template),
        EVAL_FAMILY: ((EVAL_FAMILY,), real_vocabulary, QUOTA),
    }
    kept: list[Sample] = []
    template_reports = []
    dropped = {}
    benign_only = {}
    hard_slots: list[str] = []
    for name, (families, group_vocabulary, requested) in groups.items():
        templates = [(f, t) for f in families for t in REGISTRY.templates(f)]
        wanted = [(f, t.template_id) for f, t in templates]
        rendered, shortfall = render(group_vocabulary, SEED, wanted, requested)
        group_kept, dropped[name] = _screen(rendered, others, sanity_texts)
        word_set = set(group_vocabulary)
        benign_only[name] = sum(not has_non_benign_token(s.text, word_set) for s in group_kept)
        for family, template in templates:
            key = (family, template.template_id)
            mine = [s for s in rendered if (s.family, s.template_id) == key]
            mine_kept = [s for s in group_kept if (s.family, s.template_id) == key]
            template_reports.append(
                _template_report(family, template, requested, mine, mine_kept, shortfall[key]))
            if name == "training" and template.hard:
                hard_slots += [t for s in mine_kept for t in s.text.split() if t in word_set]
        kept += group_kept
    texts = [s.text for s in kept]

    stats = length_stats(kept, real)
    distinct = {f: len({s.text for s in kept if s.family == f}) for f in ALL_FAMILIES}
    distinct[WANDS_KEY] = len({normalise(q) for q in real})
    with_span = {p.positive_index for p in near_duplicates(real, span_texts, NEGATIVES_THRESHOLD)}
    pairs = near_duplicates(texts, span_texts, NEGATIVES_THRESHOLD)

    parameters = {
        "negatives_size": n,
        "per_template": per_template,
        "vocabulary_cap": VOCABULARY_CAP,
    }
    body = {
        "vocabulary": {
            "cap": VOCABULARY_CAP,
            "span_occurrences": len(vocabulary),
            "real_query_occurrences": len(real_vocabulary),
        },
        "spans": [s._asdict() for s in spans],
        "positives": [s._asdict() for s in kept],
        "part_numbers": [s._asdict() for s in part_numbers],
        "positives_dropped": dropped,
        "benign_only_outputs": benign_only,
        "templates": template_reports,
        "length_stats": {f: v._asdict() for f, v in stats.items()},
        "distinct_counts": distinct,
        "span_audit": {
            "removed": removed,
            "real_queries_total": len(real),
            "real_queries_with_span_above_0.6": {
                "count": len(with_span), "share": len(with_span) / len(real)},
        },
        "near_duplicates": {
            "threshold": NEGATIVES_THRESHOLD,
            "count": len(pairs),
            "positives_involved": len({p.positive_index for p in pairs}),
            "pairs": [list(p) for p in pairs],
        },
        "sanity": {
            "count": len(sanity.queries),
            "above_0.5": sanity_overlaps(sanity.queries, texts, SANITY_THRESHOLD),
        },
        "one_off_shares": {
            "hard_positive_word_slots": one_off_share(hard_slots, vocabulary),
            "span_token_occurrences": one_off_share(vocabulary, vocabulary),
        },
    }
    return parameters, body


def _generate(
    n: int,
    query_path: Path = RAW_PATH,
    product_path: Path = PRODUCT_PATH,
    checksums_path: Path = CHECKSUMS_PATH,
    sanity_path: Path = SANITY_PATH,
    results_dir: Path = RESULTS_DIR,
) -> int:
    """Run `generate`, write the report and print counts; 1 with a message on failure."""
    started = time.monotonic()
    try:
        parameters, body = build_generate_report(
            n, query_path, product_path, checksums_path, sanity_path)
        path = write_report(results_dir, "generate", parameters, body)
    except SanityError:
        # The loader's message can quote a query; keep sanity text out of the output.
        print(f"bqc generate: {sanity_path} is invalid (details withheld)", file=sys.stderr)
        return 1
    except (CommandError, WandsError, PoolExhausted, OSError) as e:
        print(f"bqc generate: {e}", file=sys.stderr)
        return 1
    print(f"wrote {path}")
    print(f"spans: {len(body['spans'])}, positives: {len(body['positives'])}, "
          f"part numbers: {len(body['part_numbers'])}")
    print(f"span removals: {body['span_audit']['removed']}")
    print(f"positives dropped: {body['positives_dropped']}")
    print(f"near-duplicate pairs above {NEGATIVES_THRESHOLD}: {body['near_duplicates']['count']}")
    print(f"sanity queries above {SANITY_THRESHOLD}: {len(body['sanity']['above_0.5'])}")
    print(f"seconds: {time.monotonic() - started:.1f}")
    return 0


class _FitTimer:
    """A scorer factory that times the one `fit` of the scorer it makes (section 10, 13.9).

    `fit` returns the inner fitted scorer, so `fit_final_model` keeps that and never the timer.
    """

    def __init__(self, make_scorer: Callable[[], Scorer]) -> None:
        self._make_scorer = make_scorer
        self.seconds: float | None = None

    def __call__(self) -> "_FitTimer":
        self._scorer = self._make_scorer()
        return self

    def fit(self, texts: Sequence[str], y: Sequence[int]) -> Scorer:
        began = time.perf_counter()
        fitted = self._scorer.fit(texts, y)
        self.seconds = time.perf_counter() - began
        return fitted


def _not_run_reasons(
    arms: Sequence[str], models: Sequence[str], requested: Sequence[str], flag: str,
    comparison_mlp_never: bool,
) -> dict[str, dict[str, str]]:
    """arm -> model -> reason, for the models a section is not run for (10, 8.6 item 1)."""
    reasons: dict[str, dict[str, str]] = {arm: {} for arm in arms}
    for arm in arms:
        for model in models:
            if model not in requested:
                reasons[arm][model] = (
                    f"not requested: {flag} does not name {model} (section 10)")
            elif comparison_mlp_never and arm == COMPARISON and model == MLP:
                reasons[arm][model] = (
                    "the MLP is never run in the comparison arm's leave-one-family-out "
                    "(8.6 item 1)")
    return reasons


def _evaluate(
    negatives: Sequence[str],
    n: int,
    models: Sequence[str],
    target_fpr: float,
    lofo: Sequence[str] | None = None,
    catalogue_gap: Sequence[str] | None = None,
    query_path: Path = RAW_PATH,
    product_path: Path = PRODUCT_PATH,
    checksums_path: Path = CHECKSUMS_PATH,
    sanity_path: Path = SANITY_PATH,
    results_dir: Path = RESULTS_DIR,
    registry: Registry = REGISTRY,
    vocabulary_cap: int = VOCABULARY_CAP,
    **folds: int,
) -> int:
    """Run `evaluate` (8, 10): the run report of `build_run_report` plus the sanity section, the
    part-number diagnostic and the MLP fit time, written once through `write_report` after one
    final model per arm and model is fitted and pickled; 1 with a message on failure, and no
    run JSON then.

    `lofo` and `catalogue_gap` are None or empty (off) or the models to run the section for;
    the main fits and the final models cover every model in `models`. `folds` are the fold sizes of
    `build_run_report` (`n_splits`, `repeats`, ...), for tests.
    """
    started = time.monotonic()
    try:
        queries, products, sanity = _load_inputs(
            query_path, product_path, checksums_path, sanity_path)
        real = [q.query for q in queries]
        sanity_texts = [q.query for q in sanity.queries]
        part_numbers = _part_numbers(token_occurrences(real), real, sanity_texts)
        part_number_texts = [s.text for s in part_numbers]
        factories = {name: MODEL_FACTORIES[name] for name in models}
        lofo_not_run = _not_run_reasons(negatives, models, lofo or (), "--lofo", True)
        gap_not_run = _not_run_reasons(
            negatives, models, catalogue_gap or (), "--catalogue-gap", False)
        parameters, body = build_run_report(
            real_queries=real, products=products, part_numbers=part_number_texts,
            sanity=sanity_texts, models=factories, target_fpr=target_fpr, arms=list(negatives),
            n=n, registry=registry, vocabulary_cap=vocabulary_cap, lofo=bool(lofo),
            catalogue_gap=bool(catalogue_gap),
            query_classes=[q.query_class for q in queries] if catalogue_gap else None,
            lofo_not_run=lofo_not_run, gap_not_run=gap_not_run, **folds)
        parameters = {**parameters, "lofo_models": lofo, "catalogue_gap_models": catalogue_gap}
        finals = []
        mlp_seconds = None  # the primary arm's final MLP fit, not its spans or threshold scores
        for arm in negatives:
            for model, make_scorer in factories.items():
                timer = _FitTimer(make_scorer) if arm == PRIMARY and model == MLP else None
                finals.append(fit_final_model(
                    arm, timer or make_scorer, target_fpr, model, real, products,
                    part_number_texts, sanity_texts, n=n, registry=registry,
                    vocabulary_cap=vocabulary_cap))
                if timer:
                    mlp_seconds = timer.seconds
        body["sanity"] = sanity_section(sanity_path, queries, finals)
        body["part_number_diagnostic"] = part_number_section(part_numbers, finals)
        body["timing"] = {"mlp_final_fit_seconds": mlp_seconds}
        for final in finals:
            save_final_model(final, results_dir)
        path = write_report(results_dir, "evaluate", parameters, body)
    except SanityError:
        # The loader's message can quote a query; keep sanity text out of the output.
        print(f"bqc evaluate: {sanity_path} is invalid (details withheld)", file=sys.stderr)
        return 1
    except (CommandError, WandsError, PoolExhausted, FoldError, NonFiniteScoreError,
            OSError) as e:
        print(f"bqc evaluate: {e}", file=sys.stderr)
        return 1
    print(f"wrote {path}")
    print(f"sanity: {body['sanity']['status']}")
    if mlp_seconds is not None:
        note = " (exceeds 5 minutes, section 10)" if mlp_seconds > MLP_FIT_LIMIT_SECONDS else ""
        print(f"MLP final fit, primary arm: {mlp_seconds:.1f} s{note}")
    print(f"seconds: {time.monotonic() - started:.1f}")
    return 0


def _fetch() -> int:
    try:
        for target, url in ((RAW_PATH, URL), (PRODUCT_PATH, PRODUCT_URL)):
            fetch(target, CHECKSUMS_PATH, url)
    except WandsError as e:
        print(f"bqc fetch: {e}", file=sys.stderr)
        return 1
    return 0


def _check_negatives_size(parser: argparse.ArgumentParser, n: int) -> None:
    if n < TRAINING_TEMPLATE_COUNT:
        parser.error(
            f"--negatives-size must be at least {TRAINING_TEMPLATE_COUNT} "
            "(one render per training template)")


def _check_evaluate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    _check_negatives_size(parser, args.negatives_size)
    for flag, requested in (("--lofo", args.lofo), ("--catalogue-gap", args.catalogue_gap)):
        outside = [m for m in requested or () if m not in args.models]
        if outside:
            parser.error(f"{flag} names {', '.join(outside)}, which --models does not run")


def _section_models(requested: list[str] | None, models: Sequence[str]) -> list[str] | None:
    """None (flag absent), the models named, or all of `--models` for a bare flag."""
    if requested is None:
        return None
    return requested or list(models)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch on the command; argparse exits with 2 on bad usage."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "fetch":
        return _fetch()
    if args.command == "generate":
        _check_negatives_size(parser, args.negatives_size)
        return _generate(args.negatives_size)
    if args.command == "evaluate":
        _check_evaluate_args(parser, args)
        return _evaluate(
            args.negatives, args.negatives_size, args.models, args.target_fpr,
            _section_models(args.lofo, args.models),
            _section_models(args.catalogue_gap, args.models))
    raise AssertionError(f"unhandled command {args.command!r}")  # required=True prevents it
