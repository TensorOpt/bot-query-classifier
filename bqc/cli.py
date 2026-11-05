"""Command line entry point and command implementations."""

import argparse
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bqc import SEED
from bqc.negatives import (
    DEFAULT_N,
    VOCABULARY_CAP,
    PoolExhausted,
    build_spans,
    token_occurrences,
)
from bqc.preprocess import normalise
from bqc.report import RESULTS_DIR, write_report
from bqc.sanity import SANITY_PATH, SanityError, SanityQuery, load_sanity
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
    fetch,
    load_products,
    load_queries,
)

MIN_DISTINCT_SHARE = 0.5  # a template under this share of distinct strings is a finding (4)

TRAINING_TEMPLATE_COUNT = 24  # templates of A-D: positives per template are n // 24


class GenerateError(Exception):
    """`generate` can't finish; the message holds no sanity or removed text."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bqc")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    commands.add_parser("fetch", help="download WANDS query.csv and product.csv into data/raw/")
    generate = commands.add_parser(
        "generate", help="build the default training set and write a report to results/")
    generate.add_argument(
        "--negatives-size", type=_positive_int, default=DEFAULT_N, metavar="N",
        help=f"number of product-name spans (default {DEFAULT_N})")
    return parser


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be positive, got {number}")
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
        raise GenerateError(
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


def build_generate_report(
    n: int,
    query_path: Path,
    product_path: Path,
    checksums_path: Path,
    sanity_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build everything of `generate` (3.2, 4, 10) and return (parameters, report body).

    Raises GenerateError, WandsError, SanityError or PoolExhausted; none of the messages
    holds sanity text.
    """
    for path in (query_path, product_path):
        if not path.exists():
            raise GenerateError(f"missing {path}; run `python -m bqc fetch` first")
    queries = load_queries(query_path, checksums_path)
    products = load_products(product_path, checksums_path)
    sanity = load_sanity(sanity_path, queries)
    real = [q.query for q in queries]
    sanity_texts = [q.query for q in sanity.queries]

    real_vocabulary = token_occurrences(real)
    try:
        part_numbers = generate_part_numbers(real_vocabulary, SEED, real, sanity_texts)
    except AuditError:
        raise GenerateError(
            "a part-number string equals a real or sanity query (text withheld)") from None
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
    except (GenerateError, WandsError, PoolExhausted, OSError) as e:
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


def _fetch() -> int:
    try:
        for target, url in ((RAW_PATH, URL), (PRODUCT_PATH, PRODUCT_URL)):
            fetch(target, CHECKSUMS_PATH, url)
    except WandsError as e:
        print(f"bqc fetch: {e}", file=sys.stderr)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch on the command; argparse exits with 2 on bad usage."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "fetch":
        return _fetch()
    if args.command == "generate":
        if args.negatives_size < TRAINING_TEMPLATE_COUNT:
            parser.error(
                f"--negatives-size must be at least {TRAINING_TEMPLATE_COUNT} "
                "(one render per training template)")
        return _generate(args.negatives_size)
    raise AssertionError(f"unhandled command {args.command!r}")  # required=True prevents it
