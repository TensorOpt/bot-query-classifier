"""Sample type, template registry and generation entry point (DESIGN.md 4, 2.2).

Also the one home of the part-number regex and the letter/digit block count
(DESIGN.md 2.2).
"""
import random
import re
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import NamedTuple

TRAINING_FAMILIES = ("A", "B", "C", "D")
EVAL_FAMILY = "E"
ALL_FAMILIES = TRAINING_FAMILIES + (EVAL_FAMILY,)
TEMPLATES_PER_FAMILY = 6
MIN_HARD_TEMPLATES = 2
QUOTA = 20

PART_NUMBER_RE = re.compile(r"^[a-z]{0,4}[-/]?\d{3,}[a-z0-9]{0,2}$")
MIN_GIBBERISH_BLOCKS = 4
MAX_NUMBER_BLOCKS = 3
NUMBER_SHAPED_RE = re.compile(r'^(?=.*\d)[a-z0-9]+([-/.x][a-z0-9]+)*"?$')
_BLOCK_RE = re.compile(r"[a-z]+|\d+")


class Sample(NamedTuple):
    text: str
    family: str
    template_id: str
    seed: int


@dataclass(frozen=True)
class Template:
    """One sub-generator. `render(rng, vocabulary)` returns one query string."""
    family: str
    template_id: str
    hard: bool
    render: Callable[[random.Random, Sequence[str]], str]
    quota: int = QUOTA


def block_count(token: str) -> int:
    """Number of contiguous letter blocks plus digit blocks in a token (2.2).

    Other characters (`-`, `/`) separate blocks: `xk9qz7vplm` has 5, `wf-2049x` has 3.
    """
    return len(_BLOCK_RE.findall(token))


def is_benign_token(token: str, vocabulary: Collection[str]) -> bool:
    """True if `token` is a vocabulary word or number-shaped with at most 3 blocks (2.2)."""
    return token in vocabulary or (
        bool(NUMBER_SHAPED_RE.match(token)) and block_count(token) <= MAX_NUMBER_BLOCKS)


def has_non_benign_token(text: str, vocabulary: Collection[str]) -> bool:
    """True if at least one whitespace token of `text` is not benign (2.2)."""
    words = vocabulary if isinstance(vocabulary, (set, frozenset)) else set(vocabulary)
    return any(not is_benign_token(t, words) for t in text.split())


def digit_tokens_below_block_rule(text: str) -> list[str]:
    """Digit-bearing tokens with fewer than MIN_GIBBERISH_BLOCKS blocks (2.2)."""
    return [
        t for t in text.split()
        if any(c.isdigit() for c in t) and block_count(t) < MIN_GIBBERISH_BLOCKS
    ]


class Registry:
    def __init__(self) -> None:
        self._templates: dict[str, list[Template]] = {f: [] for f in ALL_FAMILIES}

    def register(self, template: Template) -> None:
        """Add a template; unknown family or duplicate id raises ValueError."""
        if template.family not in self._templates:
            raise ValueError(f"unknown family {template.family!r}")
        if any(t.template_id == template.template_id
               for t in self._templates[template.family]):
            raise ValueError(
                f"duplicate template {template.family}/{template.template_id}")
        self._templates[template.family].append(template)

    def templates(self, family: str) -> list[Template]:
        return list(self._templates[family])

    def validate(self, families: Sequence[str]) -> None:
        """Enforce section 4 rule 6 for each family; ValueError otherwise."""
        for family in families:
            if family not in self._templates:
                raise ValueError(f"unknown family {family!r}")
            templates = self._templates[family]
            if len(templates) != TEMPLATES_PER_FAMILY:
                raise ValueError(
                    f"family {family} has {len(templates)} templates, "
                    f"need {TEMPLATES_PER_FAMILY}")
            bad = [t.template_id for t in templates if t.quota != QUOTA]
            if bad:
                raise ValueError(f"family {family} templates {bad} have quota != {QUOTA}")
            if sum(t.hard for t in templates) < MIN_HARD_TEMPLATES:
                raise ValueError(
                    f"family {family} needs at least {MIN_HARD_TEMPLATES} "
                    "hard-positive templates")

    def group_sizes(
        self, families: Sequence[str] = TRAINING_FAMILIES
    ) -> dict[tuple[str, str], int]:
        """Sample count per (family, template_id) group, without generating text."""
        self.validate(families)
        return {
            (f, t.template_id): t.quota
            for f in families for t in self._templates[f]
        }

    def generate(
        self,
        vocabulary: Sequence[str],
        seed: int,
        families: Sequence[str] = TRAINING_FAMILIES,
    ) -> list[Sample]:
        if not vocabulary:
            raise ValueError("vocabulary is empty")
        self.validate(families)
        samples = []
        for family in families:
            for t in self._templates[family]:
                # A str seed is hashed with sha512, so it ignores PYTHONHASHSEED.
                rng = random.Random(f"{seed}/{family}/{t.template_id}")
                samples.extend(
                    Sample(t.render(rng, vocabulary), family, t.template_id, seed)
                    for _ in range(t.quota)
                )
        return samples


REGISTRY = Registry()


def register(template: Template) -> None:
    REGISTRY.register(template)


def group_sizes(
    families: Sequence[str] = TRAINING_FAMILIES,
) -> dict[tuple[str, str], int]:
    return REGISTRY.group_sizes(families)


def generate(
    vocabulary: Sequence[str],
    seed: int,
    families: Sequence[str] = TRAINING_FAMILIES,
) -> list[Sample]:
    """Default output is the 480 training positives; E only when requested."""
    return REGISTRY.generate(vocabulary, seed, families)
