"""Stub templates and a registry builder shared by the synth tests."""
import random
from collections.abc import Sequence

from bqc.synth.registry import (
    ALL_FAMILIES,
    MIN_HARD_TEMPLATES,
    TEMPLATES_PER_FAMILY,
    Registry,
    Template,
)

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def gibberish(rng: random.Random, blocks: int = 5) -> str:
    """Alternate letter and digit blocks, starting with letters: >= `blocks` blocks."""
    parts = []
    for i in range(blocks):
        if i % 2 == 0:
            parts.append("".join(rng.choice(LETTERS) for _ in range(rng.randint(2, 4))))
        else:
            parts.append(str(rng.randint(1, 9)))
    return "".join(parts)


def pattern_render(family: str, k: int):
    def render(rng: random.Random, vocabulary: Sequence[str]) -> str:
        word = "".join(rng.choice(LETTERS) for _ in range(rng.randint(3, 8)))
        if family == "A":
            return f"{gibberish(rng)} {word}"
        return f"{family.lower()}{k} {word} {rng.randint(0, 99)}"
    return render


def hard_render(family: str, k: int):
    def render(rng: random.Random, vocabulary: Sequence[str]) -> str:
        word = rng.choice(vocabulary)
        suffix = "".join(rng.choice(LETTERS) for _ in range(rng.randint(3, 8)))
        return f"{word} {suffix}{k}x" if family != "A" else f"{word} {suffix}"
    return render


def build_family(registry: Registry, family: str) -> None:
    for k in range(TEMPLATES_PER_FAMILY):
        hard = k < MIN_HARD_TEMPLATES
        registry.register(Template(
            family, f"{family.lower()}{k}", hard,
            hard_render(family, k) if hard else pattern_render(family, k),
        ))


def build_registry(families=ALL_FAMILIES) -> Registry:
    registry = Registry()
    for family in families:
        build_family(registry, family)
    return registry
