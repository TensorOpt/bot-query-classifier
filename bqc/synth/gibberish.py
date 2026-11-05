"""Family A: gibberish (DESIGN.md 2, 2.2, 4).

Six templates: four pattern templates (keyboard mashing, repeated characters,
interleaved letter/digit tokens, UUID/hash-like hex) and two hard positives
(vocabulary words plus a gibberish token, never a digit).
"""
import random
import string
from collections.abc import Callable, Sequence

from bqc.synth.registry import Template, register

FAMILY = "A"


KEYBOARD_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
HASH_LENGTHS = (32, 40, 64)  # md5, sha1, sha256 hex digests
_LETTERS = string.ascii_lowercase
_HEX_LETTERS = "abcdef"
_HEX_DIGITS = "0123456789"
_HEX = _HEX_DIGITS + _HEX_LETTERS


def _fresh(rng: random.Random, vocabulary: Sequence[str],
           make: Callable[[random.Random], str]) -> str:
    """A token from `make` that is not a vocabulary word (2.2): redraw on collision."""
    words = set(vocabulary)
    while True:
        token = make(rng)
        if token not in words:
            return token


def _mash_token(rng: random.Random) -> str:
    """5-12 keys drawn from a window of 4-6 adjacent keys of one keyboard row."""
    row = rng.choice(KEYBOARD_ROWS)
    width = rng.randint(4, min(6, len(row)))
    start = rng.randint(0, len(row) - width)
    window = row[start:start + width]
    return "".join(rng.choice(window) for _ in range(rng.randint(5, 12)))


def _repeat_token(rng: random.Random) -> str:
    """A 1-3 letter unit repeated to 6-16 characters."""
    unit = "".join(rng.choice(_LETTERS) for _ in range(rng.randint(1, 3)))
    copies = rng.randint(-(-6 // len(unit)), 16 // len(unit))
    return unit * copies


def _alnum_token(rng: random.Random) -> str:
    """7-9 alternating letter/digit blocks, letters first: always at least 4 blocks (2.2).

    At least 11 characters in total.
    """
    parts = []
    for i in range(rng.randint(7, 9)):
        if i % 2 == 0:
            parts.append("".join(rng.choice(_LETTERS) for _ in range(rng.randint(2, 4))))
        else:
            parts.append("".join(rng.choice(_HEX_DIGITS) for _ in range(rng.randint(1, 2))))
    return "".join(parts)


def _hex_token(rng: random.Random) -> str:
    """UUID v4 or a 32/40/64 hex digest.

    The first four characters are digit, letter, digit, letter, so any digit-bearing
    token has at least 4 blocks whatever the random rest is (2.2).
    """
    kind = rng.choice(("uuid",) + HASH_LENGTHS)
    length = 32 if kind == "uuid" else kind
    chars = [rng.choice(_HEX_DIGITS), rng.choice(_HEX_LETTERS),
             rng.choice(_HEX_DIGITS), rng.choice(_HEX_LETTERS)]
    chars += [rng.choice(_HEX) for _ in range(length - 4)]
    if kind != "uuid":
        return "".join(chars)
    chars[12] = "4"
    chars[16] = rng.choice("89ab")
    h = "".join(chars)
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def _words(rng: random.Random, vocabulary: Sequence[str], low: int, high: int) -> list[str]:
    """Vocabulary words without digits: real words such as `18x18` would break 2.2."""
    pool = [w for w in vocabulary if not any(c.isdigit() for c in w)]
    if not pool:
        raise ValueError("vocabulary has no word without a digit")
    return [rng.choice(pool) for _ in range(rng.randint(low, high))]


def _mash(rng: random.Random, vocabulary: Sequence[str]) -> str:
    return " ".join(_fresh(rng, vocabulary, _mash_token) for _ in range(rng.randint(1, 3)))


def _repeat(rng: random.Random, vocabulary: Sequence[str]) -> str:
    return _fresh(rng, vocabulary, _repeat_token)


def _alnum(rng: random.Random, vocabulary: Sequence[str]) -> str:
    return " ".join(_alnum_token(rng) for _ in range(rng.randint(1, 2)))


def _hex(rng: random.Random, vocabulary: Sequence[str]) -> str:
    return _hex_token(rng)


def _hard_mash(rng: random.Random, vocabulary: Sequence[str]) -> str:
    """`accent chair asdfgh`: vocabulary words, then a mash token."""
    return " ".join(_words(rng, vocabulary, 1, 3)
                    + [_fresh(rng, vocabulary, _mash_token)])


def _hard_repeat(rng: random.Random, vocabulary: Sequence[str]) -> str:
    """`area rug aaaaaaa`: a repeated-character token at a random position."""
    tokens = _words(rng, vocabulary, 1, 3)
    tokens.insert(rng.randint(0, len(tokens)), _fresh(rng, vocabulary, _repeat_token))
    return " ".join(tokens)


for _id, _hard, _render in [
    ("mash", False, _mash),
    ("repeat", False, _repeat),
    ("alnum", False, _alnum),
    ("hex", False, _hex),
    ("hard_mash", True, _hard_mash),
    ("hard_repeat", True, _hard_repeat),
]:
    register(Template(FAMILY, _id, _hard, _render))
