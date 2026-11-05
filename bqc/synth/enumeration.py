"""Family C: enumeration and scraping (DESIGN.md 2, 2.2, 4).

Six templates: four pattern templates (dictionary walks, URL paths, query-string
fragments with random keys, pagination and offset sweeps) and two hard positives
(product words plus a query-string fragment, product words in a search-URL form).
No output is a SKU, part number or bare number (2.2).
"""
import random
import string
from collections.abc import Sequence

from bqc.synth.registry import Template, register

FAMILY = "C"


_WALK_CANDIDATES = (
    list(string.ascii_lowercase)
    + [a + b for a in string.ascii_lowercase for b in string.ascii_lowercase]
)
_SEGMENTS = ("category", "categories", "products", "product", "catalog", "shop",
             "browse", "collections", "item", "sofas", "rugs", "chairs", "tables",
             "lighting", "beds", "outdoor", "sale", "new", "brands", "all")
_PAGING_KEYS = ("page", "p", "offset", "limit", "start", "per_page", "rows", "skip",
                "size", "from", "pagesize")
_LIMITS = (10, 20, 24, 25, 48, 50, 100)
_SEARCH_FORMS = ("/search?q={}", "/search?query={}", "/s?k={}", "/products?search={}",
                 "/catalogsearch/result/?q={}", "/search.php?keyword={}", "?q={}",
                 "/search?term={}&page=1")


def _words(rng: random.Random, vocabulary: Sequence[str]) -> list[str]:
    """1-3 vocabulary words; repeats in `vocabulary` are fine, nothing assumes distinct."""
    return [rng.choice(vocabulary) for _ in range(rng.randint(1, 3))]


def _paging_pairs(rng: random.Random) -> str:
    """Pagination or offset pairs: fixed key names, varying numbers."""
    limit = rng.choice(_LIMITS)
    kind = rng.choice(("page", "offset", "start", "skip", "from", "size"))
    if kind == "page":
        key = rng.choice(("page", "p"))
        out = f"{key}={rng.randint(1, 999)}"
        if rng.random() < 0.5:
            out += f"&{rng.choice(('per_page', 'pagesize', 'limit', 'rows'))}={limit}"
        return out
    if kind == "size":
        return f"size={limit}&page={rng.randint(1, 999)}"
    key = {"offset": "offset", "start": "start", "skip": "skip", "from": "from"}[kind]
    out = f"{key}={limit * rng.randint(1, 250)}"
    if rng.random() < 0.7:
        out += f"&{rng.choice(('limit', 'rows', 'size'))}={limit}"
    return out


def _walk(rng: random.Random, vocabulary: Sequence[str]) -> str:
    """A 1-2 letter string that is not a vocabulary word (2.2)."""
    words = set(vocabulary)
    candidates = [c for c in _WALK_CANDIDATES if c not in words]
    if not candidates:
        raise ValueError("vocabulary covers every 1-2 letter string")
    return rng.choice(candidates)


def _path(rng: random.Random, vocabulary: Sequence[str]) -> str:
    """A URL path such as `/category/sofas?page=3`: always a word segment first, so
    never the `/123` shape the part-number regex matches."""
    segments = [rng.choice(_SEGMENTS) for _ in range(rng.randint(1, 3))]
    if rng.random() < 0.3:
        segments.append(str(rng.randint(1, 9999)))
    path = "/" + "/".join(segments)
    if rng.random() < 0.3:
        path += "/"
    if rng.random() < 0.5:
        path += f"?page={rng.randint(1, 99)}"
    return path


def _random_key(rng: random.Random) -> str:
    """2-8 random letters, never one of the paging key names (template 4's)."""
    while True:
        key = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(2, 8)))
        if key not in _PAGING_KEYS:
            return key


def _random_keys(rng: random.Random, vocabulary: Sequence[str]) -> str:
    """`qzr=12&vbn=7`: random keys, numbers always 1-99 (template 4 varies the numbers)."""
    fragment = "&".join(f"{_random_key(rng)}={rng.randint(1, 99)}"
                        for _ in range(rng.randint(1, 3)))
    return rng.choice(("", "?")) + fragment


def _paging(rng: random.Random, vocabulary: Sequence[str]) -> str:
    """`offset=340&limit=20`: the usual key names, varying numbers."""
    return rng.choice(("", "?")) + _paging_pairs(rng)


def _hard_fragment(rng: random.Random, vocabulary: Sequence[str]) -> str:
    """`area rug&page=2`: vocabulary words, then a paging fragment."""
    return " ".join(_words(rng, vocabulary)) + rng.choice("&?") + _paging_pairs(rng)


def _hard_search_url(rng: random.Random, vocabulary: Sequence[str]) -> str:
    """`/search?q=accent+chair`: vocabulary words inside a search URL."""
    query = rng.choice(("+", "%20")).join(_words(rng, vocabulary))
    return rng.choice(_SEARCH_FORMS).format(query)


for _id, _hard, _render in [
    ("walk", False, _walk),
    ("path", False, _path),
    ("random_keys", False, _random_keys),
    ("paging", False, _paging),
    ("hard_fragment", True, _hard_fragment),
    ("hard_search_url", True, _hard_search_url),
]:
    register(Template(FAMILY, _id, _hard, _render))
