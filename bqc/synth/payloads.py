"""Family B: injection and probing payloads (DESIGN.md 2, 2.2, 4).

Six templates: four payload templates (SQL, script, template injection, path
traversal) and two hard positives (vocabulary words wrapped in SQL and NoSQL
payloads). Payloads are only ever generated as text, never executed or sent.
"""
import random
from collections.abc import Sequence

from bqc.synth.registry import Template, register

FAMILY = "B"


_TABLES = ("accounts", "orders", "customers", "admin", "products", "sessions")
_COLUMNS = ("username", "password", "email", "id", "token")
_TRAVERSAL_TARGETS = ("etc/passwd", "etc/shadow", "etc/hosts", "proc/self/environ",
                      "windows/win.ini", "boot.ini", "var/log/auth.log",
                      "proc/self/cmdline", "etc/issue", "etc/group", "etc/hostname")
_ARITHMETIC = ("7*7", "7*'7'", "3*3", "9*9", "13*37", "1337*2")
_XSS_ARGS = ("2", "9", "42", "7331", "document.cookie", "document.domain", "'xss'",
             "'x'", "string.fromcharcode(88)", "this.id", "top.location", "0")
_XSS_FUNCTIONS = ("alert", "prompt", "confirm")


def _words(rng: random.Random, vocabulary: Sequence[str]) -> str:
    """1-3 vocabulary words; repeats in `vocabulary` are fine, nothing assumes distinct."""
    return " ".join(rng.choice(vocabulary) for _ in range(rng.randint(1, 3)))


def _number(rng: random.Random) -> int:
    return rng.randint(1, 9)


def _sql_payload(rng: random.Random) -> str:
    """A quote-breaking SQL fragment, always starting with `'` (so the word before it
    becomes a non-vocabulary token): boolean tautologies, UNION SELECT, time delays and
    column reads and `order by` column-count probes."""
    n = _number(rng)
    kind = rng.choice(("or", "union", "sleep", "comment", "order"))
    if kind == "or":
        return rng.choice((f"' or {n}={n}--", f"' or {n}={n}#", f"' or 'a'='a",
                           f"\" or {n}={n}--", f"') or ({n}={n})--"))
    if kind == "union":
        cols = ",".join(["null"] * rng.randint(1, 5))
        return f"' union select {cols}--"
    if kind == "sleep":
        return f"' and sleep({rng.randint(2, 9)})--"
    if kind == "order":
        return f"' order by {rng.randint(1, 20)}--"
    return f"' union select {rng.choice(_COLUMNS)} from {rng.choice(_TABLES)}--"


def _nosql_payload(rng: random.Random) -> str:
    op = rng.choice(("$gt", "$ne", "$regex", "$where"))
    value = {"$gt": '""', "$ne": "null", "$regex": '".*"', "$where": '"1==1"'}[op]
    if op != "$where" and rng.random() < 0.3:
        return f"[{op}]="  # query-string form: `user[$ne]=`
    return f'{{"{op}":{value}}}'


def _sql(rng: random.Random, vocabulary: Sequence[str]) -> str:
    """Bare SQL injection, optionally with a leading digit or word (`1' or 1=1--`)."""
    prefix = rng.choice(("", "", str(_number(rng)), rng.choice(_COLUMNS)))
    return f"{prefix}{_sql_payload(rng)}"


def _script(rng: random.Random, vocabulary: Sequence[str]) -> str:
    """Cross-site scripting payloads: script tags, event handlers and javascript: URLs."""
    call = f"{rng.choice(_XSS_FUNCTIONS)}({rng.choice(_XSS_ARGS)})"
    return rng.choice((
        f"<script>{call}</script>",
        f"<img src=x onerror={call}>",
        f"<svg onload={call}>",
        f"\"><script>{call}</script>",
        f"javascript:{call}",
        f"<body onload={call}>",
        f"<input autofocus onfocus={call}>",
        f"<iframe src=javascript:{call}>",
        f"<details open ontoggle={call}>",
        f"<a href=# onmouseover={call}>",
        f"'><img src=x onerror={call}>",
    ))


def _template(rng: random.Random, vocabulary: Sequence[str]) -> str:
    """Server-side template injection: arithmetic in each engine's expression syntax, or a
    statement-style print tag (`{% print 12*34 %}`) with random operands."""
    expr = rng.choice(_ARITHMETIC)
    return rng.choice((
        f"${{{expr}}}",
        f"{{{{{expr}}}}}",
        f"<%= {expr} %>",
        f"#{{{expr}}}",
        f"{{% print {rng.randint(2, 99)}*{rng.randint(2, 99)} %}}",
    ))


def _traversal(rng: random.Random, vocabulary: Sequence[str]) -> str:
    """Path traversal: repeated parent-directory steps (several encodings) to a system file."""
    depth = rng.randint(3, 12)
    up = rng.choice(("../", "..\\", "..%2f", "....//", "%2e%2e%2f", "..;/"))
    target = rng.choice(_TRAVERSAL_TARGETS)
    start = rng.choice(("", "", "/", "file:///", "/var/www/"))
    end = rng.choice(("", "", "%00", "%00.jpg"))
    return start + up * depth + target + end


def _hard_sql(rng: random.Random, vocabulary: Sequence[str]) -> str:
    """`leather sofa' union select null--`: vocabulary words, then a SQL fragment."""
    return f"{_words(rng, vocabulary)}{_sql_payload(rng)}"


def _hard_nosql(rng: random.Random, vocabulary: Sequence[str]) -> str:
    """`oak desk {"$ne":null}`: vocabulary words, then a NoSQL operator."""
    return f"{_words(rng, vocabulary)} {_nosql_payload(rng)}"


for _id, _hard, _render in [
    ("sql", False, _sql),
    ("script", False, _script),
    ("template", False, _template),
    ("traversal", False, _traversal),
    ("hard_sql", True, _hard_sql),
    ("hard_nosql", True, _hard_nosql),
]:
    register(Template(FAMILY, _id, _hard, _render))
