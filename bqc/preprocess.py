"""Query preprocessing: the one home of normalise() (DESIGN.md section 5)."""
import unicodedata

MAX_CODE_POINTS = 512


def normalise(s: str) -> str:
    """Return the normalised form of a query.

    Steps, in order: drop Cc/Cf characters unless str.isspace(); NFKC; lower();
    NFKC again; collapse whitespace; truncate to 512 code points then rstrip().
    Pure and silent about truncation. Idempotence is a tested property over the
    cases in tests/test_preprocess.py, not a guarantee over all strings
    (DESIGN.md 13.16).
    """
    s = "".join(
        c for c in s
        if c.isspace() or unicodedata.category(c) not in ("Cc", "Cf")
    )
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = unicodedata.normalize("NFKC", s)
    s = " ".join(s.split())
    return s[:MAX_CODE_POINTS].rstrip()
