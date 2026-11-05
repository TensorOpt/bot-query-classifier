"""Query preprocessing: the one home of normalise() (DESIGN.md section 5)."""
import unicodedata

MAX_CODE_POINTS = 512


def normalise(s: str) -> str:
    """Return the normalised form of a query.

    Steps, in order: NFKC; drop Cc/Cf characters unless str.isspace();
    collapse whitespace; lower(); truncate to 512 code points then rstrip().
    Pure, idempotent, and silent about truncation.
    """
    s = unicodedata.normalize("NFKC", s)
    s = "".join(
        c for c in s
        if c.isspace() or unicodedata.category(c) not in ("Cc", "Cf")
    )
    s = " ".join(s.split())
    s = s.lower()
    return s[:MAX_CODE_POINTS].rstrip()
