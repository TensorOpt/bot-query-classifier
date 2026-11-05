"""Family D: spam and promotional (DESIGN.md 2, 2.2, 4).

Six templates: four spam templates (bare URLs, email addresses, phone numbers,
promotional / crypto / SEO phrases) and two hard positives (product words next to a
URL, product words next to a contact detail). Output is inert text: invented
text built at run time and never fetched or dialled (rule 9): domains are
label parts plus a common spam TLD, phone numbers use random area codes and random
7-digit shapes. No phone number is a bare digit run (2.2).
"""
import random
from collections.abc import Sequence

from bqc.synth.registry import Template, register

FAMILY = "D"

_LABELS = ("topoffer", "superprize", "bestoffer", "megasale", "cheapstuff", "winbig", "greenpick",
           "hotdeal", "luckyday", "fastcash", "bigsavings", "clickhere", "freegift",
           "seoboost", "luckwheel", "pillbox")
_TLDS = ("biz", "info", "xyz", "top", "club", "online", "shop", "site", "com", "net")
_PATHS = ("", "", "/", "/offer", "/buy-now", "/deal?id=7", "/win", "/signup", "/sale/1")
_SCHEMES = ("http://", "https://", "www.", "", "http://www.")
_USERS = ("sales", "info", "hello", "winner", "support", "offers", "admin", "bonus",
          "marketing", "contact", "ceo", "billing")
_CONTACT_VERBS = ("call", "text", "message", "dial", "contact", "ring")
_SPAM_ITEMS = ("comments", "watches", "pills", "likes", "handbags", "sunglasses",
               "diet supplements", "views", "replica shoes", "loans", "backlinks",
               "traffic", "meds", "subscribers", "diplomas")
_PROMO_OPENERS = ("buy cheap", "order cheap", "get free", "discount", "best price on",
                  "wholesale", "100% genuine", "buy now")
_PROMO_CLOSERS = ("click here", "limited offer!!!", "act now", "visit our site",
                  "free shipping $$$", "order today!", "no prescription", "click here!!!")
_CRYPTO_SEO = (
    "double your bitcoin in one day guaranteed",
    "earn $500 a day crypto trading signals",
    "free nft airdrop claim now",
    "seo services rank #1 on page one",
    "cheap backlinks high da50 sites",
    "buy website visitors $5",
    "forex robot 100% profit join now",
    "win big jackpot free credits no deposit",
    "work from home earn $$$ fast",
    "you have won a prize claim your reward",
    "crypto giveaway send one get two back",
    "increase your traffic with our seo packages",
    "guaranteed first page ranking cheap seo",
    "make money online no experience needed!!!",
    "invest today 300% returns weekly",
    "lottery winner contact claims agent",
)
_CRYPTO_TAILS = ("", "", " !!!", " today only", " click link", " $$$", " 100% safe")
_SUFFIXES = ("click here", "buy now", "free offer", "visit", "order now", "sale!!!")


def _words(rng: random.Random, vocabulary: Sequence[str]) -> str:
    """1-3 vocabulary words; repeats in `vocabulary` are fine, nothing assumes distinct."""
    return " ".join(rng.choice(vocabulary) for _ in range(rng.randint(1, 3)))


def _domain(rng: random.Random) -> str:
    return f"{rng.choice(_LABELS)}{rng.choice(('', '-', '-'))}{rng.choice(_LABELS)}" \
           f".{rng.choice(_TLDS)}"


def _url(rng: random.Random) -> str:
    """`http://megasale-winbig.biz/offer`; a scheme or `www.` is always present or
    the domain carries a dot, so it is never a vocabulary word or a bare number."""
    return f"{rng.choice(_SCHEMES)}{_domain(rng)}{rng.choice(_PATHS)}"


def _email(rng: random.Random) -> str:
    return f"{rng.choice(_USERS)}{rng.choice(('', '.', '_', '-'))}" \
           f"{rng.choice(_LABELS)}@{_domain(rng)}"


def _digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("0123456789") for _ in range(n))


def _phone_number(rng: random.Random) -> str:
    """A random area code and a random 7-digit line in one of three hyphenated shapes
    (`867-5309`, `86-75-309`, `867-53-09`), in a `(`, `+` or four-block form, so it is
    never a bare digit run or a part-number shape (2.2)."""
    area = rng.choice("23456789") + _digits(rng, 2)
    line = rng.choice((f"{_digits(rng, 3)}-{_digits(rng, 4)}",
                       f"{_digits(rng, 2)}-{_digits(rng, 2)}-{_digits(rng, 3)}",
                       f"{_digits(rng, 3)}-{_digits(rng, 2)}-{_digits(rng, 2)}"))
    form = rng.randrange(4)
    if form == 0:
        return f"({area}) {line}"
    if form == 1:
        return f"+1-{area}-{line}"
    if form == 2:
        return f"+1 ({area}) {line}"
    return f"1-{area}-{line}"


def _phone(rng: random.Random, vocabulary: Sequence[str]) -> str:
    """`call (212) 867-5309 now`: a verb, the number, sometimes a closer."""
    out = f"{rng.choice(_CONTACT_VERBS)} {_phone_number(rng)}"
    return out + rng.choice(("", "", " now", " today", " free quote", " 24/7"))


def _url_only(rng: random.Random, vocabulary: Sequence[str]) -> str:
    return _url(rng)


def _email_only(rng: random.Random, vocabulary: Sequence[str]) -> str:
    return rng.choice(("", "", "contact ", "email ", "write to ")) + _email(rng)


def _promo(rng: random.Random, vocabulary: Sequence[str]) -> str:
    """Promotional phrases: `buy cheap pills click here!!!` or a crypto / SEO line."""
    if rng.random() < 0.5:
        return rng.choice(_CRYPTO_SEO) + rng.choice(_CRYPTO_TAILS)
    return (f"{rng.choice(_PROMO_OPENERS)} {rng.choice(_SPAM_ITEMS)} "
            f"{rng.choice(_PROMO_CLOSERS)}")


def _hard_url(rng: random.Random, vocabulary: Sequence[str]) -> str:
    """`cheap area rug http://x.co`: vocabulary words next to a URL."""
    words, url = _words(rng, vocabulary), _url(rng)
    form = rng.randrange(4)
    if form == 0:
        return f"{words} {url}"
    if form == 1:
        return f"{url} {words}"
    if form == 2:
        return f"{rng.choice(('buy', 'cheap', 'best price'))} {words} {url}"
    return f"{words} {rng.choice(_SUFFIXES)} {url}"


def _hard_contact(rng: random.Random, vocabulary: Sequence[str]) -> str:
    """`leather sofa sales@megasale.biz`: vocabulary words next to an email or phone."""
    words = _words(rng, vocabulary)
    if rng.random() < 0.5:
        return f"{words} {rng.choice(_CONTACT_VERBS)} {_phone_number(rng)}"
    return f"{words} {_email(rng)}"


for _id, _hard, _render in [
    ("url", False, _url_only),
    ("email", False, _email_only),
    ("phone", False, _phone),
    ("promo", False, _promo),
    ("hard_url", True, _hard_url),
    ("hard_contact", True, _hard_contact),
]:
    register(Template(FAMILY, _id, _hard, _render))
