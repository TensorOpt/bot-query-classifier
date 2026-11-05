"""Family E: off-topic natural language, evaluation-only (DESIGN.md 2, 2.1, 2.2, 4).

Six templates: four pattern templates (trivia questions, chat prompts, Python snippets,
JavaScript snippets) and two hard positives (a question and a chat prompt wrapped around
product vocabulary). Every output is text that is not a product search, and none is an
off-domain shopping query (2.1).
"""
import random
from collections.abc import Sequence

from bqc.synth.registry import Template, register

FAMILY = "E"


def _choose(rng: random.Random, vocabulary: Sequence[str], options: Sequence[str]) -> str:
    """A random option that is not a vocabulary word, so the marker word is non-benign
    by construction (2.2)."""
    words = set(vocabulary)
    candidates = [o for o in options if o not in words]
    if not candidates:
        raise ValueError("vocabulary covers every marker option")
    return rng.choice(candidates)


def _words(rng: random.Random, vocabulary: Sequence[str]) -> str:
    """1-3 vocabulary words; repeats in `vocabulary` are fine, nothing assumes distinct."""
    return " ".join(rng.choice(vocabulary) for _ in range(rng.randint(1, 3)))


_PLACES = ("france", "japan", "brazil", "canada", "egypt", "norway", "kenya", "peru",
           "india", "chile", "germany", "spain", "italy", "mexico", "australia",
           "vietnam", "poland", "sweden", "turkey", "argentina")
_FACTS = ("capital", "population", "currency", "national anthem", "largest city",
          "official language", "time zone", "highest mountain")
_TRIVIA_FRAMES = (
    "what is the {fact} of {place}",
    "what's the {fact} of {place}",
    "tell me the {fact} of {place}",
    "do you know the {fact} of {place}",
    "how many people live in {place}",
    "when did {place} become a country",
)


def _trivia(rng: random.Random, vocabulary: Sequence[str]) -> str:
    """`what is the capital of france`."""
    return rng.choice(_TRIVIA_FRAMES).format(
        fact=rng.choice(_FACTS), place=_choose(rng, vocabulary, _PLACES))


_FORMS = ("poem", "haiku", "limerick", "sonnet", "essay", "song", "riddle", "fable")
_TOPICS = ("friendship", "the ocean", "black holes", "dragons", "the moon", "time travel",
           "volcanoes", "pirates", "photosynthesis", "my cat", "a rainy monday")
_CONCEPTS = ("quantum computing", "blockchain", "inflation", "photosynthesis",
             "machine learning", "relativity", "recursion", "the stock market")
_LANGUAGES = ("spanish", "french", "german", "japanese", "italian", "latin")
_GREETINGS = ("hello", "good morning", "thank you", "how are you", "see you later")


def _chat(rng: random.Random, vocabulary: Sequence[str]) -> str:
    """`write a haiku about the moon`, `explain recursion like i am five`."""
    kind = rng.randint(0, 3)
    if kind == 0:
        return (f"{rng.choice(('write', 'compose', 'give me'))} a "
                f"{_choose(rng, vocabulary, _FORMS)} about {rng.choice(_TOPICS)}")
    if kind == 1:
        return (f"explain {rng.choice(_CONCEPTS)} like i am "
                f"{rng.choice(('five', 'a beginner', 'in a hurry'))}")
    if kind == 2:
        return (f"translate {rng.choice(_GREETINGS)} into "
                f"{_choose(rng, vocabulary, _LANGUAGES)}")
    return (f"can you {rng.choice(('summarize', 'rewrite', 'shorten'))} this "
            f"{_choose(rng, vocabulary, _FORMS)} for me")


_LIBS = ("numpy", "pandas", "torch", "requests", "flask", "scipy", "matplotlib",
         "sklearn", "django", "pytest")
_VARS = ("i", "j", "k", "n", "x", "y", "idx", "item", "val", "res", "tmp", "data")
_FUNCS = ("foo", "bar", "compute", "helper", "run", "handler", "calc", "process")


def _python(rng: random.Random, vocabulary: Sequence[str]) -> str:
    """`for i in range(10): print(i)`."""
    v, w = rng.choice(_VARS), rng.choice(_VARS)
    n = rng.randint(2, 99)
    kind = rng.randint(0, 4)
    if kind == 0:
        return f"for {v} in range({n}): print({v})"
    if kind == 1:
        return f"def {rng.choice(_FUNCS)}({v}): return {v} * {n}"
    if kind == 2:
        alias = rng.choice(("np", "pd", "m", "lib"))
        return f"import {_choose(rng, vocabulary, _LIBS)} as {alias}"
    if kind == 3:
        return f"{w} = [{v} for {v} in range({n}) if {v} % 2 == 0]"
    return f"while {v} < {n}: {v} += 1"


def _javascript(rng: random.Random, vocabulary: Sequence[str]) -> str:
    """`console.log(x);`, `const f = (a, b) => a + b;`."""
    v, w = rng.choice(_VARS), rng.choice(_VARS)
    n = rng.randint(2, 99)
    kind = rng.randint(0, 4)
    if kind == 0:
        return f"console.log({v});"
    if kind == 1:
        return f"const {rng.choice(_FUNCS)} = ({v}, {w}) => {v} + {w};"
    if kind == 2:
        return f"if ({v} === null) {{ return; }}"
    if kind == 3:
        return f"document.getelementbyid('{rng.choice(_FUNCS)}{n}').click();"
    return f"for (let {v} = 0; {v} < {n}; {v}++) {{}}"


_QUESTION_FRAMES = (
    ("what is the {m} of {w}", ("history", "origin", "etymology", "symbolism", "philosophy")),
    ("who {m} the {w}", ("invented", "discovered", "popularised")),
    ("why is the {w} so important in {m}", ("mythology", "literature", "geography")),
)
_CHAT_FRAMES = (
    ("write a {m} about {w}", ("poem", "haiku", "limerick", "sonnet", "essay", "song")),
    ("tell me a {m} about {w}", ("story", "joke", "riddle", "fable")),
    ("pretend you are a {w} and {m} me a story", ("narrate", "recite", "whisper")),
)


def _hard(frames: Sequence[tuple[str, Sequence[str]]]):
    def render(rng: random.Random, vocabulary: Sequence[str]) -> str:
        frame, markers = rng.choice(frames)
        return frame.format(m=_choose(rng, vocabulary, markers), w=_words(rng, vocabulary))
    return render


for _id, _hard_flag, _render in [
    ("trivia", False, _trivia),
    ("chat", False, _chat),
    ("python", False, _python),
    ("javascript", False, _javascript),
    ("hard_question", True, _hard(_QUESTION_FRAMES)),
    ("hard_chat", True, _hard(_CHAT_FRAMES)),
]:
    register(Template(FAMILY, _id, _hard_flag, _render))
