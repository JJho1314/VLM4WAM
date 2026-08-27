"""Pick the noun the probe should localise: the head noun of the MANIPULATED object.

Two mistakes this guards against, both seen in real LIBERO instructions:

1. Destination instead of object. "pick up the orange juice and place it in the basket" used to
   resolve to "basket", because "orange"/"juice" were out of vocabulary and the first hit won. The
   probe then correctly highlighted the basket, which made the figure look wrong.
2. Modifier instead of head. English noun phrases are head-final, so "alphabet soup" -> "soup" and
   "yellow and white mug" -> "mug". Taking the FIRST hit returned "alphabet" and "yellow", which
   localise far worse than the head noun.

So: cut the object phrase at the clause boundary (not at an "and" inside the phrase), then take the
LAST vocabulary hit in it.
"""
import re

NOUNS = [
    "bowl", "plate", "mug", "pot", "cabinet", "drawer", "stove", "basket", "soup", "banana", "cheese",
    "cream", "ketchup", "milk", "butter", "sauce", "tomato", "bottle", "cup", "frying", "pan", "moka",
    "wine", "book", "caddy", "mustard", "box", "rack", "turkey", "salad", "dressing",
    "orange", "juice", "chocolate", "pudding", "alphabet", "ramekin", "cookie", "microwave", "bin",
    "akita", "porcelain",
]
_MODIFIERS = {"white", "yellow", "black", "red", "alphabet", "chocolate", "orange", "cream", "frying",
              "moka", "akita", "porcelain", "tomato"}          # only used if nothing else is found

_VERB = r"(?:pick up|put|place|push|open|close|turn on)\s+(?:both\s+)?(?:the\s+)?"
# clause boundary: a second instruction ("and place/put/..."), or a preposition introducing the target
_STOP = re.compile(r"\s+(?:and\s+(?:place|put|then|close|open|turn)|in|on|onto|into|to|from|next\s+to|"
                   r"between|at|of|top\s+of)\b")


def _phrase(instruction: str) -> str:
    low = str(instruction).lower()
    m = re.match(_VERB, low)
    rest = low[m.end():] if m else low
    stop = _STOP.search(rest)
    return rest[: stop.start()] if stop else rest


def target_nouns(instruction: str) -> list[str]:
    """Vocabulary hits inside the manipulated-object phrase, in sentence order."""
    for scope in (_phrase(instruction), str(instruction).lower()):
        words = [w.rstrip("s") if w.rstrip("s") in NOUNS else w for w in re.findall(r"[a-z]+", scope)]
        hits = [w for w in dict.fromkeys(words) if w in NOUNS]
        if hits:
            return hits
    return []


def target_noun(instruction: str, fallback: str = "") -> str:
    """Head noun of the manipulated object (last hit), preferring a non-modifier word."""
    hits = target_nouns(instruction)
    if not hits:
        return fallback
    heads = [w for w in hits if w not in _MODIFIERS]
    return heads[-1] if heads else hits[-1]
