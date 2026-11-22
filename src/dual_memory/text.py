"""Small text helpers: token counting, word splitting and value normalising."""

from __future__ import annotations

import re
from collections.abc import Callable

TokenCounter = Callable[[str], int]

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
_PIECE_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# A small stop-word list, used only for keyword queries and near-duplicate checks.
STOP_WORDS: frozenset[str] = frozenset(
    """a an and are as at be been but by did do does for from had has have he her his
    how i i'm in is it its me my of on or our she so that the their them they this to
    was we were what when where which who whom why will with you your now before
    after then than there here about into over just also very really""".split()
)


def approx_tokens(text: str) -> int:
    """Estimate the number of LLM tokens in ``text`` without a tokenizer.

    Words count as one token, plus one extra token per four characters beyond
    the first four (long words split into several pieces), and punctuation
    counts as one token each. It is close to BPE counts for English prose and
    never returns less than the number of words.
    """
    count = 0
    for piece in _PIECE_RE.findall(text):
        count += 1 + max(0, (len(piece) - 4) // 4) if piece[0].isalnum() else 1
    return count


def words(text: str) -> list[str]:
    """Lower-cased words of ``text`` (letters, digits and inner apostrophes)."""
    return [w.lower() for w in _WORD_RE.findall(text)]


def content_words(text: str) -> list[str]:
    """Words of ``text`` without stop words."""
    return [w for w in words(text) if w not in STOP_WORDS]


def normalize_value(value: str) -> str:
    """Normalise an answer or slot value for exact comparison."""
    v = value.strip().lower()
    v = re.sub(r"^(the|a|an)\s+", "", v)
    v = re.sub(r"[^\w\s'-]", "", v)
    return re.sub(r"\s+", " ", v).strip()


def jaccard(a: str, b: str) -> float:
    """Jaccard overlap of the content words of two strings."""
    sa, sb = set(content_words(a)), set(content_words(b))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def fts_query(text: str, max_terms: int = 12) -> str:
    """Turn free text into a safe SQLite FTS5 ``OR`` query.

    Every term is quoted, so user text can never inject FTS5 operators.
    Returns an empty string when the text has no usable words.
    """
    seen: list[str] = []
    for w in content_words(text):
        w = w.replace('"', "")
        if w and w not in seen:
            seen.append(w)
        if len(seen) >= max_terms:
            break
    return " OR ".join(f'"{w}"' for w in seen)
