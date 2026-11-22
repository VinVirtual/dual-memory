"""Summarisers fold old turns into a short running summary."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Protocol

from .extract import RuleExtractor, split_sentences
from .text import TokenCounter, approx_tokens, content_words
from .types import Turn


class Summarizer(Protocol):
    def summarize(self, previous: str, turns: Sequence[Turn], max_tokens: int) -> str: ...


def date_tag(ts: float | None) -> str:
    """``[YYYY-MM-DD] `` for a timestamp, or an empty string for ``None``."""
    return f"[{time.strftime('%Y-%m-%d', time.gmtime(ts))}] " if ts is not None else ""


class RuleSummarizer:
    """Extractive, deterministic summariser.

    It keeps whole user sentences, each tagged with its date and speaker.
    Sentences that state a fact (according to ``RuleExtractor``) come first,
    then other user sentences with at least ``min_content_words`` content
    words. Lines from the previous summary are kept, and when the result is
    over ``max_tokens`` the oldest lines are dropped first. It does not merge
    or rewrite facts, so a changed fact appears twice until the older line
    falls off.
    """

    def __init__(
        self, user_name: str = "User", min_content_words: int = 4, counter: TokenCounter = approx_tokens
    ) -> None:
        self.user_name = user_name
        self.min_content_words = min_content_words
        self.count = counter
        self._extractor = RuleExtractor()

    def _select(self, turns: Sequence[Turn]) -> list[str]:
        facts: list[str] = []
        others: list[str] = []
        for turn in turns:
            if turn.role != "user":
                continue
            for sentence in split_sentences(turn.content):
                line = f"{date_tag(turn.ts)}{self.user_name}: {sentence}"
                if self._extractor.extract_sentence(sentence, self.user_name):
                    facts.append(line)
                elif len(content_words(sentence)) >= self.min_content_words:
                    others.append(line)
        return facts + others

    def summarize(self, previous: str, turns: Sequence[Turn], max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        lines = [ln for ln in previous.splitlines() if ln.strip()]
        for line in self._select(turns):
            if line not in lines:
                lines.append(line)
        # Keep fact lines over chatter, then newest over oldest, within budget.
        is_fact = [bool(self._fact_line(ln)) for ln in lines]
        order = sorted(range(len(lines)), key=lambda i: (not is_fact[i], -i))
        kept: set[int] = set()
        used = 0
        for i in order:
            cost = self.count(lines[i]) + 1
            if used + cost > max_tokens:
                continue
            kept.add(i)
            used += cost
        return "\n".join(lines[i] for i in range(len(lines)) if i in kept)

    def _fact_line(self, line: str) -> bool:
        body = line.split(": ", 1)[-1]
        return bool(self._extractor.extract_sentence(body, self.user_name))
