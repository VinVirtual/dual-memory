"""Short-term working memory: recent turns within a token budget."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .summarize import Summarizer
from .text import TokenCounter, approx_tokens
from .types import Turn

EvictHook = Callable[[Sequence[Turn]], None]


class WorkingMemory:
    """Recent turns, bounded by tokens, with a running summary of older ones.

    When the turns plus the summary go over ``max_tokens``, the oldest turns
    are evicted until the total is at or below ``target_ratio * max_tokens``.
    Evicted turns are folded into the summary (at most ``summary_tokens``)
    and passed to ``on_evict``, which is where ``DualMemory`` hands them to
    long-term memory. The newest turn is never evicted, even when it alone
    is over budget.
    """

    def __init__(
        self,
        summarizer: Summarizer,
        max_tokens: int = 1024,
        summary_tokens: int = 256,
        target_ratio: float = 0.6,
        counter: TokenCounter = approx_tokens,
        on_evict: EvictHook | None = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0 <= summary_tokens < max_tokens:
            raise ValueError("summary_tokens must be in [0, max_tokens)")
        if not 0.0 < target_ratio <= 1.0:
            raise ValueError("target_ratio must be in (0, 1]")
        self.summarizer = summarizer
        self.max_tokens = max_tokens
        self.summary_tokens = summary_tokens
        self.target_ratio = target_ratio
        self.count = counter
        self.on_evict = on_evict
        self._turns: list[Turn] = []
        self._costs: list[int] = []
        self.summary = ""
        self.evictions = 0

    @property
    def turns(self) -> tuple[Turn, ...]:
        return tuple(self._turns)

    def tokens(self) -> int:
        return sum(self._costs) + (self.count(self.summary) if self.summary else 0)

    def add(self, turn: Turn) -> list[Turn]:
        """Add a turn; returns the turns evicted by this call (often none)."""
        self._turns.append(turn)
        self._costs.append(self.count(turn.content) + 4)  # +4: role and separators
        if self.tokens() <= self.max_tokens:
            return []
        return self._evict()

    def _evict(self) -> list[Turn]:
        target = int(self.max_tokens * self.target_ratio)
        summary_cost = min(self.summary_tokens, self.count(self.summary) if self.summary else 0)
        evicted: list[Turn] = []
        while len(self._turns) > 1 and sum(self._costs) + max(summary_cost, 1) > target:
            evicted.append(self._turns.pop(0))
            self._costs.pop(0)
            summary_cost = self.summary_tokens
        if evicted:
            self.summary = self.summarizer.summarize(self.summary, evicted, self.summary_tokens)
            self.evictions += 1
            if self.on_evict is not None:
                self.on_evict(evicted)
        return evicted

    def flush(self) -> list[Turn]:
        """Remove every turn (end of session). They go to ``on_evict``; the summary stays."""
        evicted, self._turns, self._costs = self._turns, [], []
        if evicted and self.on_evict is not None:
            self.on_evict(evicted)
        return evicted

    def clear(self) -> None:
        self._turns, self._costs, self.summary = [], [], ""

    def messages(self) -> list[dict[str, str]]:
        """Chat messages: the summary (as a system message, if any), then the turns."""
        out: list[dict[str, str]] = []
        if self.summary:
            out.append({"role": "system", "content": "Summary of earlier conversation:\n" + self.summary})
        out.extend({"role": t.role, "content": t.content} for t in self._turns)
        return out
