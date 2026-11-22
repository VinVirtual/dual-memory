"""``DualMemory``: working memory and long-term memory behind one small API."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from types import TracebackType

from .embedders import Embedder, HashingEmbedder
from .extract import Extractor, RuleExtractor
from .llm import LLM, LLMExtractor, LLMSummarizer
from .longterm import LongTermConfig, LongTermMemory
from .store import SQLiteStore
from .summarize import RuleSummarizer, Summarizer, date_tag
from .text import TokenCounter, approx_tokens
from .types import Fact, MemoryItem, Provenance, RememberResult, Role, ScoredMemory, Turn
from .working import WorkingMemory


def render_item(item: MemoryItem) -> list[str]:
    """Context lines for one memory, each tagged with its date."""
    if item.kind == "episode":
        return [ln for ln in item.text.splitlines() if ln.strip()]
    line = f"{date_tag(item.event_time)}{item.text}"
    if item.status == "superseded" and item.superseded_at:
        line += f" (no longer true since {date_tag(item.superseded_at).strip()[1:-1]})"
    return [line]


class DualMemory:
    """Short-term working memory plus long-term SQLite memory for one user.

    Turns go into working memory. When it overflows, or when a session ends,
    the oldest turns are summarised and their facts are written to long-term
    memory (deduplicated, with superseding of changed facts). ``context``
    builds a memory block for a prompt from long-term memory, and
    ``messages`` adds the working memory after it.
    """

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        user_name: str = "User",
        embedder: Embedder | None = None,
        llm: LLM | None = None,
        extractor: Extractor | None = None,
        summarizer: Summarizer | None = None,
        working_tokens: int = 1024,
        summary_tokens: int = 256,
        episode_tokens: int = 160,
        config: LongTermConfig | None = None,
        counter: TokenCounter = approx_tokens,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not user_name.strip():
            raise ValueError("user_name must not be empty")
        self.user_name = user_name
        self.embedder: Embedder = embedder or HashingEmbedder()
        self.clock = clock
        self.count = counter
        self.episode_tokens = episode_tokens
        if extractor is None:
            extractor = LLMExtractor(llm) if llm is not None else RuleExtractor()
        if summarizer is None:
            summarizer = LLMSummarizer(llm, user_name) if llm is not None else RuleSummarizer(user_name)
        self.summarizer = summarizer
        self.store = SQLiteStore(path, embedder_name=self.embedder.name, dim=self.embedder.dim)
        self.long = LongTermMemory(self.store, self.embedder, extractor, config)
        self.working = WorkingMemory(
            summarizer, working_tokens, summary_tokens, counter=counter, on_evict=self._on_evict
        )
        self.session_id = "s1"
        self._session_no = 1
        self._turn_index = 0
        self._session_turns: list[Turn] = []
        self.last_results: list[RememberResult] = []

    # -- writing -----------------------------------------------------------
    def add(self, role: Role, content: str, ts: float | None = None) -> list[RememberResult]:
        """Add one turn. Returns what long-term memory did, if anything was evicted."""
        turn = Turn(role, content, self.clock() if ts is None else ts, self.session_id, self._turn_index)
        self._turn_index += 1
        self._session_turns.append(turn)
        self.last_results = []
        self.working.add(turn)
        return self.last_results

    def _on_evict(self, turns: Sequence[Turn]) -> None:
        self.last_results = self.long.ingest(turns, self.user_name, now=self.clock())

    def end_session(self, next_session_id: str | None = None) -> list[RememberResult]:
        """Move the rest of working memory to long-term memory and store an episode.

        The episode is a summary of the whole session (kind ``episode``).
        The working-memory summary is cleared, so the next session starts fresh.
        """
        results: list[RememberResult] = []
        self.last_results = []
        self.working.flush()
        results.extend(self.last_results)
        if self._session_turns and self.episode_tokens > 0:
            text = self.summarizer.summarize("", self._session_turns, self.episode_tokens)
            if text.strip():
                last = self._session_turns[-1]
                episode = Fact(
                    text=text,
                    kind="episode",
                    importance=0.4,
                    event_time=last.ts,
                    provenance=Provenance(self.session_id, last.index, ""),
                )
                results.append(self.long.remember(episode, now=self.clock()))
        self.working.clear()
        self._session_turns = []
        self._session_no += 1
        self.session_id = next_session_id or f"s{self._session_no}"
        self._turn_index = 0
        return results

    def remember(self, fact: Fact | str) -> RememberResult:
        """Store a fact directly (a plain string becomes an unslotted fact)."""
        if isinstance(fact, str):
            fact = Fact(text=fact, event_time=self.clock())
        return self.long.remember(fact, now=self.clock())

    def forget(self, item_id: int) -> bool:
        return self.long.forget(item_id)

    # -- reading -----------------------------------------------------------
    def recall(self, query: str, k: int = 5, include_superseded: bool = False) -> list[ScoredMemory]:
        return self.long.search(query, k, now=self.clock(), include_superseded=include_superseded)

    def context(self, query: str, k: int = 8, max_tokens: int = 512, include_history: bool = True) -> str:
        """A dated, plain-text memory block for ``query``, at most ``max_tokens``.

        With ``include_history`` every retrieved fact brings the earlier
        values of its slot, marked "no longer true since ...", so a model
        can answer questions like "where did they live before?".
        """
        lines: list[str] = []
        seen_ids: set[int] = set()
        for hit in self.recall(query, k):
            chain = [hit.item]
            if include_history and hit.item.attribute:
                chain = self.long.history(hit.item.subject, hit.item.attribute)
            for item in chain:
                if item.id in seen_ids:
                    continue
                seen_ids.add(item.id)
                for line in render_item(item):
                    if line not in lines:
                        lines.append(line)
        out: list[str] = []
        used = 0
        for line in lines:
            cost = self.count(line) + 1
            if used + cost > max_tokens:
                break
            out.append(line)
            used += cost
        return "\n".join(out)

    def messages(self, query: str, k: int = 8, max_tokens: int = 512) -> list[dict[str, str]]:
        """Chat messages: a memory system message, then working memory."""
        block = self.context(query, k, max_tokens)
        head = [{"role": "system", "content": "Long-term memory:\n" + block}] if block else []
        return head + self.working.messages()

    def stats(self) -> dict[str, int]:
        return {
            "memories": self.store.count(),
            "active": self.store.count(active_only=True),
            "working_turns": len(self.working.turns),
            "working_tokens": self.working.tokens(),
            "bytes": self.store.size_bytes(),
        }

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> DualMemory:
        return self

    def __exit__(
        self, et: type[BaseException] | None, ev: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()
