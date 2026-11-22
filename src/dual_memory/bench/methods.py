"""The memory methods the benchmark compares.

Every method sees the same sessions in the same order and must produce a
plain-text context of at most ``budget`` tokens for each question. The same
reader then answers from that context, so differences in accuracy come
from what each method chose to keep and retrieve.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from ..embedders import Embedder
from ..extract import Extractor, split_sentences
from ..longterm import LongTermConfig, RetrievalWeights
from ..memory import DualMemory, render_item
from ..store import SQLiteStore
from ..summarize import Summarizer, date_tag
from ..text import TokenCounter, approx_tokens
from ..types import Fact, Provenance, Turn


@dataclass
class Retrieved:
    """What a method hands the reader: the context, plus ranked units for recall@k."""

    context: str
    units: list[str] = field(default_factory=list)


class Method(Protocol):
    name: str

    def ingest_session(self, turns: Sequence[Turn]) -> None: ...

    def retrieve(self, question: str, now: float) -> Retrieved: ...

    def size_bytes(self) -> int: ...

    def close(self) -> None: ...


def render_turn(turn: Turn, user_name: str) -> str:
    who = user_name if turn.role == "user" else "Assistant"
    return f"{date_tag(turn.ts)}{who}: {turn.content}"


def fit(
    lines: Sequence[str], budget: int, counter: TokenCounter = approx_tokens, skip_overflow: bool = False
) -> list[str]:
    """Keep lines in order until the token budget is used.

    With ``skip_overflow`` a line that does not fit is skipped and later,
    shorter lines may still fill the remaining budget.
    """
    out: list[str] = []
    used = 0
    for line in lines:
        cost = counter(line) + 1
        if used + cost > budget:
            if skip_overflow:
                continue
            break
        out.append(line)
        used += cost
    return out


class NoMemory:
    """Baseline: the model gets no memory at all."""

    name = "no-memory"

    def __init__(self, user_name: str, budget: int) -> None:
        self.user_name = user_name

    def ingest_session(self, turns: Sequence[Turn]) -> None:
        return None

    def retrieve(self, question: str, now: float) -> Retrieved:
        return Retrieved("")

    def size_bytes(self) -> int:
        return 0

    def close(self) -> None:
        return None


class FullHistory:
    """Baseline: the whole transcript, cut to the most recent ``budget`` tokens."""

    name = "full-history"

    def __init__(self, user_name: str, budget: int) -> None:
        self.user_name = user_name
        self.budget = budget
        self.lines: list[str] = []

    def ingest_session(self, turns: Sequence[Turn]) -> None:
        self.lines.extend(render_turn(t, self.user_name) for t in turns)

    def retrieve(self, question: str, now: float) -> Retrieved:
        kept = list(reversed(fit(list(reversed(self.lines)), self.budget)))
        return Retrieved("\n".join(kept))

    def size_bytes(self) -> int:
        return sum(len(ln.encode()) + 1 for ln in self.lines)

    def close(self) -> None:
        return None


class VectorRAG:
    """Baseline: plain vector retrieval over the raw conversation.

    Variants (all embed the text with the same embedder and rank by cosine):

    - ``naive-rag``: every turn, user and assistant, top ``k`` turns. With
      short turns this leaves much of the budget unused.
    - ``rag-user-turns``: user turns only, top ``k``.
    - ``rag-budget-filled``: user turns only, ranked, added until the
      budget is full.
    - ``rag-sentences``: user turns split into sentences, ranked, added
      until the budget is full.
    """

    def __init__(
        self,
        user_name: str,
        budget: int,
        embedder: Embedder,
        k: int = 8,
        *,
        name: str = "naive-rag",
        user_only: bool = False,
        fill_budget: bool = False,
        sentences: bool = False,
        candidates: int = 60,
    ) -> None:
        self.name = name
        self.user_name = user_name
        self.budget = budget
        self.k = k
        self.user_only = user_only
        self.fill_budget = fill_budget
        self.sentences = sentences
        self.candidates = candidates
        self.embedder = embedder
        self.store = SQLiteStore(":memory:", embedder_name=embedder.name, dim=embedder.dim)

    def _chunks(self, turns: Sequence[Turn]) -> list[tuple[Turn, str]]:
        out: list[tuple[Turn, str]] = []
        for t in turns:
            if self.user_only and t.role != "user":
                continue
            parts = split_sentences(t.content) if self.sentences else [t.content]
            out.extend((t, p) for p in parts if p.strip())
        return out

    def ingest_session(self, turns: Sequence[Turn]) -> None:
        chunks = self._chunks(turns)
        if not chunks:
            return
        vecs = self.embedder.embed([c for _, c in chunks])
        for (turn, chunk), vec in zip(chunks, vecs, strict=True):
            text = render_turn(Turn(turn.role, chunk, turn.ts, turn.session_id, turn.index), self.user_name)
            fact = Fact(
                text=text,
                kind="turn",
                event_time=turn.ts,
                provenance=Provenance(turn.session_id, turn.index, ""),
            )
            self.store.insert(fact, vec, now=turn.ts)

    def retrieve(self, question: str, now: float) -> Retrieved:
        q = self.embedder.embed([question])[0]
        hits = self.store.vector_search(q, self.candidates if self.fill_budget else self.k)
        items = self.store.get_many(i for i, _ in hits)
        lines = fit([items[i].text for i, _ in hits], self.budget, skip_overflow=self.fill_budget)
        return Retrieved("\n".join(lines), lines)

    def size_bytes(self) -> int:
        return self.store.size_bytes()

    def close(self) -> None:
        self.store.close()


RAG_VARIANTS: dict[str, dict[str, bool]] = {
    "naive-rag": {},
    "rag-user-turns": {"user_only": True},
    "rag-budget-filled": {"user_only": True, "fill_budget": True},
    "rag-sentences": {"user_only": True, "fill_budget": True, "sentences": True},
}


class SummarizeEverything:
    """Baseline: one running summary, updated after every session, used as the context."""

    name = "summarize-all"

    def __init__(self, user_name: str, budget: int, summarizer: Summarizer) -> None:
        self.budget = budget
        self.summarizer = summarizer
        self.summary = ""

    def ingest_session(self, turns: Sequence[Turn]) -> None:
        self.summary = self.summarizer.summarize(self.summary, turns, self.budget)

    def retrieve(self, question: str, now: float) -> Retrieved:
        return Retrieved("\n".join(fit(self.summary.splitlines(), self.budget)))

    def size_bytes(self) -> int:
        return len(self.summary.encode())

    def close(self) -> None:
        return None


class DualMemoryMethod:
    """This library: working memory + long-term facts/episodes with hybrid retrieval."""

    def __init__(
        self,
        user_name: str,
        budget: int,
        embedder: Embedder,
        *,
        k: int = 8,
        name: str = "dual-memory",
        config: LongTermConfig | None = None,
        extractor: Extractor | None = None,
        summarizer: Summarizer | None = None,
        include_history: bool = True,
        kinds: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.budget = budget
        self.k = k
        self.include_history = include_history
        self.kinds = kinds
        self._now = 0.0
        self.mem = DualMemory(
            user_name=user_name,
            embedder=embedder,
            extractor=extractor,
            summarizer=summarizer,
            config=config,
            clock=lambda: self._now,
        )

    def ingest_session(self, turns: Sequence[Turn]) -> None:
        for t in turns:
            self._now = t.ts if t.ts is not None else self._now
            self.mem.add(t.role, t.content, ts=t.ts)
        self.mem.end_session()

    def retrieve(self, question: str, now: float) -> Retrieved:
        self._now = now
        hits = self.mem.recall(question, self.k, kinds=self.kinds, relevant_only=True)
        units = ["\n".join(render_item(h.item)) for h in hits]
        ctx = self.mem.context(
            question, self.k, self.budget, include_history=self.include_history, kinds=self.kinds
        )
        return Retrieved(ctx, units)

    def size_bytes(self) -> int:
        return self.mem.store.size_bytes()

    def close(self) -> None:
        self.mem.close()


@dataclass(frozen=True)
class DualVariant:
    config: LongTermConfig = field(default_factory=LongTermConfig)
    include_history: bool = True
    kinds: tuple[str, ...] = ()


def dual_variants() -> dict[str, DualVariant]:
    """dual-memory and its ablations."""
    return {
        "dual-memory": DualVariant(),
        "dual: no supersede": DualVariant(LongTermConfig(supersede=False)),
        "dual: vector only": DualVariant(LongTermConfig(weights=RetrievalWeights(1.0, 0.0, 0.0, 0.0))),
        "dual: keyword only": DualVariant(LongTermConfig(weights=RetrievalWeights(0.0, 1.0, 0.0, 0.0))),
        "dual: no history": DualVariant(include_history=False),
        "dual: facts only": DualVariant(kinds=("fact",)),
        "dual: episodes only": DualVariant(kinds=("episode",)),
    }
