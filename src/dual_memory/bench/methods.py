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
from ..extract import Extractor
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


def fit(lines: Sequence[str], budget: int, counter: TokenCounter = approx_tokens) -> list[str]:
    """Keep lines in order until the token budget is used."""
    out: list[str] = []
    used = 0
    for line in lines:
        cost = counter(line) + 1
        if used + cost > budget:
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
    """Baseline: embed every turn; retrieve the top ``k`` turns by cosine similarity."""

    name = "vector-rag"

    def __init__(self, user_name: str, budget: int, embedder: Embedder, k: int = 8) -> None:
        self.user_name = user_name
        self.budget = budget
        self.k = k
        self.embedder = embedder
        self.store = SQLiteStore(":memory:", embedder_name=embedder.name, dim=embedder.dim)

    def ingest_session(self, turns: Sequence[Turn]) -> None:
        texts = [render_turn(t, self.user_name) for t in turns]
        vecs = self.embedder.embed([t.content for t in turns])
        for turn, text, vec in zip(turns, texts, vecs, strict=True):
            fact = Fact(
                text=text,
                kind="turn",
                event_time=turn.ts,
                provenance=Provenance(turn.session_id, turn.index, ""),
            )
            self.store.insert(fact, vec, now=turn.ts)

    def retrieve(self, question: str, now: float) -> Retrieved:
        q = self.embedder.embed([question])[0]
        hits = self.store.vector_search(q, self.k)
        items = self.store.get_many(i for i, _ in hits)
        lines = fit([items[i].text for i, _ in hits], self.budget)
        return Retrieved("\n".join(lines), lines)

    def size_bytes(self) -> int:
        return self.store.size_bytes()

    def close(self) -> None:
        self.store.close()


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
    ) -> None:
        self.name = name
        self.budget = budget
        self.k = k
        self.include_history = include_history
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
            self._now = t.ts
            self.mem.add(t.role, t.content, ts=t.ts)
        self.mem.end_session()

    def retrieve(self, question: str, now: float) -> Retrieved:
        self._now = now
        hits = self.mem.recall(question, self.k)
        units = ["\n".join(render_item(h.item)) for h in hits]
        ctx = self.mem.context(question, self.k, self.budget, include_history=self.include_history)
        return Retrieved(ctx, units)

    def size_bytes(self) -> int:
        return self.mem.store.size_bytes()

    def close(self) -> None:
        self.mem.close()


def dual_variants() -> dict[str, tuple[LongTermConfig, bool]]:
    """Ablations of dual-memory: name -> (config, include_history)."""
    return {
        "dual-memory": (LongTermConfig(), True),
        "dual: no supersede": (LongTermConfig(supersede=False), True),
        "dual: vector only": (LongTermConfig(weights=RetrievalWeights(1.0, 0.0, 0.0, 0.0)), True),
        "dual: keyword only": (LongTermConfig(weights=RetrievalWeights(0.0, 1.0, 0.0, 0.0)), True),
        "dual: no history": (LongTermConfig(), False),
    }
