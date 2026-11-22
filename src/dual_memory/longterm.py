"""Long-term memory: remember facts with dedupe and supersede, recall by hybrid score."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .embedders import Embedder
from .extract import Extractor, RuleExtractor
from .store import SQLiteStore
from .text import jaccard, normalize_value
from .types import Fact, MemoryItem, RememberResult, ScoredMemory, ScoreParts, Turn

DAY = 86400.0


@dataclass(frozen=True)
class RetrievalWeights:
    """Weights of the four parts of the hybrid score. They need not sum to 1."""

    vector: float = 0.55
    keyword: float = 0.25
    recency: float = 0.10
    importance: float = 0.10

    def __post_init__(self) -> None:
        values = (self.vector, self.keyword, self.recency, self.importance)
        if any(v < 0 for v in values) or sum(values) == 0:
            raise ValueError("weights must be non-negative and not all zero")


@dataclass(frozen=True)
class LongTermConfig:
    weights: RetrievalWeights = field(default_factory=RetrievalWeights)
    half_life_days: float = 30.0
    candidates: int = 50
    dedupe_threshold: float = 0.92
    supersede: bool = True
    reinforce_step: float = 0.05

    def __post_init__(self) -> None:
        if self.half_life_days <= 0:
            raise ValueError("half_life_days must be positive")
        if not 0.0 < self.dedupe_threshold <= 1.0:
            raise ValueError("dedupe_threshold must be in (0, 1]")
        if self.candidates < 1:
            raise ValueError("candidates must be at least 1")


def _minmax(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class LongTermMemory:
    """Facts and episodes in SQLite, with provenance and a supersede history."""

    def __init__(
        self,
        store: SQLiteStore,
        embedder: Embedder,
        extractor: Extractor | None = None,
        config: LongTermConfig | None = None,
    ) -> None:
        if store.dim != embedder.dim:
            raise ValueError(f"store dim {store.dim} != embedder dim {embedder.dim}")
        self.store = store
        self.embedder = embedder
        self.extractor: Extractor = extractor or RuleExtractor()
        self.config = config or LongTermConfig()

    # -- writing -----------------------------------------------------------
    def ingest(self, turns: Sequence[Turn], user_name: str, now: float | None = None) -> list[RememberResult]:
        """Extract facts from ``turns`` and remember each one."""
        return [self.remember(f, now=now) for f in self.extractor.extract(turns, user_name)]

    def remember(self, fact: Fact, now: float | None = None) -> RememberResult:
        """Store one fact.

        - Same slot, same value: the existing memory is *reinforced* (mention
          count, last-seen time and importance go up). Nothing new is stored.
        - Exclusive slot, different value, and the new fact is not older than
          the current one: the old memory is marked *superseded* by the new
          one (both are kept, linked both ways, with the time of the change).
        - No slot: a near-duplicate (cosine and word overlap above the
          threshold) is reinforced; anything else is inserted.
        """
        now = time.time() if now is None else now
        vec = self.embedder.embed([fact.text])[0]
        if fact.has_slot:
            return self._remember_slot(fact, vec, now)
        dup = self._near_duplicate(fact, vec)
        if dup is not None:
            return RememberResult("reinforced", self._reinforce(dup, fact, now))
        return RememberResult("inserted", self.store.insert(fact, vec, now))

    def _reinforce(self, item: MemoryItem, fact: Fact, now: float) -> MemoryItem:
        importance = max(item.importance, fact.importance) + self.config.reinforce_step
        return self.store.reinforce(item.id, fact.event_time or now, importance)

    def _remember_slot(self, fact: Fact, vec: np.ndarray, now: float) -> RememberResult:  # type: ignore[type-arg]
        active = self.store.find_slot(fact.subject, fact.attribute)
        value = normalize_value(fact.value)
        for item in active:
            if normalize_value(item.value) == value:
                return RememberResult("reinforced", self._reinforce(item, fact, now))
        if not (fact.exclusive and self.config.supersede and active):
            return RememberResult("inserted", self.store.insert(fact, vec, now))
        current = active[-1]
        if (fact.event_time or now) < current.event_time:
            # An older statement arriving late: keep it as history, already superseded.
            item = self.store.insert(fact, vec, now)
            self.store.mark_superseded(item.id, current.id, current.event_time)
            refreshed = self.store.get(item.id)
            assert refreshed is not None
            return RememberResult("inserted", refreshed)
        item = self.store.insert(fact, vec, now, supersedes=current.id)
        for old in active:
            self.store.mark_superseded(old.id, item.id, fact.event_time or now)
        replaced = self.store.get(current.id)
        return RememberResult("superseded", item, replaced)

    def _near_duplicate(self, fact: Fact, vec: np.ndarray) -> MemoryItem | None:  # type: ignore[type-arg]
        for item_id, sim in self.store.vector_search(vec, 5, kinds=(fact.kind,)):
            if sim < self.config.dedupe_threshold:
                break
            item = self.store.get(item_id)
            if item is not None and not item.attribute and jaccard(item.text, fact.text) >= 0.6:
                return item
        return None

    def forget(self, item_id: int) -> bool:
        return self.store.delete(item_id)

    # -- reading -----------------------------------------------------------
    def history(self, subject: str, attribute: str) -> list[MemoryItem]:
        """Every value a slot has had, oldest first, superseded ones included."""
        return self.store.find_slot(subject, attribute, active_only=False)

    def search(
        self,
        query: str,
        k: int = 5,
        now: float | None = None,
        include_superseded: bool = False,
        kinds: tuple[str, ...] = (),
    ) -> list[ScoredMemory]:
        """Hybrid search: cosine + BM25 + recency + importance, top ``k``."""
        if k <= 0 or not query.strip():
            return []
        now = time.time() if now is None else now
        cfg = self.config
        active_only = not include_superseded
        qvec = self.embedder.embed([query])[0]
        vec = dict(self.store.vector_search(qvec, cfg.candidates, active_only, kinds))
        kw = _minmax(dict(self.store.keyword_search(query, cfg.candidates, active_only, kinds)))
        ids = set(vec) | set(kw)
        items = self.store.get_many(ids)
        missing_vec = [i for i in ids if i not in vec]
        if missing_vec:
            # Keyword-only hits still deserve a real cosine score.
            mat = self.embedder.embed([items[i].text for i in missing_vec])
            vec.update({i: float(s) for i, s in zip(missing_vec, mat @ qvec, strict=True)})
        lam = math.log(2) / (cfg.half_life_days * DAY)
        w = cfg.weights
        out: list[ScoredMemory] = []
        for i, item in items.items():
            parts = ScoreParts(
                vector=max(0.0, vec.get(i, 0.0)),
                keyword=kw.get(i, 0.0),
                recency=math.exp(-lam * max(0.0, now - item.last_seen)),
                importance=item.importance,
            )
            score = (
                w.vector * parts.vector
                + w.keyword * parts.keyword
                + w.recency * parts.recency
                + w.importance * parts.importance
            )
            out.append(ScoredMemory(item, score, parts))
        out.sort(key=lambda s: (-s.score, -s.item.id))
        return out[:k]
