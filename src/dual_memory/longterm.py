"""Long-term memory: remember facts with dedupe and supersede, recall by hybrid score."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from .embedders import Embedder, Vectors
from .extract import Extractor, RuleExtractor
from .store import SQLiteStore, unit
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
    """Tuning for long-term memory.

    ``keyword_half``: a BM25 score of this size maps to 0.5 in the keyword
    part (``bm25 / (bm25 + keyword_half)``), so one weak match stays weak
    instead of being stretched to 1.0.
    ``min_cosine``: a memory with no keyword match needs at least this cosine
    similarity to be returned by ``search(..., relevant_only=True)``, which
    ``DualMemory.context`` uses so that unrelated memories do not pad a prompt.
    """

    weights: RetrievalWeights = field(default_factory=RetrievalWeights)
    half_life_days: float = 30.0
    candidates: int = 50
    dedupe_threshold: float = 0.92
    supersede: bool = True
    reinforce_step: float = 0.05
    keyword_half: float = 2.0
    min_cosine: float = 0.25

    def __post_init__(self) -> None:
        if self.half_life_days <= 0:
            raise ValueError("half_life_days must be positive")
        if not 0.0 < self.dedupe_threshold <= 1.0:
            raise ValueError("dedupe_threshold must be in (0, 1]")
        if self.candidates < 1:
            raise ValueError("candidates must be at least 1")
        if self.keyword_half <= 0:
            raise ValueError("keyword_half must be positive")
        if not -1.0 <= self.min_cosine <= 1.0:
            raise ValueError("min_cosine must be in [-1, 1]")


def saturate(score: float, half: float) -> float:
    """Map a non-negative BM25 score into [0, 1): ``score / (score + half)``."""
    score = max(0.0, score)
    return score / (score + half)


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

        - Same slot, same value as the current one, or as the historical value
          that was current at the fact's time: that memory is *reinforced*
          (mention count, last-seen time and importance go up).
        - Exclusive slot, new value, not older than the current one: the old
          memory is *superseded* (both kept, linked both ways, with the time).
        - Exclusive slot, new value, older than the current one (a statement
          arriving late): it is inserted as history and spliced into the
          chain at its place in time; the current value stays current.
        - No slot: a near-duplicate (cosine and word overlap above the
          threshold) is reinforced; anything else is inserted.
        """
        now = time.time() if now is None else now
        vec = unit(self.embedder.embed([fact.text])[0], self.store.dim)
        if fact.has_slot:
            return self._remember_slot(fact, vec, now)
        dup = self._near_duplicate(fact, vec)
        if dup is not None:
            return RememberResult("reinforced", self._reinforce(dup, fact, now))
        return RememberResult("inserted", self.store.insert(fact, vec, now))

    @staticmethod
    def _when(fact: Fact, now: float) -> float:
        return fact.event_time if fact.event_time is not None else now

    def _reinforce(self, item: MemoryItem, fact: Fact, now: float) -> MemoryItem:
        importance = max(item.importance, fact.importance) + self.config.reinforce_step
        return self.store.reinforce(item.id, self._when(fact, now), importance)

    def _remember_slot(self, fact: Fact, vec: Vectors, now: float) -> RememberResult:
        when = self._when(fact, now)
        value = normalize_value(fact.value)
        active = self.store.find_slot(fact.subject, fact.attribute)
        for item in active:
            if normalize_value(item.value) == value:
                return RememberResult("reinforced", self._reinforce(item, fact, now))
        if not (fact.exclusive and self.config.supersede and active):
            return RememberResult("inserted", self.store.insert(fact, vec, now))
        current = active[-1]
        if when < current.event_time:
            return self._insert_late(fact, vec, now, when)
        item = self.store.insert(fact, vec, now, supersedes=current.id)
        for old in active:
            self.store.mark_superseded(old.id, item.id, when)
        replaced = self.store.get(current.id)
        return RememberResult("superseded", item, replaced)

    def _insert_late(self, fact: Fact, vec: Vectors, now: float, when: float) -> RememberResult:
        """A statement older than the current value: reinforce or splice it into the history."""
        history = self.store.find_slot(fact.subject, fact.attribute, active_only=False)
        before = [h for h in history if h.event_time <= when]
        after = [h for h in history if h.event_time > when]
        if before and normalize_value(before[-1].value) == normalize_value(fact.value):
            return RememberResult("reinforced", self._reinforce(before[-1], fact, now))
        successor = after[0]
        if normalize_value(successor.value) == normalize_value(fact.value):
            # Same value as the next one, just stated earlier: one memory, not two.
            return RememberResult("reinforced", self._reinforce(successor, fact, now))
        predecessor = before[-1] if before else None
        item = self.store.insert(fact, vec, now, supersedes=predecessor.id if predecessor else None)
        self.store.set_links(
            item.id, status="superseded", superseded_by=successor.id, superseded_at=successor.event_time
        )
        self.store.set_links(successor.id, supersedes=item.id)
        if predecessor is not None:
            self.store.set_links(predecessor.id, superseded_by=item.id, superseded_at=when)
        refreshed = self.store.get(item.id)
        assert refreshed is not None
        return RememberResult("inserted", refreshed)

    def _near_duplicate(self, fact: Fact, vec: Vectors) -> MemoryItem | None:
        for item_id, sim in self.store.vector_search(vec, 5, kinds=(fact.kind,)):
            if sim < self.config.dedupe_threshold:
                break
            item = self.store.get(item_id)
            if item is not None and not item.attribute and jaccard(item.text, fact.text) >= 0.6:
                return item
        return None

    def forget(self, item_id: int, scrub: bool = True) -> bool:
        """Delete one memory. With ``scrub`` (default), copies of its text go too.

        Copies are: the source text kept on other memories read from the same
        turn (blanked), and episodes of the same session that mention the
        forgotten value or text (deleted). The file is then purged (see
        ``SQLiteStore.purge``).
        """
        item = self.store.get(item_id)
        if item is None:
            return False
        if scrub and item.provenance.session_id:
            needle = (item.value or item.text).casefold()
            for ep_id in self.store.ids_from_source(item.provenance.session_id, kinds=("episode",)):
                ep = self.store.get(ep_id)
                if ep is not None and ep.id != item.id and needle in ep.text.casefold():
                    self.store.delete(ep.id, purge=False)
            if item.provenance.turn_index >= 0:
                self.store.scrub_source(item.provenance.session_id, item.provenance.turn_index)
        return self.store.delete(item_id)

    def forget_source(self, session_id: str, turn_index: int) -> int:
        """Delete everything read from one turn, and the episode of its session.

        Use it when a user asks to erase something they said. Returns the
        number of memories deleted.
        """
        ids = self.store.ids_from_source(session_id, turn_index)
        ids += [i for i in self.store.ids_from_source(session_id, kinds=("episode",)) if i not in ids]
        for i in ids:
            self.store.delete(i, purge=False)
        self.store.scrub_source(session_id, turn_index)
        self.store.purge()
        return len(ids)

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
        relevant_only: bool = False,
    ) -> list[ScoredMemory]:
        """Hybrid search: cosine + BM25 + recency + importance, top ``k``.

        With ``relevant_only``, a memory must match a query word (BM25) or
        reach ``config.min_cosine``; recency and importance alone never
        qualify a memory.
        """
        if k <= 0 or not query.strip():
            return []
        now = time.time() if now is None else now
        cfg = self.config
        active_only = not include_superseded
        qvec = unit(self.embedder.embed([query])[0], self.store.dim)
        vec = dict(self.store.vector_search(qvec, cfg.candidates, active_only, kinds))
        raw_kw = dict(self.store.keyword_search(query, cfg.candidates, active_only, kinds))
        kw = {i: saturate(v, cfg.keyword_half) for i, v in raw_kw.items()}
        ids = set(vec) | set(kw)
        items = self.store.get_many(ids)
        missing_vec = [i for i in ids if i not in vec]
        if missing_vec:
            # Keyword-only hits still deserve a real cosine score.
            mat = self.embedder.embed([items[i].text for i in missing_vec])
            sims = [float(unit(row, self.store.dim) @ qvec) for row in mat]
            vec.update(dict(zip(missing_vec, sims, strict=True)))
        lam = math.log(2) / (cfg.half_life_days * DAY)
        w = cfg.weights
        out: list[ScoredMemory] = []
        for i, item in items.items():
            if relevant_only and i not in raw_kw and vec.get(i, 0.0) < cfg.min_cosine:
                continue
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
