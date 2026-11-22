"""Plain data types shared by every part of the library."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Role = Literal["user", "assistant", "system"]
MemoryKind = Literal["fact", "episode", "turn"]
MemoryStatus = Literal["active", "superseded", "deleted"]

VALID_ROLES: tuple[str, ...] = ("user", "assistant", "system")
VALID_KINDS: tuple[str, ...] = ("fact", "episode", "turn")


@dataclass(frozen=True)
class Turn:
    """One message in a conversation.

    ``ts`` is a Unix timestamp (seconds), or ``None`` when unknown. ``session_id`` and ``index`` say where
    the turn came from, so memories built from it can point back to it.
    """

    role: Role
    content: str
    ts: float | None = None
    session_id: str = ""
    index: int = -1

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}, got {self.role!r}")
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")


@dataclass(frozen=True)
class Provenance:
    """Where a memory came from: a session, a turn and the text it was read from."""

    session_id: str = ""
    turn_index: int = -1
    source_text: str = ""


@dataclass(frozen=True)
class Fact:
    """A candidate memory produced by an extractor, before it is stored.

    A fact with ``subject`` and ``attribute`` fills a *slot* (for example
    ``("Mira", "city")``). When ``exclusive`` is true the slot holds one value at
    a time, so a new value supersedes the old one. Facts without a slot are
    deduplicated by similarity only.
    """

    text: str
    subject: str = ""
    attribute: str = ""
    value: str = ""
    exclusive: bool = True
    importance: float = 0.5
    kind: MemoryKind = "fact"
    event_time: float | None = None
    provenance: Provenance = field(default_factory=Provenance)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("fact text must not be empty")
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError("importance must be between 0 and 1")
        if self.kind not in VALID_KINDS:
            raise ValueError(f"kind must be one of {VALID_KINDS}")

    @property
    def has_slot(self) -> bool:
        return bool(self.subject and self.attribute)


@dataclass(frozen=True)
class MemoryItem:
    """A memory as stored in the long-term store."""

    id: int
    kind: MemoryKind
    text: str
    subject: str
    attribute: str
    value: str
    exclusive: bool
    importance: float
    created_at: float
    event_time: float
    last_seen: float
    mentions: int
    status: MemoryStatus
    superseded_by: int | None
    superseded_at: float | None
    supersedes: int | None
    provenance: Provenance

    @property
    def is_active(self) -> bool:
        return self.status == "active"


@dataclass(frozen=True)
class ScoreParts:
    """The pieces of a hybrid retrieval score, each in [0, 1]."""

    vector: float = 0.0
    keyword: float = 0.0
    recency: float = 0.0
    importance: float = 0.0


@dataclass(frozen=True)
class ScoredMemory:
    item: MemoryItem
    score: float
    parts: ScoreParts


@dataclass(frozen=True)
class RememberResult:
    """What ``LongTermMemory.remember`` did with a fact."""

    action: Literal["inserted", "reinforced", "superseded"]
    item: MemoryItem
    replaced: MemoryItem | None = None
