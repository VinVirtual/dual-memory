"""SQLite storage for long-term memory: rows, embeddings and an FTS5 index."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any, cast

import numpy as np

from .embedders import Vectors
from .text import fts_query, slot_key
from .types import Fact, MemoryItem, MemoryKind, MemoryStatus, Provenance

SCHEMA_VERSION = "1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    subject TEXT NOT NULL DEFAULT '',
    attribute TEXT NOT NULL DEFAULT '',
    value TEXT NOT NULL DEFAULT '',
    subject_key TEXT NOT NULL DEFAULT '',
    attribute_key TEXT NOT NULL DEFAULT '',
    exclusive INTEGER NOT NULL DEFAULT 1,
    importance REAL NOT NULL,
    created_at REAL NOT NULL,
    event_time REAL NOT NULL,
    last_seen REAL NOT NULL,
    mentions INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'active',
    superseded_by INTEGER,
    superseded_at REAL,
    supersedes INTEGER,
    session_id TEXT NOT NULL DEFAULT '',
    turn_index INTEGER NOT NULL DEFAULT -1,
    source_text TEXT NOT NULL DEFAULT '',
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_slot ON memories (subject_key, attribute_key, status);
CREATE INDEX IF NOT EXISTS idx_status ON memories (status, kind);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    text, content='memories', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF text ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
END;
"""

_COLUMNS = (
    "id, kind, text, subject, attribute, value, exclusive, importance, created_at, event_time, "
    "last_seen, mentions, status, superseded_by, superseded_at, supersedes, session_id, "
    "turn_index, source_text"
)


def unit(vec: Any, dim: int) -> Vectors:
    """``vec`` as a flat float32 unit vector of length ``dim`` (zero stays zero)."""
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.shape[0] != dim:
        raise ValueError(f"embedding has dim {v.shape[0]}, store expects {dim}")
    if not np.all(np.isfinite(v)):
        raise ValueError("embedding contains NaN or infinite values")
    norm = float(np.linalg.norm(v))
    out: Vectors = v / norm if norm > 0 else v
    return out


class EmbedderMismatchError(RuntimeError):
    """The database was built with a different embedder than the one given."""


def _row_to_item(row: tuple[Any, ...]) -> MemoryItem:
    return MemoryItem(
        id=row[0],
        kind=cast(MemoryKind, row[1]),
        text=row[2],
        subject=row[3],
        attribute=row[4],
        value=row[5],
        exclusive=bool(row[6]),
        importance=row[7],
        created_at=row[8],
        event_time=row[9],
        last_seen=row[10],
        mentions=row[11],
        status=cast(MemoryStatus, row[12]),
        superseded_by=row[13],
        superseded_at=row[14],
        supersedes=row[15],
        provenance=Provenance(row[16], row[17], row[18]),
    )


class SQLiteStore:
    """One SQLite file (or ``":memory:"``) holding memories for one agent or user.

    The store refuses to open a file that was written with a different
    embedder (name or dimension), because mixing vector spaces silently
    ruins retrieval. Vectors are normalised to unit length on the way in,
    so cosine similarity is a plain dot product whatever the embedder does.

    A store is meant for one thread (``sqlite3``'s ``check_same_thread``
    stays on). Several processes or connections may share a file: the
    vector cache is refreshed when another connection commits.
    """

    def __init__(self, path: str | Path = ":memory:", *, embedder_name: str, dim: int) -> None:
        self.path = str(path)
        self.dim = dim
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA secure_delete = ON")  # forget() overwrites the bytes
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(_SCHEMA)
        self._check_meta(embedder_name, dim)
        self._version = 0
        self._cache_key: tuple[int, int, bool, tuple[str, ...]] | None = None
        self._cache: tuple[np.ndarray[Any, Any], Vectors] | None = None

    # -- lifecycle ---------------------------------------------------------
    def _check_meta(self, embedder_name: str, dim: int) -> None:
        meta = dict(self._conn.execute("SELECT key, value FROM meta").fetchall())
        if not meta:
            with self._conn:
                self._conn.executemany(
                    "INSERT INTO meta (key, value) VALUES (?, ?)",
                    [("schema", SCHEMA_VERSION), ("embedder", embedder_name), ("dim", str(dim))],
                )
            return
        if meta.get("embedder") != embedder_name or meta.get("dim") != str(dim):
            self._conn.close()
            raise EmbedderMismatchError(
                f"{self.path} was built with embedder {meta.get('embedder')!r} "
                f"(dim {meta.get('dim')}), not {embedder_name!r} (dim {dim})"
            )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SQLiteStore:
        return self

    def __exit__(
        self, et: type[BaseException] | None, ev: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()

    def _data_version(self) -> int:
        """Changes when another connection commits to the same file."""
        return int(self._conn.execute("PRAGMA data_version").fetchone()[0])

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._conn:
            yield
        self._version += 1

    # -- writes ------------------------------------------------------------
    def insert(
        self, fact: Fact, embedding: Vectors, now: float | None = None, supersedes: int | None = None
    ) -> MemoryItem:
        vec = unit(embedding, self.dim)
        now = time.time() if now is None else now
        event_time = fact.event_time if fact.event_time is not None else now
        with self.transaction():
            cur = self._conn.execute(
                "INSERT INTO memories (kind, text, subject, attribute, value, subject_key, "
                "attribute_key, exclusive, importance, created_at, event_time, last_seen, "
                "supersedes, session_id, turn_index, source_text, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    fact.kind,
                    fact.text,
                    fact.subject,
                    fact.attribute,
                    fact.value,
                    slot_key(fact.subject),
                    slot_key(fact.attribute),
                    int(fact.exclusive),
                    fact.importance,
                    now,
                    event_time,
                    event_time,
                    supersedes,
                    fact.provenance.session_id,
                    fact.provenance.turn_index,
                    fact.provenance.source_text,
                    vec.tobytes(),
                ),
            )
        item = self.get(cast(int, cur.lastrowid))
        assert item is not None
        return item

    def set_links(self, item_id: int, **fields: Any) -> None:
        """Update supersede links: status, supersedes, superseded_by, superseded_at."""
        allowed = {"status", "supersedes", "superseded_by", "superseded_at"}
        if not fields or not set(fields) <= allowed:
            raise ValueError(f"set_links accepts only {sorted(allowed)}")
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self.transaction():
            self._conn.execute(f"UPDATE memories SET {cols} WHERE id = ?", [*fields.values(), item_id])

    def mark_superseded(self, old_id: int, new_id: int, at: float) -> None:
        with self.transaction():
            self._conn.execute(
                "UPDATE memories SET status = 'superseded', superseded_by = ?, superseded_at = ? "
                "WHERE id = ?",
                (new_id, at, old_id),
            )

    def reinforce(self, item_id: int, seen_at: float, importance: float) -> MemoryItem:
        with self.transaction():
            self._conn.execute(
                "UPDATE memories SET mentions = mentions + 1, last_seen = MAX(last_seen, ?), "
                "importance = ? WHERE id = ?",
                (seen_at, min(1.0, max(0.0, importance)), item_id),
            )
        item = self.get(item_id)
        if item is None:
            raise KeyError(item_id)
        return item

    def delete(self, item_id: int, purge: bool = True) -> bool:
        """Hard-delete one memory and splice it out of its supersede chain.

        If B superseded A and C superseded B, deleting B links A straight to
        C. Deleting the current value does not revive the older one; it stays
        superseded. With ``purge`` the deleted text is also removed from the
        full-text index pages and the write-ahead log (see ``purge``).
        """
        item = self.get(item_id)
        if item is None:
            return False
        with self.transaction():
            self._conn.execute(
                "UPDATE memories SET superseded_by = ? WHERE superseded_by = ?", (item.superseded_by, item_id)
            )
            self._conn.execute(
                "UPDATE memories SET supersedes = ? WHERE supersedes = ?", (item.supersedes, item_id)
            )
            self._conn.execute("DELETE FROM memories WHERE id = ?", (item_id,))
        if purge:
            self.purge()
        return True

    def scrub_source(self, session_id: str, turn_index: int) -> int:
        """Blank the copied source text of every memory read from one turn."""
        with self.transaction():
            cur = self._conn.execute(
                "UPDATE memories SET source_text = '' WHERE session_id = ? AND turn_index = ?",
                (session_id, turn_index),
            )
        return cur.rowcount

    def ids_from_source(
        self, session_id: str, turn_index: int | None = None, kinds: tuple[str, ...] = ()
    ) -> list[int]:
        sql = "SELECT id FROM memories WHERE session_id = ?"
        args: list[Any] = [session_id]
        if turn_index is not None:
            sql += " AND turn_index = ?"
            args.append(turn_index)
        if kinds:
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            args.extend(kinds)
        return [int(r[0]) for r in self._conn.execute(sql + " ORDER BY id", args)]

    def purge(self, vacuum: bool = False) -> None:
        """Make deleted text unrecoverable from this file, as far as SQLite allows.

        Rebuilds the FTS5 index segments (old segments hold deleted terms),
        checkpoints and truncates the write-ahead log, and with ``vacuum``
        rewrites the whole file. ``secure_delete`` is on, so freed pages are
        zeroed. Copies outside this file (backups, logs, text already sent
        to a model) are out of reach.
        """
        with self.transaction():
            self._conn.execute("INSERT INTO memories_fts(memories_fts) VALUES ('rebuild')")
            self._conn.execute("INSERT INTO memories_fts(memories_fts) VALUES ('optimize')")
        if vacuum:
            self._conn.execute("VACUUM")
        if self.path != ":memory:":
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    # -- reads -------------------------------------------------------------
    def get(self, item_id: int) -> MemoryItem | None:
        row = self._conn.execute(f"SELECT {_COLUMNS} FROM memories WHERE id = ?", (item_id,)).fetchone()
        return _row_to_item(row) if row else None

    def get_many(self, ids: Iterable[int]) -> dict[int, MemoryItem]:
        ids = list(ids)
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self._conn.execute(f"SELECT {_COLUMNS} FROM memories WHERE id IN ({marks})", ids)
        return {r[0]: _row_to_item(r) for r in rows}

    def find_slot(self, subject: str, attribute: str, active_only: bool = True) -> list[MemoryItem]:
        sql = f"SELECT {_COLUMNS} FROM memories WHERE subject_key = ? AND attribute_key = ?"
        if active_only:
            sql += " AND status = 'active'"
        sql += " ORDER BY event_time, id"
        rows = self._conn.execute(sql, (slot_key(subject), slot_key(attribute)))
        return [_row_to_item(r) for r in rows]

    def all_items(self, active_only: bool = False, kinds: tuple[str, ...] = ()) -> list[MemoryItem]:
        sql, args = self._filter_sql(active_only, kinds)
        rows = self._conn.execute(f"SELECT {_COLUMNS} FROM memories{sql} ORDER BY id", args)
        return [_row_to_item(r) for r in rows]

    @staticmethod
    def _filter_sql(active_only: bool, kinds: tuple[str, ...]) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        args: list[Any] = []
        if active_only:
            clauses.append("status = 'active'")
        if kinds:
            clauses.append(f"kind IN ({','.join('?' * len(kinds))})")
            args.extend(kinds)
        return (" WHERE " + " AND ".join(clauses) if clauses else ""), args

    def _matrix(self, active_only: bool, kinds: tuple[str, ...]) -> tuple[np.ndarray[Any, Any], Vectors]:
        key = (self._version, self._data_version(), active_only, kinds)
        if self._cache_key == key and self._cache is not None:
            return self._cache
        sql, args = self._filter_sql(active_only, kinds)
        rows = self._conn.execute(f"SELECT id, embedding FROM memories{sql}", args).fetchall()
        ids = np.array([r[0] for r in rows], dtype=np.int64)
        mat = (
            np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), self.dim)
            if rows
            else np.zeros((0, self.dim), dtype=np.float32)
        )
        self._cache_key, self._cache = key, (ids, mat)
        return ids, mat

    def vector_search(
        self, query: Vectors, n: int, active_only: bool = True, kinds: tuple[str, ...] = ()
    ) -> list[tuple[int, float]]:
        """Top ``n`` memories by cosine similarity (vectors are unit length)."""
        ids, mat = self._matrix(active_only, kinds)
        if len(ids) == 0 or n <= 0:
            return []
        sims = mat @ unit(query, self.dim)
        n = min(n, len(ids))
        top = np.argpartition(-sims, n - 1)[:n]
        top = top[np.argsort(-sims[top], kind="stable")]
        return [(int(ids[i]), float(sims[i])) for i in top]

    def keyword_search(
        self, text: str, n: int, active_only: bool = True, kinds: tuple[str, ...] = ()
    ) -> list[tuple[int, float]]:
        """Top ``n`` memories by FTS5 BM25. Scores are positive; higher is better."""
        q = fts_query(text)
        if not q or n <= 0:
            return []
        sql = (
            "SELECT m.id, -bm25(memories_fts) AS s FROM memories_fts "
            "JOIN memories m ON m.id = memories_fts.rowid WHERE memories_fts MATCH ?"
        )
        args: list[Any] = [q]
        if active_only:
            sql += " AND m.status = 'active'"
        if kinds:
            sql += f" AND m.kind IN ({','.join('?' * len(kinds))})"
            args.extend(kinds)
        sql += " ORDER BY s DESC LIMIT ?"
        args.append(n)
        return [(int(r[0]), float(r[1])) for r in self._conn.execute(sql, args)]

    def count(self, active_only: bool = False) -> int:
        sql = "SELECT COUNT(*) FROM memories" + (" WHERE status = 'active'" if active_only else "")
        return int(self._conn.execute(sql).fetchone()[0])

    def size_bytes(self) -> int:
        """Bytes used by the database (pages in use, so it also works in memory)."""
        page_count = self._conn.execute("PRAGMA page_count").fetchone()[0]
        page_size = self._conn.execute("PRAGMA page_size").fetchone()[0]
        free = self._conn.execute("PRAGMA freelist_count").fetchone()[0]
        return int((page_count - free) * page_size)
