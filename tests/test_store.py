from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dual_memory import EmbedderMismatchError, Fact, HashingEmbedder, Provenance, SQLiteStore


def _add(store: SQLiteStore, emb: HashingEmbedder, text: str, **kw: object):
    return store.insert(Fact(text=text, **kw), emb.embed([text])[0], now=100.0)  # type: ignore[arg-type]


def test_insert_and_get_round_trip(store: SQLiteStore, embedder: HashingEmbedder) -> None:
    item = _add(
        store,
        embedder,
        "Mira lives in Oslo.",
        subject="Mira",
        attribute="city",
        value="Oslo",
        provenance=Provenance("s1", 3, "I live in Oslo."),
    )
    got = store.get(item.id)
    assert got == item
    assert got is not None and got.provenance.turn_index == 3 and got.is_active


def test_event_time_defaults_to_now(store: SQLiteStore, embedder: HashingEmbedder) -> None:
    item = _add(store, embedder, "something happened")
    assert item.event_time == 100.0 and item.created_at == 100.0


def test_insert_rejects_wrong_dimension(store: SQLiteStore) -> None:
    with pytest.raises(ValueError):
        store.insert(Fact(text="x"), np.zeros(7, dtype=np.float32))


def test_find_slot_is_case_insensitive(store: SQLiteStore, embedder: HashingEmbedder) -> None:
    _add(store, embedder, "Mira lives in Oslo.", subject="Mira", attribute="city", value="Oslo")
    assert len(store.find_slot("mira", "CITY")) == 1


def test_vector_search_ranks_best_first(store: SQLiteStore, embedder: HashingEmbedder) -> None:
    a = _add(store, embedder, "Mira lives in Lisbon")
    _add(store, embedder, "octopus documentary night")
    hits = store.vector_search(embedder.embed(["Mira lives in Lisbon"])[0], 2)
    assert hits[0][0] == a.id and hits[0][1] > hits[1][1]


def test_vector_search_empty_store(store: SQLiteStore, embedder: HashingEmbedder) -> None:
    assert store.vector_search(embedder.embed(["x"])[0], 5) == []


def test_vector_cache_invalidated_on_write(store: SQLiteStore, embedder: HashingEmbedder) -> None:
    q = embedder.embed(["chess club"])[0]
    assert store.vector_search(q, 5) == []
    _add(store, embedder, "chess club on Tuesdays")
    assert len(store.vector_search(q, 5)) == 1


def test_keyword_search_bm25(store: SQLiteStore, embedder: HashingEmbedder) -> None:
    a = _add(store, embedder, "My partner Robin loves sailing")
    _add(store, embedder, "The weather was strange")
    hits = store.keyword_search("who loves sailing?", 5)
    assert [h[0] for h in hits] == [a.id] and hits[0][1] > 0


def test_keyword_search_stemming(store: SQLiteStore, embedder: HashingEmbedder) -> None:
    a = _add(store, embedder, "She was running every morning")
    assert [h[0] for h in store.keyword_search("runs", 5)] == [a.id]


def test_keyword_search_is_injection_safe(store: SQLiteStore, embedder: HashingEmbedder) -> None:
    _add(store, embedder, "plain text")
    for nasty in ['") OR 1=1 --', "NEAR(a b)", "text*", '"""', "col:text", "'; DROP TABLE memories; --"]:
        store.keyword_search(nasty, 5)  # must not raise
    assert store.count() == 1


def test_superseded_hidden_from_active_search(store: SQLiteStore, embedder: HashingEmbedder) -> None:
    old = _add(store, embedder, "Mira lives in Oslo")
    new = _add(store, embedder, "Mira lives in Riga")
    store.mark_superseded(old.id, new.id, 200.0)
    q = embedder.embed(["Mira lives"])[0]
    assert old.id not in [i for i, _ in store.vector_search(q, 5)]
    assert old.id in [i for i, _ in store.vector_search(q, 5, active_only=False)]
    assert old.id not in [i for i, _ in store.keyword_search("Mira lives", 5)]


def test_reinforce_counts_and_clamps(store: SQLiteStore, embedder: HashingEmbedder) -> None:
    item = _add(store, embedder, "x y z")
    again = store.reinforce(item.id, 500.0, 7.0)
    assert again.mentions == 2 and again.last_seen == 500.0 and again.importance == 1.0


def test_delete_removes_from_index_and_unlinks(store: SQLiteStore, embedder: HashingEmbedder) -> None:
    old = _add(store, embedder, "Mira lives in Oslo")
    new = _add(store, embedder, "Mira lives in Riga")
    store.mark_superseded(old.id, new.id, 200.0)
    assert store.delete(new.id)
    assert store.get(new.id) is None
    remaining = store.get(old.id)
    assert remaining is not None and remaining.superseded_by is None
    assert store.keyword_search("Riga", 5) == []
    assert not store.delete(new.id)


def test_kind_filter(store: SQLiteStore, embedder: HashingEmbedder) -> None:
    _add(store, embedder, "turn text here", kind="turn")
    f = _add(store, embedder, "fact text here")
    q = embedder.embed(["text here"])[0]
    assert [i for i, _ in store.vector_search(q, 5, kinds=("fact",))] == [f.id]


def test_size_and_count(store: SQLiteStore, embedder: HashingEmbedder) -> None:
    before = store.size_bytes()
    for i in range(50):
        _add(store, embedder, f"memory number {i} about something")
    assert store.count() == 50 and store.size_bytes() > before


def test_file_store_persists_and_checks_embedder(tmp_path: Path) -> None:
    path = tmp_path / "m.sqlite"
    emb = HashingEmbedder(dim=64)
    with SQLiteStore(path, embedder_name=emb.name, dim=emb.dim) as s:
        s.insert(Fact(text="kept"), emb.embed(["kept"])[0])
    with SQLiteStore(path, embedder_name=emb.name, dim=emb.dim) as s:
        assert s.count() == 1
    with pytest.raises(EmbedderMismatchError):
        SQLiteStore(path, embedder_name="other", dim=64)
    with pytest.raises(EmbedderMismatchError):
        SQLiteStore(path, embedder_name=emb.name, dim=32)
