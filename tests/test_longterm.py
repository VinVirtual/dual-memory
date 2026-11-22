from __future__ import annotations

import pytest

from dual_memory import (
    Fact,
    HashingEmbedder,
    LongTermConfig,
    LongTermMemory,
    RetrievalWeights,
    SQLiteStore,
    Turn,
)
from tests.conftest import DAY, T0


def slot(value: str, t: float, attr: str = "city", subject: str = "Mira", exclusive: bool = True) -> Fact:
    return Fact(
        text=f"{subject} {attr} {value}",
        subject=subject,
        attribute=attr,
        value=value,
        exclusive=exclusive,
        event_time=t,
    )


def test_insert_new_fact(ltm: LongTermMemory) -> None:
    r = ltm.remember(slot("Oslo", T0), now=T0)
    assert r.action == "inserted" and r.item.is_active


def test_same_value_reinforces(ltm: LongTermMemory) -> None:
    ltm.remember(slot("Oslo", T0), now=T0)
    r = ltm.remember(slot("oslo.", T0 + DAY), now=T0 + DAY)
    assert r.action == "reinforced" and r.item.mentions == 2
    assert r.item.last_seen == T0 + DAY
    assert ltm.store.count() == 1


def test_new_value_supersedes_with_provenance_chain(ltm: LongTermMemory) -> None:
    first = ltm.remember(slot("Oslo", T0), now=T0).item
    r = ltm.remember(slot("Riga", T0 + 10 * DAY), now=T0 + 10 * DAY)
    assert r.action == "superseded"
    assert r.replaced is not None and r.replaced.id == first.id
    assert r.replaced.status == "superseded" and r.replaced.superseded_by == r.item.id
    assert r.replaced.superseded_at == T0 + 10 * DAY
    assert r.item.supersedes == first.id
    assert [i.value for i in ltm.store.find_slot("Mira", "city")] == ["Riga"]


def test_history_keeps_every_value_in_order(ltm: LongTermMemory) -> None:
    for i, city in enumerate(["Oslo", "Riga", "Lyon"]):
        ltm.remember(slot(city, T0 + i * DAY), now=T0 + i * DAY)
    assert [h.value for h in ltm.history("Mira", "city")] == ["Oslo", "Riga", "Lyon"]
    assert [h.status for h in ltm.history("Mira", "city")] == ["superseded", "superseded", "active"]


def test_returning_to_an_old_value_supersedes_again(ltm: LongTermMemory) -> None:
    ltm.remember(slot("Oslo", T0), now=T0)
    ltm.remember(slot("Riga", T0 + DAY), now=T0 + DAY)
    r = ltm.remember(slot("Oslo", T0 + 2 * DAY), now=T0 + 2 * DAY)
    assert r.action == "superseded"
    assert [i.value for i in ltm.store.find_slot("Mira", "city")] == ["Oslo"]


def test_late_arriving_older_fact_does_not_override(ltm: LongTermMemory) -> None:
    ltm.remember(slot("Riga", T0 + 5 * DAY), now=T0 + 5 * DAY)
    r = ltm.remember(slot("Oslo", T0), now=T0 + 6 * DAY)
    assert r.action == "inserted" and r.item.status == "superseded"
    assert [i.value for i in ltm.store.find_slot("Mira", "city")] == ["Riga"]


def test_non_exclusive_slot_accumulates(ltm: LongTermMemory) -> None:
    ltm.remember(slot("peanuts", T0, "allergy", exclusive=False), now=T0)
    ltm.remember(slot("shellfish", T0 + DAY, "allergy", exclusive=False), now=T0 + DAY)
    assert {i.value for i in ltm.store.find_slot("Mira", "allergy")} == {"peanuts", "shellfish"}


def test_slots_are_per_subject(ltm: LongTermMemory) -> None:
    ltm.remember(slot("Oslo", T0), now=T0)
    ltm.remember(slot("Riga", T0 + DAY, subject="Tomas"), now=T0 + DAY)
    assert [i.value for i in ltm.store.find_slot("Mira", "city")] == ["Oslo"]


def test_supersede_disabled_keeps_both_active(store: SQLiteStore, embedder: HashingEmbedder) -> None:
    ltm = LongTermMemory(store, embedder, config=LongTermConfig(supersede=False))
    ltm.remember(slot("Oslo", T0), now=T0)
    ltm.remember(slot("Riga", T0 + DAY), now=T0 + DAY)
    assert len(store.find_slot("Mira", "city")) == 2


def test_unslotted_near_duplicate_is_reinforced(ltm: LongTermMemory) -> None:
    ltm.remember(Fact(text="Mira prefers window seats on long flights"), now=T0)
    r = ltm.remember(Fact(text="Mira prefers window seats on long flights!"), now=T0 + DAY)
    assert r.action == "reinforced" and ltm.store.count() == 1


def test_unslotted_different_fact_is_inserted(ltm: LongTermMemory) -> None:
    ltm.remember(Fact(text="Mira prefers window seats on long flights"), now=T0)
    r = ltm.remember(Fact(text="Mira collects vintage postcards from Japan"), now=T0)
    assert r.action == "inserted" and ltm.store.count() == 2


def test_ingest_uses_extractor(ltm: LongTermMemory) -> None:
    res = ltm.ingest([Turn("user", "I live in Oslo. I work as a nurse.", T0, "s1", 0)], "Mira", now=T0)
    assert sorted(r.item.attribute for r in res) == ["city", "job"]


def test_search_hybrid_parts_and_order(ltm: LongTermMemory) -> None:
    ltm.ingest(
        [Turn("user", "I live in Oslo.", T0), Turn("user", "I have a cat called Pip.", T0)], "Mira", now=T0
    )
    hits = ltm.search("Where does Mira live?", k=2, now=T0)
    assert hits[0].item.attribute == "city"
    assert hits[0].score >= hits[1].score
    p = hits[0].parts
    assert 0 <= p.vector <= 1 and 0 <= p.keyword <= 1 and 0 < p.recency <= 1


def test_search_excludes_superseded_unless_asked(ltm: LongTermMemory) -> None:
    ltm.remember(slot("Oslo", T0), now=T0)
    ltm.remember(slot("Riga", T0 + DAY), now=T0 + DAY)
    assert {h.item.value for h in ltm.search("Mira city", 5, now=T0)} == {"Riga"}
    assert {h.item.value for h in ltm.search("Mira city", 5, now=T0, include_superseded=True)} == {
        "Oslo",
        "Riga",
    }


def test_recency_decays_with_half_life(ltm: LongTermMemory) -> None:
    ltm.remember(Fact(text="ancient note about kayaks", event_time=T0), now=T0)
    hit = ltm.search("kayaks", 1, now=T0 + 30 * DAY)[0]
    assert hit.parts.recency == pytest.approx(0.5, rel=1e-6)


def test_recency_weight_breaks_ties(store: SQLiteStore, embedder: HashingEmbedder) -> None:
    ltm = LongTermMemory(store, embedder, config=LongTermConfig(weights=RetrievalWeights(0, 0, 1, 0)))
    ltm.remember(Fact(text="note alpha kayaks", event_time=T0), now=T0)
    ltm.remember(Fact(text="note beta kayaks", event_time=T0 + 20 * DAY), now=T0 + 20 * DAY)
    assert ltm.search("kayaks", 1, now=T0 + 21 * DAY)[0].item.text == "note beta kayaks"


def test_search_edge_cases(ltm: LongTermMemory) -> None:
    assert ltm.search("anything", 3) == []
    ltm.remember(Fact(text="x marks"), now=T0)
    assert ltm.search("", 3) == [] and ltm.search("x", 0) == []


def test_forget(ltm: LongTermMemory) -> None:
    item = ltm.remember(Fact(text="secret note"), now=T0).item
    assert ltm.forget(item.id) and ltm.search("secret note", 3, now=T0) == []


@pytest.mark.parametrize(
    "kwargs",
    [{"half_life_days": 0}, {"dedupe_threshold": 0}, {"dedupe_threshold": 1.5}, {"candidates": 0}],
)
def test_config_validation(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        LongTermConfig(**kwargs)  # type: ignore[arg-type]


def test_weights_validation() -> None:
    with pytest.raises(ValueError):
        RetrievalWeights(0, 0, 0, 0)
    with pytest.raises(ValueError):
        RetrievalWeights(-1, 1, 0, 0)


def test_dim_mismatch_rejected(store: SQLiteStore) -> None:
    with pytest.raises(ValueError):
        LongTermMemory(store, HashingEmbedder(dim=64))
