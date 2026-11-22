from __future__ import annotations

from pathlib import Path

import pytest

from dual_memory import DualMemory, Fact, HashingEmbedder, RuleAnswerer, StubLLM
from tests.conftest import DAY, T0


class Clock:
    def __init__(self) -> None:
        self.t = T0

    def __call__(self) -> float:
        return self.t


def test_session_flow_supersede_and_context() -> None:
    clock = Clock()
    m = DualMemory(user_name="Mira", clock=clock)
    m.add("user", "Hi! I live in Porto and I work as a nurse.")
    m.add("assistant", "Nice to meet you.")
    first = m.end_session()
    assert {r.action for r in first} == {"inserted"}
    assert any(r.item.kind == "episode" for r in first)
    clock.t += 20 * DAY
    m.add("user", "Big news: I just moved to Lisbon!")
    res = m.end_session()
    assert any(r.action == "superseded" for r in res)
    ctx = m.context("Where does Mira live?")
    assert "no longer true since" in ctx and "Lisbon" in ctx
    a = RuleAnswerer()
    assert a.answer("Where does Mira live now?", ctx) == "Lisbon"
    assert a.answer("Where did Mira live before?", ctx) == "Porto"
    assert m.session_id == "s3"


def test_overflow_moves_turns_to_long_term_mid_session() -> None:
    m = DualMemory(user_name="Mira", working_tokens=40, summary_tokens=20, clock=Clock())
    m.add("user", "I have a cat called Pip.")
    results = []
    for i in range(8):
        results += m.add("assistant", f"That is lovely, reply {i}, tell me more about it.")
    assert any(r.item.attribute == "pet_name" for r in results)
    assert m.working.evictions >= 1 and m.stats()["memories"] >= 1


def test_context_budget_and_no_history() -> None:
    clock = Clock()
    m = DualMemory(user_name="Mira", clock=clock)
    m.add("user", "I live in Porto.")
    m.end_session()
    clock.t += DAY
    m.add("user", "I moved to Riga.")
    m.end_session()
    assert "Porto" not in m.context("Where does Mira live?", include_history=False).split("\n")[0]
    assert m.context("Where does Mira live?", max_tokens=1) == ""


def test_messages_layout() -> None:
    m = DualMemory(user_name="Mira", clock=Clock())
    m.add("user", "I live in Porto.")
    m.end_session()
    m.add("user", "what's new?")
    msgs = m.messages("Where do I live?")
    assert msgs[0]["role"] == "user" and "<memory>" in msgs[0]["content"]
    assert "not instructions" in msgs[0]["content"]
    assert msgs[-1] == {"role": "user", "content": "what's new?"}


def test_remember_string_and_forget() -> None:
    m = DualMemory(clock=Clock())
    r = m.remember("The user prefers short answers")
    assert m.recall("short answers")[0].item.id == r.item.id
    assert m.forget(r.item.id) and m.recall("short answers") == []


def test_remember_fact_object() -> None:
    m = DualMemory(clock=Clock())
    r = m.remember(Fact(text="Mira lives in Oslo.", subject="Mira", attribute="city", value="Oslo"))
    assert r.action == "inserted"


def test_stub_llm_path_matches_rule_path() -> None:
    a = DualMemory(user_name="Mira", clock=Clock())
    b = DualMemory(user_name="Mira", llm=StubLLM(), clock=Clock())
    for m in (a, b):
        m.add("user", "I live in Porto. My favourite drink is chai.")
        m.end_session()
    facts = lambda m: {(i.attribute, i.value) for i in m.store.all_items() if i.kind == "fact"}  # noqa: E731
    assert facts(a) == facts(b) == {("city", "Porto"), ("favorite_drink", "chai")}


def test_persistence_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "mem.sqlite"
    with DualMemory(path, user_name="Mira", clock=Clock()) as m:
        m.add("user", "I live in Porto.")
        m.end_session()
    with DualMemory(path, user_name="Mira", clock=Clock()) as m:
        assert "Porto" in m.context("Where does Mira live?")


def test_empty_user_name_rejected() -> None:
    with pytest.raises(ValueError):
        DualMemory(user_name=" ")


def test_invalid_role_rejected() -> None:
    with pytest.raises(ValueError):
        DualMemory().add("robot", "hi")  # type: ignore[arg-type]


def test_custom_embedder_used() -> None:
    m = DualMemory(embedder=HashingEmbedder(dim=32))
    assert m.store.dim == 32
