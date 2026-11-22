from __future__ import annotations

from collections.abc import Sequence

import pytest

from dual_memory import RuleSummarizer, Turn, WorkingMemory


def make(
    max_tokens: int = 60, summary_tokens: int = 20, **kw: object
) -> tuple[WorkingMemory, list[list[Turn]]]:
    evicted: list[list[Turn]] = []

    def hook(turns: Sequence[Turn]) -> None:
        evicted.append(list(turns))

    wm = WorkingMemory(RuleSummarizer("Mira"), max_tokens, summary_tokens, on_evict=hook, **kw)  # type: ignore[arg-type]
    return wm, evicted


def test_under_budget_keeps_everything() -> None:
    wm, evicted = make()
    wm.add(Turn("user", "hello there"))
    assert len(wm.turns) == 1 and evicted == [] and wm.summary == ""


def test_overflow_evicts_oldest_and_summarises() -> None:
    wm, evicted = make(max_tokens=100, summary_tokens=40)
    wm.add(Turn("user", "I live in Oslo and I love the fjords around the city a lot.", 1.0))
    for i in range(10):
        wm.add(Turn("assistant", f"reply number {i} with a few extra words in it"))
    assert evicted and evicted[0][0].content.startswith("I live in Oslo")
    assert "Oslo" in wm.summary
    assert wm.tokens() <= wm.max_tokens


def test_newest_turn_never_evicted_even_if_huge() -> None:
    wm, _ = make()
    wm.add(Turn("user", "word " * 500))
    assert len(wm.turns) == 1


def test_summary_respects_its_budget() -> None:
    wm, _ = make(max_tokens=80, summary_tokens=25)
    for i in range(30):
        wm.add(Turn("user", f"I live in City{i} with my many plants and books.", float(i)))
    from dual_memory.text import approx_tokens

    assert approx_tokens(wm.summary) <= 25 + 5


def test_flush_hands_all_turns_over_and_keeps_summary() -> None:
    wm, evicted = make(max_tokens=500, summary_tokens=50)
    wm.add(Turn("user", "a"))
    wm.add(Turn("user", "b"))
    wm.summary = "kept"
    assert [t.content for t in wm.flush()] == ["a", "b"]
    assert evicted[-1][0].content == "a" and wm.turns == () and wm.summary == "kept"


def test_messages_include_summary_first() -> None:
    wm, _ = make(max_tokens=500, summary_tokens=50)
    wm.summary = "[2025-01-01] Mira: I live in Oslo."
    wm.add(Turn("user", "hi"))
    msgs = wm.messages()
    assert msgs[0]["role"] == "system" and "Oslo" in msgs[0]["content"]
    assert msgs[1] == {"role": "user", "content": "hi"}


def test_clear() -> None:
    wm, _ = make(max_tokens=500, summary_tokens=50)
    wm.add(Turn("user", "hi"))
    wm.summary = "s"
    wm.clear()
    assert wm.turns == () and wm.summary == "" and wm.tokens() == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_tokens": 0},
        {"summary_tokens": 100, "max_tokens": 100},
        {"target_ratio": 0},
        {"target_ratio": 1.5},
    ],
)
def test_validation(kwargs: dict[str, float]) -> None:
    args = {"max_tokens": 100, "summary_tokens": 10, **kwargs}
    with pytest.raises(ValueError):
        WorkingMemory(RuleSummarizer(), **args)  # type: ignore[arg-type]
