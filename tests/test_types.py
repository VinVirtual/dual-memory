from __future__ import annotations

import pytest

from dual_memory import Fact, Turn


def test_turn_rejects_bad_role_and_content() -> None:
    with pytest.raises(ValueError):
        Turn("bot", "hi")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Turn("user", 42)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs", [{"text": "  "}, {"text": "x", "importance": 1.5}, {"text": "x", "kind": "note"}]
)
def test_fact_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Fact(**kwargs)  # type: ignore[arg-type]


def test_fact_slot_property() -> None:
    assert Fact(text="x", subject="Mira", attribute="city").has_slot
    assert not Fact(text="x", subject="Mira").has_slot


def test_types_are_immutable() -> None:
    t = Turn("user", "hi")
    with pytest.raises(AttributeError):
        t.content = "changed"  # type: ignore[misc]
