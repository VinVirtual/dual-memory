from __future__ import annotations

import pytest

from dual_memory import RuleAnswerer, exact_match, parse_question
from dual_memory.answer import read_candidates

CTX = """[2025-01-06] Mira: I live in Porto.
[2025-01-06] Assistant: Oh, I live in Lisbon myself.
[2025-03-01] Mira: Big news: I just moved to Lisbon!
[2025-02-01] Mira: My friend Tomas lives in Oslo.
[2025-04-01] Mira lives in Riga.
[2025-01-10] Mira's favorite drink is green tea.
[2025-01-12] Mira: I'm thinking about moving to Seville someday."""


@pytest.mark.parametrize(
    ("question", "subject", "attribute", "mode"),
    [
        ("Where does Mira live now?", "Mira", "city", "current"),
        ("Where did Mira live before?", "Mira", "city", "previous"),
        ("Where does Mira's friend Tomas live?", "Tomas", "city", "current"),
        ("What does Mira's colleague Ana do for work?", "Ana", "job", "current"),
        ("What did Mira do for work before?", "Mira", "job", "previous"),
        ("What is Mira's favorite drink now?", "Mira", "favorite_drink", "current"),
        ("What was Mira's favorite drink before?", "Mira", "favorite_drink", "previous"),
        ("What diet does Mira follow now?", "Mira", "diet", "current"),
        ("What car did Mira drive before?", "Mira", "car", "previous"),
        ("What is the name of Mira's pet?", "Mira", "pet_name", "current"),
        ("What is Mira's partner's name?", "Mira", "partner", "current"),
        ("What is Mira's main hobby now?", "Mira", "hobby", "current"),
        ("What is Mira's blood type?", "Mira", "blood_type", "current"),
        ("What is Mira's favorite cuisine?", "Mira", "favorite_cuisine", "current"),
    ],
)
def test_parse_question(question: str, subject: str, attribute: str, mode: str) -> None:
    q = parse_question(question)
    assert q is not None and (q.subject, q.attribute, q.mode) == (subject, attribute, mode)


def test_parse_question_unknown() -> None:
    assert parse_question("Tell me a joke") is None


def test_current_is_latest_dated_value() -> None:
    assert RuleAnswerer().answer("Where does Mira live now?", CTX) == "Riga"


def test_previous_is_value_before_latest() -> None:
    assert RuleAnswerer().answer("Where did Mira live before?", CTX) == "Lisbon"


def test_friend_and_assistant_lines() -> None:
    a = RuleAnswerer()
    assert a.answer("Where does Mira's friend Tomas live?", CTX) == "Oslo"
    cands = read_candidates(CTX, "Mira", "city")
    assert "Seville" not in [c.value for c in cands]  # hypothetical
    assert [c.date for c in cands] == sorted(c.date for c in cands)


def test_unknown_when_missing_or_no_previous() -> None:
    a = RuleAnswerer()
    assert a.answer("What is Mira's blood type?", CTX) == "unknown"
    assert a.answer("What was Mira's favorite drink before?", CTX) == "unknown"
    assert a.answer("Where does Mira live now?", "") == "unknown"
    assert a.answer("Tell me a joke", CTX) == "unknown"


def test_undated_lines_use_order() -> None:
    ctx = "Mira lives in Oslo.\nMira lives in Riga."
    assert RuleAnswerer().answer("Where does Mira live?", ctx) == "Riga"


@pytest.mark.parametrize(
    ("a", "b", "same"), [("Lisbon", "lisbon.", True), ("a nurse", "Nurse", True), ("Riga", "Oslo", False)]
)
def test_exact_match(a: str, b: str, same: bool) -> None:
    assert exact_match(a, b) is same
