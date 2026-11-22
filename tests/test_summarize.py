from __future__ import annotations

from dual_memory import RuleSummarizer, Turn
from dual_memory.summarize import date_tag
from dual_memory.text import approx_tokens
from tests.conftest import DAY, T0


def test_keeps_fact_sentences_with_date_and_speaker() -> None:
    s = RuleSummarizer("Mira").summarize("", [Turn("user", "Hi! I live in Oslo.", T0)], 100)
    assert s == f"{date_tag(T0)}Mira: I live in Oslo."


def test_ignores_assistant_and_short_chatter() -> None:
    turns = [Turn("assistant", "I live in the cloud, ha."), Turn("user", "ok thanks!")]
    assert RuleSummarizer("Mira").summarize("", turns, 100) == ""


def test_drops_oldest_when_over_budget_and_prefers_facts() -> None:
    rs = RuleSummarizer("Mira")
    prev = rs.summarize("", [Turn("user", "I live in Oslo.", T0)], 100)
    turns = [
        Turn("user", "Documentaries about deep ocean creatures fascinate me greatly.", T0 + DAY),
        Turn("user", "I moved to Riga.", T0 + 2 * DAY),
    ]
    tight = rs.summarize(prev, turns, approx_tokens(f"{date_tag(T0)}Mira: I moved to Riga.") + 1)
    assert "Riga" in tight and "Oslo" not in tight and "ocean" not in tight


def test_zero_budget() -> None:
    assert RuleSummarizer().summarize("x", [Turn("user", "I live in Oslo.")], 0) == ""


def test_date_tag() -> None:
    assert date_tag(None) == "" and date_tag(T0).startswith("[2025-01-")
    assert date_tag(0.0) == "[1970-01-01] "  # 0 is a real time, not "unknown"
