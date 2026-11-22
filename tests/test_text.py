from __future__ import annotations

import pytest

from dual_memory.text import approx_tokens, content_words, fts_query, jaccard, normalize_value, words


def test_approx_tokens_counts_words_and_punctuation() -> None:
    assert approx_tokens("I live in Oslo.") == 5


def test_approx_tokens_long_words_cost_more() -> None:
    assert approx_tokens("internationalisation") > approx_tokens("cat")


def test_approx_tokens_empty() -> None:
    assert approx_tokens("") == 0


def test_approx_tokens_never_below_word_count() -> None:
    text = "the quick brown fox jumps over the lazy dog"
    assert approx_tokens(text) >= len(text.split())


def test_words_lowercases_and_keeps_apostrophes() -> None:
    assert words("Mira's Cat, PIP!") == ["mira's", "cat", "pip"]


def test_content_words_drop_stop_words() -> None:
    assert content_words("Where does she live now?") == ["live"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("The Lisbon.", "lisbon"),
        ("  Green   Tea ", "green tea"),
        ("a nurse", "nurse"),
        ("gluten-free", "gluten-free"),
    ],
)
def test_normalize_value(raw: str, expected: str) -> None:
    assert normalize_value(raw) == expected


def test_jaccard_identical_and_disjoint() -> None:
    assert jaccard("green tea please", "green tea please") == 1.0
    assert jaccard("green tea", "black coffee") == 0.0


def test_fts_query_quotes_every_term() -> None:
    assert fts_query("Where does Mira live?") == '"mira" OR "live"'


def test_fts_query_neutralises_operators_and_quotes() -> None:
    q = fts_query('NEAR("x" OR y) AND * col:z "')
    assert '""' not in q
    for term in q.split(" OR "):
        assert term.startswith('"') and term.endswith('"')


def test_fts_query_empty_for_stop_words_only() -> None:
    assert fts_query("what is the") == ""


def test_fts_query_limits_terms() -> None:
    q = fts_query(" ".join(f"word{i}" for i in range(50)), max_terms=5)
    assert q.count(" OR ") == 4
