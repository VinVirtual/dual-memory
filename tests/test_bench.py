from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from dual_memory import HashingEmbedder
from dual_memory.bench.generate import generate, load, save
from dual_memory.bench.methods import FullHistory, fit, render_turn
from dual_memory.bench.run import (
    BenchConfig,
    Components,
    make_method,
    markdown_table,
    run_benchmark,
    write_results,
)
from dual_memory.extract import RuleExtractor, split_sentences
from dual_memory.types import Turn


@pytest.fixture(scope="module")
def small() -> dict[str, Any]:
    return generate(seed=3, personas=4, sessions=6)


@pytest.fixture(scope="module")
def result(small: dict[str, Any]) -> dict[str, Any]:
    return run_benchmark(small, BenchConfig(budget=256), Components(embedder=HashingEmbedder()))


def test_generate_is_deterministic() -> None:
    assert generate(seed=11, personas=3, sessions=5) == generate(seed=11, personas=3, sessions=5)
    assert generate(seed=11, personas=3, sessions=5) != generate(seed=12, personas=3, sessions=5)


@pytest.mark.parametrize(
    "kwargs",
    [{"personas": 0}, {"personas": 999}, {"sessions": 1}, {"paraphrase_rate": 1.5}, {"n_friends": 50}],
)
def test_generate_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        generate(**kwargs)


def test_dataset_shape(small: dict[str, Any]) -> None:
    p = small["personas"][0]
    assert len(p["sessions"]) == 6
    roles = [t["role"] for s in p["sessions"] for t in s["turns"]]
    assert roles[0::2] == ["user"] * (len(roles) // 2) and roles[1::2] == ["assistant"] * (len(roles) // 2)
    times = [t["ts"] for s in p["sessions"] for t in s["turns"]]
    assert times == sorted(times)
    types = Counter(q["type"] for q in p["questions"])
    assert types["abstain"] == 1 and types["current"] == 3 and types["static"] >= 1


def test_covered_flag_matches_rule_extractor() -> None:
    data = generate(seed=5, personas=8, sessions=8)
    ex = RuleExtractor()
    for p in data["personas"]:
        turns = {(s["id"], i): t for s in p["sessions"] for i, t in enumerate(s["turns"])}
        for st in p["statements"]:
            text = turns[(st["session"], st["turn"])]["content"]
            got = [x for s in split_sentences(text) for x in ex.extract_sentence(s, p["name"])]
            assert ((st["subject"], st["attribute"], st["value"]) in got) == st["covered"], text


def test_distractor_turns_extract_nothing(small: dict[str, Any]) -> None:
    ex = RuleExtractor()
    for p in small["personas"]:
        stated = {(s["session"], s["turn"]) for s in p["statements"]}
        for s in p["sessions"]:
            for i, t in enumerate(s["turns"]):
                if t["role"] == "user" and (s["id"], i) not in stated:
                    assert [
                        x
                        for sent in split_sentences(t["content"])
                        for x in ex.extract_sentence(sent, p["name"])
                    ] == []


def test_gold_answers_are_latest_and_previous(small: dict[str, Any]) -> None:
    for p in small["personas"]:
        for q in p["questions"]:
            if q["type"] == "current":
                stmts = [
                    s
                    for s in p["statements"]
                    if s["subject"] == p["name"] and s["attribute"] == q["attribute"]
                ]
                assert q["answer"] == stmts[-1]["value"]
            if q["type"] == "previous":
                stmts = [
                    s
                    for s in p["statements"]
                    if s["subject"] == p["name"] and s["attribute"] == q["attribute"]
                ]
                assert q["answer"] != stmts[-1]["value"]


def test_save_and_load(tmp_path: Path, small: dict[str, Any]) -> None:
    save(small, tmp_path / "d.json")
    assert load(tmp_path / "d.json") == small
    (tmp_path / "bad.json").write_text("[1, 2]")
    with pytest.raises(ValueError):
        load(tmp_path / "bad.json")


def test_fit_respects_budget() -> None:
    assert fit(["one two", "three four", "five six"], 7) == ["one two", "three four"]
    assert fit(["x"], 0) == []


def test_full_history_keeps_most_recent() -> None:
    fh = FullHistory("Mira", budget=30)
    fh.ingest_session([Turn("user", f"message number {i}", float(i)) for i in range(10)])
    ctx = fh.retrieve("q", 0).context
    assert "message number 9" in ctx and "message number 0" not in ctx


def test_render_turn() -> None:
    assert render_turn(Turn("assistant", "hi", 0), "Mira") == "Assistant: hi"


def test_unknown_method() -> None:
    with pytest.raises(ValueError):
        make_method("telepathy", "Mira", BenchConfig(), Components(embedder=HashingEmbedder()))


def test_result_structure(result: dict[str, Any], small: dict[str, Any]) -> None:
    names = [r["method"] for r in result["results"]]
    assert names[:5] == ["no-memory", "full-history", "vector-rag", "summarize-all", "dual-memory"]
    n_q = sum(len(p["questions"]) for p in small["personas"])
    assert all(r["questions"] == n_q for r in result["results"])
    assert len(result["rows"]) == n_q * len(names)


def test_no_memory_only_gets_abstentions(result: dict[str, Any]) -> None:
    nm = next(r for r in result["results"] if r["method"] == "no-memory")
    assert nm["by_type"]["abstain"] == 1.0 and nm["by_type"]["current"] == 0.0
    assert nm["storage_bytes"] == 0


def test_dual_memory_beats_no_memory_and_tracks_changes(result: dict[str, Any]) -> None:
    res = {r["method"]: r for r in result["results"]}
    assert res["dual-memory"]["accuracy"] > res["no-memory"]["accuracy"]
    assert res["dual-memory"]["slot_consistency"] > res["dual: no supersede"]["slot_consistency"]
    assert res["dual-memory"]["by_type"]["previous"] >= res["dual: no history"]["by_type"]["previous"]


def test_context_never_exceeds_budget(result: dict[str, Any]) -> None:
    assert max(r["context_tokens"] for r in result["rows"]) <= 256


def test_rows_are_consistent(result: dict[str, Any]) -> None:
    from dual_memory.answer import exact_match

    for row in result["rows"]:
        assert row["correct"] == exact_match(row["answer"], row["gold"])
        if row["gold"] == "unknown":
            assert "in_context" not in row


def test_write_results_and_chart(tmp_path: Path, result: dict[str, Any]) -> None:
    pytest.importorskip("matplotlib")
    from dual_memory.bench.chart import render_chart

    paths = write_results(result, tmp_path)
    summary = json.loads(paths["json"].read_text())
    assert "rows" not in summary and summary["config"]["budget_tokens"] == 256
    assert len(paths["rows"].read_text().splitlines()) == len(result["rows"])
    assert "| dual-memory |" in paths["md"].read_text()
    png = render_chart(
        result, [result, {**result, "config": {**result["config"], "budget_tokens": 512}}], tmp_path / "c.png"
    )
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_markdown_table_has_all_methods(result: dict[str, Any]) -> None:
    table = markdown_table(result)
    assert all(f"| {r['method']} |" in table for r in result["results"])


def test_write_results_without_rows(tmp_path: Path, result: dict[str, Any]) -> None:
    paths = write_results(result, tmp_path, rows=False)
    assert "rows" not in paths and not (tmp_path / "questions.jsonl").exists()
