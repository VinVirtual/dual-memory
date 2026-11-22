from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from dual_memory import (
    LLMAnswerer,
    LLMError,
    LLMExtractor,
    LLMJudge,
    LLMSummarizer,
    OpenAICompatibleLLM,
    StubLLM,
    Turn,
)
from tests.conftest import T0


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.seen: list[Any] = []

    def complete(self, messages: Any, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
        self.seen.append(messages)
        return self.reply


def test_stub_extract_matches_rule_extractor() -> None:
    facts = LLMExtractor(StubLLM()).extract(
        [
            Turn("user", "I live in Oslo. My friend Ana lives in Riga.", T0, "s1", 2),
            Turn("assistant", "I live in a datacenter.", T0),
        ],
        "Mira",
    )
    assert {(f.subject, f.attribute, f.value) for f in facts} == {
        ("Mira", "city", "Oslo"),
        ("Ana", "city", "Riga"),
    }
    assert all(f.event_time == T0 for f in facts)


def test_stub_summarize_answer_judge() -> None:
    llm = StubLLM()
    s = LLMSummarizer(llm, "Mira").summarize("", [Turn("user", "I work as a nurse.", T0)], 100)
    assert "nurse" in s and s.startswith("[2025-01-")
    assert LLMAnswerer(llm).answer("What does Mira do for work?", s) == "nurse"
    judge = LLMJudge(llm)
    assert judge.judge("q", "Nurse.", "nurse") and not judge.judge("q", "pilot", "nurse")
    assert llm.calls == 4


def test_stub_unknown_task_returns_empty() -> None:
    assert StubLLM().complete([{"role": "user", "content": "hello"}]) == ""


def test_llm_extractor_validates_items() -> None:
    reply = json.dumps(
        [
            {"subject": "Mira", "attribute": "Favorite Drink", "value": "chai", "importance": 9},
            {"subject": "", "attribute": "city", "value": "Oslo"},
            "not an object",
            {"subject": "Mira", "attribute": "city", "value": "x" * 500},
            {
                "subject": "Mira",
                "attribute": "job",
                "value": "nurse",
                "importance": "high",
                "exclusive": False,
            },
        ]
    )
    facts = LLMExtractor(FakeLLM("Sure! " + reply)).extract([Turn("user", "...", T0)], "Mira")
    assert [(f.attribute, f.value) for f in facts] == [("favorite_drink", "chai"), ("job", "nurse")]
    assert facts[0].importance == 1.0 and facts[1].importance == 0.5 and not facts[1].exclusive


@pytest.mark.parametrize("reply", ["no json here", "[not valid json", '{"a": 1}'])
def test_llm_extractor_rejects_bad_output(reply: str) -> None:
    with pytest.raises(LLMError):
        LLMExtractor(FakeLLM(reply)).extract([Turn("user", "hi")], "Mira")


def test_llm_extractor_skips_empty_input() -> None:
    fake = FakeLLM("[]")
    assert LLMExtractor(fake).extract([Turn("user", "  ")], "Mira") == [] and fake.seen == []


def test_llm_answerer_empty_reply_is_unknown() -> None:
    assert LLMAnswerer(FakeLLM("  ")).answer("q", "ctx") == "unknown"


class _Handler(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []
    status = 200
    body: Any = {"choices": [{"message": {"content": "Lisbon"}}]}

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers["Content-Length"])
        type(self).requests.append(
            {
                "path": self.path,
                "auth": self.headers.get("Authorization"),
                "json": json.loads(self.rfile.read(length)),
            }
        )
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(type(self).body).encode())

    def log_message(self, *args: Any) -> None:
        return None


@pytest.fixture
def server() -> Iterator[str]:
    _Handler.requests, _Handler.status = [], 200
    _Handler.body = {"choices": [{"message": {"content": "Lisbon"}}]}
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1"
    httpd.shutdown()


def test_openai_client_sends_request_with_env_key(server: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DM_TEST_KEY", "sk-test-123")
    llm = OpenAICompatibleLLM("tiny-model", server, api_key_env="DM_TEST_KEY")
    out = llm.complete([{"role": "user", "content": "hi"}], max_tokens=7)
    assert out == "Lisbon"
    req = _Handler.requests[0]
    assert req["path"] == "/v1/chat/completions" and req["auth"] == "Bearer sk-test-123"
    assert req["json"]["model"] == "tiny-model" and req["json"]["max_tokens"] == 7


def test_openai_client_without_key_sends_no_auth(server: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DM_TEST_KEY", raising=False)
    OpenAICompatibleLLM("m", server, api_key_env="DM_TEST_KEY").complete([{"role": "user", "content": "x"}])
    assert _Handler.requests[0]["auth"] is None


def test_openai_client_http_error_does_not_leak_key(server: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DM_TEST_KEY", "sk-secret-999")
    _Handler.status = 401
    with pytest.raises(LLMError) as err:
        OpenAICompatibleLLM("m", server, api_key_env="DM_TEST_KEY").complete(
            [{"role": "user", "content": "x"}]
        )
    assert "401" in str(err.value) and "sk-secret" not in str(err.value)


def test_openai_client_bad_shape(server: str) -> None:
    _Handler.body = {"unexpected": True}
    with pytest.raises(LLMError):
        OpenAICompatibleLLM("m", server).complete([{"role": "user", "content": "x"}])


def test_openai_client_unreachable() -> None:
    with pytest.raises(LLMError):
        OpenAICompatibleLLM("m", "http://127.0.0.1:9", timeout=2).complete([{"role": "user", "content": "x"}])


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com", "localhost:8000"])
def test_openai_client_rejects_non_http_urls(url: str) -> None:
    with pytest.raises(ValueError):
        OpenAICompatibleLLM("m", url)
