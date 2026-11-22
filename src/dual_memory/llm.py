"""Pluggable language models and the LLM-backed extractor, summariser, reader and judge.

- ``OpenAICompatibleLLM`` talks to any ``/v1/chat/completions`` endpoint
  (OpenAI, a local llama.cpp / vLLM / Ollama server, ...). The key is read
  from an environment variable, never from code.
- ``StubLLM`` is deterministic and offline. It recognises the task marker
  in the system prompts below and answers with the rule-based components,
  in the same format a real model is asked for. Tests and the default
  benchmark use it, so every LLM code path runs without a key.
"""

from __future__ import annotations

import calendar
import ipaddress
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from typing import Any, Protocol

from .answer import UNKNOWN, RuleAnswerer, exact_match
from .extract import SLOTS, RuleExtractor, render_fact, split_sentences
from .summarize import RuleSummarizer, date_tag
from .text import escape_tags
from .types import Fact, Provenance, Turn

Message = dict[str, str]


class LLM(Protocol):
    def complete(
        self, messages: Sequence[Message], *, max_tokens: int = 512, temperature: float = 0.0
    ) -> str: ...


class LLMError(RuntimeError):
    """The model endpoint failed or returned something unusable."""


def is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect, so the Authorization header never follows one."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


class OpenAICompatibleLLM:
    """Minimal client for an OpenAI-compatible chat completions endpoint.

    Safety rules for the key: it is read from ``api_key_env`` at call time
    and never stored; it is only sent over HTTPS, or over plain HTTP to a
    loopback host (a local server); redirects are refused, so it never
    follows one to another origin; and requests to a loopback host ignore
    proxy settings.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 60.0,
    ) -> None:
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("base_url must be an http:// or https:// URL with a host")
        self.host = parsed.hostname
        self.scheme = parsed.scheme
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout = timeout

    def complete(
        self, messages: Sequence[Message], *, max_tokens: int = 512, temperature: float = 0.0
    ) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": list(messages),
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        ).encode()
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env, "")
        local = is_loopback(self.host)
        if key:
            if self.scheme != "https" and not local:
                raise LLMError(
                    "refusing to send an API key over plain http to a non-loopback host; use https"
                )
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers=headers)
        handlers: list[urllib.request.BaseHandler] = [_NoRedirect()]
        if local:
            handlers.append(urllib.request.ProxyHandler({}))
        opener = urllib.request.build_opener(*handlers)
        try:
            with opener.open(req, timeout=self.timeout) as resp:
                data: Any = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            raise LLMError(f"model endpoint returned HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMError(f"model endpoint failed: {exc}") from exc
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("unexpected response shape from model endpoint") from exc


# -- prompts -----------------------------------------------------------------
EXTRACT_SYSTEM = (
    "<<task:extract>> You extract durable facts about people from a conversation. "
    'Return only a JSON array. Each item: {"subject": person\'s name, "attribute": '
    "snake_case attribute (e.g. city, job, favorite_drink, hobby, diet, car, pet_name, partner, "
    'allergy), "value": short value, "exclusive": true if a person has one value at a time, '
    '"importance": 0..1}. "I" means the user, whose name is given. Skip negated, '
    "hypothetical and joking statements. Return [] when there is nothing."
)
SUMMARIZE_SYSTEM = (
    "<<task:summarize>> Update the running summary of a conversation with the new turns. "
    "Keep durable facts with their dates, one per line as '[YYYY-MM-DD] Name: statement'. "
    "Stay within the token budget."
)
ANSWER_SYSTEM = (
    "<<task:answer>> Answer the question using only the memory context. Reply with the "
    "answer value only, a few words, no sentence. If the context does not say, reply 'unknown'."
)
JUDGE_SYSTEM = (
    "<<task:judge>> Decide if the predicted answer means the same as the gold answer. "
    "Reply with exactly 'yes' or 'no'."
)


def _section(text: str, name: str) -> str:
    m = re.search(rf"<{name}>\n?(.*?)\n?</{name}>", text, re.S)
    return m.group(1) if m else ""


def _turn_lines(turns: Sequence[Turn], user_name: str) -> str:
    return "\n".join(
        f"{date_tag(t.ts)}{user_name if t.role == 'user' else t.role.capitalize()}: {escape_tags(t.content)}"
        for t in turns
    )


class StubLLM:
    """Deterministic offline model that follows the task markers above."""

    def __init__(self) -> None:
        self.calls = 0
        self._answerer = RuleAnswerer()

    def complete(
        self, messages: Sequence[Message], *, max_tokens: int = 512, temperature: float = 0.0
    ) -> str:
        self.calls += 1
        system = " ".join(m["content"] for m in messages if m["role"] == "system")
        user = "\n".join(m["content"] for m in messages if m["role"] == "user")
        if "<<task:extract>>" in system:
            return self._extract(user)
        if "<<task:summarize>>" in system:
            return self._summarize(user, max_tokens)
        if "<<task:answer>>" in system:
            return self._answerer.answer(_section(user, "question"), _section(user, "context"))
        if "<<task:judge>>" in system:
            same = exact_match(_section(user, "prediction"), _section(user, "gold"))
            return "yes" if same else "no"
        return ""

    @staticmethod
    def _extract(user: str) -> str:
        name = _section(user, "user_name") or "User"
        ex = RuleExtractor()
        out = []
        for line in _section(user, "conversation").splitlines():
            speaker, _, body = line.partition(": ")
            speaker = re.sub(r"^\[[^\]]*\]\s*", "", speaker)
            if speaker != name:
                continue
            triples = [t for sent in split_sentences(body) for t in ex.extract_sentence(sent, name)]
            for subj, attr, val in triples:
                out.append(
                    {
                        "subject": subj,
                        "attribute": attr,
                        "value": val,
                        "exclusive": SLOTS[attr].exclusive if attr in SLOTS else True,
                        "importance": SLOTS[attr].importance if attr in SLOTS else 0.5,
                    }
                )
        return json.dumps(out)

    @staticmethod
    def _summarize(user: str, max_tokens: int) -> str:
        name = _section(user, "user_name") or "User"
        turns = []
        for line in _section(user, "new_turns").splitlines():
            m = re.match(r"^(?:\[(\d{4}-\d{2}-\d{2})\]\s*)?([^:]+): (.*)$", line)
            if not m:
                continue
            ts: float | None = None
            if m.group(1):
                ts = float(calendar.timegm(time.strptime(m.group(1), "%Y-%m-%d")))
            role = "user" if m.group(2) == name else "assistant"
            turns.append(Turn(role, m.group(3), ts))  # type: ignore[arg-type]
        return RuleSummarizer(name).summarize(_section(user, "summary"), turns, max_tokens)


def _parse_json_array(text: str) -> list[Any]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise LLMError("model did not return a JSON array")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        raise LLMError("model returned invalid JSON") from exc
    if not isinstance(data, list):
        raise LLMError("model did not return a JSON array")
    return data


class LLMExtractor:
    """Fact extraction by a language model, validated before use."""

    def __init__(self, llm: LLM, max_facts: int = 50) -> None:
        self.llm = llm
        self.max_facts = max_facts

    def extract(self, turns: Sequence[Turn], user_name: str) -> list[Fact]:
        turns = [t for t in turns if t.content.strip()]
        if not turns:
            return []
        prompt = (
            f"<user_name>{escape_tags(user_name)}</user_name>\n<conversation>\n"
            f"{_turn_lines(turns, user_name)}\n</conversation>"
        )
        raw = self.llm.complete(
            [{"role": "system", "content": EXTRACT_SYSTEM}, {"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        last = turns[-1]
        prov = Provenance(last.session_id, last.index, "\n".join(t.content for t in turns)[:2000])
        facts: list[Fact] = []
        for obj in _parse_json_array(raw)[: self.max_facts]:
            fact = self._to_fact(obj, last.ts, prov)
            if fact is not None:
                facts.append(fact)
        return facts

    @staticmethod
    def _to_fact(obj: Any, ts: float | None, prov: Provenance) -> Fact | None:
        if not isinstance(obj, dict):
            return None
        subj, attr, val = (str(obj.get(k, "")).strip() for k in ("subject", "attribute", "value"))
        attr = re.sub(r"[^a-z0-9_]", "_", attr.lower()).strip("_")
        if not (subj and attr and val) or len(val) > 200 or len(subj) > 80:
            return None
        try:
            importance = min(1.0, max(0.0, float(obj.get("importance", 0.5))))
        except (TypeError, ValueError):
            importance = 0.5
        return Fact(
            text=render_fact(subj, attr, val),
            subject=subj,
            attribute=attr,
            value=val,
            exclusive=bool(obj.get("exclusive", True)),
            importance=importance,
            event_time=ts,
            provenance=prov,
        )


class LLMSummarizer:
    def __init__(self, llm: LLM, user_name: str = "User") -> None:
        self.llm = llm
        self.user_name = user_name

    def summarize(self, previous: str, turns: Sequence[Turn], max_tokens: int) -> str:
        prompt = (
            f"<user_name>{escape_tags(self.user_name)}</user_name>\n<summary>\n{escape_tags(previous)}\n</summary>\n"
            f"<new_turns>\n{_turn_lines(turns, self.user_name)}\n</new_turns>\n"
            f"Token budget: {max_tokens}"
        )
        return self.llm.complete(
            [{"role": "system", "content": SUMMARIZE_SYSTEM}, {"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        ).strip()


class LLMAnswerer:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def answer(self, question: str, context: str) -> str:
        prompt = (
            f"<context>\n{escape_tags(context)}\n</context>\n<question>{escape_tags(question)}</question>"
        )
        out = self.llm.complete(
            [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": prompt}], max_tokens=32
        )
        return out.strip().strip(".") or UNKNOWN


class LLMJudge:
    """Model-judged equivalence. An opinion, not ground truth; report it next to exact match."""

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def judge(self, question: str, prediction: str, gold: str) -> bool:
        prompt = (
            f"<question>{escape_tags(question)}</question>\n<gold>{escape_tags(gold)}</gold>\n"
            f"<prediction>{escape_tags(prediction)}</prediction>"
        )
        out = self.llm.complete(
            [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}], max_tokens=3
        )
        return out.strip().lower().startswith("yes")
