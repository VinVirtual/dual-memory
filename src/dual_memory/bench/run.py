"""Run the memory benchmark and write results (JSON, Markdown table, per-question JSONL)."""

from __future__ import annotations

import json
import math
import platform
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from ..answer import UNKNOWN, Answerer, RuleAnswerer, exact_match
from ..embedders import Embedder
from ..extract import Extractor
from ..llm import LLMJudge
from ..summarize import RuleSummarizer, Summarizer
from ..text import approx_tokens, normalize_value
from ..types import Role, Turn
from .methods import (
    RAG_VARIANTS,
    DualMemoryMethod,
    FullHistory,
    Method,
    NoMemory,
    SummarizeEverything,
    VectorRAG,
    dual_variants,
)

QUESTION_TYPES = ("current", "previous", "static", "friend", "abstain")
BASELINES = ("no-memory", "full-history", *RAG_VARIANTS, "summarize-all")
SWEEP_METHODS = (*BASELINES, "dual-memory")


@dataclass
class BenchConfig:
    budget: int = 384
    k: int = 8
    methods: Sequence[str] = field(default_factory=lambda: [*BASELINES, *dual_variants()])
    embedder_label: str = ""


@dataclass
class Components:
    """The pluggable parts. Defaults are deterministic (stub mode)."""

    embedder: Embedder
    answerer: Answerer = field(default_factory=RuleAnswerer)
    extractor: Extractor | None = None
    summarizer_factory: Callable[[str], Summarizer] = RuleSummarizer
    judge: LLMJudge | None = None
    mode: str = "stub"


def persona_turns(persona: dict[str, Any]) -> list[list[Turn]]:
    sessions = []
    for s in persona["sessions"]:
        sessions.append(
            [
                Turn(cast(Role, t["role"]), t["content"], float(t["ts"]), s["id"], i)
                for i, t in enumerate(s["turns"])
            ]
        )
    return sessions


def make_method(name: str, user: str, cfg: BenchConfig, comp: Components) -> Method:
    if name == "no-memory":
        return NoMemory(user, cfg.budget)
    if name == "full-history":
        return FullHistory(user, cfg.budget)
    if name in RAG_VARIANTS:
        return VectorRAG(user, cfg.budget, comp.embedder, cfg.k, name=name, **RAG_VARIANTS[name])
    if name == "summarize-all":
        return SummarizeEverything(user, cfg.budget, comp.summarizer_factory(user))
    variants = dual_variants()
    if name in variants:
        v = variants[name]
        return DualMemoryMethod(
            user,
            cfg.budget,
            comp.embedder,
            k=cfg.k,
            name=name,
            config=v.config,
            extractor=comp.extractor,
            summarizer=comp.summarizer_factory(user),
            include_history=v.include_history,
            kinds=v.kinds,
        )
    raise ValueError(f"unknown method {name!r}; known: {[*BASELINES, *variants]}")


def _contains(text: str, value: str) -> bool:
    return f" {normalize_value(value)} " in f" {normalize_value(text)} "


def _slot_consistency(method: Method, persona: dict[str, Any]) -> tuple[int, int] | None:
    if not isinstance(method, DualMemoryMethod):
        return None
    name = persona["name"]
    ok = total = 0
    by_slot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in persona["statements"]:
        if s["subject"] == name:
            by_slot[s["attribute"]].append(s)
    for attr, stmts in by_slot.items():
        if len({normalize_value(s["value"]) for s in stmts}) < 2:
            continue
        total += 1
        active = method.mem.store.find_slot(name, attr)
        ok += int(len(active) == 1 and exact_match(active[0].value, stmts[-1]["value"]))
    return ok, total


def run_persona(persona: dict[str, Any], name: str, cfg: BenchConfig, comp: Components) -> dict[str, Any]:
    user = persona["name"]
    method = make_method(name, user, cfg, comp)
    sessions = persona_turns(persona)
    n_turns = sum(len(s) for s in sessions)
    t0 = time.perf_counter()
    for turns in sessions:
        method.ingest_session(turns)
    ingest_s = time.perf_counter() - t0
    now = (sessions[-1][-1].ts or 0.0) + 86400
    old_values = defaultdict(set)
    for s in persona["statements"]:
        old_values[(s["subject"], s["attribute"])].add(normalize_value(s["value"]))
    rows = []
    for q in persona["questions"]:
        t1 = time.perf_counter()
        got = method.retrieve(q["question"], now)
        t2 = time.perf_counter()
        answer = comp.answerer.answer(q["question"], got.context)
        t3 = time.perf_counter()
        gold = q["answer"]
        row = {
            "id": q["id"],
            "method": name,
            "type": q["type"],
            "covered": q["covered"],
            "question": q["question"],
            "gold": gold,
            "answer": answer,
            "correct": exact_match(answer, gold),
            "retrieve_ms": (t2 - t1) * 1000,
            "answer_ms": (t3 - t2) * 1000,
            "context_tokens": approx_tokens(got.context),
        }
        if gold != UNKNOWN:
            row["in_context"] = _contains(got.context, gold)
            if got.units:
                row["recall@1"] = any(_contains(u, gold) for u in got.units[:1])
                row["recall@3"] = any(_contains(u, gold) for u in got.units[:3])
                row[f"recall@{cfg.k}"] = any(_contains(u, gold) for u in got.units[: cfg.k])
        if q["type"] == "current":
            stale = old_values[(q["subject"], q["attribute"])] - {normalize_value(gold)}
            row["stale"] = normalize_value(answer) in stale
        if comp.judge is not None:
            row["judge_correct"] = comp.judge.judge(q["question"], answer, gold)
        rows.append(row)
    result = {
        "rows": rows,
        "ingest_s": ingest_s,
        "turns": n_turns,
        "bytes": method.size_bytes(),
        "consistency": _slot_consistency(method, persona),
    }
    method.close()
    return result


def _mean(xs: Sequence[float]) -> float | None:
    return round(statistics.fmean(xs), 4) if xs else None


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval for a proportion (95% by default)."""
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def _pct(xs: Sequence[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(p * len(s)))], 3)


def summarize_rows(name: str, parts: list[dict[str, Any]], k: int) -> dict[str, Any]:
    rows = [r for p in parts for r in p["rows"]]

    def acc(rs: list[dict[str, Any]]) -> float | None:
        return _mean([float(r["correct"]) for r in rs])

    out: dict[str, Any] = {
        "method": name,
        "questions": len(rows),
        "accuracy": acc(rows),
        "accuracy_ci95": wilson(sum(bool(r["correct"]) for r in rows), len(rows)),
        "accuracy_covered": acc([r for r in rows if r["covered"]]),
        "by_type": {t: acc([r for r in rows if r["type"] == t]) for t in QUESTION_TYPES},
        "stale_rate": _mean([float(r["stale"]) for r in rows if "stale" in r]),
        "answer_in_context": _mean([float(r["in_context"]) for r in rows if "in_context" in r]),
    }
    for key in ("recall@1", "recall@3", f"recall@{k}"):
        vals = [float(r[key]) for r in rows if key in r]
        out[key] = _mean(vals)
    if any("judge_correct" in r for r in rows):
        out["judge_accuracy"] = _mean([float(r["judge_correct"]) for r in rows if "judge_correct" in r])
    lat = [r["retrieve_ms"] for r in rows]
    out["retrieve_ms_p50"], out["retrieve_ms_p95"] = _pct(lat, 0.5), _pct(lat, 0.95)
    turns = sum(p["turns"] for p in parts)
    out["ingest_ms_per_turn"] = round(1000 * sum(p["ingest_s"] for p in parts) / max(turns, 1), 4)
    out["storage_bytes"] = sum(p["bytes"] for p in parts)
    out["avg_context_tokens"] = _mean([float(r["context_tokens"]) for r in rows])
    cons = [p["consistency"] for p in parts if p["consistency"] is not None]
    if cons:
        total = sum(c[1] for c in cons)
        out["slot_consistency"] = round(sum(c[0] for c in cons) / total, 4) if total else None
    return out


def run_benchmark(
    dataset: dict[str, Any], cfg: BenchConfig, comp: Components, progress: Callable[[str], None] | None = None
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for name in cfg.methods:
        t0 = time.perf_counter()
        parts = [run_persona(p, name, cfg, comp) for p in dataset["personas"]]
        summaries.append(summarize_rows(name, parts, cfg.k))
        all_rows.extend(r for p in parts for r in p["rows"])
        if progress:
            progress(f"{name}: accuracy {summaries[-1]['accuracy']} ({time.perf_counter() - t0:.1f}s)")
    return {
        "config": {
            "embedder_revision": getattr(comp.embedder, "revision", None),
            "embedder_files_sha256": getattr(comp.embedder, "files_sha256", None),
            "budget_tokens": cfg.budget,
            "k": cfg.k,
            "embedder": comp.embedder.name,
            "embedder_label": cfg.embedder_label or comp.embedder.name,
            "mode": comp.mode,
            "answerer": type(comp.answerer).__name__,
            "judge": "llm" if comp.judge else "exact-match",
        },
        "dataset": {
            "seed": dataset.get("seed"),
            "personas": len(dataset["personas"]),
            "questions": sum(len(p["questions"]) for p in dataset["personas"]),
            "turns": sum(len(s["turns"]) for p in dataset["personas"] for s in p["sessions"]),
            "paraphrase_rate": dataset.get("paraphrase_rate"),
        },
        "machine": {"python": platform.python_version(), "platform": platform.platform(terse=True)},
        "results": summaries,
        "rows": all_rows,
    }


def _fmt(v: Any, pct: bool = True) -> str:
    if v is None:
        return "n/a"
    return f"{100 * v:.1f}" if pct else f"{v}"


def _ci(r: dict[str, Any]) -> str:
    ci = r.get("accuracy_ci95")
    return f"{100 * ci[0]:.1f}-{100 * ci[1]:.1f}" if ci else "n/a"


def markdown_table(result: dict[str, Any]) -> str:
    k = result["config"]["k"]
    head = (
        "| method | accuracy % | 95% CI | current | previous | static | friend | abstain | stale % "
        f"| answer in context % | recall@{k} % | context tokens | retrieve p50 ms | storage KB |"
    )
    lines = [head, "|" + "---|" * 14]
    for r in result["results"]:
        bt = r["by_type"]
        lines.append(
            f"| {r['method']} | {_fmt(r['accuracy'])} | {_ci(r)} | {_fmt(bt['current'])} "
            f"| {_fmt(bt['previous'])} | {_fmt(bt['static'])} | {_fmt(bt['friend'])} | {_fmt(bt['abstain'])} "
            f"| {_fmt(r['stale_rate'])} | {_fmt(r['answer_in_context'])} | {_fmt(r.get(f'recall@{k}'))} "
            f"| {r['avg_context_tokens']:.0f} | {_fmt(r['retrieve_ms_p50'], False)} "
            f"| {r['storage_bytes'] / 1024:.0f} |"
        )
    return "\n".join(lines)


def write_results(result: dict[str, Any], out_dir: str | Path, rows: bool = True) -> dict[str, Path]:
    """Write results.json and results.md, plus questions.jsonl (one line per answer) if ``rows``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": out / "results.json", "md": out / "results.md"}
    summary = {k: v for k, v in result.items() if k != "rows"}
    paths["json"].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if rows:
        paths["rows"] = out / "questions.jsonl"
        with paths["rows"].open("w", encoding="utf-8") as fh:
            for row in result["rows"]:
                clean = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}
                fh.write(json.dumps(clean) + "\n")
    c, d = result["config"], result["dataset"]
    paths["md"].write_text(
        f"# Benchmark results\n\nEmbedder `{c['embedder_label']}`, mode `{c['mode']}`, "
        f"answerer `{c['answerer']}`, judge `{c['judge']}`, context budget {c['budget_tokens']} tokens, "
        f"k = {c['k']}. Dataset: seed {d['seed']}, {d['personas']} personas, {d['turns']} turns, "
        f"{d['questions']} questions.\n\n{markdown_table(result)}\n",
        encoding="utf-8",
    )
    return paths
