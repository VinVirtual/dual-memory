"""Command line: generate the dataset, run the benchmark, draw the chart, check the setup."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .embedders import Embedder, HashingEmbedder, OnnxEmbedder
from .llm import LLMAnswerer, LLMExtractor, LLMJudge, LLMSummarizer, OpenAICompatibleLLM
from .summarize import Summarizer

log = logging.getLogger("dual_memory")


def _embedder(kind: str, model_dir: str | None) -> tuple[Embedder, str]:
    if kind == "hashing":
        return HashingEmbedder(), "hashing (no model)"
    if kind == "onnx":
        emb = OnnxEmbedder(model_dir=model_dir)
        return emb, "all-MiniLM-L6-v2 (ONNX, CPU)"
    raise SystemExit(f"unknown embedder {kind!r} (use hashing or onnx)")


def cmd_generate(a: argparse.Namespace) -> int:
    from .bench.generate import generate, save

    data = generate(
        seed=a.seed,
        personas=a.personas,
        sessions=a.sessions,
        paraphrase_rate=a.paraphrase_rate,
        n_friends=a.friends,
    )
    save(data, a.out)
    n_q = sum(len(p["questions"]) for p in data["personas"])
    n_t = sum(len(s["turns"]) for p in data["personas"] for s in p["sessions"])
    print(f"wrote {a.out}: {len(data['personas'])} personas, {n_t} turns, {n_q} questions")
    return 0


def _components(a: argparse.Namespace) -> Any:
    from .bench.run import Components

    emb, label = _embedder(a.embedder, a.model_dir)
    if not a.llm_model:
        return Components(embedder=emb), label
    llm = OpenAICompatibleLLM(a.llm_model, a.llm_base_url, a.llm_key_env)

    def summarizer(user: str) -> Summarizer:
        return LLMSummarizer(llm, user)

    comp = Components(
        embedder=emb,
        answerer=LLMAnswerer(llm),
        extractor=LLMExtractor(llm),
        summarizer_factory=summarizer,
        judge=LLMJudge(llm),
        mode=f"llm:{a.llm_model}",
    )
    return comp, label


def cmd_bench(a: argparse.Namespace) -> int:
    from .bench.chart import render_chart
    from .bench.generate import load
    from .bench.run import SWEEP_METHODS, BenchConfig, markdown_table, run_benchmark, write_results

    data = load(a.data)
    comp, label = _components(a)
    methods = a.methods.split(",") if a.methods else None
    budgets = [int(b) for b in a.budgets.split(",")] if a.budgets else [a.budget]
    if a.budget not in budgets:
        budgets.append(a.budget)
    budgets.sort()
    out = Path(a.out)
    runs: dict[int, dict[str, Any]] = {}
    for b in budgets:
        cfg = BenchConfig(budget=b, k=a.k, embedder_label=label)
        if methods:
            cfg.methods = methods
        elif b != a.budget:
            cfg.methods = list(SWEEP_METHODS)
        print(f"budget {b} tokens", flush=True)
        runs[b] = run_benchmark(data, cfg, comp, progress=lambda s: print("  " + s, flush=True))
        write_results(runs[b], out / f"budget-{b}", rows=False)
    main = runs[a.budget]
    write_results(main, out)
    sweep = [runs[b] for b in budgets]
    (out / "sweep.json").write_text(
        json.dumps(
            [
                {
                    "budget_tokens": b,
                    "results": [{k: v for k, v in r.items() if k != "by_type"} for r in runs[b]["results"]],
                }
                for b in budgets
            ],
            indent=2,
        )
        + "\n"
    )
    print()
    print(markdown_table(main))
    if not a.no_chart:
        try:
            path = render_chart(main, sweep, out / "chart.png")
            print(f"\nchart: {path}")
        except ImportError:
            print('\nmatplotlib is not installed; pip install "dual-memory[bench]" for the chart')
    return 0


def cmd_chart(a: argparse.Namespace) -> int:
    from .bench.chart import render_chart

    base = Path(a.results)
    main = json.loads((base / "results.json").read_text())
    sweep = [
        json.loads(p.read_text())
        for p in sorted(base.glob("budget-*/results.json"), key=lambda p: int(p.parent.name.split("-")[1]))
    ]
    print(render_chart(main, sweep, a.out or base / "chart.png"))
    return 0


def cmd_check(a: argparse.Namespace) -> int:
    import sqlite3

    ok = True
    print(f"python {sys.version.split()[0]}: ok" if sys.version_info >= (3, 10) else "python: need 3.10+")
    ok &= sys.version_info >= (3, 10)
    try:
        sqlite3.connect(":memory:").execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        print(f"sqlite {sqlite3.sqlite_version} with FTS5: ok")
    except sqlite3.OperationalError:
        ok = False
        print(
            "sqlite: FTS5 missing. Use a Python whose sqlite3 has FTS5 (python.org, uv and conda builds do)."
        )
    for mod, extra in (("onnxruntime", "onnx"), ("tokenizers", "onnx"), ("matplotlib", "bench")):
        try:
            __import__(mod)
            print(f"{mod}: ok")
        except ImportError:
            print(f'{mod}: not installed (optional) -> pip install "dual-memory[{extra}]"')
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dual-memory", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write the synthetic benchmark dataset")
    g.add_argument("--out", default="bench-data.json")
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--personas", type=int, default=30)
    g.add_argument("--sessions", type=int, default=12)
    g.add_argument("--friends", type=int, default=4)
    g.add_argument("--paraphrase-rate", type=float, default=0.15)
    g.set_defaults(func=cmd_generate)

    b = sub.add_parser("bench", help="run the benchmark")
    b.add_argument("--data", required=True)
    b.add_argument("--out", default="bench-results")
    b.add_argument("--embedder", choices=["hashing", "onnx"], default="hashing")
    b.add_argument("--model-dir", default=None, help="folder with model.onnx and tokenizer.json")
    b.add_argument("--budget", type=int, default=256, help="context budget in tokens")
    b.add_argument("--budgets", default="", help="comma list for a budget sweep, e.g. 128,256,512,1024")
    b.add_argument("--k", type=int, default=8)
    b.add_argument("--methods", default="", help="comma list; default: all")
    b.add_argument("--llm-model", default="", help="use an OpenAI-compatible model instead of the stub")
    b.add_argument("--llm-base-url", default="https://api.openai.com/v1")
    b.add_argument("--llm-key-env", default="OPENAI_API_KEY", help="env var holding the key")
    b.add_argument("--no-chart", action="store_true")
    b.set_defaults(func=cmd_bench)

    c = sub.add_parser("chart", help="redraw the chart from a results folder")
    c.add_argument("results")
    c.add_argument("--out", default=None)
    c.set_defaults(func=cmd_chart)

    k = sub.add_parser("check", help="check Python, SQLite FTS5 and optional extras")
    k.set_defaults(func=cmd_check)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
