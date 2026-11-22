# dual-memory

[![test](https://github.com/VinVirtual/dual-memory/actions/workflows/test.yml/badge.svg)](https://github.com/VinVirtual/dual-memory/actions/workflows/test.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Memory for AI agents in two parts: a **working memory** of recent turns that stays inside a token budget, and a **long-term memory** of facts and episodes in one SQLite file, found again by vector, keyword, recency and importance together. When a fact changes, the new one supersedes the old one, and both are kept with their dates and sources.

It comes with a **reproducible benchmark** on synthetic multi-session conversations, run against four baselines. Every number below was produced by the commands in this README on one machine, without an API key.

![Benchmark chart: dual-memory against no memory, full history, vector RAG and summarise-everything, by question type and by context budget](examples/benchmark/minilm/chart.png)

## What it does

- **Working memory**: turns are kept until they pass `working_tokens`; then the oldest are summarised into a short running summary and handed to long-term memory. The newest turn is never dropped.
- **Long-term memory**: facts (`subject`, `attribute`, `value`) and session episodes, each with an embedding, importance, timestamps, a mention count and provenance (session, turn, source text). Stored in SQLite with an FTS5 index.
- **Hybrid retrieval**: cosine similarity + BM25 + recency decay + importance, with the weights exposed and every score part returned.
- **Dedupe and contradictions**: the same fact again reinforces the old row. A new value for a one-value-at-a-time fact (city, job, diet...) supersedes the old row, linked both ways, with the time of the change. A late-arriving older statement never overrides a newer one. `history()` returns every value a fact has had.
- **Prompt context**: `context()` writes a dated text block from relevant memories only, and earlier values come along marked "no longer true since ...", so a model can answer "where did they live before?". `messages()` passes memory as user data in `<memory>` tags, never as system instructions.
- **Forgetting**: `forget()` and `forget_source()` delete a memory or everything from one turn, including copies in episodes and summaries, and purge the file.
- **Pluggable parts**: embedder (hashing fallback, or MiniLM via onnxruntime on the CPU), LLM (any OpenAI-compatible endpoint, or a deterministic stub), extractor, summariser, token counter, clock.

## Quick start

```bash
git clone https://github.com/VinVirtual/dual-memory && cd dual-memory
python -m venv .venv && . .venv/bin/activate
pip install -e .            # numpy only; add ".[onnx]" for MiniLM, ".[bench]" for the chart, ".[dev]" for tests
./check_env.sh
```

```python
from dual_memory import DualMemory, RuleAnswerer

day = 86400.0
mem = DualMemory("quickstart.sqlite", user_name="Mira")  # HashingEmbedder + rule extractor by default

mem.add("user", "Hi! I live in Porto and I work as a nurse.", ts=1_736_000_000)
mem.add("assistant", "Nice to meet you, Mira!", ts=1_736_000_030)
mem.end_session()  # facts go to long-term memory, the session becomes an episode

mem.add("user", "Big news: I just moved to Lisbon!", ts=1_736_000_000 + 30 * day)
mem.end_session()  # "Lisbon" supersedes "Porto"; both are kept, linked, with dates

for hit in mem.recall("Where does Mira live?", k=3):
    print(f"{hit.score:.2f}  {hit.item.status:<10} {hit.item.text}")  # score includes recency

context = mem.context("Where does Mira live?")  # dated text block for your prompt
print(context)
reader = RuleAnswerer()  # deterministic stand-in for an LLM
print(reader.answer("Where does Mira live now?", context))  # Lisbon
print(reader.answer("Where did Mira live before?", context))  # Porto
mem.close()
```

This is [`examples/quickstart.py`](examples/quickstart.py). It prints:

```text
0.21  active     Mira lives in Lisbon.
0.19  active     Mira works as a nurse.
0.17  active     [2025-01-04] Mira: I live in Porto and I work as a nurse.
[2025-01-04] Mira lives in Porto. (no longer true since 2025-02-03)
[2025-02-03] Mira lives in Lisbon.
[2025-01-04] Mira works as a nurse.
[2025-01-04] Mira: I live in Porto and I work as a nurse.
[2025-02-03] Mira: Big news: I just moved to Lisbon!
Lisbon
Porto
```

With a real model and real embeddings:

```python
from dual_memory import DualMemory, OnnxEmbedder, OpenAICompatibleLLM

llm = OpenAICompatibleLLM("your-model", base_url="http://localhost:8000/v1", api_key_env="MY_LLM_KEY")
mem = DualMemory("mira.sqlite", user_name="Mira", embedder=OnnxEmbedder(), llm=llm)
messages = mem.messages("What should I cook tonight?") + [{"role": "user", "content": "What should I cook tonight?"}]
```

The key is read from the environment variable you name, at call time. It is never written anywhere.

## How it works

1. **Add**: each turn goes into working memory with its time, session and index.
2. **Overflow or end of session**: the oldest turns are summarised (rule-based extractive summary, or the LLM) and passed to the extractor, which turns them into facts.
3. **Remember**: each fact is reinforced, superseded or inserted (rules in [docs/api.md](docs/api.md)). At the end of a session, a short episode summary is stored too.
4. **Recall**: 50 candidates by cosine and 50 by BM25 are merged and scored; the top `k` are returned with their score parts.
5. **Context**: the top memories, plus earlier values of each fact, are written as dated lines within a token budget.

## The benchmark

Synthetic conversations, generated by a seeded script: **30 fictional personas, 12 sessions each over four to six months, 6,552 turns and 450 questions**. Facts change over time, friends' facts act as distractors, there are negated and hypothetical statements, and each of a persona's own fact statements has a 15% chance of using wording the rule extractor does not know (60 of 485 in the default dataset). Questions ask for the current value of a changed fact, the value *before* it, facts that never changed, a friend's fact, and things never mentioned (the right answer is "unknown").

Each method must fit what it remembers into the same **256-token context**. A deterministic reader then answers from that context and answers are scored by exact match. **This stub mode needs no API key, and it measures whether the right, dated fact reached the context, not how well a model reasons.** What every mode and metric means is in [docs/benchmark.md](docs/benchmark.md).

Headline: seed 7, MiniLM embedder (`all-MiniLM-L6-v2`, ONNX on the CPU, pinned revision and file hashes), **the 256-token column** of the budget sweep. "Vector RAG" is the strongest of four plain-RAG variants: user turns ranked by cosine and added until the budget is full. From [`examples/benchmark/minilm/results.md`](examples/benchmark/minilm/results.md):

| method | accuracy % | 95% CI | current | previous | static | friend | abstain | stale % | answer in context % | context tokens |
|---|---|---|---|---|---|---|---|---|---|---|
| no-memory | 6.7 | 4.7-9.4 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 0 |
| full-history (last 256 tokens) | 10.7 | 8.1-13.9 | 11.1 | 0.0 | 4.4 | 0.0 | 100.0 | 0.0 | 6.7 | 235 |
| summarize-all | 60.2 | 55.6-64.6 | 91.1 | 41.1 | 51.1 | 50.0 | 100.0 | 6.7 | 60.0 | 235 |
| naive RAG (k=8 turns, all roles) | 76.7 | 72.5-80.3 | 71.1 | 43.3 | 84.4 | 100.0 | 100.0 | 16.7 | 86.4 | 150 |
| **vector RAG** (user turns, budget-filled) | **82.4** | 78.7-85.7 | 80.0 | 54.4 | 88.9 | 100.0 | 100.0 | 17.8 | 92.6 | 237 |
| **dual-memory** | **90.4** | 87.4-92.8 | 91.1 | 77.8 | 91.7 | 100.0 | 100.0 | 7.8 | 91.7 | 224 |

> **Read this before quoting the table.**
> - The data generator, the rule extractor and the deterministic reader were written together, by the same author. dual-memory's 98.3% on the 408 questions with known wording is therefore an **upper bound** for rule extraction, not an estimate for real text. All 43 of its misses involve a statement of that fact in wording the extractor skips.
> - The reader always takes the **latest dated line**. That helps every method that keeps dates (both RAGs, the summary, dual-memory), and it is why switching superseding off barely changes accuracy (90.0%).
> - Naive RAG fills only about 150 of the 256 tokens (8 short turns, half of them assistant replies). Ranking user turns and filling the budget lifts plain RAG to **82.4%**; that is the fair comparison, and the gap to dual-memory is **8.0 points**.
> - The remaining gap comes from **slot extraction plus supersede history**: on "before" questions dual-memory gets 77.8% against 54.4% for vector RAG. On current, static and friend questions the gap is 11.1, 2.8 and 0 points.
> - Every method gets the 30 abstain questions right (the reader says "unknown" when it finds nothing), so 6.7 points of every score are free.
> - The 95% intervals are Wilson intervals over 450 questions. The retrieval ablations (vector only, keyword only, facts only) tie with dual-memory at 90.4%, so **this benchmark does not separate retrieval-ranking quality**; it measures what reaches the context.
> - "answer in context" is whether the gold value appears anywhere in the context; it is comparable across methods. recall@8 (in [`results.md`](examples/benchmark/minilm/results.md)) counts only the top 8 retrieved units, which are turns for RAG and memories for dual-memory, so it is not comparable between them.
> - Synthetic data, one dataset family. Seeds 1 and 2 give the same ordering (dual-memory 88.9 / 88.0, vector RAG 80.0 / 78.2, summarize-all 62.4 / 58.2). No LLM reader was run.

More of what the runs show:

- **Budget.** At 128 tokens dual-memory stays at 90.4% while vector RAG falls to 73.8% and the summary to 42.2%. At 1,024 tokens vector RAG reaches 89.3% and the summary 90.4%: once a persona's facts fit, the summary is as good, and much smaller (21 KB for all personas against 3,296 KB).
- **Superseding** keeps the store consistent: for 91.1% of the facts that changed, exactly one active value remains and it is the latest; with superseding off, 10.0%.
- **Episodes are not pulling their weight here.** Retrieving only facts scores 90.4%, only episodes 55.8%. With the hashing embedder (no model), episodes even crowd out facts: dual-memory 87.1% against 90.4% for facts only ([`examples/benchmark/hashing/results.md`](examples/benchmark/hashing/results.md)). Hashing vector RAG gets 60.0%.
- **Storage** across the 30 personas: vector RAG over user turns 8,264 KB, naive RAG 15,132 KB, dual-memory 3,296 KB. **Latency** per question, p50: about 5 ms for vector RAG and 11 ms for dual-memory with MiniLM (one query embedding plus SQLite), on a shared laptop CPU (AMD Ryzen 7 255) with other jobs running.

Reproduce:

```bash
pip install -e ".[onnx,bench]"
dual-memory generate --out data.json                      # --seed 1 / --seed 2 for the other seeds
dual-memory bench --data data.json --out results --embedder onnx --budgets 128,256,512,1024
```

CI re-runs the hashing benchmark on every push and fails if any quality number differs from the committed results; a weekly job does the same for MiniLM.

## Commands

```bash
dual-memory generate --out data.json [--seed 7 --personas 30 --sessions 12 --friends 4 --paraphrase-rate 0.15]
dual-memory bench --data data.json --out results [--embedder hashing|onnx] [--budget 256] [--budgets 128,256,512,1024] [--k 8]
                  [--methods no-memory,full-history,vector-rag,summarize-all,dual-memory,...]
                  [--llm-model NAME --llm-base-url URL --llm-key-env VAR]
dual-memory chart results            # redraw chart.png from a results folder
dual-memory check                    # Python, SQLite FTS5, optional extras
```

## Limits

- **The rule extractor knows a fixed list of English phrasings** ([docs/extraction.md](docs/extraction.md)) and skips any negated or hypothetical sentence. Use an LLM extractor for real conversations.
- **Facts are slots** (subject, attribute, value). Relations between facts, events with durations and multi-hop reasoning are not modelled; episodes hold the rest as text.
- **Vector search is exact** (a NumPy product over a user's active memories). It is fast for tens of thousands of memories per user, not millions; there is no ANN index.
- **One SQLite file per user or agent**, one writer at a time. There is no access control inside a file; keep files per tenant.
- `forget()` and `forget_source()` remove the text from the database file and its write-ahead log (secure delete, FTS index rebuild, WAL checkpoint; a test scans the raw bytes). They cannot reach backups or copies of the file, free blocks the operating system still holds, logs, or text already sent to a model.
- Turns still in working memory are not searched by `recall()` or `context()` until they are evicted or the session ends; `messages()` includes them.
- One `DualMemory` per thread (SQLite's thread check stays on). Several processes may share a file.
- The keyword part of the score is saturated (`bm25 / (bm25 + 2)`) and `context()` drops memories with neither a keyword match nor cosine >= 0.25. Both are defaults chosen before the re-run; they were not tuned on the benchmark (with the hashing embedder a smaller `keyword_half` of 0.5 scored higher, and 2 was kept).
- Token counts are estimated (`approx_tokens`) unless you pass a real tokenizer as `counter`.
- The benchmark is synthetic and its numbers come from a deterministic reader; see the limits in [docs/benchmark.md](docs/benchmark.md). LLM mode is implemented and tested against a local fake server, but no LLM-mode results were run or are reported.
- Tested on Linux with Python 3.10 (and 3.12 in CI).

## Development

```bash
pip install -e ".[dev]"
pytest --cov=dual_memory        # 248 tests; the ONNX tests are skipped without onnxruntime or the model
ruff check src tests && mypy
```

Third-party assets and their licences are listed in [ASSETS.md](ASSETS.md).

## Licence

MIT, see [LICENSE](LICENSE).
