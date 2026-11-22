# API

Everything below is importable from `dual_memory`. The package is typed
(`py.typed`) and checked with `mypy --strict`.

## DualMemory

```python
DualMemory(
    path=":memory:",            # or a file path; one SQLite file per user/agent
    *,
    user_name="User",           # who "I" is in user turns
    embedder=None,              # default HashingEmbedder(); OnnxEmbedder() for real embeddings
    llm=None,                   # set it to use LLMExtractor + LLMSummarizer
    extractor=None,             # override extraction directly
    summarizer=None,            # override summarising directly
    working_tokens=1024,        # working-memory budget
    summary_tokens=256,         # running-summary budget inside working memory
    episode_tokens=160,         # per-session episode summary (0 = no episodes)
    config=None,                # LongTermConfig
    counter=approx_tokens,      # token counter, e.g. a tiktoken-based callable
    clock=time.time,            # injectable clock (tests, replays)
)
```

| Method | What it does |
|---|---|
| `add(role, content, ts=None)` | Add a turn to working memory. Returns the `RememberResult`s if an overflow pushed turns into long-term memory. |
| `end_session(next_session_id=None)` | Move the rest of working memory to long-term memory, store a session episode, start a new session. |
| `remember(fact_or_text)` | Store a `Fact` (or a plain string) directly. |
| `forget(id)` | Hard-delete one memory (and unlink it from supersede chains). |
| `recall(query, k=5, include_superseded=False)` | Hybrid search; returns `ScoredMemory` with the score and its parts. |
| `context(query, k=8, max_tokens=512, include_history=True)` | A dated plain-text block for a prompt. Earlier values of retrieved facts are included and marked `(no longer true since YYYY-MM-DD)`. |
| `messages(query, ...)` | `[system memory block] + working-memory messages`, ready for a chat API. |
| `stats()` | Counts, working-memory tokens and database bytes. |

`DualMemory` is a context manager (`with DualMemory(...) as mem:`).

## Long-term memory rules

`LongTermMemory.remember(fact)` does one of three things:

- **reinforced**: same subject, attribute and (normalised) value as an
  active memory. Mentions, last-seen time and importance go up.
- **superseded**: same subject and an exclusive attribute, new value, not
  older than the current one. The old row gets `status="superseded"`,
  `superseded_by` and `superseded_at`; the new row gets `supersedes`. Both
  keep their provenance (session, turn index, source text).
- **inserted**: anything else. A fact older than the current value is stored
  as history, already superseded. Facts with no slot are deduplicated by
  cosine similarity (>= `dedupe_threshold`) plus word overlap.

`history(subject, attribute)` returns every value, oldest first.

## Retrieval score

```
score = w_vector * cosine            (clipped to [0, 1])
      + w_keyword * BM25              (FTS5, min-max scaled over the candidates)
      + w_recency * 0.5 ** (age / half_life)
      + w_importance * importance
```

Defaults: `RetrievalWeights(vector=0.55, keyword=0.25, recency=0.10,
importance=0.10)`, `half_life_days=30`, `candidates=50` from each of the
vector and keyword searches. Vector search is exact (a NumPy matrix product
over the active rows, cached between writes). That is fast for tens of
thousands of memories per user; it is not an ANN index.

## Pluggable parts

| Protocol | Built in |
|---|---|
| `Embedder` (`name`, `dim`, `embed(list[str]) -> float32 array`) | `HashingEmbedder`, `OnnxEmbedder` |
| `LLM` (`complete(messages, max_tokens=, temperature=) -> str`) | `OpenAICompatibleLLM`, `StubLLM` |
| `Extractor` (`extract(turns, user_name) -> list[Fact]`) | `RuleExtractor`, `LLMExtractor` |
| `Summarizer` (`summarize(previous, turns, max_tokens) -> str`) | `RuleSummarizer`, `LLMSummarizer` |
| `Answerer` (`answer(question, context) -> str`) | `RuleAnswerer`, `LLMAnswerer` |

`OpenAICompatibleLLM(model, base_url, api_key_env="OPENAI_API_KEY")` reads
the key from that environment variable at call time and never stores it.
Only `http://` and `https://` base URLs are accepted.

A SQLite file remembers the embedder name and dimension it was built with
and refuses to open with a different one (`EmbedderMismatchError`).
