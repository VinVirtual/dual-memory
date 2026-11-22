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
| `forget(id, scrub=True)` | Delete one memory and splice it out of its supersede chain. With `scrub`, also blank the copied source text on memories from the same turn, delete episodes of that session that mention it, and drop matching lines from the working-memory summary. Then purge the file (below). |
| `forget_source(session_id, turn_index)` | Erase everything read from one turn: its facts, its session's episode, copies of its text, and the turn itself if it is still in working memory. |
| `recall(query, k=5, include_superseded=False, kinds=(), relevant_only=False)` | Hybrid search of long-term memory; returns `ScoredMemory` with the score and its parts. |
| `context(query, k=8, max_tokens=512, include_history=True, kinds=())` | A dated plain-text block for a prompt, from relevant memories only. Earlier values of retrieved facts are included and marked `(no longer true since YYYY-MM-DD)`. Turns still in working memory are not in it. |
| `messages(query, ...)` | The memory block as a **user** message inside `<memory>` tags (memory is data written by users, never system instructions; tag-like text is escaped), then the working-memory messages (its summary also as user data). |
| `stats()` | Counts, working-memory tokens and database bytes. |

`DualMemory` is a context manager (`with DualMemory(...) as mem:`).

## Long-term memory rules

`LongTermMemory.remember(fact)` does one of three things:

- **reinforced**: same subject, attribute and (normalised) value as an
  active memory, or, for a statement that arrives late, as the value that
  was current at its time. Mentions, last-seen time and importance go up.
- **superseded**: same subject and an exclusive attribute, new value, not
  older than the current one. The old row gets `status="superseded"`,
  `superseded_by` and `superseded_at`; the new row gets `supersedes`. Both
  keep their provenance (session, turn index, source text).
- **inserted**: anything else. A new value older than the current one is
  stored as history and spliced into the chain at its place in time
  (predecessor -> it -> successor); the current value stays current. Facts
  with no slot are deduplicated by cosine similarity (>= `dedupe_threshold`)
  plus word overlap. Subjects and attributes are matched case-folded with
  spaces collapsed. A fact with `event_time=None` takes the time it is
  stored; `0.0` is a real time.

Deleting a memory splices the chain (A -> B -> C becomes A -> C). Deleting
the current value does not revive the older one.

`history(subject, attribute)` returns every value, oldest first.

## Forgetting

`SQLiteStore` runs with `secure_delete` on, so freed pages are zeroed. After
a delete, `purge()` rebuilds and optimises the FTS5 index (old index
segments still hold deleted words) and checkpoints and truncates the
write-ahead log; `purge(vacuum=True)` also rewrites the file. A test reads
the raw bytes of the database and its WAL after `forget()` and finds no
trace of the forgotten text. What this cannot reach: backups and copies of
the file, the operating system's free disk blocks, logs, and any text
already sent to a model.

## Threads and processes

A store (and so a `DualMemory`) belongs to the thread that created it;
SQLite's `check_same_thread` stays on, so using it from another thread
raises an error. Several processes or connections may open the same file:
writes are serialised by SQLite, and the in-memory vector cache is
refreshed when `PRAGMA data_version` shows another connection committed.

## Retrieval score

```
score = w_vector * cosine            (clipped to [0, 1]; vectors are normalised on the way in)
      + w_keyword * bm25 / (bm25 + keyword_half)   (FTS5 BM25, saturated; keyword_half = 2)
      + w_recency * 0.5 ** (age / half_life)
      + w_importance * importance
```

Defaults: `RetrievalWeights(vector=0.55, keyword=0.25, recency=0.10,
importance=0.10)`, `half_life_days=30`, `candidates=50` from each of the
vector and keyword searches. With `relevant_only` (used by `context()`), a
memory must match a query word or have cosine >= `min_cosine` (0.25);
recency and importance alone never put a memory in a prompt. Vector search is exact (a NumPy matrix product
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
It sends the key only over HTTPS, or over plain HTTP to a loopback host;
it refuses redirects (so the key never follows one to another origin); and
requests to a loopback host ignore proxy settings. Text placed inside
prompt sections is tag-escaped.

`OnnxEmbedder()` downloads the default model at a pinned Hub commit and
checks the SHA-256 of `model.onnx` and `tokenizer.json`
(`ModelIntegrityError` on a mismatch). Benchmark results record both.

A SQLite file remembers the embedder name and dimension it was built with
and refuses to open with a different one (`EmbedderMismatchError`).
