# The memory benchmark

A reproducible test of what an agent's memory puts in front of the model,
on synthetic conversations that are hard in the ways real ones are: facts
change, old facts linger, friends have facts too, people say what they
*don't* do and what they *might* do, and most turns are chatter.

```bash
dual-memory generate --out data.json                       # seed 7, 30 personas, 12 sessions each
dual-memory bench --data data.json --out results --budgets 128,256,512,1024
dual-memory bench --data data.json --out results-minilm --embedder onnx --budgets 128,256,512,1024
```

Both commands are deterministic in stub mode: the same seed gives the same
dataset, and the same dataset gives the same answers (only the latency
columns vary between runs and machines).

## The data

`dual_memory/bench/generate.py` builds, for each fictional persona:

- **12 sessions** spread over four to six months (4 to 21 days apart), each
  with 5 to 14 user turns (9 on average) and an assistant reply to each.
- **9 facts** stated in the first three sessions: city, job, favourite drink,
  hobby, diet, car, pet's name, partner's name, favourite cuisine.
- **3 facts that change** later, one or two times each, at random sessions.
  Some unchanged facts are restated later on.
- **4 friends or colleagues** with their own city and job, mentioned in passing.
- **Distractors**: negations ("I don't drink espresso anymore"),
  hypotheticals ("I'm thinking about moving to Lyon someday", "If I ever
  quit, I would work as a florist") and general chatter.
- **Paraphrases**: each of a persona's own fact statements has a 15% chance
  (60 of 485 in the default dataset) of using wording the rule extractor
  does not know ("Home is Riga these days"). This models extraction misses.
  Each question records whether its gold statement was one (`covered`).

The default dataset has 30 personas, 6,552 turns and 450 questions.

## The questions

| Type | Example | Gold answer |
|---|---|---|
| `current` | Where does Mira live now? | the latest value of a fact that changed |
| `previous` | Where did Mira live before? | the value just before the latest one |
| `static` | What is the name of Mira's pet? | a fact stated once or twice, never changed |
| `friend` | Where does Mira's friend Kofi live? | a friend's fact (the persona's own value is a distractor) |
| `abstain` | What is Mira's blood type? | `unknown` (never mentioned) |

## The methods

Every method sees the same sessions in order and must build a plain-text
context of at most the **budget** (default 256 tokens) for each question.

| Method | What it keeps | What it gives the reader |
|---|---|---|
| `no-memory` | nothing | an empty context |
| `full-history` | the whole transcript | the most recent turns that fit the budget |
| `naive-rag` | every turn (user and assistant), embedded | the top 8 turns by cosine. Short turns leave much of the budget unused (about 150 of 256 tokens) |
| `rag-user-turns` | user turns only | the top 8 user turns |
| `rag-budget-filled` | user turns only | user turns ranked by cosine, added until the budget is full |
| `rag-sentences` | user turns split into sentences | sentences ranked by cosine, added until the budget is full. Every synthetic user turn is one sentence, so on this data it equals `rag-budget-filled` |
| `summarize-all` | one running summary, updated after each session, at most the budget | the summary |
| `dual-memory` | this library: facts and session episodes in SQLite, with superseding | relevant memories, top 8 by hybrid score, plus each fact's earlier values |
| `dual: no supersede` | a changed fact is added instead of superseding | same |
| `dual: vector only` / `keyword only` | same store | ranked by cosine only / BM25 only |
| `dual: no history` | same store | earlier values of facts are not added |
| `dual: facts only` / `episodes only` | same store | only facts / only session episodes are retrieved |

"vector RAG" in the README and the chart means the strongest RAG variant at
the main budget.

## The reader and the judge (what "stub mode" means)

With no API key the benchmark uses a **deterministic reader**
(`RuleAnswerer`). It parses the question into a slot (who, what attribute,
now or before), reads every dated line of the context with the same rule
extractor, and answers with the latest value (or, for "before", the value
just before the latest). With nothing to go on it answers `unknown`.
Answers are judged by **normalised exact match**.

So in stub mode, **accuracy measures whether the right fact, with its date,
reached the context**. It does not measure how well a language model reads
or reasons. A real model might also guess, or might be misled by stale
lines that the deterministic reader orders by date. The reader and the
extractor share their rules, so a paraphrased statement is invisible to
every method alike; the `covered` subset removes that effect.

With `--llm-model` (any OpenAI-compatible endpoint; the key comes from the
environment variable named by `--llm-key-env`), extraction, summarising and
answering use the model, and an LLM judge is reported next to exact match.
**No LLM-mode numbers are published here: they were not run.**

## The metrics

| Column | Meaning |
|---|---|
| accuracy | share of all questions answered correctly (exact match) |
| current / previous / static / friend / abstain | accuracy per question type |
| stale % | of `current` questions, the share answered with an *older* value of the same fact |
| answer in context % | share of answerable questions whose gold value appears anywhere in the context |
| recall@k % | share of answerable questions whose gold value appears in the top k retrieved units (turns for RAG, memories for dual-memory; history lines are not counted). Only for ranked methods |
| slot consistency | dual-memory only: of the facts that changed, the share whose store holds exactly one active value and it is the latest one |
| retrieve ms p50 / p95 | wall time to build the context for one question |
| ingest ms per turn | wall time to ingest the whole history, divided by the number of turns |
| storage KB | bytes on disk (SQLite pages in use) or of the text kept, summed over personas |

"Appears" means the normalised value occurs as whole words in the
normalised text. It can be a false positive (a friend lives in the same
city), so treat recall as an upper bound.

## Results

Seed 7, 30 personas, 6,552 turns, 450 questions (408 with known wording),
k = 8, stub mode. **The headline is the 256-token budget.** Full tables,
every answer (`questions.jsonl`), the per-budget results and the embedder's
pinned revision and file hashes are in `examples/benchmark/minilm/` and
`examples/benchmark/hashing/`; seeds 1 and 2 are in `examples/benchmark/seeds/`.
Intervals are 95% Wilson intervals over the 450 questions.

**MiniLM embedder, 256 tokens:**

| method | accuracy % | 95% CI | known wording % | previous % | stale % | slot consistency % | answer in context % | context tokens | ingest ms/turn | retrieve p50 / p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| no-memory | 6.7 | 4.7-9.4 | 7.3 | 0.0 | 0.0 | n/a | 0.0 | 0 | 0.000 | 0.001 / 0.001 |
| full-history | 10.7 | 8.1-13.9 | 11.8 | 0.0 | 0.0 | n/a | 6.7 | 235 | 0.002 | 0.13 / 0.14 |
| naive-rag | 76.7 | 72.5-80.3 | 83.1 | 43.3 | 16.7 | n/a | 86.4 | 150 | 3.16 | 4.6 / 6.8 |
| rag-user-turns | 78.7 | 74.7-82.2 | 85.3 | 48.9 | 15.6 | n/a | 88.1 | 149 | 1.48 | 4.4 / 5.9 |
| rag-budget-filled | 82.4 | 78.7-85.7 | 89.5 | 54.4 | 17.8 | n/a | 92.6 | 237 | 1.46 | 5.3 / 6.8 |
| rag-sentences | 82.4 | 78.7-85.7 | 89.5 | 54.4 | 17.8 | n/a | 92.6 | 237 | 1.48 | 5.5 / 7.1 |
| summarize-all | 60.2 | 55.6-64.6 | 65.4 | 41.1 | 6.7 | n/a | 60.0 | 235 | 0.054 | 0.07 / 0.11 |
| dual-memory | 90.4 | 87.4-92.8 | 98.3 | 77.8 | 7.8 | 91.1 | 91.7 | 224 | 2.19 | 11.1 / 16.2 |
| dual: no supersede | 90.0 | 86.9-92.4 | 97.8 | 76.7 | 8.9 | 10.0 | 91.9 | 186 | 2.18 | 11.4 / 17.8 |
| dual: vector only | 90.4 | 87.4-92.8 | 98.3 | 77.8 | 7.8 | 91.1 | 93.1 | 222 | 2.01 | 10.7 / 15.2 |
| dual: keyword only | 90.4 | 87.4-92.8 | 98.3 | 77.8 | 7.8 | 91.1 | 91.4 | 233 | 2.26 | 11.4 / 19.5 |
| dual: no history | 78.7 | 74.7-82.2 | 85.3 | 18.9 | 7.8 | 91.1 | 80.5 | 206 | 2.28 | 11.6 / 20.2 |
| dual: facts only | 90.4 | 87.4-92.8 | 98.3 | 77.8 | 7.8 | 91.1 | 91.4 | 193 | 2.18 | 10.8 / 18.7 |
| dual: episodes only | 55.8 | 51.2-60.3 | 60.8 | 16.7 | 21.1 | 91.1 | 61.9 | 234 | 2.27 | 11.1 / 20.2 |

**Accuracy (%) against the context budget, MiniLM:**

| budget | full-history | naive-rag | rag-user-turns | rag-budget-filled | summarize-all | dual-memory |
|---|---|---|---|---|---|---|
| 128 | 9.6 | 73.1 | 73.3 | 73.8 | 42.2 | 90.4 |
| **256** | 10.7 | 76.7 | 78.7 | 82.4 | 60.2 | 90.4 |
| 512 | 13.8 | 76.7 | 78.7 | 86.7 | 90.4 | 90.4 |
| 1024 | 20.7 | 76.7 | 78.7 | 89.3 | 90.4 | 90.4 |

**Hashing embedder (no model), 256 tokens:** naive-rag 49.8, rag-user-turns
54.7, rag-budget-filled 60.0, summarize-all 60.2, dual-memory 87.1 (83.7-89.9),
dual: vector only 75.1, keyword only 90.4, facts only 90.4, episodes only
53.8, no history 77.6. Sweep for dual-memory: 83.8 / 87.1 / 88.7 / 88.9 at
128 / 256 / 512 / 1024 tokens; summarize-all reaches 90.4 from 512 tokens,
so with this embedder the summary beats dual-memory at large budgets.

**Other seeds, 256 tokens** (same generator, different random draws):

| seed | embedder | dual-memory | rag-budget-filled | naive-rag | summarize-all |
|---|---|---|---|---|---|
| 1 | MiniLM | 88.9 | 80.0 | 72.7 | 62.4 |
| 2 | MiniLM | 88.0 | 78.2 | 73.8 | 58.2 |
| 1 | hashing | 85.8 | 56.0 | 45.6 | 62.4 |
| 2 | hashing | 85.6 | 55.8 | 48.2 | 58.2 |

The ordering of the main methods is the same on all three seeds with MiniLM.

Reading these honestly:

- **What separates the methods is history.** On "before" questions
  dual-memory gets 77.8% and the best plain RAG 54.4%. On current, static
  and friend questions the gap to the best RAG is 11.1, 2.8 and 0 points.
- **Retrieval ranking is not measured.** Vector-only, keyword-only and
  facts-only retrieval all tie with dual-memory under MiniLM: with about 20
  facts per persona, almost any ranking puts the right fact in the top 8.
  Only the weak hashing embedder separates them (vector only 75.1).
- **Episodes do not help here, and can hurt.** Facts-only retrieval is at
  least as good as dual-memory with both embedders, and with hashing it is
  3.3 points better: episodes compete with facts for the top 8.
- The **known-wording gap** (90.4 vs 98.3) is the extractor: a paraphrased
  statement is never stored as a fact, and the reader cannot read it either.
  With MiniLM all 43 dual-memory misses involve such a statement; with the
  hashing embedder 14 of its 58 misses do not (retrieval misses).
- **stale %** for dual-memory (7.8) comes from changes stated in unknown
  wording: the store never learned the new value.
- **answer in context** is comparable across methods (gold value anywhere in
  the context). **recall@k** counts only the top-k units, which are turns for
  RAG and memories for dual-memory, and excludes the history lines that
  `context()` adds, so it is not comparable between the two families.
- Every method answers the 30 abstain questions correctly (6.7 points).
- `rag-sentences` equals `rag-budget-filled` because every synthetic user
  turn is a single sentence.

## Limits

- **Synthetic data.** Templates, fixed attributes and short values. Real
  conversations are messier; expect lower numbers everywhere, and run the
  LLM mode on your own data before trusting any of this.
- **The stub reader is generous to dated contexts.** It always picks the
  latest dated value, so superseding barely changes stub accuracy
  (`dual: no supersede` scores about the same). Its benefit shows in slot
  consistency and in what a real model sees: one current value, older ones
  marked "no longer true".
- **The extractor and the data were written by the same author.** Every
  covered phrasing is extracted and no distractor is, by construction
  (there is a test for that). Accuracy on the covered subset is therefore
  an upper bound for rule extraction, not an estimate for real text.
- `summarize-all` uses an extractive rule summariser. An LLM summariser
  would compress better, and also lose or blur details differently.
- Latency was measured on a shared 16-core CPU machine with other jobs
  running; treat it as a rough scale, not a benchmark of SQLite.
