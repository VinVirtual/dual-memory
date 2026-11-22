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
| `vector-rag` | every turn, embedded, in SQLite | the top 8 turns by cosine similarity to the question |
| `summarize-all` | one running summary, updated after each session, at most the budget | the summary |
| `dual-memory` | this library: facts and session episodes in SQLite, with superseding | top 8 by hybrid score, plus each fact's earlier values |
| `dual: no supersede` | as above, but a changed fact is added instead of superseding | same |
| `dual: vector only` | as dual-memory | ranked by cosine only |
| `dual: keyword only` | as dual-memory | ranked by BM25 only |
| `dual: no history` | as dual-memory | earlier values of facts are not added |

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

Both runs: seed 7, 30 personas, 6,552 turns, 450 questions (408 with known
wording), context budget 256 tokens, k = 8, stub mode. Full tables, every
answer (`questions.jsonl`) and the per-budget results are in
`examples/benchmark/minilm/` and `examples/benchmark/hashing/`.

**MiniLM embedder** (`all-MiniLM-L6-v2`, ONNX, CPU):

| method | accuracy % | on known wording % | previous % | stale % | slot consistency % | recall@1 % | recall@3 % | ingest ms/turn | retrieve p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| no-memory | 6.7 | 7.4 | 0.0 | 0.0 | n/a | n/a | n/a | 0.000 | 0.003 |
| full-history | 10.7 | 11.8 | 0.0 | 0.0 | n/a | n/a | n/a | 0.002 | 0.155 |
| vector-rag | 76.7 | 83.1 | 43.3 | 16.7 | n/a | 53.8 | 71.7 | 3.21 | 7.75 |
| summarize-all | 60.2 | 65.4 | 41.1 | 6.7 | n/a | n/a | n/a | 0.050 | 0.109 |
| dual-memory | 90.4 | 98.3 | 77.8 | 7.8 | 91.1 | 74.5 | 77.6 | 1.50 | 11.7 |
| dual: no supersede | 90.0 | 97.8 | 76.7 | 8.9 | 10.0 | 74.5 | 91.7 | 1.37 | 9.27 |
| dual: vector only | 90.4 | 98.3 | 77.8 | 7.8 | 91.1 | 67.4 | 75.2 | 1.41 | 9.87 |
| dual: keyword only | 90.4 | 98.3 | 77.8 | 7.8 | 91.1 | 71.4 | 80.5 | 1.42 | 10.8 |
| dual: no history | 78.0 | 84.6 | 15.6 | 7.8 | 91.1 | 74.5 | 77.6 | 1.45 | 11.0 |

**Hashing embedder** (no model):

| method | accuracy % | on known wording % | previous % | recall@1 % | recall@3 % |
|---|---|---|---|---|---|
| vector-rag | 49.8 | 53.7 | 13.3 | 38.3 | 45.5 |
| summarize-all | 60.2 | 65.4 | 41.1 | n/a | n/a |
| dual-memory | 90.4 | 98.3 | 77.8 | 70.0 | 76.4 |
| dual: vector only | 74.9 | 81.1 | 57.8 | 47.4 | 62.9 |
| dual: keyword only | 90.4 | 98.3 | 77.8 | 71.4 | 80.5 |

**Accuracy (%) against the context budget** (MiniLM; the hashing run gives
the same figures except vector RAG at 48.0 / 49.8 / 49.8 / 49.8 and
dual-memory at 89.3 for 128 tokens):

| budget (tokens) | full-history | vector-rag | summarize-all | dual-memory |
|---|---|---|---|---|
| 128 | 9.6 | 73.1 | 42.2 | 90.4 |
| 256 | 10.7 | 76.7 | 60.2 | 90.4 |
| 512 | 13.8 | 76.7 | 90.4 | 90.4 |
| 1024 | 20.7 | 76.7 | 90.4 | 90.4 |

Reading these honestly:

- The 9.6-point gap between all questions and known wording is the
  extractor: a paraphrased statement is never stored as a fact, and the
  deterministic reader cannot read it either.
- `stale %` for dual-memory (7.8) comes from changes stated in unknown
  wording: the store never learned the new value, so it answered the old one.
- recall@k counts only the top-k memories, not the earlier values that
  `context()` adds, which is why dual-memory's recall@8 (82.1) is below its
  answer-in-context rate (91.9). With superseding off, old values stay
  active and rank in the top k themselves (recall@3 91.7).
- Once a persona's facts fit in the summary budget (512 tokens and up),
  summarize-all equals dual-memory. The summary's advantage is size (21 KB
  for all personas); its weakness is that it has nowhere to put history
  once the budget is tight.

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
