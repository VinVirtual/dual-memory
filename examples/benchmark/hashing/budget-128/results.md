# Benchmark results

Embedder `hashing (no model)`, mode `stub`, answerer `RuleAnswerer`, judge `exact-match`, context budget 128 tokens, k = 8. Dataset: seed 7, 30 personas, 6552 turns, 450 questions.

| method | accuracy % | current | previous | static | friend | abstain | stale % | answer in context % | recall@8 % | retrieve p50 ms | storage KB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| no-memory | 6.7 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | n/a | 0.001 | 0 |
| full-history | 9.6 | 6.7 | 0.0 | 3.9 | 0.0 | 100.0 | 0.0 | 4.0 | n/a | 0.038 | 410 |
| vector-rag | 48.0 | 43.3 | 12.2 | 43.9 | 95.0 | 100.0 | 15.6 | 49.5 | 49.5 | 0.303 | 15132 |
| summarize-all | 42.2 | 76.7 | 13.3 | 31.1 | 38.3 | 100.0 | 1.1 | 39.3 | n/a | 0.048 | 10 |
| dual-memory | 89.3 | 90.0 | 74.4 | 91.1 | 100.0 | 100.0 | 8.9 | 90.5 | 82.1 | 1.053 | 3296 |
