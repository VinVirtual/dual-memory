# Benchmark results

Embedder `hashing (no model)`, mode `stub`, answerer `RuleAnswerer`, judge `exact-match`, context budget 1024 tokens, k = 8. Dataset: seed 7, 30 personas, 6552 turns, 450 questions.

| method | accuracy % | current | previous | static | friend | abstain | stale % | answer in context % | recall@8 % | retrieve p50 ms | storage KB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| no-memory | 6.7 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | n/a | 0.001 | 0 |
| full-history | 20.7 | 37.8 | 5.6 | 13.3 | 0.0 | 100.0 | 0.0 | 21.2 | n/a | 0.434 | 410 |
| vector-rag | 49.8 | 43.3 | 13.3 | 47.8 | 95.0 | 100.0 | 15.6 | 51.2 | 51.2 | 0.256 | 15132 |
| summarize-all | 90.4 | 91.1 | 77.8 | 91.7 | 100.0 | 100.0 | 7.8 | 92.9 | n/a | 0.352 | 90 |
| dual-memory | 90.4 | 91.1 | 77.8 | 91.7 | 100.0 | 100.0 | 7.8 | 92.1 | 82.1 | 1.812 | 3296 |
