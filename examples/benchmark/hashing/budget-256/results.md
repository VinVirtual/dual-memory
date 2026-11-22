# Benchmark results

Embedder `hashing (no model)`, mode `stub`, answerer `RuleAnswerer`, judge `exact-match`, context budget 256 tokens, k = 8. Dataset: seed 7, 30 personas, 6552 turns, 450 questions.

| method | accuracy % | current | previous | static | friend | abstain | stale % | answer in context % | recall@8 % | retrieve p50 ms | storage KB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| no-memory | 6.7 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | n/a | 0.001 | 0 |
| full-history | 10.7 | 11.1 | 0.0 | 4.4 | 0.0 | 100.0 | 0.0 | 6.7 | n/a | 0.066 | 410 |
| vector-rag | 49.8 | 43.3 | 13.3 | 47.8 | 95.0 | 100.0 | 15.6 | 51.2 | 51.2 | 0.196 | 15132 |
| summarize-all | 60.2 | 91.1 | 41.1 | 51.1 | 50.0 | 100.0 | 6.7 | 60.0 | n/a | 0.104 | 21 |
| dual-memory | 90.4 | 91.1 | 77.8 | 91.7 | 100.0 | 100.0 | 7.8 | 91.4 | 82.1 | 1.357 | 3296 |
| dual: no supersede | 90.0 | 90.0 | 76.7 | 91.7 | 100.0 | 100.0 | 8.9 | 91.4 | 91.9 | 1.526 | 3296 |
| dual: vector only | 74.9 | 73.3 | 57.8 | 80.6 | 73.3 | 100.0 | 7.8 | 76.7 | 79.3 | 1.317 | 3296 |
| dual: keyword only | 90.4 | 91.1 | 77.8 | 91.7 | 100.0 | 100.0 | 7.8 | 91.4 | 89.3 | 1.549 | 3296 |
| dual: no history | 78.2 | 91.1 | 16.7 | 91.7 | 100.0 | 100.0 | 7.8 | 79.5 | 82.1 | 1.443 | 3296 |
