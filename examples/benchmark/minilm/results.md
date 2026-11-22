# Benchmark results

Embedder `all-MiniLM-L6-v2 (ONNX, CPU)`, mode `stub`, answerer `RuleAnswerer`, judge `exact-match`, context budget 256 tokens, k = 8. Dataset: seed 7, 30 personas, 6552 turns, 450 questions.

| method | accuracy % | current | previous | static | friend | abstain | stale % | answer in context % | recall@8 % | retrieve p50 ms | storage KB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| no-memory | 6.7 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | n/a | 0.001 | 0 |
| full-history | 10.7 | 11.1 | 0.0 | 4.4 | 0.0 | 100.0 | 0.0 | 6.7 | n/a | 0.122 | 410 |
| vector-rag | 76.7 | 71.1 | 43.3 | 84.4 | 100.0 | 100.0 | 16.7 | 86.4 | 86.4 | 4.907 | 15132 |
| summarize-all | 60.2 | 91.1 | 41.1 | 51.1 | 50.0 | 100.0 | 6.7 | 60.0 | n/a | 0.066 | 21 |
| dual-memory | 90.4 | 91.1 | 77.8 | 91.7 | 100.0 | 100.0 | 7.8 | 91.9 | 82.1 | 8.798 | 3296 |
| dual: no supersede | 90.0 | 90.0 | 76.7 | 91.7 | 100.0 | 100.0 | 8.9 | 91.9 | 92.1 | 8.45 | 3296 |
| dual: vector only | 90.4 | 91.1 | 77.8 | 91.7 | 100.0 | 100.0 | 7.8 | 93.1 | 82.9 | 8.534 | 3296 |
| dual: keyword only | 90.4 | 91.1 | 77.8 | 91.7 | 100.0 | 100.0 | 7.8 | 91.4 | 89.3 | 8.663 | 3296 |
| dual: no history | 78.0 | 91.1 | 15.6 | 91.7 | 100.0 | 100.0 | 7.8 | 80.0 | 82.1 | 8.68 | 3296 |
