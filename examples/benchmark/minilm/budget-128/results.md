# Benchmark results

Embedder `all-MiniLM-L6-v2 (ONNX, CPU)`, mode `stub`, answerer `RuleAnswerer`, judge `exact-match`, context budget 128 tokens, k = 8. Dataset: seed 7, 30 personas, 6552 turns, 450 questions.

| method | accuracy % | current | previous | static | friend | abstain | stale % | answer in context % | recall@8 % | retrieve p50 ms | storage KB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| no-memory | 6.7 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | n/a | 0.001 | 0 |
| full-history | 9.6 | 6.7 | 0.0 | 3.9 | 0.0 | 100.0 | 0.0 | 4.0 | n/a | 0.064 | 410 |
| vector-rag | 73.1 | 63.3 | 40.0 | 81.1 | 100.0 | 100.0 | 18.9 | 82.4 | 82.4 | 4.619 | 15132 |
| summarize-all | 42.2 | 76.7 | 13.3 | 31.1 | 38.3 | 100.0 | 1.1 | 39.3 | n/a | 0.055 | 10 |
| dual-memory | 90.4 | 91.1 | 77.8 | 91.7 | 100.0 | 100.0 | 7.8 | 91.7 | 82.1 | 10.993 | 3296 |
