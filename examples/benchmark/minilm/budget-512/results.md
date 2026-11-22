# Benchmark results

Embedder `all-MiniLM-L6-v2 (ONNX, CPU)`, mode `stub`, answerer `RuleAnswerer`, judge `exact-match`, context budget 512 tokens, k = 8. Dataset: seed 7, 30 personas, 6552 turns, 450 questions.

| method | accuracy % | current | previous | static | friend | abstain | stale % | answer in context % | recall@8 % | retrieve p50 ms | storage KB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| no-memory | 6.7 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | n/a | 0.001 | 0 |
| full-history | 13.8 | 21.1 | 0.0 | 7.2 | 0.0 | 100.0 | 0.0 | 11.4 | n/a | 0.123 | 410 |
| vector-rag | 76.7 | 71.1 | 43.3 | 84.4 | 100.0 | 100.0 | 16.7 | 86.4 | 86.4 | 4.038 | 15132 |
| summarize-all | 90.4 | 91.1 | 77.8 | 91.7 | 100.0 | 100.0 | 7.8 | 91.9 | n/a | 0.118 | 44 |
| dual-memory | 90.4 | 91.1 | 77.8 | 91.7 | 100.0 | 100.0 | 7.8 | 92.6 | 82.1 | 9.133 | 3296 |
