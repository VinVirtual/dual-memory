# Benchmark results

Embedder `all-MiniLM-L6-v2 (ONNX, CPU)`, mode `stub`, answerer `RuleAnswerer`, judge `exact-match`, context budget 128 tokens, k = 8. Dataset: seed 7, 30 personas, 6552 turns, 450 questions.

| method | accuracy % | 95% CI | current | previous | static | friend | abstain | stale % | answer in context % | recall@8 % | context tokens | retrieve p50 ms | storage KB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no-memory | 6.7 | 4.7-9.4 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | n/a | 0 | 0.001 | 0 |
| full-history | 9.6 | 7.2-12.6 | 6.7 | 0.0 | 3.9 | 0.0 | 100.0 | 0.0 | 4.0 | n/a | 113 | 0.035 | 410 |
| naive-rag | 73.1 | 68.8-77.0 | 63.3 | 40.0 | 81.1 | 100.0 | 100.0 | 18.9 | 82.4 | 82.4 | 111 | 4.577 | 15132 |
| rag-user-turns | 73.3 | 69.1-77.2 | 63.3 | 41.1 | 81.1 | 100.0 | 100.0 | 18.9 | 82.9 | 82.9 | 112 | 4.528 | 8264 |
| rag-budget-filled | 73.8 | 69.5-77.6 | 63.3 | 41.1 | 82.2 | 100.0 | 100.0 | 18.9 | 83.6 | 83.6 | 115 | 4.987 | 8264 |
| rag-sentences | 73.8 | 69.5-77.6 | 63.3 | 41.1 | 82.2 | 100.0 | 100.0 | 18.9 | 83.6 | 83.6 | 115 | 5.224 | 8264 |
| summarize-all | 42.2 | 37.7-46.8 | 76.7 | 13.3 | 31.1 | 38.3 | 100.0 | 1.1 | 39.3 | n/a | 111 | 0.034 | 10 |
| dual-memory | 90.4 | 87.4-92.8 | 91.1 | 77.8 | 91.7 | 100.0 | 100.0 | 7.8 | 91.7 | 82.6 | 112 | 9.998 | 3296 |
