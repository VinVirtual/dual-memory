# Benchmark results

Embedder `all-MiniLM-L6-v2 (ONNX, CPU)`, mode `stub`, answerer `RuleAnswerer`, judge `exact-match`, context budget 1024 tokens, k = 8. Dataset: seed 7, 30 personas, 6552 turns, 450 questions.

| method | accuracy % | 95% CI | current | previous | static | friend | abstain | stale % | answer in context % | recall@8 % | context tokens | retrieve p50 ms | storage KB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no-memory | 6.7 | 4.7-9.4 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | n/a | 0 | 0.001 | 0 |
| full-history | 20.7 | 17.2-24.6 | 37.8 | 5.6 | 13.3 | 0.0 | 100.0 | 0.0 | 21.2 | n/a | 965 | 0.331 | 410 |
| naive-rag | 76.7 | 72.5-80.3 | 71.1 | 43.3 | 84.4 | 100.0 | 100.0 | 16.7 | 86.4 | 86.4 | 150 | 4.809 | 15132 |
| rag-user-turns | 78.7 | 74.7-82.2 | 73.3 | 48.9 | 85.6 | 100.0 | 100.0 | 15.6 | 88.1 | 88.1 | 149 | 4.585 | 8264 |
| rag-budget-filled | 89.3 | 86.1-91.9 | 90.0 | 74.4 | 91.1 | 100.0 | 100.0 | 8.9 | 98.8 | 88.1 | 966 | 5.883 | 8264 |
| rag-sentences | 89.3 | 86.1-91.9 | 90.0 | 74.4 | 91.1 | 100.0 | 100.0 | 8.9 | 98.8 | 88.1 | 966 | 6.147 | 8264 |
| summarize-all | 90.4 | 87.4-92.8 | 91.1 | 77.8 | 91.7 | 100.0 | 100.0 | 7.8 | 92.9 | n/a | 964 | 0.321 | 90 |
| dual-memory | 90.4 | 87.4-92.8 | 91.1 | 77.8 | 91.7 | 100.0 | 100.0 | 7.8 | 92.6 | 82.6 | 337 | 11.43 | 3296 |
