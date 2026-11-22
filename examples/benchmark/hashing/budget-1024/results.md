# Benchmark results

Embedder `hashing (no model)`, mode `stub`, answerer `RuleAnswerer`, judge `exact-match`, context budget 1024 tokens, k = 8. Dataset: seed 7, 30 personas, 6552 turns, 450 questions.

| method | accuracy % | 95% CI | current | previous | static | friend | abstain | stale % | answer in context % | recall@8 % | context tokens | retrieve p50 ms | storage KB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no-memory | 6.7 | 4.7-9.4 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | n/a | 0 | 0.0 | 0 |
| full-history | 20.7 | 17.2-24.6 | 37.8 | 5.6 | 13.3 | 0.0 | 100.0 | 0.0 | 21.2 | n/a | 965 | 0.245 | 410 |
| naive-rag | 49.8 | 45.2-54.4 | 43.3 | 13.3 | 47.8 | 95.0 | 100.0 | 15.6 | 51.2 | 51.2 | 141 | 0.337 | 15132 |
| rag-user-turns | 54.7 | 50.0-59.2 | 44.4 | 14.4 | 57.8 | 98.3 | 100.0 | 18.9 | 57.4 | 57.4 | 141 | 0.347 | 8264 |
| rag-budget-filled | 68.2 | 63.8-72.4 | 55.6 | 37.8 | 73.9 | 100.0 | 100.0 | 27.8 | 76.7 | 57.4 | 965 | 1.519 | 8264 |
| rag-sentences | 68.2 | 63.8-72.4 | 55.6 | 37.8 | 73.9 | 100.0 | 100.0 | 27.8 | 76.7 | 57.4 | 965 | 1.163 | 8264 |
| summarize-all | 90.4 | 87.4-92.8 | 91.1 | 77.8 | 91.7 | 100.0 | 100.0 | 7.8 | 92.9 | n/a | 964 | 0.251 | 90 |
| dual-memory | 88.9 | 85.7-91.5 | 88.9 | 75.6 | 90.0 | 100.0 | 100.0 | 7.8 | 91.2 | 83.6 | 417 | 1.74 | 3296 |
