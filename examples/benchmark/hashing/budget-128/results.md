# Benchmark results

Embedder `hashing (no model)`, mode `stub`, answerer `RuleAnswerer`, judge `exact-match`, context budget 128 tokens, k = 8. Dataset: seed 7, 30 personas, 6552 turns, 450 questions.

| method | accuracy % | 95% CI | current | previous | static | friend | abstain | stale % | answer in context % | recall@8 % | context tokens | retrieve p50 ms | storage KB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no-memory | 6.7 | 4.7-9.4 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | n/a | 0 | 0.0 | 0 |
| full-history | 9.6 | 7.2-12.6 | 6.7 | 0.0 | 3.9 | 0.0 | 100.0 | 0.0 | 4.0 | n/a | 113 | 0.032 | 410 |
| naive-rag | 48.0 | 43.4-52.6 | 43.3 | 12.2 | 43.9 | 95.0 | 100.0 | 15.6 | 49.5 | 49.5 | 114 | 0.246 | 15132 |
| rag-user-turns | 52.7 | 48.0-57.2 | 44.4 | 13.3 | 53.3 | 98.3 | 100.0 | 18.9 | 55.2 | 55.2 | 114 | 0.313 | 8264 |
| rag-budget-filled | 53.1 | 48.5-57.7 | 44.4 | 15.6 | 53.3 | 98.3 | 100.0 | 18.9 | 55.5 | 55.5 | 117 | 1.132 | 8264 |
| rag-sentences | 53.1 | 48.5-57.7 | 44.4 | 15.6 | 53.3 | 98.3 | 100.0 | 18.9 | 55.5 | 55.5 | 117 | 1.016 | 8264 |
| summarize-all | 42.2 | 37.7-46.8 | 76.7 | 13.3 | 31.1 | 38.3 | 100.0 | 1.1 | 39.3 | n/a | 111 | 0.052 | 10 |
| dual-memory | 83.8 | 80.1-86.9 | 84.4 | 65.6 | 86.7 | 93.3 | 100.0 | 8.9 | 84.5 | 83.6 | 112 | 1.726 | 3296 |
