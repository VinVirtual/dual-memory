# Benchmark results

Embedder `hashing (no model)`, mode `stub`, answerer `RuleAnswerer`, judge `exact-match`, context budget 512 tokens, k = 8. Dataset: seed 7, 30 personas, 6552 turns, 450 questions.

| method | accuracy % | 95% CI | current | previous | static | friend | abstain | stale % | answer in context % | recall@8 % | context tokens | retrieve p50 ms | storage KB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no-memory | 6.7 | 4.7-9.4 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | n/a | 0 | 0.0 | 0 |
| full-history | 13.8 | 10.9-17.3 | 21.1 | 0.0 | 7.2 | 0.0 | 100.0 | 0.0 | 11.4 | n/a | 478 | 0.123 | 410 |
| naive-rag | 49.8 | 45.2-54.4 | 43.3 | 13.3 | 47.8 | 95.0 | 100.0 | 15.6 | 51.2 | 51.2 | 141 | 0.214 | 15132 |
| rag-user-turns | 54.7 | 50.0-59.2 | 44.4 | 14.4 | 57.8 | 98.3 | 100.0 | 18.9 | 57.4 | 57.4 | 141 | 0.27 | 8264 |
| rag-budget-filled | 62.4 | 57.9-66.8 | 47.8 | 24.4 | 70.0 | 100.0 | 100.0 | 25.6 | 69.5 | 57.4 | 480 | 0.951 | 8264 |
| rag-sentences | 62.4 | 57.9-66.8 | 47.8 | 24.4 | 70.0 | 100.0 | 100.0 | 25.6 | 69.5 | 57.4 | 480 | 1.108 | 8264 |
| summarize-all | 90.4 | 87.4-92.8 | 91.1 | 77.8 | 91.7 | 100.0 | 100.0 | 7.8 | 91.9 | n/a | 479 | 0.151 | 44 |
| dual-memory | 88.7 | 85.4-91.3 | 88.9 | 75.6 | 89.4 | 100.0 | 100.0 | 7.8 | 90.0 | 83.6 | 370 | 1.128 | 3296 |
