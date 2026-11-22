# Benchmark results

Embedder `all-MiniLM-L6-v2 (ONNX, CPU)`, mode `stub`, answerer `RuleAnswerer`, judge `exact-match`, context budget 512 tokens, k = 8. Dataset: seed 7, 30 personas, 6552 turns, 450 questions.

| method | accuracy % | 95% CI | current | previous | static | friend | abstain | stale % | answer in context % | recall@8 % | context tokens | retrieve p50 ms | storage KB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no-memory | 6.7 | 4.7-9.4 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | n/a | 0 | 0.001 | 0 |
| full-history | 13.8 | 10.9-17.3 | 21.1 | 0.0 | 7.2 | 0.0 | 100.0 | 0.0 | 11.4 | n/a | 478 | 0.184 | 410 |
| naive-rag | 76.7 | 72.5-80.3 | 71.1 | 43.3 | 84.4 | 100.0 | 100.0 | 16.7 | 86.4 | 86.4 | 150 | 5.108 | 15132 |
| rag-user-turns | 78.7 | 74.7-82.2 | 73.3 | 48.9 | 85.6 | 100.0 | 100.0 | 15.6 | 88.1 | 88.1 | 149 | 5.141 | 8264 |
| rag-budget-filled | 86.7 | 83.2-89.5 | 86.7 | 65.6 | 90.6 | 100.0 | 100.0 | 12.2 | 97.4 | 88.1 | 480 | 5.931 | 8264 |
| rag-sentences | 86.7 | 83.2-89.5 | 86.7 | 65.6 | 90.6 | 100.0 | 100.0 | 12.2 | 97.4 | 88.1 | 480 | 6.016 | 8264 |
| summarize-all | 90.4 | 87.4-92.8 | 91.1 | 77.8 | 91.7 | 100.0 | 100.0 | 7.8 | 91.9 | n/a | 479 | 0.125 | 44 |
| dual-memory | 90.4 | 87.4-92.8 | 91.1 | 77.8 | 91.7 | 100.0 | 100.0 | 7.8 | 92.4 | 82.6 | 330 | 10.667 | 3296 |
