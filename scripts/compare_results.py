"""Compare two benchmark results.json files on every quality metric (latency is ignored).

Exit 1 when any method's accuracy, per-type accuracy, stale rate, answer-in-context,
recall, storage or slot consistency differs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

KEYS = (
    "accuracy",
    "accuracy_ci95",
    "accuracy_covered",
    "by_type",
    "stale_rate",
    "answer_in_context",
    "recall@1",
    "recall@3",
    "recall@8",
    "slot_consistency",
    "avg_context_tokens",
    "questions",
)


def main(new_path: str, ref_path: str) -> int:
    new = {r["method"]: r for r in json.loads(Path(new_path).read_text())["results"]}
    ref = {r["method"]: r for r in json.loads(Path(ref_path).read_text())["results"]}
    problems = []
    for method, r in ref.items():
        if method not in new:
            problems.append(f"{method}: missing from new results")
            continue
        for key in KEYS:
            a, b = new[method].get(key), r.get(key)
            if a != b and not (isinstance(a, list) and isinstance(b, list) and list(a) == list(b)):
                problems.append(f"{method}.{key}: {a} != reference {b}")
    for line in problems:
        print(line)
    print("results match" if not problems else f"{len(problems)} differences")
    return 1 if problems else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: compare_results.py NEW.json REFERENCE.json")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
