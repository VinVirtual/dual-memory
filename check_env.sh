#!/usr/bin/env bash
# Checks what dual-memory needs and says how to fix anything missing.
set -u
fail=0
PY="${PYTHON:-python3}"
if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; fi

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "python: not found. Install Python 3.10 or newer (https://www.python.org/downloads/)."
  exit 1
fi
if "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "python: $("$PY" -c 'import sys; print(sys.version.split()[0])') ok"
else
  echo "python: 3.10 or newer is needed. Install it, or set PYTHON=/path/to/python3.10."
  fail=1
fi
if "$PY" -c 'import sqlite3; sqlite3.connect(":memory:").execute("create virtual table t using fts5(x)")' 2>/dev/null; then
  echo "sqlite3 with FTS5: ok"
else
  echo "sqlite3 FTS5: missing. Use a Python build with FTS5 (python.org, uv and conda builds have it)."
  fail=1
fi
if "$PY" -c 'import dual_memory' 2>/dev/null; then
  echo "dual_memory: installed"
else
  echo "dual_memory: not installed. Run: $PY -m pip install -e \".[dev]\"  (inside a virtual environment)"
  fail=1
fi
for mod in numpy:core pytest:dev matplotlib:bench onnxruntime:onnx tokenizers:onnx; do
  name="${mod%%:*}"; extra="${mod##*:}"
  if "$PY" -c "import $name" 2>/dev/null; then
    echo "$name: ok"
  elif [ "$extra" = "core" ]; then
    echo "$name: missing. Run: $PY -m pip install -e ."
    fail=1
  else
    echo "$name: not installed (optional). For it: $PY -m pip install -e \".[$extra]\""
  fi
done
if [ $fail -eq 0 ]; then echo "All required parts are present."; fi
exit $fail
