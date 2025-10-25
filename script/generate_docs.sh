#!/usr/bin/env bash
set -euo pipefail
mkdir -p docs
if command -v pdoc >/dev/null 2>&1; then
  PYTHONPATH=src pdoc -o docs txori
else
  echo "pdoc not installed; skipping docs generation" >&2
fi
