#!/usr/bin/env bash
set -uo pipefail
failed=0
for i in {0..13}; do
  f="./data/test_files/notes${i}.txt"
  [[ -f "$f" ]] || { echo "skip: $f (missing)"; continue; }
  echo "==> $f"
  uv run python -m ingest.cli "$f" || { echo "FAILED: $f"; ((failed++)); }
done
echo "done, $failed failed"
exit $(( failed > 0 ))
