#!/usr/bin/env bash
set -euo pipefail

SUITE="${1:-evals/suites/smoke.json}"
shift || true

python -m evals.runner \
  --suite "$SUITE" \
  --backend "${BLENDER_COPILOT_BACKEND:-http://127.0.0.1:8765}" \
  "$@"
