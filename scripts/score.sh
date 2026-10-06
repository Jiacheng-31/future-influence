#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
METHOD="${1:?usage: score.sh {value-gradient|value-ablation|nll} [score options]}"
shift
case "$METHOD" in
  value-gradient) SCRIPT="$ROOT/token_selection/scoring/offline_token_fast_fix_score.py" ;;
  value-ablation) SCRIPT="$ROOT/token_selection/scoring/offline_token_value_ablation_score.py" ;;
  nll) SCRIPT="$ROOT/token_selection/entropy/offline_token_entropy_score.py" ;;
  *) echo "Unknown scoring method: $METHOD" >&2; exit 2 ;;
esac
cd "$ROOT"
exec "${TOKEN_SELECTION_PYTHON:-python3}" "$SCRIPT" "$@"
