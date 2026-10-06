#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:?usage: train.sh {baseline|selective|think} CONFIG.yaml [key=value ...]}"
CONFIG="${2:?usage: train.sh {baseline|selective|think} CONFIG.yaml [key=value ...]}"
shift 2

PYTHON="${TOKEN_SELECTION_PYTHON:-python3}"
LLAMAFACTORY_SRC="${LLAMAFACTORY_SRC:-$ROOT/../Qwen/LLaMA-Factory/src}"
if [[ ! -d "$LLAMAFACTORY_SRC/llamafactory" ]]; then
  echo "LLaMAFactory source not found: $LLAMAFACTORY_SRC" >&2
  exit 1
fi
if [[ "$CONFIG" != /* ]]; then
  CONFIG="$ROOT/$CONFIG"
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Training config not found: $CONFIG" >&2
  exit 1
fi

case "$MODE" in
  baseline) LAUNCHER="$LLAMAFACTORY_SRC/llamafactory/launcher.py" ;;
  selective) LAUNCHER="$ROOT/token_selection/training/launcher.py" ;;
  think) LAUNCHER="$ROOT/token_selection/training/think_launcher.py" ;;
  *) echo "Unknown training mode: $MODE" >&2; exit 2 ;;
esac

export PYTHONPATH="$LLAMAFACTORY_SRC${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
exec "$PYTHON" -m torch.distributed.run \
  --nnodes "${NNODES:-1}" \
  --node_rank "${NODE_RANK:-0}" \
  --nproc_per_node "${NPROC_PER_NODE:-1}" \
  --master_addr "${MASTER_ADDR:-127.0.0.1}" \
  --master_port "${MASTER_PORT:-29500}" \
  "$LAUNCHER" "$CONFIG" "$@"
