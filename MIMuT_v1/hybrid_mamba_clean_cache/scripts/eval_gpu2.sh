#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  echo "Usage: scripts/eval_gpu2.sh <muscriptor-eval arguments>" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-2}"

exec python -m muscriptor.evaluation.cli "$@"
