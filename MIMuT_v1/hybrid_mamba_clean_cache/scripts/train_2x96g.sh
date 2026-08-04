#!/usr/bin/env bash
set -euo pipefail

experiment_config="${1:-configs/main_312m.yaml}"

export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:-0,1}"

python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=2 \
  -m muscriptor.training.train \
  --config "${experiment_config}"
