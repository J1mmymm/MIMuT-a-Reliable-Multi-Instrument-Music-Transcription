#!/usr/bin/env bash
set -euo pipefail

experiment_config="${1:-configs/large_1_39b.yaml}"

export CUDA_VISIBLE_DEVICES="${THREE_GPU_TRAIN_GPUS:-0,1,2}"

python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=3 \
  -m muscriptor.training.train \
  --config "${experiment_config}"
