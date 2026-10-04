#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL="${1:-NNnPH}"
DATASET="${2:-Pbc3}"
RUN_ID="${3:-0}"

python experiments.py \
    --setting FL \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --device cuda:0

		