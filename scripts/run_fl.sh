#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL="${1:-NNnPH}"
DATASET="${2:-Pbc3}"

case "$MODEL" in
    LinearPH|NNPH|NNnPH)
        MODULE="LinearPH_NNPH_NNnPH.experiments"
        ;;
    AFT_Weibull)
        MODULE="AFT_Weibull.experiments"
        ;;
    DeepHit)
        MODULE="DeepHit.experiments"
        ;;
    DeepPseudo)
        MODULE="DeepPseudo.experiments"
        ;;
    Fed_CoxPH)
        MODULE="Fed_CoxPH.experiments"
        ;;
    FedPDNN|FedPLSTM|FedPAttn)
        MODULE="FedPseudo_Flare.experiments"
        ;;
    *)
        echo "Unsupported model: $MODEL"
        exit 1
        ;;
esac

python -m "$MODULE" \
    --partition FL \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --device cuda:0