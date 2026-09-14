#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-results/sft}
TRAIN_SIZE=${TRAIN_SIZE:-1000}
EVAL_SIZE=${EVAL_SIZE:-100}
EPOCHS=${EPOCHS:-1}
MAX_LENGTH=${MAX_LENGTH:-768}
BATCH_SIZE=${BATCH_SIZE:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-8}
LR=${LR:-5e-5}

common=(
  --output-root "$ROOT"
  --train-size "$TRAIN_SIZE"
  --eval-size "$EVAL_SIZE"
  --epochs "$EPOCHS"
  --max-length "$MAX_LENGTH"
  --batch-size "$BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --grad-accum "$GRAD_ACCUM"
  --lr "$LR"
)

python sft_experiment.py --mode base "${common[@]}"
python sft_experiment.py --mode lora --rank 1 "${common[@]}"
python sft_experiment.py --mode lora --rank 4 "${common[@]}"
python sft_experiment.py --mode lora --rank 16 "${common[@]}"
python sft_experiment.py --mode full "${common[@]}"
python plot_sft.py --root "$ROOT"
