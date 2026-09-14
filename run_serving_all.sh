#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-results/serving}
QUANTIZATION=${QUANTIZATION:-fp8}
REPEATS=${REPEATS:-3}
PREFILL_BATCH=${PREFILL_BATCH:-2}
DECODE_TOKENS=${DECODE_TOKENS:-256}
GPU_MEM=${GPU_MEM:-0.90}

common=(
  --quantization "$QUANTIZATION"
  --kv-cache-dtype fp8
  --repeats "$REPEATS"
  --prefill-batch "$PREFILL_BATCH"
  --decode-tokens "$DECODE_TOKENS"
  --gpu-memory-utilization "$GPU_MEM"
)

python serving_benchmark.py \
  --model allenai/OLMo-2-0425-1B-Instruct \
  --name dense_1b \
  --output-dir "$ROOT/dense_1b" \
  "${common[@]}"

python serving_benchmark.py \
  --model allenai/OLMoE-1B-7B-0924-Instruct \
  --name moe_1b7b \
  --output-dir "$ROOT/moe_1b7b" \
  "${common[@]}"

python serving_benchmark.py \
  --model allenai/OLMo-2-1124-7B \
  --name dense_7b \
  --output-dir "$ROOT/dense_7b" \
  "${common[@]}"

python plot_serving.py --root "$ROOT"
