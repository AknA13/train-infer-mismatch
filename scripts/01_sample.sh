#!/bin/bash
# Stage 1: sample the RL-style corpus with vLLM for every model in TIM_MODELS.
# Writes $DATA/corpus/<tag>/samples.jsonl (+ samples_bi.jsonl: same prompts,
# batch-invariant sampler). Idempotent per file.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
need_data
export VLLM_CACHE_ROOT="${TIM_VLLM_CACHE:-$DATA/vllm-cache}"; mkdir -p "$VLLM_CACHE_ROOT"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
MODELS="${1:-${TIM_MODELS:-$MODEL}}"
for m in $MODELS; do
  step "sample $m"
  wait_free "${TIM_MIN_FREE_MIB:-60000}"
  run "$PY" -m mismatch.sample --model "$m" --n "${TIM_SAMPLE_N:-512}" --max-tokens "${TIM_SAMPLE_MAXTOK:-2048}"
  step "sample $m (batch-invariant sampler)"
  wait_free "${TIM_MIN_FREE_MIB:-60000}"
  VLLM_BATCH_INVARIANT=1 VLLM_ATTENTION_BACKEND=FLASH_ATTN \
    run "$PY" -m mismatch.sample --model "$m" --n "${TIM_SAMPLE_N:-512}" --max-tokens "${TIM_SAMPLE_MAXTOK:-2048}" --variant bi
done
echo "[ok] stage 1 complete"
