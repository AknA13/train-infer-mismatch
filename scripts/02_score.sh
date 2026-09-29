#!/bin/bash
# Stage 2: score every view on the corpus (one subprocess per view, resumable).
#   scripts/02_score.sh                      # all models in TIM_MODELS
#   scripts/02_score.sh Qwen/Qwen3-1.7B      # one model
#   TIM_ONLY=vllm_bf16_fa,hf_fp32_eager scripts/02_score.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
need_data
export VLLM_CACHE_ROOT="${TIM_VLLM_CACHE:-$DATA/vllm-cache}"; mkdir -p "$VLLM_CACHE_ROOT"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
MODELS="${1:-${TIM_MODELS:-$MODEL}}"
EXTRA=(); [ -n "${TIM_ONLY:-}" ] && EXTRA+=(--only "$TIM_ONLY"); [ -n "${TIM_SKIP:-}" ] && EXTRA+=(--skip "$TIM_SKIP")
[ -n "${TIM_LIMIT:-}" ] && EXTRA+=(--limit "$TIM_LIMIT")
for m in $MODELS; do
  step "score $m"
  wait_free "${TIM_MIN_FREE_MIB:-60000}"
  run_soft "$PY" -m mismatch.run_views --model "$m" "${EXTRA[@]}"
  step "score $m (batch-invariant corpus)"
  run_soft "$PY" -m mismatch.run_views --model "$m" --variant bi "${EXTRA[@]}"
done
report_soft_fails
